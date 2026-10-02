"""
Extract Whisper hidden-state vectors for each phoneme token.
"""

import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel


MODEL_NAME = "openai/whisper-medium"
LAYERS = [4, 20]

INPUT_PATH = os.path.join("data", "pipeline", "phoneme_tokens.csv")
OUTPUT_PATH = os.path.join("data", "pipeline", "features_whisper.npz")
SPEAKERS_DIR = os.path.join(
    "ru-fr_interference", "2",
    "wav_et_textgrids", "FRcorp_textgrids_only",
)

WHISPER_SR = 16_000
HOP_LENGTH = 160
FRAME_DURATION_S = HOP_LENGTH / WHISPER_SR


def find_wav(speaker_dir: str, sentence_id: int) -> str | None:
    pattern = os.path.join(speaker_dir, f"*FRcorp{sentence_id}.wav")
    matches = glob.glob(pattern)
    return matches[0] if len(matches) == 1 else None


def load_audio(wav_path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != WHISPER_SR:
        waveform = torchaudio.functional.resample(waveform, sr, WHISPER_SR)
    return waveform.squeeze(0)


def frame_times(n_frames: int) -> np.ndarray:
    return np.arange(n_frames) * FRAME_DURATION_S


def overlapping_frames(centres: np.ndarray, onset: float, offset: float) -> np.ndarray:
    mask = (centres >= onset) & (centres <= offset)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        indices = np.array([np.argmin(np.abs(centres - (onset + offset) / 2))])
    return indices


def register_hooks(model: WhisperModel) -> tuple[dict[int, torch.Tensor], list]:
    storage: dict[int, torch.Tensor] = {}
    handles = []
    encoder_layers = model.encoder.layers

    for layer_idx in LAYERS:
        def make_hook(idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                storage[idx] = hidden.detach().cpu()
            return hook

        handle = encoder_layers[layer_idx - 1].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    return storage, handles


def main() -> None:
    print(f"Model : {MODEL_NAME}")
    print(f"Layers: {LAYERS}")

    tokens = pd.read_csv(INPUT_PATH)
    n_tokens = len(tokens)
    print(f"Loaded {n_tokens} phoneme tokens from '{INPUT_PATH}'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    hidden_dim = model.config.d_model
    hook_storage, hook_handles = register_hooks(model)

    hiddens = {
        layer: np.full((n_tokens, hidden_dim), np.nan, dtype=np.float32)
        for layer in LAYERS
    }
    token_index = np.arange(n_tokens, dtype=np.int32)

    groups = defaultdict(list)
    for row_idx, row in tokens.iterrows():
        groups[(row["speaker"], int(row["sentence_id"]))].append(row_idx)

    n_groups = len(groups)
    print(f"Processing {n_groups} unique (speaker, sentence) pairs…")

    for group_idx, ((speaker, sent_id), row_indices) in enumerate(groups.items(), 1):
        speaker_dir = os.path.join(SPEAKERS_DIR, speaker)
        wav_path = find_wav(speaker_dir, sent_id)

        if wav_path is None:
            print(f"  [{group_idx}/{n_groups}] WAV not found for speaker={speaker}, sentence={sent_id}")
            continue

        try:
            audio = load_audio(wav_path)
        except Exception as exc:
            print(f"  [{group_idx}/{n_groups}] could not load {wav_path}: {exc}")
            continue

        inputs = processor(
            audio.numpy(),
            sampling_rate=WHISPER_SR,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)

        hook_storage.clear()
        with torch.no_grad():
            model.encoder(input_features)

        n_frames = hook_storage[LAYERS[0]].shape[1]
        centres = frame_times(n_frames)

        if group_idx % 20 == 0 or group_idx == 1:
            print(f"  [{group_idx}/{n_groups}] speaker={speaker}, sentence={sent_id}, frames={n_frames}")

        for row_idx in row_indices:
            row = tokens.iloc[row_idx]
            frame_idx = overlapping_frames(
                centres,
                float(row["onset"]),
                float(row["offset"]),
            )

            for layer in LAYERS:
                layer_hidden = hook_storage[layer][0]
                hiddens[layer][row_idx] = layer_hidden[frame_idx].mean(dim=0).numpy()

    for handle in hook_handles:
        handle.remove()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    save_dict = {f"hiddens_layer{layer}": hiddens[layer] for layer in LAYERS}
    save_dict["token_index"] = token_index
    save_dict["layers"] = np.array(LAYERS, dtype=np.int32)
    save_dict["model"] = np.bytes_(MODEL_NAME.encode())

    np.savez(OUTPUT_PATH, **save_dict)
    print(f"Saved features for {n_tokens} tokens to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()