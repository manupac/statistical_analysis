"""
Add spectral-centre-of-gravity features to the acoustic table.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth


INPUT_PATH = Path("data/pipeline/features_acoustic.csv")
OUTPUT_PATH = Path("data/pipeline/features_acoustic_extended.csv")
SPEAKERS_DIR = Path("ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only")
MID_WINDOW_SECONDS = 0.025


def find_wav(speaker_dir: Path, sentence_id: int) -> Path:
    matches = glob.glob(str(speaker_dir / f"*FRcorp{sentence_id}.wav"))
    if len(matches) != 1:
        raise FileNotFoundError(f"No unique WAV for {speaker_dir}, sentence {sentence_id}")
    return Path(matches[0])


def safe_extract_part(sound, start, end):
    start = max(float(start), sound.xmin)
    end = min(float(end), sound.xmax)
    if end <= start:
        return None
    return sound.extract_part(from_time=start, to_time=end, preserve_times=False)


def safe_scg(segment) -> float:
    if segment is None:
        return np.nan
    try:
        value = segment.to_spectrum().get_centre_of_gravity(power=2.0)
        return float(value) if value is not None and np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    sound_cache = {}
    scg_mean = []
    scg_mid = []

    for idx, row in df.iterrows():
        speaker = str(row["speaker"])
        sent_id = int(row["sentence_id"])
        onset = float(row["onset"])
        offset = float(row["offset"])
        midpoint = (onset + offset) / 2.0
        half = MID_WINDOW_SECONDS / 2.0

        key = (speaker, sent_id)
        if key not in sound_cache:
            wav_path = find_wav(SPEAKERS_DIR / speaker, sent_id)
            sound_cache[key] = parselmouth.Sound(str(wav_path))

        sound = sound_cache[key]
        scg_mean.append(safe_scg(safe_extract_part(sound, onset, offset)))
        scg_mid.append(safe_scg(safe_extract_part(sound, midpoint - half, midpoint + half)))

        if (idx + 1) % 5000 == 0:
            print(f"processed {idx + 1:,}/{len(df):,} tokens")

    out = df.copy()
    out["SCG_mean"] = scg_mean
    out["SCG_mid"] = scg_mid

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()