"""
Apply Lobanov normalisation to acoustic formants and PCA/UMAP
reduction to Whisper and XLS-R representations.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DATA_DIR = Path("data/pipeline")
DIMS = [2, 5, 50]

TOKENS_PATH = DATA_DIR / "phoneme_tokens.csv"
ACOUSTIC_PATH = DATA_DIR / "features_acoustic.csv"
WHISPER_PATH = DATA_DIR / "features_whisper.npz"
XLSR_PATH = DATA_DIR / "features_xlsr.npz"

OUT_ACOUSTIC = DATA_DIR / "features_acoustic_norm.csv"
OUT_WHISPER = DATA_DIR / "features_whisper_pca.npz"
OUT_XLSR = DATA_DIR / "features_xlsr_pca.npz"

FRENCH_VOWELS = {
    "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə",
    "ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "ɑ̃ː", "aː", "u:",
    "a̰", "ɛ̰", "ɑ̰̃", "ə̰", "ø̰",
    "ɪ", "ʉ", "ɨ", "æ", "i̥", "y̥",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def is_vowel(phonemes: pd.Series) -> pd.Series:
    return phonemes.isin(FRENCH_VOWELS)


def lobanov_normalise(acoustic: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    df = acoustic.copy()

    if "phoneme" not in df.columns:
        df = df.merge(
            tokens[["speaker", "sentence_id", "repetition_index", "phoneme"]],
            how="left",
            on=["speaker", "sentence_id", "repetition_index"],
        )

    vowel_rows = df[is_vowel(df["phoneme"])]

    formant_pairs = [("F1_mid", "F2_mid")]
    for suffix in ("25", "75"):
        if f"F1_{suffix}" in df.columns and f"F2_{suffix}" in df.columns:
            formant_pairs.append((f"F1_{suffix}", f"F2_{suffix}"))

    for f1_col, f2_col in formant_pairs:
        stats = vowel_rows.groupby("speaker")[[f1_col, f2_col]].agg(["mean", "std"])
        stats.columns = ["_".join(col) for col in stats.columns]
        stats = stats.rename(columns={
            f"{f1_col}_mean": "F1_mean",
            f"{f1_col}_std": "F1_std",
            f"{f2_col}_mean": "F2_mean",
            f"{f2_col}_std": "F2_std",
        })

        df = df.merge(stats, on="speaker", how="left")
        suffix_tag = f1_col.replace("F1_", "")

        df[f"F1_lob_{suffix_tag}"] = (df[f1_col] - df["F1_mean"]) / df["F1_std"]
        df[f"F2_lob_{suffix_tag}"] = (df[f2_col] - df["F2_mean"]) / df["F2_std"]

        df.loc[df["F1_std"] == 0, f"F1_lob_{suffix_tag}"] = np.nan
        df.loc[df["F2_std"] == 0, f"F2_lob_{suffix_tag}"] = np.nan
        df = df.drop(columns=["F1_mean", "F1_std", "F2_mean", "F2_std"])

    return df


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def pca_reduce(npz_path: Path, label: str) -> dict:
    log.info("Loading %s features from %s", label, npz_path)
    data = np.load(npz_path, allow_pickle=True)
    hidden_keys = [key for key in data.files if key.startswith("hiddens_")]

    out = {key: data[key] for key in data.files if key not in hidden_keys}

    for key in hidden_keys:
        hidden = l2_normalise(data[key].astype(np.float32))
        _, dim = hidden.shape
        log.info("  %s: shape %s", key, hidden.shape)

        for n_components in DIMS:
            if n_components >= dim:
                continue
            pca = PCA(n_components=n_components, random_state=42)
            projection = pca.fit_transform(hidden).astype(np.float32)
            out[f"{key}_pca{n_components}"] = projection
            out[f"{key}_pca{n_components}_explained_variance_ratio"] = (
                pca.explained_variance_ratio_.astype(np.float32)
            )
            log.info(
                "    PCA d=%d explained %.1f%%",
                n_components,
                100 * pca.explained_variance_ratio_.sum(),
            )

    return out


def umap_reduce(npz_path: Path, label: str, out: dict) -> dict:
    import umap  # type: ignore

    data = np.load(npz_path, allow_pickle=True)
    hidden_keys = [key for key in data.files if key.startswith("hiddens_")]

    for key in hidden_keys:
        hidden = l2_normalise(data[key].astype(np.float32))
        log.info("  UMAP d=2 for %s / %s", label, key)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            low_memory=False,
        )
        out[f"{key}_umap2"] = reducer.fit_transform(hidden).astype(np.float32)

    return out


def check_inputs() -> None:
    for path in (TOKENS_PATH, ACOUSTIC_PATH, WHISPER_PATH, XLSR_PATH):
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    check_inputs()

    log.info("Loading manifest and acoustic features")
    tokens = pd.read_csv(TOKENS_PATH)
    acoustic = pd.read_csv(ACOUSTIC_PATH)

    log.info("Lobanov normalisation")
    acoustic_norm = lobanov_normalise(acoustic, tokens)
    acoustic_norm.to_csv(OUT_ACOUSTIC, index=False)
    log.info("Saved %s", OUT_ACOUSTIC)

    log.info("PCA - Whisper")
    whisper_out = pca_reduce(WHISPER_PATH, "Whisper")
    log.info("UMAP - Whisper")
    whisper_out = umap_reduce(WHISPER_PATH, "Whisper", whisper_out)
    np.savez_compressed(OUT_WHISPER, **whisper_out)
    log.info("Saved %s", OUT_WHISPER)

    log.info("PCA - XLS-R")
    xlsr_out = pca_reduce(XLSR_PATH, "XLS-R")
    log.info("UMAP - XLS-R")
    xlsr_out = umap_reduce(XLSR_PATH, "XLS-R", xlsr_out)
    np.savez_compressed(OUT_XLSR, **xlsr_out)
    log.info("Saved %s", OUT_XLSR)


if __name__ == "__main__":
    main()