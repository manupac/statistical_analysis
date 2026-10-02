"""
Compute neural descriptive statistics and plots.

Full all-phoneme outputs:
    results/descriptive_neural/

Minimal vowel-only outputs used for Q2/Q3:
    results/neural_vowels/
"""

from pathlib import Path
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("data/pipeline")
FULL_OUT_DIR = Path("results/descriptive_neural")
VOWEL_OUT_DIR = Path("results/neural_vowels")
SEED = 42

ORAL_VOWELS = {"a", "e", "i", "o", "u", "y", "ø", "œ", "ɑ", "ə", "ɛ"}
ORAL_VOWELS_ORDER = ["a", "e", "i", "o", "u", "y", "ø", "œ", "ɑ", "ə", "ɛ"]

PHONEME_CMAP = "tab20"
L1_PALETTE = {"fr": "#2166ac", "ru": "#d6604d"}
GENDER_PALETTE = {"f": "#7b2d8b", "m": "#1a7a1a"}

MODELS = {
    "whisper": {"npz": "features_whisper_pca.npz", "layers": [4, 20]},
    "xlsr": {"npz": "features_xlsr_pca.npz", "layers": [4, 12, 20]},
}

DR_METHODS = ["pca2", "umap2"]


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)


def load_data():
    df_raw = pd.read_csv(DATA_DIR / "features_acoustic_norm.csv")

    keep_mask = ~df_raw["phoneme"].str.contains("ding", na=False).values
    df = df_raw[keep_mask].reset_index(drop=True)

    npz = {}
    for model, cfg in MODELS.items():
        raw = np.load(DATA_DIR / cfg["npz"])
        filtered = {}

        for key in raw.files:
            arr = raw[key]
            if arr.ndim >= 1 and arr.shape[0] == len(df_raw):
                filtered[key] = arr[keep_mask]
            else:
                filtered[key] = arr

        npz[model] = filtered

    return df, npz


def get_projection(npz_obj, layer: int, dr: str):
    key = f"hiddens_layer{layer}_{dr}"
    return npz_obj[key] if key in npz_obj else None


def plot_projection(df, proj, model: str, layer: int, dr: str, colour_by: str, out_dir: Path) -> None:
    x, y = proj[:, 0], proj[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))

    if colour_by == "phoneme":
        labels = sorted(df["phoneme"].unique())
        cmap = plt.get_cmap(PHONEME_CMAP)
        colours = {p: cmap(i / max(len(labels) - 1, 1)) for i, p in enumerate(labels)}
        values = df["phoneme"].values

        for label in labels:
            mask = values == label
            ax.scatter(
                x[mask], y[mask],
                s=6, alpha=0.45,
                color=colours[label],
                label=label,
                rasterized=True,
            )

        ax.legend(markerscale=2, fontsize=7, ncol=3, loc="upper right", framealpha=0.6)

    elif colour_by == "L1":
        values = df["L1"].values
        for label, colour in L1_PALETTE.items():
            mask = values == label
            ax.scatter(
                x[mask], y[mask],
                s=6, alpha=0.45,
                color=colour,
                label=label,
                rasterized=True,
            )
        ax.legend(markerscale=2, fontsize=9)

    elif colour_by == "gender":
        values = df["gender"].values
        for label, colour in GENDER_PALETTE.items():
            mask = values == label
            ax.scatter(
                x[mask], y[mask],
                s=6, alpha=0.45,
                color=colour,
                label=label,
                rasterized=True,
            )
        ax.legend(markerscale=2, fontsize=9)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(f"{model.upper()} layer {layer} {dr.upper()} — by {colour_by}")
    fig.tight_layout()

    path = out_dir / "figures" / f"{model}_layer{layer}_{dr}_by_{colour_by}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def between_class_variance_ratio(df, proj) -> float:
    phonemes = df["phoneme"].values
    grand_mean = proj.mean(axis=0)
    total_scatter = ((proj - grand_mean) ** 2).sum()

    between_scatter = 0.0
    for phoneme in np.unique(phonemes):
        mask = phonemes == phoneme
        centroid = proj[mask].mean(axis=0)
        between_scatter += mask.sum() * ((centroid - grand_mean) ** 2).sum()

    return float(between_scatter / total_scatter)


def cosine_similarity_ratio(df, proj, rng, n_sample_pairs=50_000):
    phonemes = df["phoneme"].values
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    proj_norm = proj / norms

    n = len(proj_norm)
    indices = np.arange(n)

    within_sims = []
    for phoneme in np.unique(phonemes):
        ph_idx = indices[phonemes == phoneme]
        if len(ph_idx) < 2:
            continue

        n_pairs = max(1, int(n_sample_pairs * len(ph_idx) / n))
        i_idx = rng.choice(ph_idx, size=n_pairs, replace=True)
        j_idx = rng.choice(ph_idx, size=n_pairs, replace=True)

        keep = i_idx != j_idx
        i_idx = i_idx[keep]
        j_idx = j_idx[keep]

        if len(i_idx):
            within_sims.append((proj_norm[i_idx] * proj_norm[j_idx]).sum(axis=1))

    i_idx = rng.choice(n, size=n_sample_pairs * 3, replace=True)
    j_idx = rng.choice(n, size=n_sample_pairs * 3, replace=True)
    keep = phonemes[i_idx] != phonemes[j_idx]

    i_cross = i_idx[keep][:n_sample_pairs]
    j_cross = j_idx[keep][:n_sample_pairs]

    within_mean = np.concatenate(within_sims).mean() if within_sims else np.nan
    between_mean = (
        (proj_norm[i_cross] * proj_norm[j_cross]).sum(axis=1).mean()
        if len(i_cross)
        else np.nan
    )
    ratio = within_mean / between_mean if between_mean != 0 else np.nan

    return within_mean, between_mean, ratio


def interspeaker_variance_ratio(df_vowels, proj_vowels, phoneme: str) -> float:
    mask = df_vowels["phoneme"].values == phoneme
    ph_proj = proj_vowels[mask]
    ph_speakers = df_vowels["speaker"].values[mask]

    speakers = np.unique(ph_speakers)
    if len(speakers) < 2 or len(ph_proj) < 2:
        return np.nan

    grand_centroid = ph_proj.mean(axis=0)
    ss_total = ((ph_proj - grand_centroid) ** 2).sum()

    if ss_total == 0:
        return np.nan

    ss_inter = 0.0
    for speaker in speakers:
        sp_proj = ph_proj[ph_speakers == speaker]
        sp_centroid = sp_proj.mean(axis=0)
        ss_inter += len(sp_proj) * ((sp_centroid - grand_centroid) ** 2).sum()

    return float(ss_inter / ss_total)


def run_full_analysis(df, npz, rng) -> None:
    bcr_rows = []
    cos_rows = []

    for model, cfg in MODELS.items():
        for layer in cfg["layers"]:
            for dr in DR_METHODS:
                proj = get_projection(npz[model], layer, dr)

                if proj is None:
                    warnings.warn(f"Missing projection: {model} layer {layer} {dr}")
                    continue

                for colour_by in ["phoneme", "L1", "gender"]:
                    plot_projection(df, proj, model, layer, dr, colour_by, FULL_OUT_DIR)

                bcr_rows.append({
                    "model": model,
                    "layer": layer,
                    "dr": dr,
                    "between_class_variance_ratio": round(
                        between_class_variance_ratio(df, proj), 6
                    ),
                })

                within, between, ratio = cosine_similarity_ratio(df, proj, rng)
                cos_rows.append({
                    "model": model,
                    "layer": layer,
                    "dr": dr,
                    "within_phoneme_cosine_sim": round(within, 6),
                    "between_phoneme_cosine_sim": round(between, 6),
                    "within_over_between_ratio": round(ratio, 6),
                })

    pd.DataFrame(bcr_rows).to_csv(
        FULL_OUT_DIR / "tables" / "between_class_variance_ratio.csv",
        index=False,
    )
    pd.DataFrame(cos_rows).to_csv(
        FULL_OUT_DIR / "tables" / "cosine_similarity_ratios.csv",
        index=False,
    )


def run_vowel_question_outputs(df, npz) -> None:
    vowel_mask = df["phoneme"].isin(ORAL_VOWELS).values
    df_vowels = df[vowel_mask].reset_index(drop=True)

    for model in ["whisper", "xlsr"]:
        proj_full = get_projection(npz[model], 20, "umap2")
        if proj_full is None:
            warnings.warn(f"Missing vowel UMAP projection: {model} layer 20")
            continue

        proj_vowels = proj_full[vowel_mask]
        plot_projection(
            df_vowels,
            proj_vowels,
            model=model,
            layer=20,
            dr="umap2",
            colour_by="phoneme",
            out_dir=VOWEL_OUT_DIR,
        )

    rows = []
    for phoneme in ORAL_VOWELS_ORDER:
        if phoneme not in set(df_vowels["phoneme"]):
            continue

        row = {"phoneme": phoneme}

        for model in ["whisper", "xlsr"]:
            for dr in DR_METHODS:
                proj_full = get_projection(npz[model], 20, dr)

                if proj_full is None:
                    row[f"{model}_layer20_{dr}"] = np.nan
                    continue

                proj_vowels = proj_full[vowel_mask]
                value = interspeaker_variance_ratio(df_vowels, proj_vowels, phoneme)
                row[f"{model}_layer20_{dr}"] = round(value, 4) if not np.isnan(value) else np.nan

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        VOWEL_OUT_DIR / "tables" / "neural_interspeaker_variability.csv",
        index=False,
    )


def main() -> None:
    reset_output_dir(FULL_OUT_DIR)
    reset_output_dir(VOWEL_OUT_DIR)

    rng = np.random.default_rng(SEED)

    print("Loading data ...")
    df, npz = load_data()
    print(f"Tokens after artifact removal: {len(df):,}")

    print("Running full all-phoneme neural descriptive analysis ...")
    run_full_analysis(df, npz, rng)

    print("Running minimal vowel-only question-support analysis ...")
    run_vowel_question_outputs(df, npz)

    print(f"Full outputs saved to {FULL_OUT_DIR}")
    print(f"Vowel-only outputs saved to {VOWEL_OUT_DIR}")


if __name__ == "__main__":
    main()