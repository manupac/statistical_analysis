"""
Compute acoustic descriptive statistics and plots.
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse


warnings.filterwarnings("ignore", category=FutureWarning)


DATA_DIR = Path("data/pipeline")
RESULTS_DIR = Path("results/acoustics")
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

ORAL_VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "œ", "ɑ", "ə", "ɛ"]
VIOLIN_SUBSET = ["a", "e", "ɛ", "i", "o", "u", "y"]

GROUPS = ["L1/F", "L1/M", "L2/F", "L2/M"]
GROUP_PALETTE = {
    "L1/F": "#2166ac",
    "L1/M": "#92c5de",
    "L2/F": "#d6604d",
    "L2/M": "#f4a582",
}

DPI = 150


def load_data() -> pd.DataFrame:
    path = DATA_DIR / "features_acoustic_norm.csv"
    df = pd.read_csv(path)

    l1_map = {
        "fr": "L1", "fra": "L1", "l1": "L1", "french": "L1", "native": "L1",
        "ru": "L2", "rus": "L2", "l2": "L2", "russian": "L2",
    }
    gender_map = {
        "f": "F", "female": "F", "fem": "F",
        "m": "M", "male": "M",
    }

    df["L1_norm"] = df["L1"].astype(str).str.lower().str.strip().map(l1_map)
    df["gender_norm"] = df["gender"].astype(str).str.lower().str.strip().map(gender_map)
    df["group"] = df["L1_norm"] + "/" + df["gender_norm"]

    return df


def vowels_only(df: pd.DataFrame, vowel_set=ORAL_VOWELS) -> pd.DataFrame:
    return df[df["phoneme"].isin(vowel_set)].copy()


def confidence_ellipse(x, y, ax, n_std=1.96, **kwargs) -> None:
    if len(x) < 3:
        return

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        **kwargs,
    )
    ax.add_patch(ellipse)


def compute_desc_stats(df: pd.DataFrame) -> pd.DataFrame:
    vow = vowels_only(df)
    records = []

    for phoneme in ORAL_VOWELS:
        for group in GROUPS:
            sub = vow[(vow["phoneme"] == phoneme) & (vow["group"] == group)]

            for formant, col in (("F1", "F1_lob_mid"), ("F2", "F2_lob_mid")):
                vals = sub[col].dropna()
                if vals.empty:
                    continue

                q1, q3 = vals.quantile([0.25, 0.75])
                mean = vals.mean()
                sd = vals.std(ddof=1)

                records.append({
                    "phoneme": phoneme,
                    "group": group,
                    "formant": formant,
                    "n": len(vals),
                    "mean": round(mean, 4),
                    "median": round(vals.median(), 4),
                    "SD": round(sd, 4),
                    "IQR": round(q3 - q1, 4),
                    "CV": round(sd / mean, 4) if mean != 0 else np.nan,
                })

    return pd.DataFrame(records)


def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    vow = vowels_only(df)
    records = []

    for phoneme in ORAL_VOWELS:
        sub = vow[vow["phoneme"] == phoneme][["speaker", "F1_lob_mid"]].dropna()
        if len(sub) < 2:
            continue

        n_tokens = len(sub)
        df_total = n_tokens - 1
        grand_mean = sub["F1_lob_mid"].mean()

        ss_total = ((sub["F1_lob_mid"] - grand_mean) ** 2).sum()

        speaker_stats = sub.groupby("speaker")["F1_lob_mid"].agg(["mean", "count"])
        ss_inter = (
            speaker_stats["count"] * (speaker_stats["mean"] - grand_mean) ** 2
        ).sum()
        ss_intra = ss_total - ss_inter

        sigma2_total = ss_total / df_total
        sigma2_inter = ss_inter / df_total
        sigma2_intra = ss_intra / df_total
        sigma2_residual = sigma2_total - sigma2_inter - sigma2_intra

        pct_inter = 100 * ss_inter / ss_total if ss_total > 0 else np.nan
        pct_intra = 100 * ss_intra / ss_total if ss_total > 0 else np.nan
        pct_residual = 100 - pct_inter - pct_intra if ss_total > 0 else np.nan

        records.append({
            "phoneme": phoneme,
            "n_tokens": int(n_tokens),
            "n_speakers": int(speaker_stats.shape[0]),
            "sigma2_total": round(sigma2_total, 4),
            "sigma2_inter": round(sigma2_inter, 4),
            "sigma2_intra": round(sigma2_intra, 4),
            "sigma2_residual": round(abs(sigma2_residual), 6),
            "pct_inter": round(pct_inter, 1),
            "pct_intra": round(pct_intra, 1),
            "pct_residual": round(pct_residual, 1),
        })

    return pd.DataFrame(records)


def plot_vowel_chart(df: pd.DataFrame, out_path: Path) -> None:
    vow = vowels_only(df)

    f1 = vow["F1_lob_mid"].dropna()
    f2 = vow["F2_lob_mid"].dropna()
    f1_min, f1_max = f1.quantile(0.01), f1.quantile(0.99)
    f2_min, f2_max = f2.quantile(0.01), f2.quantile(0.99)
    pad = 0.2

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, group in zip(axes, GROUPS):
        sub = vow[vow["group"] == group]
        color = GROUP_PALETTE[group]

        for phoneme in ORAL_VOWELS:
            ph = sub[sub["phoneme"] == phoneme][["F2_lob_mid", "F1_lob_mid"]].dropna()
            if len(ph) < 3:
                continue

            x = ph["F2_lob_mid"].values
            y = ph["F1_lob_mid"].values

            confidence_ellipse(
                x, y, ax,
                edgecolor=color,
                facecolor=color,
                alpha=0.15,
                linewidth=1.0,
            )

            ax.scatter(x.mean(), y.mean(), color=color, s=60, zorder=5)
            ax.annotate(
                phoneme,
                (x.mean(), y.mean()),
                fontsize=9,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )

        ax.set_title(f"Group: {group}", fontsize=11, fontweight="bold")
        ax.set_xlabel("F2 (Lobanov) → back", fontsize=9)
        ax.set_ylabel("F1 (Lobanov) → open", fontsize=9)
        ax.set_xlim(f2_max + pad, f2_min - pad)
        ax.set_ylim(f1_max + pad, f1_min - pad)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Vowel chart — Lobanov-normalised F1/F2\n"
        "Centroids + 95% confidence ellipses by speaker group",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, out_dir: Path, min_tokens: int = 5) -> None:
    vow = vowels_only(df)
    token_counts = vow.groupby("phoneme")["F1_lob_mid"].count()

    valid_phonemes = [
        phoneme for phoneme in ORAL_VOWELS
        if token_counts.get(phoneme, 0) >= min_tokens
    ]
    vow = vow[vow["phoneme"].isin(valid_phonemes)]

    for formant, col in (("F1", "F1_lob_mid"), ("F2", "F2_lob_mid")):
        fig, ax = plt.subplots(figsize=(14, 5))

        sns.boxplot(
            data=vow,
            x="phoneme",
            y=col,
            hue="group",
            order=valid_phonemes,
            hue_order=GROUPS,
            palette=GROUP_PALETTE,
            width=0.7,
            fliersize=2,
            linewidth=0.8,
            ax=ax,
        )

        ax.set_title(
            f"{formant} (Lobanov) per phoneme — stratified by L1 status and gender",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Phoneme", fontsize=10)
        ax.set_ylabel(f"{formant} Lobanov z-score", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.legend(title="Group", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        fig.savefig(out_dir / f"boxplot_{formant}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_intraspeaker_violin(df: pd.DataFrame, out_path: Path) -> None:
    vow = vowels_only(df, VIOLIN_SUBSET)

    speaker_sd = (
        vow.groupby(["phoneme", "speaker", "L1_norm", "gender_norm"])["F1_lob_mid"]
        .std(ddof=1)
        .reset_index()
        .rename(columns={"F1_lob_mid": "intra_sd"})
        .dropna(subset=["intra_sd"])
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.violinplot(
        data=speaker_sd,
        x="phoneme",
        y="intra_sd",
        hue="L1_norm",
        order=VIOLIN_SUBSET,
        hue_order=["L1", "L2"],
        palette={"L1": "#2166ac", "L2": "#d6604d"},
        split=True,
        inner=None,
        linewidth=0.8,
        ax=ax,
    )

    sns.stripplot(
        data=speaker_sd,
        x="phoneme",
        y="intra_sd",
        hue="L1_norm",
        order=VIOLIN_SUBSET,
        hue_order=["L1", "L2"],
        palette={"L1": "#2166ac", "L2": "#d6604d"},
        dodge=True,
        size=4,
        alpha=0.7,
        jitter=True,
        ax=ax,
        legend=False,
    )

    ax.set_title(
        "Intra-speaker variability — within-speaker SD of F1 (Lobanov)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Phoneme", fontsize=10)
    ax.set_ylabel("Within-speaker SD of F1 (Lobanov)", fontsize=10)
    ax.legend(title="L1 status", labels=["L1 French", "L1 Russian"], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    required = {"phoneme", "L1", "gender", "F1_lob_mid", "F2_lob_mid", "speaker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    desc = compute_desc_stats(df)
    desc.to_csv(TABLE_DIR / "desc_stats.csv", index=False)

    vardecomp = variance_decomposition(df)
    vardecomp.to_csv(TABLE_DIR / "variance_decomposition.csv", index=False)

    plot_vowel_chart(df, FIGURE_DIR / "vowel_chart.png")
    plot_boxplots(df, FIGURE_DIR)
    plot_intraspeaker_violin(df, FIGURE_DIR / "violin_intraspeaker.png")

    print(f"Saved tables to {TABLE_DIR}")
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()