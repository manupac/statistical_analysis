"""
Run group-comparison tests for acoustic and neural representations.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, wilcoxon


warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = "data/pipeline"
OUT_DIR = "results/group_comparisons"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUT_DIR, "figures")

ORAL_VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "œ", "ɑ", "ə", "ɛ"]

MODELS = {
    "whisper": {"npz": "features_whisper_pca.npz", "layers": [20]},
    "xlsr": {"npz": "features_xlsr_pca.npz", "layers": [20]},
}

ALPHA = 0.05
SEED = 42
N_PERMUTATIONS = 5000
SHAPIRO_SUBSAMPLE = 5000


def load_acoustic():
    df_raw = pd.read_csv(os.path.join(DATA_DIR, "features_acoustic_norm.csv"))

    artifact_mask = df_raw["phoneme"].str.contains("ding", na=False)
    keep_mask = ~artifact_mask.values
    df = df_raw[keep_mask].reset_index(drop=True)

    l1_map = {"fr": "L1", "fra": "L1", "l1": "L1", "ru": "L2", "rus": "L2", "l2": "L2"}
    gender_map = {"f": "F", "female": "F", "fem": "F", "m": "M", "male": "M"}

    df["L1_norm"] = df["L1"].str.lower().str.strip().map(l1_map)
    df["gender_norm"] = df["gender"].str.lower().str.strip().map(gender_map)

    return df, keep_mask


def load_neural(model: str, layer: int) -> np.ndarray:
    npz_path = os.path.join(DATA_DIR, MODELS[model]["npz"])
    npz = np.load(npz_path)

    for d in [50, 5, 2]:
        key = f"hiddens_layer{layer}_pca{d}"
        if key in npz:
            print(f"    {model} layer {layer}: using {key}")
            return npz[key]

    raise KeyError(f"No PCA key found for {model} layer {layer} in {npz_path}")


def bh_correction(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    if n == 0:
        return np.array([])

    order = np.argsort(p_values)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)

    adj = p_values * n / ranked
    adj_sorted = adj[order]

    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])

    adj[order] = adj_sorted
    return np.minimum(adj, 1.0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(1.0 - np.dot(a, b) / (na * nb))


def safe_phoneme_filename(phoneme: str) -> str:
    mapping = {
        "ɑ": "alpha",
        "ə": "schwa",
        "ɛ": "epsilon",
        "ø": "o_slash",
        "œ": "oe",
        "ɔ": "open_o",
        "̃": "nasal",
        "ː": "long",
    }

    name = phoneme
    for old, new in mapping.items():
        name = name.replace(old, new)

    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def plot_qq(g1: np.ndarray, g2: np.ndarray, phoneme: str, formant: str) -> None:
    safe_phoneme = safe_phoneme_filename(phoneme)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, data, label in [(axes[0], g1, "L1"), (axes[1], g2, "L2")]:
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"{phoneme} / {formant} — {label} (n={len(data)})", fontsize=9)
        ax.get_lines()[1].set_color("red")

    fig.suptitle(f"Q-Q plots: {phoneme} — {formant}", fontsize=10)
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, f"qqplot_{safe_phoneme}_{formant}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def acoustic_l1l2_tests(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []

    for formant, col in [("F1", "F1_lob_mid"), ("F2", "F2_lob_mid")]:
        for phoneme in ORAL_VOWELS:
            sub = df[(df["phoneme"] == phoneme) & df["L1_norm"].isin(["L1", "L2"])]

            g1 = sub[sub["L1_norm"] == "L1"][col].dropna().values
            g2 = sub[sub["L1_norm"] == "L2"][col].dropna().values

            if len(g1) < 3 or len(g2) < 3:
                continue

            s1 = g1 if len(g1) <= SHAPIRO_SUBSAMPLE else rng.choice(g1, SHAPIRO_SUBSAMPLE, replace=False)
            s2 = g2 if len(g2) <= SHAPIRO_SUBSAMPLE else rng.choice(g2, SHAPIRO_SUBSAMPLE, replace=False)

            _, sw_p1 = shapiro(s1)
            _, sw_p2 = shapiro(s2)

            normal1 = sw_p1 > ALPHA
            normal2 = sw_p2 > ALPHA
            both_normal = normal1 and normal2

            plot_qq(g1, g2, phoneme, formant)

            _, lev_p = levene(g1, g2, center="median")
            equal_var = lev_p > ALPHA

            if both_normal and equal_var:
                test_name = "t-test (Student)"
                stat, p_raw = ttest_ind(g1, g2, equal_var=True)
            elif both_normal and not equal_var:
                test_name = "t-test (Welch)"
                stat, p_raw = ttest_ind(g1, g2, equal_var=False)
            else:
                test_name = "Mann-Whitney U"
                stat, p_raw = mannwhitneyu(g1, g2, alternative="two-sided")

            rows.append({
                "formant": formant,
                "phoneme": phoneme,
                "n_L1": len(g1),
                "n_L2": len(g2),
                "mean_L1": round(g1.mean(), 4),
                "mean_L2": round(g2.mean(), 4),
                "SW_p_L1": round(sw_p1, 4),
                "SW_p_L2": round(sw_p2, 4),
                "normal_L1": bool(normal1),
                "normal_L2": bool(normal2),
                "Levene_p": round(lev_p, 4),
                "equal_var": bool(equal_var),
                "test": test_name,
                "stat": round(float(stat), 4),
                "p_raw": round(float(p_raw), 6),
            })

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        return result_df

    result_df["p_adj_BH"] = bh_correction(result_df["p_raw"].values).round(6)
    result_df["significant_raw"] = result_df["p_raw"] < ALPHA
    result_df["significant_BH"] = result_df["p_adj_BH"] < ALPHA

    return result_df


def gender_residual_test(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for formant, col in [("F1", "F1_lob_mid"), ("F2", "F2_lob_mid")]:
        speaker_means = (
            df[df["L1_norm"].notna() & df["gender_norm"].notna()]
            .groupby(["speaker", "gender_norm", "L1_norm"])[col]
            .mean()
            .reset_index()
        )

        f_means = speaker_means[speaker_means["gender_norm"] == "F"][col].dropna().values
        m_means = speaker_means[speaker_means["gender_norm"] == "M"][col].dropna().values

        if len(f_means) < 3 or len(m_means) < 3:
            rows.append({
                "formant": formant,
                "test": "skipped",
                "n_F": len(f_means),
                "n_M": len(m_means),
                "mean_F": np.nan,
                "mean_M": np.nan,
                "stat": np.nan,
                "p_raw": np.nan,
            })
            continue

        if len(f_means) == len(m_means):
            test_name = "Wilcoxon signed-rank (paired)"
            stat, p_raw = wilcoxon(f_means, m_means)
        else:
            test_name = "Mann-Whitney U (unpaired)"
            stat, p_raw = mannwhitneyu(f_means, m_means, alternative="two-sided")

        rows.append({
            "formant": formant,
            "test": test_name,
            "n_F": len(f_means),
            "n_M": len(m_means),
            "mean_F": round(float(f_means.mean()), 4),
            "mean_M": round(float(m_means.mean()), 4),
            "stat": round(float(stat), 4),
            "p_raw": round(float(p_raw), 6),
        })

    return pd.DataFrame(rows)


def neural_l1l2_tests(df: pd.DataFrame, keep_mask: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    vowel_mask = df["phoneme"].isin(ORAL_VOWELS).values
    df_vowels = df[vowel_mask].reset_index(drop=True)

    all_rows = []

    for model, cfg in MODELS.items():
        for layer in cfg["layers"]:
            print(f"  Neural test: {model} layer {layer}")

            proj_raw = load_neural(model, layer)
            proj_aligned = proj_raw[keep_mask]
            proj_vowels = proj_aligned[vowel_mask]

            speakers = df_vowels["speaker"].values
            l1_values = df_vowels["L1_norm"].values
            phonemes = df_vowels["phoneme"].values

            for phoneme in ORAL_VOWELS:
                ph_mask = phonemes == phoneme

                proj_ph = proj_vowels[ph_mask]
                l1_ph = l1_values[ph_mask]
                spk_ph = speakers[ph_mask]

                l1_mask = l1_ph == "L1"
                l2_mask = l1_ph == "L2"

                if l1_mask.sum() < 2 or l2_mask.sum() < 2:
                    continue

                c_l1 = proj_ph[l1_mask].mean(axis=0)
                c_l2 = proj_ph[l2_mask].mean(axis=0)
                d_obs = cosine_distance(c_l1, c_l2)

                if np.isnan(d_obs):
                    continue

                unique_speakers = np.unique(spk_ph)
                speaker_l1 = {
                    speaker: l1_ph[spk_ph == speaker][0]
                    for speaker in unique_speakers
                    if l1_ph[spk_ph == speaker][0] in ("L1", "L2")
                }

                labeled_speakers = list(speaker_l1.keys())
                n_speakers = len(labeled_speakers)
                n_l1_speakers = sum(1 for speaker in labeled_speakers if speaker_l1[speaker] == "L1")

                count_extreme = 0

                for _ in range(N_PERMUTATIONS):
                    perm_l1_flags = np.zeros(n_speakers, dtype=bool)
                    chosen = rng.choice(n_speakers, size=n_l1_speakers, replace=False)
                    perm_l1_flags[chosen] = True

                    perm_speaker_l1 = {
                        speaker: perm_l1_flags[i]
                        for i, speaker in enumerate(labeled_speakers)
                    }

                    token_l1 = np.array([perm_speaker_l1.get(speaker, None) for speaker in spk_ph])
                    valid = np.array([value is not None for value in token_l1])
                    token_l1 = np.array([bool(value) for value in token_l1[valid]])

                    proj_valid = proj_ph[valid]

                    if token_l1.sum() < 1 or (~token_l1).sum() < 1:
                        continue

                    c1 = proj_valid[token_l1].mean(axis=0)
                    c2 = proj_valid[~token_l1].mean(axis=0)
                    d_perm = cosine_distance(c1, c2)

                    if not np.isnan(d_perm) and d_perm >= d_obs:
                        count_extreme += 1

                p_raw = (count_extreme + 1) / (N_PERMUTATIONS + 1)

                all_rows.append({
                    "model": model,
                    "layer": layer,
                    "phoneme": phoneme,
                    "n_L1_tokens": int(l1_mask.sum()),
                    "n_L2_tokens": int(l2_mask.sum()),
                    "n_L1_spk": int(n_l1_speakers),
                    "n_L2_spk": int(n_speakers - n_l1_speakers),
                    "cosine_dist_obs": round(float(d_obs), 6),
                    "p_raw": round(float(p_raw), 6),
                })

    result_df = pd.DataFrame(all_rows)

    if result_df.empty:
        return result_df

    result_df["p_adj_BH"] = bh_correction(result_df["p_raw"].values).round(6)
    result_df["significant_raw"] = result_df["p_raw"] < ALPHA
    result_df["significant_BH"] = result_df["p_adj_BH"] < ALPHA

    return result_df


def main() -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("Loading data ...")
    df, keep_mask = load_acoustic()
    print(f"  Tokens loaded: {len(df):,}")

    print("Running acoustic L1/L2 tests ...")
    acoustic_df = acoustic_l1l2_tests(df)
    acoustic_df.to_csv(os.path.join(TABLE_DIR, "acoustic_l1l2_tests.csv"), index=False)

    print("Running residual gender tests ...")
    gender_df = gender_residual_test(df)
    gender_df.to_csv(os.path.join(TABLE_DIR, "acoustic_gender_test.csv"), index=False)

    print(f"Running neural L1/L2 permutation tests (B={N_PERMUTATIONS}) ...")
    neural_df = neural_l1l2_tests(df, keep_mask)
    neural_df.to_csv(os.path.join(TABLE_DIR, "neural_l1l2_tests.csv"), index=False)

    print(f"Saved tables to {TABLE_DIR}")
    print(f"Saved Q-Q plots to {FIGURE_DIR}")


if __name__ == "__main__":
    main()