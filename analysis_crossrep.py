"""
Compute cross-representation RSMs and Mantel correlations.
"""

import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


DATA_DIR = "data/pipeline"
OUT_DIR = "results/cross_representation"

WHISPER_LAYER = 20
XLSR_LAYER = 20
N_PERMUTATIONS = 9999
MAX_TOKENS = 2000
SEED = 42

VOWEL_SET = {
    "a", "e", "i", "o", "u", "y", "ø", "œ", "ə", "ɛ", "ɔ", "ɑ",
    "ɛ̃", "œ̃", "ɔ̃", "ɑ̃",
    "aː", "ɑ̃ː", "u:",
    "a̰", "ɛ̰", "ɔ̰", "ḛ", "ḭ", "y̰",
    "i̥", "y̥",
    "ɪ", "ʉ", "ɨ", "æ",
}

MODELS = {
    "whisper": {"npz": "features_whisper_pca.npz"},
    "xlsr": {"npz": "features_xlsr_pca.npz"},
}


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "features_acoustic_norm.csv"))

    vowel_mask = df["phoneme"].isin(VOWEL_SET).values
    df_vowels = df[vowel_mask].reset_index(drop=True)
    vowel_indices = np.where(vowel_mask)[0]

    ac_cols = ["F1_lob_mid", "F2_lob_mid"]
    ac_mat_full = df_vowels[ac_cols].values.astype(float)
    valid_ac = ~np.isnan(ac_mat_full).any(axis=1)

    print(f"  Vowel tokens         : {len(df_vowels):,}")
    print(f"  With valid acoustics : {valid_ac.sum():,}")

    wh_mat_vowels = load_neural(MODELS["whisper"]["npz"], WHISPER_LAYER, vowel_indices)
    xl_mat_vowels = load_neural(MODELS["xlsr"]["npz"], XLSR_LAYER, vowel_indices)

    df_final = df_vowels[valid_ac].reset_index(drop=True)
    ac_final = ac_mat_full[valid_ac]
    wh_final = wh_mat_vowels[valid_ac]
    xl_final = xl_mat_vowels[valid_ac]

    return df_final, ac_final, wh_final, xl_final


def load_neural(npz_name: str, layer: int, indices: np.ndarray) -> np.ndarray:
    npz = np.load(os.path.join(DATA_DIR, npz_name))

    for d in [50, 5, 2]:
        key = f"hiddens_layer{layer}_pca{d}"
        if key in npz:
            print(f"    {npz_name} layer {layer}: using {key}, shape={npz[key].shape}")
            return npz[key][indices]

    raise KeyError(f"No PCA key found for layer {layer} in {npz_name}")


def stratified_subsample(df, ac, wh, xl, rng):
    n = len(df)
    if n <= MAX_TOKENS:
        return df, ac, wh, xl

    speakers = df["speaker"].values
    unique_speakers = np.unique(speakers)
    per_speaker = max(1, MAX_TOKENS // len(unique_speakers))

    keep_idx = []
    for speaker in unique_speakers:
        speaker_idx = np.where(speakers == speaker)[0]
        chosen = rng.choice(
            speaker_idx,
            size=min(per_speaker, len(speaker_idx)),
            replace=False,
        )
        keep_idx.append(chosen)

    keep_idx = np.sort(np.concatenate(keep_idx))[:MAX_TOKENS]

    print(f"  Subsampled {len(keep_idx):,} tokens from {n:,}")
    return (
        df.iloc[keep_idx].reset_index(drop=True),
        ac[keep_idx],
        wh[keep_idx],
        xl[keep_idx],
    )


def build_acoustic_rsm(ac_mat):
    dist = squareform(pdist(ac_mat, metric="euclidean"))
    return -dist


def build_neural_rsm(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat_norm = mat / norms
    return np.clip(mat_norm @ mat_norm.T, -1.0, 1.0)


def upper_triangle(mat):
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def mantel_test(rsm_a, rsm_b, rng):
    from scipy.stats import rankdata

    tri_a = upper_triangle(rsm_a)
    tri_b = upper_triangle(rsm_b)

    r_obs, _ = spearmanr(tri_a, tri_b)

    n = rsm_b.shape[0]
    row_idx, col_idx = np.triu_indices(n, k=1)

    ranked_a = rankdata(tri_a).astype(np.float64)
    ranked_a -= ranked_a.mean()
    norm_a = np.linalg.norm(ranked_a)
    if norm_a > 0:
        ranked_a /= norm_a

    flat_b = rsm_b[row_idx, col_idx]
    rank_lookup = rankdata(flat_b).astype(np.float64)

    rank_mat_b = np.zeros((n, n), dtype=np.float64)
    rank_mat_b[row_idx, col_idx] = rank_lookup
    rank_mat_b[col_idx, row_idx] = rank_lookup

    mean_rank = rank_lookup.mean()

    count_extreme = 0
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(n)
        ranked_b_perm = rank_mat_b[perm[row_idx], perm[col_idx]]
        ranked_b_perm -= mean_rank

        norm_b = np.linalg.norm(ranked_b_perm)
        if norm_b == 0:
            continue

        ranked_b_perm /= norm_b
        r_perm = np.dot(ranked_a, ranked_b_perm)

        if abs(r_perm) >= abs(r_obs):
            count_extreme += 1

    p_val = (count_extreme + 1) / (N_PERMUTATIONS + 1)
    return r_obs, p_val


def main():
    table_dir = os.path.join(OUT_DIR, "tables")
    os.makedirs(table_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)

    print("Loading data ...")
    df, ac_mat, wh_mat, xl_mat = load_data()
    print(f"  Aligned token count  : {len(df):,}")

    print("Subsampling for RSM tractability ...")
    df, ac_mat, wh_mat, xl_mat = stratified_subsample(df, ac_mat, wh_mat, xl_mat, rng)
    n_tokens = len(df)

    print(f"Building RSMs (N={n_tokens}) ...")
    rsm_ac = build_acoustic_rsm(ac_mat)
    rsm_wh = build_neural_rsm(wh_mat)
    rsm_xl = build_neural_rsm(xl_mat)

    print(f"Running Mantel tests (B={N_PERMUTATIONS}) ...")
    pairs = [
        ("Acoustic vs Whisper", rsm_ac, rsm_wh),
        ("Acoustic vs XLS-R", rsm_ac, rsm_xl),
        ("Whisper vs XLS-R", rsm_wh, rsm_xl),
    ]

    rows = []
    for label, rsm_1, rsm_2 in pairs:
        r, p = mantel_test(rsm_1, rsm_2, rng)
        print(f"  {label:25s} r={r:.4f} p={p:.4f}")
        rows.append({
            "pair": label,
            "mantel_r": round(r, 6),
            "p_value": round(p, 6),
            "n_tokens": n_tokens,
            "n_permutations": N_PERMUTATIONS,
            "whisper_layer": WHISPER_LAYER,
            "xlsr_layer": XLSR_LAYER,
        })

    mantel_path = os.path.join(table_dir, "mantel_results.csv")
    pd.DataFrame(rows).to_csv(mantel_path, index=False)

    print(f"Saved Mantel table to {mantel_path}")


if __name__ == "__main__":
    main()