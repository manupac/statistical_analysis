"""
analysis_hierarchical_clustering.py — Stage 9: Hierarchical Clustering

Implements §9.1, §9.2, §9.3 and §9.4. The consonant-vowel
analysis uses extended acoustic features when available and neural PCA features directly.

Inputs:
    data/pipeline/features_acoustic_norm.csv  or features_acoustics_norm.csv
    data/pipeline/features_acoustic_extended.csv  optional, for consonant/vowel
    data/pipeline/features_whisper_pca.npz
    data/pipeline/features_xlsr_pca.npz
    data/pipeline/phoneme_tokens.csv

Outputs:
    results/clustering/tables/*.csv
    results/clustering/figures/*.png

Implementation notes:
    - Vowels: exact set /a e i o u y ø œ ɑ ə ɛ/.
    - Vowel clustering uses phoneme centroids.
    - Speaker clustering uses concatenated per-speaker, per-vowel mean vectors.
    - Acoustic: Ward linkage, Euclidean distance.
    - Neural: Ward linkage, cosine distance, following the lab wording.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path("data/pipeline")
OUT_DIR = Path("results/clustering")

TARGET_VOWELS = ["a", "e", "i", "o", "u", "y", "ø", "œ", "ɑ", "ə", "ɛ"]
TARGET_VOWELS_SET = set(TARGET_VOWELS)

HEIGHT_LABELS = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid", "ø": "mid", "œ": "mid", "o": "mid", "ɛ": "mid", "ə": "mid",
    "a": "low", "ɑ": "low",
}

# Teacher-inspired expected structure: anterior/front, posterior/back,
# front-rounded, and central schwa.
FB_ROUND_LABELS = {
    "i": "front", "e": "front", "ɛ": "front", "a": "front",
    "u": "back", "o": "back", "ɑ": "back",
    "y": "front_rounded", "ø": "front_rounded", "œ": "front_rounded",
    "ə": "central",
}

# Consonant inventory used for §9.2. The balanced selector tries to include
# different manner classes before falling back to a purely availability-based list.
CONSONANT_GROUPS = {
    "stop": ["p", "t", "k", "b", "d", "g"],
    "fricative": ["f", "s", "ʃ", "v", "z", "ʒ"],
    "nasal": ["m", "n", "ɲ"],
    "liquid_glide": ["l", "ʁ", "r", "j", "w", "ɥ"],
}
CONSONANT_PRIORITY = [ph for group in CONSONANT_GROUPS.values() for ph in group]

# Fixed consonant set used for §9.2.
FIXED_CONSONANTS = ["p", "f", "n", "l", "t", "k"]

REPRESENTATIONS = {
    "acoustic": {
        "kind": "acoustic",
        "metric": "euclidean",
        "linkage_metric": "euclidean",
        "feature_cols": ["F1_lob_mid", "F2_lob_mid"],
    },
    "whisper": {
        "kind": "neural",
        "metric": "cosine",
        "linkage_metric": "precomputed_cosine",
        "npz": "features_whisper_pca.npz",
        "key": "hiddens_layer20_pca5",
    },
    "xlsr": {
        "kind": "neural",
        "metric": "cosine",
        "linkage_metric": "precomputed_cosine",
        "npz": "features_xlsr_pca.npz",
        "key": "hiddens_layer20_pca5",
    },
}


def find_acoustic_norm_file(data_dir: Path) -> Path:
    candidates = [
        data_dir / "features_acoustic_norm.csv",
        data_dir / "features_acoustics_norm.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find features_acoustic_norm.csv or features_acoustics_norm.csv")


def load_manifest(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "phoneme_tokens.csv"
    df = pd.read_csv(path)
    df["manifest_row"] = df.index
    df = df[~df["phoneme"].str.contains("ding", na=False)].copy()

    l1_map = {
        "fr": "L1", "fra": "L1", "l1": "L1", "french": "L1", "native": "L1",
        "ru": "L2", "rus": "L2", "l2": "L2", "russian": "L2",
    }
    g_map = {"f": "F", "female": "F", "fem": "F", "m": "M", "male": "M"}
    df["L1_norm"] = df["L1"].str.lower().str.strip().map(l1_map)
    df["gender_norm"] = df["gender"].str.lower().str.strip().map(g_map)
    if df["L1_norm"].isna().any():
        bad = sorted(df.loc[df["L1_norm"].isna(), "L1"].dropna().unique())
        raise ValueError(f"Unmapped L1 values: {bad}")
    if df["gender_norm"].isna().any():
        bad = sorted(df.loc[df["gender_norm"].isna(), "gender"].dropna().unique())
        raise ValueError(f"Unmapped gender values: {bad}")
    return df


def load_acoustic_vowels(data_dir: Path) -> pd.DataFrame:
    path = find_acoustic_norm_file(data_dir)
    df = pd.read_csv(path)
    df = df[~df["phoneme"].str.contains("ding", na=False)].copy()
    df = df[df["phoneme"].astype(str).isin(TARGET_VOWELS_SET)].copy()
    return df


def _choose_token_index_alignment(pca_df: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    original = manifest.copy()
    filtered = manifest.copy().reset_index(drop=True)
    filtered["filtered_row"] = filtered.index
    meta_cols = ["speaker", "phoneme", "L1_norm", "gender_norm"]

    merged_original = pca_df.merge(
        original[["manifest_row"] + meta_cols],
        left_on="token_index", right_on="manifest_row", how="left"
    )
    merged_filtered = pca_df.merge(
        filtered[["filtered_row", "manifest_row"] + meta_cols],
        left_on="token_index", right_on="filtered_row", how="left"
    )

    def score(m: pd.DataFrame) -> tuple[int, int]:
        matched = int(m["speaker"].notna().sum())
        vowels = int(m["phoneme"].astype(str).isin(TARGET_VOWELS_SET).sum())
        return matched, vowels

    score_original = score(merged_original)
    score_filtered = score(merged_filtered)
    token_index = pca_df["token_index"].to_numpy()
    max_idx = int(np.nanmax(token_index)) if len(token_index) else -1

    if max_idx >= len(filtered) and max_idx < (manifest["manifest_row"].max() + 1):
        return merged_original, "original_manifest_row"
    if score_original >= score_filtered:
        return merged_original, "original_manifest_row"
    return merged_filtered, "artifact_filtered_row"


def load_neural_pca(data_dir: Path, manifest: pd.DataFrame, npz_name: str, key: str) -> pd.DataFrame:
    path = data_dir / npz_name
    z = np.load(path, allow_pickle=True)
    if key not in z:
        raise KeyError(f"{key} not found in {path}. Available keys: {z.files}")
    X = z[key]
    token_index = z["token_index"].astype(int)
    cols = [f"PC{i+1}" for i in range(X.shape[1])]
    pca_df = pd.DataFrame(X, columns=cols)
    pca_df["token_index"] = token_index
    merged, mode = _choose_token_index_alignment(pca_df, manifest)
    print(f"  {npz_name}: token_index alignment = {mode}")
    merged = merged[merged["phoneme"].astype(str).isin(TARGET_VOWELS_SET)].copy()
    return merged


def load_neural_pca_all(data_dir: Path, manifest: pd.DataFrame, npz_name: str, key: str) -> pd.DataFrame:
    """Load neural PCA rows for all non-artifact phonemes, not only vowels.

    This is used for §9.2 consonant-vowel clustering, where consonants must be
    retained. The vowel-only loader above is still used for §9.1 and §9.3.
    """
    path = data_dir / npz_name
    z = np.load(path, allow_pickle=True)
    if key not in z:
        raise KeyError(f"{key} not found in {path}. Available keys: {z.files}")
    X = z[key]
    token_index = z["token_index"].astype(int)
    cols = [f"PC{i+1}" for i in range(X.shape[1])]
    pca_df = pd.DataFrame(X, columns=cols)
    pca_df["token_index"] = token_index
    merged, mode = _choose_token_index_alignment(pca_df, manifest)
    print(f"  {npz_name}: token_index alignment for all phonemes = {mode}")
    merged = merged[merged["phoneme"].notna()].copy()
    return merged


def centroid_table(df: pd.DataFrame, group_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    out = df.dropna(subset=feature_cols).groupby(group_cols, observed=True)[feature_cols].mean().reset_index()
    return out


def build_linkage(X: np.ndarray, metric: str, linkage_metric: str):
    if X.shape[0] < 2:
        raise ValueError("Need at least two observations for clustering")
    if linkage_metric == "precomputed_cosine":
        d = pdist(X, metric="cosine")
        # Lab asks for Ward + cosine for neural. SciPy accepts a condensed
        # distance vector, although Ward is most standard with Euclidean data.
        Z = linkage(d, method="ward")
        return Z, d
    Z = linkage(X, method="ward", metric=metric)
    return Z, pdist(X, metric=metric)


def assign_clusters(Z, labels: list[str], k: int) -> pd.Series:
    raw = fcluster(Z, t=k, criterion="maxclust")
    return pd.Series(raw, index=labels, name=f"cluster_k{k}")


def safe_silhouette(X: np.ndarray, labels: np.ndarray, metric: str) -> float:
    try:
        if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
            return np.nan
        return float(silhouette_score(X, labels, metric=metric))
    except Exception:
        return np.nan


def height_gap_for_k(Z: np.ndarray, k: int) -> float:
    # Difference between the merge height that yields k clusters and the next merge.
    # Larger values suggest a more stable cut.
    n = Z.shape[0] + 1
    if k < 2 or k >= n:
        return np.nan
    heights = Z[:, 2]
    # merges are ascending; cut to k clusters happens around merge n-k
    idx = n - k
    if idx <= 0 or idx >= len(heights):
        return np.nan
    return float(heights[idx] - heights[idx - 1])


def plot_dendrogram(Z, labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.45), 4.8))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45)
    ax.set_title(title)
    ax.set_ylabel("Dendrogram height")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_vowel_clustering(rep_data: dict[str, pd.DataFrame], out_dir: Path, k_min: int, k_max: int):
    rows_ari = []
    rows_assign = []
    rows_diag = []
    rows_mis = []

    for rep, df in rep_data.items():
        spec = REPRESENTATIONS[rep]
        if rep == "acoustic":
            feature_cols = spec["feature_cols"]
        else:
            feature_cols = [c for c in df.columns if c.startswith("PC")]

        cent = centroid_table(df, ["phoneme"], feature_cols)
        cent = cent[cent["phoneme"].isin(TARGET_VOWELS_SET)].copy()
        cent["height_truth"] = cent["phoneme"].map(HEIGHT_LABELS)
        cent["fb_round_truth"] = cent["phoneme"].map(FB_ROUND_LABELS)
        cent = cent.dropna(subset=["height_truth", "fb_round_truth"] + feature_cols)
        cent = cent.sort_values("phoneme")

        labels = cent["phoneme"].astype(str).tolist()
        X = cent[feature_cols].to_numpy(dtype=float)
        if rep == "acoustic":
            # F1/F2 are already Lobanov-scaled; no extra scaling needed.
            X_cluster = X
        else:
            X_cluster = X

        Z, _dist = build_linkage(X_cluster, spec["metric"], spec["linkage_metric"])
        plot_dendrogram(Z, labels, f"Vowel clustering — {rep}", out_dir / "figures" / f"dendrogram_vowels_{rep}.png")

        # ARI against ground-truth partitions.
        for partition, truth_col in [("height", "height_truth"), ("front_back_round", "fb_round_truth")]:
            k = int(cent[truth_col].nunique())
            pred = assign_clusters(Z, labels, k)
            ari = adjusted_rand_score(cent[truth_col].astype(str), pred.loc[labels].to_numpy())
            rows_ari.append({
                "analysis": "vowel_clustering",
                "representation": rep,
                "partition": partition,
                "k": k,
                "ARI": round(float(ari), 4),
                "n_phonemes": len(labels),
            })
            for ph, truth, cl in zip(labels, cent[truth_col].astype(str), pred.loc[labels].to_numpy()):
                rows_assign.append({
                    "analysis": "vowel_clustering",
                    "representation": rep,
                    "partition": partition,
                    "phoneme": ph,
                    "truth": truth,
                    "cluster": int(cl),
                })

        # k diagnostics.
        max_k_here = min(k_max, len(labels) - 1)
        for k in range(k_min, max_k_here + 1):
            pred = assign_clusters(Z, labels, k).to_numpy()
            sil_metric = "cosine" if rep in {"whisper", "xlsr"} else "euclidean"
            sil = safe_silhouette(X_cluster, pred, sil_metric)
            rows_diag.append({
                "analysis": "vowel_clustering",
                "representation": rep,
                "k": k,
                "silhouette": round(sil, 4) if np.isfinite(sil) else np.nan,
                "height_gap": round(height_gap_for_k(Z, k), 4) if np.isfinite(height_gap_for_k(Z, k)) else np.nan,
                "n_items": len(labels),
            })

    return pd.DataFrame(rows_ari), pd.DataFrame(rows_assign), pd.DataFrame(rows_diag)


def load_extended_acoustic(data_dir: Path) -> Optional[pd.DataFrame]:
    path = data_dir / "features_acoustic_extended.csv"
    if not path.exists():
        print("  features_acoustic_extended.csv not found; skipping §9.2 consonant/vowel clustering.")
        return None
    df = pd.read_csv(path)
    df = df[~df["phoneme"].str.contains("ding", na=False)].copy()
    return df


def choose_consonants(df: pd.DataFrame, min_n: int = 0, min_consonants: int = 6,
                      mode: str = "fixed") -> tuple[list[str], dict]:
    """Return the fixed consonant set used for §9.2.

    The set is hardcoded rather than automatically selected so the analysis is
    reproducible and linguistically interpretable: /p f n l t k/ covers stops,
    a fricative, a nasal, and a liquid, while retaining sufficient token counts
    in the corpus. The input df is used only to report token counts.
    """
    counts = df["phoneme"].astype(str).value_counts() if "phoneme" in df.columns else pd.Series(dtype=int)
    chosen = FIXED_CONSONANTS.copy()
    selected_groups = {
        ph: next((g for g, phones in CONSONANT_GROUPS.items() if ph in phones), "other")
        for ph in chosen
    }
    selected_counts = {ph: int(counts.get(ph, 0)) for ph in chosen}
    info = {
        "mode": "fixed",
        "min_n": min_n,
        "selected_counts": selected_counts,
        "selected_groups": selected_groups,
    }
    return chosen, info


def _run_cv_clustering_one_rep(cent: pd.DataFrame, feature_cols: list[str], rep_label: str,
                               metric: str, linkage_metric: str, out_dir: Path,
                               selected_consonants: list[str], selection_info: dict,
                               standardise: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster phoneme centroids into consonant vs vowel classes for one representation."""
    cent = cent.dropna(subset=["phoneme", "truth_cv"]).copy()
    cent = cent[cent["phoneme"].isin(TARGET_VOWELS_SET.union(set(selected_consonants)))].copy()
    cent = cent.dropna(subset=feature_cols, how="all").copy()

    if cent["truth_cv"].nunique() < 2 or len(cent) < 4:
        return pd.DataFrame(), pd.DataFrame()

    # Impute remaining centroid-level missing values with feature means.
    X_df = cent[feature_cols].copy()
    imputed_features = {}
    for c in feature_cols:
        nmiss = int(X_df[c].isna().sum())
        imputed_features[c] = nmiss
        if X_df[c].notna().sum() == 0:
            X_df = X_df.drop(columns=[c])
        else:
            X_df[c] = X_df[c].fillna(X_df[c].mean())
    usable_cols = list(X_df.columns)
    if len(usable_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    X = X_df.to_numpy(dtype=float)
    if standardise:
        X = StandardScaler().fit_transform(X)

    labels = cent["phoneme"].astype(str).tolist()
    Z, _ = build_linkage(X, metric=metric, linkage_metric=linkage_metric)
    plot_dendrogram(
        Z, labels,
        f"Consonant vs vowel clustering — {rep_label}",
        out_dir / "figures" / f"dendrogram_consonant_vowel_{rep_label}.png",
    )

    pred = assign_clusters(Z, labels, 2).loc[labels].to_numpy()
    ari = adjusted_rand_score(cent["truth_cv"], pred)
    ari_df = pd.DataFrame([{
        "analysis": "consonant_vowel_clustering",
        "representation": rep_label,
        "k": 2,
        "ARI": round(float(ari), 4),
        "n_phonemes": len(labels),
        "selected_consonants": " ".join(selected_consonants),
        "selected_consonant_groups": str(selection_info.get("selected_groups", {})),
        "selected_consonant_counts": str(selection_info.get("selected_counts", {})),
        "consonant_selection_mode": selection_info.get("mode", ""),
        "feature_cols": " ".join(usable_cols),
        "imputed_centroid_missing_by_feature": str(imputed_features),
    }])
    assign_df = pd.DataFrame({
        "analysis": "consonant_vowel_clustering",
        "representation": rep_label,
        "phoneme": labels,
        "truth": cent["truth_cv"].astype(str).tolist(),
        "cluster": pred.astype(int),
    })
    return ari_df, assign_df


def run_consonant_vowel_clustering(df_ext: Optional[pd.DataFrame], neural_all: dict[str, pd.DataFrame],
                                   out_dir: Path, min_n: int = 0, consonant_mode: str = "fixed"):
    """Run §9.2 consonant-vowel clustering for acoustic_extended, Whisper, and XLS-R.

    Acoustic requires features_acoustic_extended.csv because it needs duration/SCG.
    Neural uses the PCA files directly and retains consonants instead of applying the
    vowel-only filter used in §9.1 and §9.3.
    """
    count_source = df_ext if df_ext is not None else next(iter(neural_all.values()))
    consonants, selection_info = choose_consonants(
        count_source, min_n=min_n, min_consonants=6, mode=consonant_mode
    )
    if len(consonants) < 6:
        print(f"  WARNING: fewer than six consonants available with min_n={min_n}: {consonants}")
    print(f"  Selected consonants (fixed): {' '.join(consonants)}")
    print(f"  Selected consonant groups: {selection_info.get('selected_groups', {})}")
    print(f"  Selected consonant counts: {selection_info.get('selected_counts', {})}")
    selected = TARGET_VOWELS + consonants

    ari_parts = []
    assign_parts = []

    # Acoustic extended: F1/F2 + duration + SCG, standardised.
    if df_ext is not None:
        df = df_ext[df_ext["phoneme"].astype(str).isin(selected)].copy()
        candidate_cols = ["F1_mid", "F2_mid", "duration_ms", "SCG_mid", "SCG_mean"]
        feature_cols = [c for c in candidate_cols if c in df.columns]
        if len(feature_cols) >= 2:
            cent = df.groupby("phoneme", observed=True)[feature_cols].mean().reset_index()
            cent["truth_cv"] = np.where(cent["phoneme"].isin(TARGET_VOWELS_SET), "vowel", "consonant")
            a, b = _run_cv_clustering_one_rep(
                cent, feature_cols, "acoustic_extended", "euclidean", "euclidean",
                out_dir, consonants, selection_info, standardise=True,
            )
            if not a.empty:
                ari_parts.append(a)
                assign_parts.append(b)
        else:
            print("  Not enough acoustic columns for consonant/vowel clustering; skipping acoustic_extended.")
    else:
        print("  features_acoustic_extended.csv not found; skipping acoustic consonant/vowel clustering.")

    # Neural PCA: same phoneme-token representations as vowels, but consonants retained.
    for rep_label, df_neu in neural_all.items():
        df = df_neu[df_neu["phoneme"].astype(str).isin(selected)].copy()
        pc_cols = [c for c in df.columns if c.startswith("PC")]
        if len(pc_cols) < 2:
            print(f"  Not enough PC columns for {rep_label} consonant/vowel clustering; skipping.")
            continue
        cent = df.groupby("phoneme", observed=True)[pc_cols].mean().reset_index()
        cent["truth_cv"] = np.where(cent["phoneme"].isin(TARGET_VOWELS_SET), "vowel", "consonant")
        a, b = _run_cv_clustering_one_rep(
            cent, pc_cols, rep_label, "cosine", "precomputed_cosine",
            out_dir, consonants, selection_info, standardise=False,
        )
        if not a.empty:
            ari_parts.append(a)
            assign_parts.append(b)

    if not ari_parts:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(ari_parts, ignore_index=True), pd.concat(assign_parts, ignore_index=True)


def common_speaker_vowels(rep_data: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    rep_speakers = []
    rep_vowels = []
    for rep, df in rep_data.items():
        speakers = set(df["speaker"].dropna().astype(str).unique())
        rep_speakers.append(speakers)
        counts = df.groupby(["speaker", "phoneme"], observed=True).size().reset_index(name="n")
        valid_vowels = []
        for ph in TARGET_VOWELS:
            sub = counts[counts["phoneme"].astype(str) == ph]
            if set(sub["speaker"].astype(str)) == speakers:
                valid_vowels.append(ph)
        rep_vowels.append(set(valid_vowels))
    common_speakers = sorted(set.intersection(*rep_speakers))
    common_vowels = [ph for ph in TARGET_VOWELS if all(ph in s for s in rep_vowels)]
    return common_speakers, common_vowels


def speaker_matrix(df: pd.DataFrame, speakers: list[str], vowels: list[str], feature_cols: list[str]) -> pd.DataFrame:
    cent = df[df["speaker"].astype(str).isin(speakers) & df["phoneme"].astype(str).isin(vowels)].copy()
    grp = cent.groupby(["speaker", "phoneme"], observed=True)[feature_cols].mean()
    rows = []
    for sp in speakers:
        row = {"speaker": sp}
        ok = True
        for ph in vowels:
            try:
                vals = grp.loc[(sp, ph)]
            except KeyError:
                ok = False
                break
            for c in feature_cols:
                row[f"{ph}_{c}"] = vals[c]
        if ok:
            rows.append(row)
    mat = pd.DataFrame(rows).dropna()
    return mat


def run_speaker_clustering(rep_data: dict[str, pd.DataFrame], manifest: pd.DataFrame, out_dir: Path, k_min: int, k_max: int):
    common_speakers, common_vowels = common_speaker_vowels(rep_data)
    print(f"  Speaker clustering common speakers: {len(common_speakers)}")
    print(f"  Speaker clustering common vowels: {common_vowels}")
    if len(common_speakers) < 3 or len(common_vowels) < 2:
        print("  Not enough common speakers/vowels for speaker clustering; skipping.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    meta = manifest.drop_duplicates("speaker").set_index("speaker")
    rows_ari = []
    rows_assign = []
    rows_diag = []

    for rep, df in rep_data.items():
        if rep == "acoustic":
            feature_cols = REPRESENTATIONS[rep]["feature_cols"]
            metric = "euclidean"
            linkage_metric = "euclidean"
        else:
            feature_cols = [c for c in df.columns if c.startswith("PC")]
            metric = "cosine"
            linkage_metric = "precomputed_cosine"

        mat = speaker_matrix(df, common_speakers, common_vowels, feature_cols)
        speakers = mat["speaker"].astype(str).tolist()
        X = mat.drop(columns=["speaker"]).to_numpy(dtype=float)
        if rep == "acoustic":
            X = StandardScaler().fit_transform(X)

        Z, _ = build_linkage(X, metric=metric, linkage_metric=linkage_metric)
        plot_dendrogram(Z, speakers, f"Speaker clustering — {rep}", out_dir / "figures" / f"dendrogram_speakers_{rep}.png")

        for factor, truth_col in [("L1", "L1_norm"), ("gender", "gender_norm")]:
            truth = [str(meta.loc[sp, truth_col]) for sp in speakers]
            pred = assign_clusters(Z, speakers, 2).loc[speakers].to_numpy()
            ari = adjusted_rand_score(truth, pred)
            rows_ari.append({
                "analysis": "speaker_clustering",
                "representation": rep,
                "factor": factor,
                "k": 2,
                "ARI": round(float(ari), 4),
                "n_speakers": len(speakers),
                "common_vowels": " ".join(common_vowels),
            })
            for sp, tr, cl in zip(speakers, truth, pred):
                rows_assign.append({
                    "analysis": "speaker_clustering",
                    "representation": rep,
                    "factor": factor,
                    "speaker": sp,
                    "truth": tr,
                    "cluster": int(cl),
                })

        for k in range(k_min, min(k_max, len(speakers) - 1) + 1):
            pred = assign_clusters(Z, speakers, k).to_numpy()
            sil = safe_silhouette(X, pred, metric)
            gap = height_gap_for_k(Z, k)
            rows_diag.append({
                "analysis": "speaker_clustering",
                "representation": rep,
                "k": k,
                "silhouette": round(sil, 4) if np.isfinite(sil) else np.nan,
                "height_gap": round(gap, 4) if np.isfinite(gap) else np.nan,
                "n_items": len(speakers),
            })

    return pd.DataFrame(rows_ari), pd.DataFrame(rows_assign), pd.DataFrame(rows_diag)


def systematic_misclassifications(vowel_assign: pd.DataFrame) -> pd.DataFrame:
    # A practical definition: wrong if its cluster's modal truth label differs from its own truth.
    rows = []
    for partition in vowel_assign["partition"].dropna().unique():
        subp = vowel_assign[vowel_assign["partition"] == partition].copy()
        wrong_records = []
        for rep in subp["representation"].unique():
            sub = subp[subp["representation"] == rep].copy()
            cluster_to_truth = (
                sub.groupby("cluster")["truth"]
                .agg(lambda x: x.value_counts().idxmax())
                .to_dict()
            )
            sub["modal_truth_for_cluster"] = sub["cluster"].map(cluster_to_truth)
            sub["wrong"] = sub["truth"] != sub["modal_truth_for_cluster"]
            wrong_records.append(sub[["representation", "partition", "phoneme", "truth", "cluster", "modal_truth_for_cluster", "wrong"]])
        wrong_df = pd.concat(wrong_records, ignore_index=True)
        pivot = wrong_df.pivot_table(index="phoneme", columns="representation", values="wrong", aggfunc="first")
        for ph, row in pivot.iterrows():
            wrong_reps = [rep for rep, val in row.items() if bool(val)]
            if len(wrong_reps) >= 2:
                rows.append({
                    "partition": partition,
                    "phoneme": ph,
                    "n_wrong_representations": len(wrong_reps),
                    "wrong_representations": " ".join(wrong_reps),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    args = parser.parse_args()

    table_dir = args.out_dir / "tables"
    fig_dir = args.out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    manifest = load_manifest(args.data_dir)
    acoustic = load_acoustic_vowels(args.data_dir)
    whisper = load_neural_pca(args.data_dir, manifest, REPRESENTATIONS["whisper"]["npz"], REPRESENTATIONS["whisper"]["key"])
    xlsr = load_neural_pca(args.data_dir, manifest, REPRESENTATIONS["xlsr"]["npz"], REPRESENTATIONS["xlsr"]["key"])
    rep_data = {"acoustic": acoustic, "whisper": whisper, "xlsr": xlsr}

    # All-phoneme neural PCA data are needed for §9.2 consonant-vowel clustering.
    whisper_all = load_neural_pca_all(args.data_dir, manifest, REPRESENTATIONS["whisper"]["npz"], REPRESENTATIONS["whisper"]["key"])
    xlsr_all = load_neural_pca_all(args.data_dir, manifest, REPRESENTATIONS["xlsr"]["npz"], REPRESENTATIONS["xlsr"]["key"])
    neural_all = {"whisper": whisper_all, "xlsr": xlsr_all}

    print("\n[9.1 + 9.4] Vowel clustering")
    vowel_ari, vowel_assign, vowel_diag = run_vowel_clustering(rep_data, args.out_dir, args.k_min, args.k_max)
    vowel_ari.to_csv(table_dir / "vowel_ari.csv", index=False)
    vowel_assign.to_csv(table_dir / "vowel_cluster_assignments.csv", index=False)
    vowel_diag.to_csv(table_dir / "k_diagnostics_vowels.csv", index=False)
    print(vowel_ari.to_string(index=False))

    print("\n[9.2] Consonant vs vowel clustering")
    ext = load_extended_acoustic(args.data_dir)
    cv_ari, cv_assign = run_consonant_vowel_clustering(
        ext, neural_all, args.out_dir, min_n=0, consonant_mode="fixed"
    )
    if not cv_ari.empty:
        cv_ari.to_csv(table_dir / "consonant_vowel_ari.csv", index=False)
        cv_assign.to_csv(table_dir / "consonant_vowel_cluster_assignments.csv", index=False)
        print(cv_ari.to_string(index=False))

    print("\n[9.3 + 9.4] Speaker clustering")
    sp_ari, sp_assign, sp_diag = run_speaker_clustering(rep_data, manifest, args.out_dir, args.k_min, args.k_max)
    if not sp_ari.empty:
        sp_ari.to_csv(table_dir / "speaker_clustering_ari.csv", index=False)
        sp_assign.to_csv(table_dir / "speaker_cluster_assignments.csv", index=False)
        sp_diag.to_csv(table_dir / "k_diagnostics_speakers.csv", index=False)
        print(sp_ari.to_string(index=False))

    mis = systematic_misclassifications(vowel_assign) if not vowel_assign.empty else pd.DataFrame()
    mis.to_csv(table_dir / "systematic_misclassifications.csv", index=False)
    if not mis.empty:
        print("\nSystematic vowel misclassifications:")
        print(mis.to_string(index=False))
    else:
        print("\nNo systematic vowel misclassifications under the current rule.")

    print(f"\nDone. Tables → {table_dir}; Figures → {fig_dir}")


if __name__ == "__main__":
    main()
