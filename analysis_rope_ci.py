#!/usr/bin/env python
"""
analysis_rope_ci.py — Stage 8: Confidence Intervals and ROPE
================================================================

Implements §8.1–§8.4 of the lab.

Acoustic part
-------------
- Reads raw-Hz acoustic features from data/pipeline/features_acoustic.csv.
- Restricts analysis to the exact target vowel set:
      /a e i o u œ ø ɑ ə ɛ/
- For each vowel and each formant F1/F2, fits:
      formant_Hz ~ L2 + Male + (1 | speaker)
- Estimates the L1/L2 contrast as the L2 coefficient in Hz.
- Computes strict 95% profile-likelihood CIs for that coefficient.
- Uses acoustic ROPE = [-20, +20] Hz.

Neural part
-----------
- Reads full layer-20 embeddings from:
      features_whisper.npz, key hiddens_layer20
      features_xlsr.npz,    key hiddens_layer20
- Aligns rows to phoneme_tokens.csv using token_index.
- For each vowel, computes cosine distance between L1 and L2 centroids.
- Computes 95% bootstrap CIs with speaker-level resampling.
- Uses one global ROPE per representation: [0, delta0], where delta0 is the
  mean same-speaker, same-phoneme intra-speaker cosine distance.

Development default B=200. Final lab run should use --B 2000.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.optimize import brentq

try:
    import statsmodels.formula.api as smf
except ImportError as e:
    raise ImportError("statsmodels is required: pip install statsmodels") from e

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*singular.*")
warnings.filterwarnings("ignore", message=".*boundary.*")
warnings.filterwarnings("ignore", message=".*Hessian.*")
warnings.filterwarnings("ignore", message=".*gradient.*")

DATA_DIR = "data/pipeline"
TARGET_VOWELS = ["a", "e", "i", "o", "u", "œ", "ø", "ɑ", "ə", "ɛ"]
TARGET_VOWELS_SET = set(TARGET_VOWELS)
ACOUSTIC_ROPE = (-20.0, 20.0)
ALPHA = 0.05
PROFILE_CRIT = float(scipy_stats.chi2.ppf(1 - ALPHA, df=1))

# Candidate raw-Hz column names. The script auto-detects the actual names.
F1_CANDIDATES = [
    "F1_mid", "F1_midpoint", "F1_hz_mid", "F1_Hz_mid", "F1_raw_mid",
    "F1", "f1", "f1_mid", "formant1", "formant_1",
]
F2_CANDIDATES = [
    "F2_mid", "F2_midpoint", "F2_hz_mid", "F2_Hz_mid", "F2_raw_mid",
    "F2", "f2", "f2_mid", "formant2", "formant_2",
]

REPRESENTATIONS = {
    "whisper": {"filename": "features_whisper.npz", "key": "hiddens_layer20"},
    "xlsr": {"filename": "features_xlsr.npz", "key": "hiddens_layer20"},
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def encode_l1_gender(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    l1_map = {
        "fr": "L1", "fra": "L1", "l1": "L1", "french": "L1", "native": "L1",
        "ru": "L2", "rus": "L2", "l2": "L2", "russian": "L2",
    }
    g_map = {"f": "F", "female": "F", "fem": "F", "m": "M", "male": "M"}

    df["L1_norm"] = df["L1"].astype(str).str.lower().str.strip().map(l1_map)
    df["gender_norm"] = df["gender"].astype(str).str.lower().str.strip().map(g_map)

    if df["L1_norm"].isna().any():
        bad = sorted(df.loc[df["L1_norm"].isna(), "L1"].dropna().unique())
        raise ValueError(f"Unmapped L1 values: {bad}")
    if df["gender_norm"].isna().any():
        bad = sorted(df.loc[df["gender_norm"].isna(), "gender"].dropna().unique())
        raise ValueError(f"Unmapped gender values: {bad}")

    df["L2"] = (df["L1_norm"] == "L2").astype(int)
    df["Male"] = (df["gender_norm"] == "M").astype(int)
    return df


def classify_rope(ci_lo: float, ci_hi: float, rope_lo: float, rope_hi: float) -> str:
    if not np.isfinite(ci_lo) or not np.isfinite(ci_hi):
        return "CI unavailable"
    if ci_lo >= rope_lo and ci_hi <= rope_hi:
        return "Equivalent"
    if ci_hi < rope_lo or ci_lo > rope_hi:
        return "Non-equivalent"
    return "Indeterminate"


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(1.0 - np.dot(a, b) / denom)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    return x / norms


def fit_mixedlm(formula: str, df: pd.DataFrame, reml: bool = False):
    md = smf.mixedlm(formula, data=df, groups=df["speaker"])
    for method, maxiter in [("powell", 5000), ("lbfgs", 5000), ("nm", 5000)]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = md.fit(reml=reml, method=method, maxiter=maxiter, disp=False)
            if res is not None:
                return res
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Acoustic profile-likelihood CIs
# ---------------------------------------------------------------------------

def detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c

    # Fuzzy fallback: prefer non-normalised/non-Lobanov midpoint columns.
    lower_map = {c.lower(): c for c in df.columns}
    label_lower = label.lower()
    possible = []
    for c in df.columns:
        cl = c.lower()
        if label_lower in cl and "lob" not in cl and "norm" not in cl and "z" not in cl:
            possible.append(c)
    if possible:
        midpoint = [c for c in possible if "mid" in c.lower()]
        return midpoint[0] if midpoint else possible[0]

    raise KeyError(
        f"Could not detect raw-Hz column for {label}. Columns found: {list(df.columns)}"
    )


def load_acoustic_raw(data_dir: str) -> tuple[pd.DataFrame, dict[str, str]]:
    path = os.path.join(data_dir, "features_acoustic.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing acoustic raw file: {path}")

    df = pd.read_csv(path)
    if "phoneme" not in df.columns:
        raise KeyError("features_acoustic.csv must contain a phoneme column")

    if "speaker" not in df.columns or "L1" not in df.columns or "gender" not in df.columns:
        raise KeyError(
            "features_acoustic.csv must contain speaker, L1, and gender columns. "
            "If not, merge it with phoneme_tokens.csv before running this script."
        )

    df = df[~df["phoneme"].astype(str).str.contains("ding", na=False)].copy()
    df = df[df["phoneme"].astype(str).isin(TARGET_VOWELS_SET)].copy()
    df = encode_l1_gender(df)

    f1_col = detect_column(df, F1_CANDIDATES, "F1")
    f2_col = detect_column(df, F2_CANDIDATES, "F2")
    print(f"  Acoustic raw columns detected: F1={f1_col}, F2={f2_col}")
    print(f"  Acoustic target-vowel tokens: {len(df):,}; speakers={df['speaker'].nunique()}")
    return df, {"F1": f1_col, "F2": f2_col}


@dataclass
class ProfileResult:
    estimate: float
    ci_lo: float
    ci_hi: float
    ll_max: float
    profile_ok: bool
    note: str


def profile_l2_ci(dff: pd.DataFrame, response_col: str) -> ProfileResult:
    """
    Strict profile-likelihood CI for the L2 coefficient in:
        y ~ L2 + Male + (1|speaker)

    For a fixed L2 coefficient b, fit:
        y - b*L2 ~ Male + (1|speaker)
    and profile over Male, intercept, and variance components.
    """
    data = dff.dropna(subset=[response_col, "speaker", "L2", "Male"]).copy()
    data = data.rename(columns={response_col: "y"})

    if data["speaker"].nunique() < 3 or data["L2"].nunique() < 2:
        return ProfileResult(np.nan, np.nan, np.nan, np.nan, False, "too few groups")

    full = fit_mixedlm("y ~ L2 + Male", data, reml=False)
    if full is None or "L2" not in full.params:
        return ProfileResult(np.nan, np.nan, np.nan, np.nan, False, "unconstrained fit failed")

    beta_hat = float(full.params["L2"])
    se = float(full.bse["L2"]) if "L2" in full.bse.index and np.isfinite(full.bse["L2"]) else 10.0
    ll_max = float(full.llf)

    def prof_ll(beta: float) -> float:
        tmp = data.copy()
        tmp["y_profile"] = tmp["y"] - beta * tmp["L2"]
        res = fit_mixedlm("y_profile ~ Male", tmp, reml=False)
        if res is None:
            return -np.inf
        return float(res.llf)

    def root_fun(beta: float) -> float:
        ll = prof_ll(beta)
        if not np.isfinite(ll):
            return np.inf
        return 2.0 * (ll_max - ll) - PROFILE_CRIT

    def find_bound(direction: int) -> tuple[float, bool]:
        step = max(abs(se), abs(beta_hat) * 0.25, 5.0)
        near = beta_hat
        f_near = root_fun(near)  # should be approx -crit
        far = beta_hat + direction * step

        for _ in range(18):
            f_far = root_fun(far)
            if np.isfinite(f_far) and f_far >= 0:
                lo, hi = (far, near) if direction < 0 else (near, far)
                try:
                    return float(brentq(root_fun, lo, hi, maxiter=60)), True
                except Exception:
                    return np.nan, False
            step *= 1.7
            far = beta_hat + direction * step
        return np.nan, False

    ci_lo, ok_lo = find_bound(-1)
    ci_hi, ok_hi = find_bound(+1)
    ok = bool(ok_lo and ok_hi)
    note = "ok" if ok else "profile bound failed"
    return ProfileResult(beta_hat, ci_lo, ci_hi, ll_max, ok, note)


def run_acoustic_profile(df: pd.DataFrame, formant_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for formant, col in formant_cols.items():
        print(f"\n[Acoustic] Profile-likelihood CIs for {formant} ({col})")
        for phoneme in TARGET_VOWELS:
            grp = df[df["phoneme"].astype(str) == phoneme].copy()
            if grp.empty:
                continue
            res = profile_l2_ci(grp, col)
            rows.append({
                "phoneme": phoneme,
                "representation": f"Acoustic {formant}",
                "contrast": "L2-L1",
                "scale": "Hz",
                "estimate": round(res.estimate, 4) if np.isfinite(res.estimate) else np.nan,
                "ci_lower_95": round(res.ci_lo, 4) if np.isfinite(res.ci_lo) else np.nan,
                "ci_upper_95": round(res.ci_hi, 4) if np.isfinite(res.ci_hi) else np.nan,
                "rope_lower": ACOUSTIC_ROPE[0],
                "rope_upper": ACOUSTIC_ROPE[1],
                "rope_classification": classify_rope(res.ci_lo, res.ci_hi, *ACOUSTIC_ROPE),
                "n_tokens": len(grp.dropna(subset=[col])),
                "n_speakers": grp["speaker"].nunique(),
                "profile_ok": res.profile_ok,
                "note": res.note,
            })
            print(
                f"  /{phoneme}/ {formant}: est={res.estimate:.2f}, "
                f"CI=[{res.ci_lo:.2f}, {res.ci_hi:.2f}], "
                f"{rows[-1]['rope_classification']}"
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Neural bootstrap CIs and neural ROPE
# ---------------------------------------------------------------------------

def load_manifest_for_alignment(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "phoneme_tokens.csv")
    manifest = pd.read_csv(path)
    manifest["manifest_row"] = manifest.index
    manifest = manifest[~manifest["phoneme"].astype(str).str.contains("ding", na=False)].copy()
    manifest = encode_l1_gender(manifest)
    manifest["is_target_vowel"] = manifest["phoneme"].astype(str).isin(TARGET_VOWELS_SET)
    return manifest


def align_embeddings(token_index: np.ndarray, emb: np.ndarray, manifest: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, str]:
    pca_df = pd.DataFrame({"token_index": token_index.astype(int)})

    original = manifest.copy()
    filtered = manifest.copy().reset_index(drop=True)
    filtered["filtered_row"] = filtered.index

    meta_cols = ["speaker", "phoneme", "L1", "gender", "L2", "Male", "is_target_vowel"]

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
        vowels = int(m["is_target_vowel"].fillna(False).sum())
        return matched, vowels

    score_original = score(merged_original)
    score_filtered = score(merged_filtered)
    max_idx = int(np.nanmax(token_index)) if len(token_index) else -1

    if max_idx >= len(filtered) and max_idx < (manifest["manifest_row"].max() + 1):
        chosen, mode = merged_original, "original_manifest_row"
    elif len(token_index) == len(filtered) and score_filtered >= score_original:
        chosen, mode = merged_filtered, "artifact_filtered_row"
    elif score_original >= score_filtered:
        chosen, mode = merged_original, "original_manifest_row"
    else:
        chosen, mode = merged_filtered, "artifact_filtered_row"

    mask = chosen["is_target_vowel"].fillna(False).to_numpy(dtype=bool)
    meta = chosen.loc[mask].reset_index(drop=True)
    emb_sub = emb[mask]
    print(f"  token_index alignment: {mode} (original={score_original}, filtered={score_filtered})")
    print(f"  target-vowel embeddings: {emb_sub.shape[0]:,} x {emb_sub.shape[1]}")
    return meta, emb_sub, mode


def load_full_embeddings(data_dir: str, rep: str, manifest: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    cfg = REPRESENTATIONS[rep]
    path = os.path.join(data_dir, cfg["filename"])
    z = np.load(path, allow_pickle=True)
    key = cfg["key"]
    if key not in z.files:
        raise KeyError(f"{path} does not contain key {key}. Available keys: {z.files}")
    if "token_index" not in z.files:
        raise KeyError(f"{path} does not contain token_index")
    emb = np.asarray(z[key], dtype=np.float64)
    token_index = np.asarray(z["token_index"], dtype=int)
    meta, emb_sub, _ = align_embeddings(token_index, emb, manifest)
    return meta, emb_sub


def compute_neural_delta0(meta: pd.DataFrame, emb: np.ndarray) -> float:
    """
    Weighted mean intra-speaker same-phoneme cosine distance.
    Uses an exact group formula on row-normalised embeddings.
    """
    emb_norm = normalize_rows(emb)
    total_dist = 0.0
    total_pairs = 0

    df = meta[["speaker", "phoneme"]].copy()
    df["row"] = np.arange(len(df))

    for (_speaker, _phoneme), grp in df.groupby(["speaker", "phoneme"], observed=True):
        idx = grp["row"].to_numpy()
        n = len(idx)
        if n < 2:
            continue
        x = emb_norm[idx]
        if np.isnan(x).any():
            x = x[~np.isnan(x).any(axis=1)]
            n = len(x)
            if n < 2:
                continue
        s = np.sum(x, axis=0)
        sum_pair_sim = (float(np.dot(s, s)) - n) / 2.0
        n_pairs = n * (n - 1) / 2.0
        mean_dist_group = 1.0 - (sum_pair_sim / n_pairs)
        total_dist += mean_dist_group * n_pairs
        total_pairs += int(n_pairs)

    if total_pairs == 0:
        return np.nan
    return float(total_dist / total_pairs)


def centroid_distance_for_phoneme(meta: pd.DataFrame, emb: np.ndarray, phoneme: str) -> float:
    mask_p = meta["phoneme"].astype(str).to_numpy() == phoneme
    mask_l1 = mask_p & (meta["L2"].to_numpy() == 0)
    mask_l2 = mask_p & (meta["L2"].to_numpy() == 1)
    if mask_l1.sum() == 0 or mask_l2.sum() == 0:
        return np.nan
    c1 = emb[mask_l1].mean(axis=0)
    c2 = emb[mask_l2].mean(axis=0)
    return cosine_distance(c1, c2)


def speaker_phoneme_centroids(meta: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    df = meta[["speaker", "phoneme", "L2"]].copy()
    df["row"] = np.arange(len(df))
    rows = []
    for (speaker, phoneme, l2), grp in df.groupby(["speaker", "phoneme", "L2"], observed=True):
        idx = grp["row"].to_numpy()
        rows.append({
            "speaker": speaker,
            "phoneme": str(phoneme),
            "L2": int(l2),
            "n_tokens": len(idx),
            "centroid": emb[idx].mean(axis=0),
        })
    return pd.DataFrame(rows)


def bootstrap_neural_distances(meta: pd.DataFrame, emb: np.ndarray, rep: str, B: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sp_cent = speaker_phoneme_centroids(meta, emb)
    speakers = np.array(sorted(meta["speaker"].dropna().unique()))
    delta0 = compute_neural_delta0(meta, emb)
    print(f"  {rep}: global neural ROPE delta0 = {delta0:.6f}")

    rows = []
    for phoneme in TARGET_VOWELS:
        observed = centroid_distance_for_phoneme(meta, emb, phoneme)
        boot = []
        for _ in range(B):
            sampled = rng.choice(speakers, size=len(speakers), replace=True)
            counts = pd.Series(sampled).value_counts()
            sub = sp_cent[sp_cent["phoneme"] == phoneme].copy()
            sub["draw_count"] = sub["speaker"].map(counts).fillna(0).astype(int)
            sub = sub[sub["draw_count"] > 0]
            if sub.empty or sub["L2"].nunique() < 2:
                continue

            centroids = {}
            for l2_val in [0, 1]:
                g = sub[sub["L2"] == l2_val]
                if g.empty:
                    continue
                weights = (g["n_tokens"].to_numpy(dtype=float) * g["draw_count"].to_numpy(dtype=float))
                vecs = np.vstack(g["centroid"].to_numpy())
                centroids[l2_val] = np.average(vecs, axis=0, weights=weights)
            if 0 in centroids and 1 in centroids:
                d = cosine_distance(centroids[0], centroids[1])
                if np.isfinite(d):
                    boot.append(d)

        if len(boot) == 0:
            ci_lo = ci_hi = np.nan
        else:
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        rows.append({
            "phoneme": phoneme,
            "representation": rep,
            "contrast": "cosine_distance_L1_L2_centroids",
            "scale": "cosine_distance",
            "estimate": round(observed, 6) if np.isfinite(observed) else np.nan,
            "ci_lower_95": round(float(ci_lo), 6) if np.isfinite(ci_lo) else np.nan,
            "ci_upper_95": round(float(ci_hi), 6) if np.isfinite(ci_hi) else np.nan,
            "rope_lower": 0.0,
            "rope_upper": round(delta0, 6) if np.isfinite(delta0) else np.nan,
            "rope_classification": classify_rope(ci_lo, ci_hi, 0.0, delta0),
            "B_requested": B,
            "B_valid": len(boot),
            "n_tokens": int((meta["phoneme"].astype(str) == phoneme).sum()),
            "n_speakers": int(meta.loc[meta["phoneme"].astype(str) == phoneme, "speaker"].nunique()),
        })
        print(
            f"  /{phoneme}/ {rep}: d={observed:.6f}, CI=[{ci_lo:.6f}, {ci_hi:.6f}], "
            f"{rows[-1]['rope_classification']} (B_valid={len(boot)})"
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def color_for_class(cls: str) -> str:
    return {
        "Equivalent": "#4dac26",
        "Non-equivalent": "#d6604d",
        "Indeterminate": "#f1a340",
        "CI unavailable": "#999999",
    }.get(cls, "#999999")


def plot_forest(df: pd.DataFrame, title: str, xlabel: str, path: str, rope: tuple[float, float]):
    d = df.copy()
    d["phoneme"] = pd.Categorical(d["phoneme"], categories=TARGET_VOWELS, ordered=True)
    d = d.sort_values("phoneme")
    d = d.dropna(subset=["estimate", "ci_lower_95", "ci_upper_95"])
    if d.empty:
        print(f"  WARNING: no data to plot for {title}")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(d) + 1.5)))
    y = np.arange(len(d))
    colors = [color_for_class(c) for c in d["rope_classification"]]

    ax.axvspan(rope[0], rope[1], color="#cccccc", alpha=0.35, label="ROPE")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    # Draw confidence intervals as explicit horizontal line segments rather
    # than Matplotlib xerr values. Percentile bootstrap CIs can occasionally
    # have ci_lower > point estimate or ci_upper < point estimate, which is
    # statistically possible and should not crash plotting.
    for yi, (_, row), color in zip(y, d.iterrows(), colors):
        lo = float(row["ci_lower_95"])
        hi = float(row["ci_upper_95"])
        est = float(row["estimate"])
        if lo > hi:
            lo, hi = hi, lo
        ax.hlines(yi, lo, hi, color="black", linewidth=0.9, zorder=1)
        ax.plot([lo, lo], [yi - 0.08, yi + 0.08], color="black", linewidth=0.9, zorder=1)
        ax.plot([hi, hi], [yi - 0.08, yi + 0.08], color="black", linewidth=0.9, zorder=1)
        ax.scatter(est, yi, color=color, s=45, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels([f"/{p}/" for p in d["phoneme"].astype(str)])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default="results/ci_rope")
    parser.add_argument("--B", type=int, default=200, help="Bootstrap iterations; final lab run should use 2000")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-acoustic", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    args = parser.parse_args()

    tbl_dir = os.path.join(args.out_dir, "tables")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(tbl_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    all_rows = []

    if not args.skip_acoustic:
        print("\n[8.1 + 8.3.1 + 8.4] Acoustic profile-likelihood CIs and ROPE")
        acoustic_df, formant_cols = load_acoustic_raw(args.data_dir)
        acoustic_res = run_acoustic_profile(acoustic_df, formant_cols)
        acoustic_path = os.path.join(tbl_dir, "acoustic_profile_ci_rope.csv")
        acoustic_res.to_csv(acoustic_path, index=False)
        print(f"  Acoustic table → {acoustic_path}")
        all_rows.append(acoustic_res)

        for formant in ["F1", "F2"]:
            sub = acoustic_res[acoustic_res["representation"] == f"Acoustic {formant}"]
            plot_forest(
                sub,
                title=f"Acoustic {formant}: L1/L2 contrast with 95% profile-likelihood CI",
                xlabel=f"L2 − L1 difference in {formant} (Hz)",
                path=os.path.join(fig_dir, f"forest_acoustic_{formant}_rope.png"),
                rope=ACOUSTIC_ROPE,
            )

    if not args.skip_neural:
        print("\n[8.2 + 8.3.2 + 8.4] Neural bootstrap CIs and ROPE")
        manifest = load_manifest_for_alignment(args.data_dir)
        neural_tables = []
        for rep in ["whisper", "xlsr"]:
            print(f"\n[Neural] {rep}")
            meta, emb = load_full_embeddings(args.data_dir, rep, manifest)
            res = bootstrap_neural_distances(meta, emb, rep, B=args.B, seed=args.seed)
            neural_tables.append(res)
            plot_forest(
                res,
                title=f"{rep.upper()}: L1/L2 centroid cosine distance with speaker-bootstrap CI",
                xlabel="Cosine distance between L1 and L2 centroids",
                path=os.path.join(fig_dir, f"forest_neural_{rep}_rope.png"),
                rope=(0.0, float(res["rope_upper"].dropna().iloc[0]) if res["rope_upper"].notna().any() else 0.0),
            )
        neural_res = pd.concat(neural_tables, ignore_index=True)
        neural_path = os.path.join(tbl_dir, "neural_bootstrap_ci_rope.csv")
        neural_res.to_csv(neural_path, index=False)
        print(f"  Neural table → {neural_path}")
        all_rows.append(neural_res)

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        summary_path = os.path.join(tbl_dir, "rope_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary table → {summary_path}")

        counts = (
            summary.groupby(["representation", "rope_classification"])
            .size().reset_index(name="n")
            .sort_values(["representation", "rope_classification"])
        )
        counts_path = os.path.join(tbl_dir, "rope_classification_counts.csv")
        counts.to_csv(counts_path, index=False)
        print(f"Classification counts → {counts_path}")
        print(counts.to_string(index=False))

    if args.B < 2000:
        print("\nREMINDER: this was a development run. For the final lab result, rerun with:")
        print("  pixi run python analysis_rope_ci.py --B 2000")


if __name__ == "__main__":
    main()
