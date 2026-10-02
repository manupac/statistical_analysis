"""
analysis_trajectory_sensitivity.py

Sensitivity analysis for the lab requirement:
"for long vowels (duration > 80 ms), additionally extract measurements at 25% and
75% of the interval and report whether using the trajectory rather than the
midpoint changes your conclusions."

This script compares midpoint-only acoustic conclusions with trajectory-based
conclusions on the same long-vowel subset.

Input:
    data/pipeline/features_acoustic_norm.csv

Expected columns:
    speaker, phoneme, L1, gender, duration_ms
    F1_lob_mid, F2_lob_mid
    F1_lob_25, F2_lob_25, F1_lob_75, F2_lob_75

Main output:
    results/trajectory_midpoint/tables/trajectory_sensitivity_summary.csv

Optional detail outputs:
    results/trajectory_midpoint/tables/trajectory_l1l2_tests.csv
    results/trajectory_midpoint/tables/trajectory_variable_vowels.csv
    results/trajectory_midpoint/tables/trajectory_classifier_accuracy.csv

Usage:
    pixi run python analysis_trajectory_sensitivity.py

Optional:
    pixi run python analysis_trajectory_sensitivity.py --vowels a,e,i,o,u,œ,ø,ɑ,ə,ɛ
    pixi run python analysis_trajectory_sensitivity.py --top-n-variable 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import levene, mannwhitneyu, shapiro, spearmanr, ttest_ind


DEFAULT_DATA = Path("data/pipeline/features_acoustic_norm.csv")
DEFAULT_OUT = Path("results/trajectory_midpoint")

# Includes /y/ by default because it is a canonical French oral vowel and appears
# in earlier acoustic analyses. Use --vowels to restrict this set if needed.
DEFAULT_VOWELS = ["a", "e", "i", "o", "u", "y", "œ", "ø", "ɑ", "ə", "ɛ"]

MID_COLS = ["F1_lob_mid", "F2_lob_mid"]
TRAJ_COLS = [
    "F1_lob_25", "F2_lob_25",
    "F1_lob_mid", "F2_lob_mid",
    "F1_lob_75", "F2_lob_75",
]

ALPHA = 0.05


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def parse_vowels(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return DEFAULT_VOWELS
    return [v.strip() for v in value.split(",") if v.strip()]


def normalise_l1(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in {"l1", "fr", "fra", "french", "native"}:
        return "L1"
    if v in {"l2", "ru", "rus", "russian"}:
        return "L2"
    return None


def normalise_gender(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in {"f", "female", "fem"}:
        return "F"
    if v in {"m", "male"}:
        return "M"
    return None


def bh_correction(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(list(p_values), dtype=float)
    n = len(p)
    if n == 0:
        return p
    out = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = np.empty(m, dtype=float)
    ranked[order] = np.arange(1, m + 1)
    adj = pv * m / ranked
    adj_sorted = adj[order]
    for i in range(m - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = np.minimum(adj_sorted, 1.0)
    out[valid] = adj
    return out


def label_change(mid_value, traj_value, kind: str = "set") -> str:
    """Return No / partly / yes."""
    if kind == "set":
        mid_set = set(mid_value)
        traj_set = set(traj_value)
        if mid_set == traj_set:
            return "No"
        if len(mid_set & traj_set) > 0:
            return "partly"
        return "yes"
    if kind == "numeric":
        diff = abs(float(traj_value) - float(mid_value))
        if diff < 0.02:
            return "No"
        if diff < 0.05:
            return "partly"
        return "yes"
    return "partly"


def fmt_list(values: Iterable[str]) -> str:
    vals = list(values)
    if not vals:
        return "none"
    return ", ".join(f"/{v}/" for v in vals)


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Input is missing required columns: " + ", ".join(missing) + "\n"
            "Make sure extract_acoustics.py extracted 25%/75% values and "
            "normalize.py created F1_lob_25/F2_lob_25/F1_lob_75/F2_lob_75."
        )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_long_vowels(path: Path, vowels: list[str], long_threshold_ms: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    df = pd.read_csv(path)
    require_columns(
        df,
        ["speaker", "phoneme", "L1", "gender", "duration_ms"] + TRAJ_COLS,
    )

    df = df.copy()
    df["L1_norm"] = df["L1"].map(normalise_l1)
    df["gender_norm"] = df["gender"].map(normalise_gender)

    # Compare midpoint and trajectory on the same complete-case long-vowel subset.
    mask = (
        df["phoneme"].isin(vowels)
        & df["L1_norm"].isin(["L1", "L2"])
        & df["duration_ms"].astype(float).gt(long_threshold_ms)
    )
    df = df.loc[mask].copy()
    df = df.dropna(subset=TRAJ_COLS + ["speaker", "phoneme", "L1_norm"])

    # Compact trajectory means for group tests.
    df["F1_lob_traj_mean"] = df[["F1_lob_25", "F1_lob_mid", "F1_lob_75"]].mean(axis=1)
    df["F2_lob_traj_mean"] = df[["F2_lob_25", "F2_lob_mid", "F2_lob_75"]].mean(axis=1)

    return df


# ---------------------------------------------------------------------
# Analysis 1: vowel-space structure
# ---------------------------------------------------------------------

def distance_structure_similarity(df: pd.DataFrame, vowels: list[str]) -> dict:
    present = [v for v in vowels if v in set(df["phoneme"])]
    if len(present) < 3:
        return {
            "midpoint": "not enough vowels",
            "trajectory": "not enough vowels",
            "changed": "yes",
            "metric": np.nan,
        }

    mid_centroids = []
    traj_centroids = []
    used = []
    for ph in present:
        sub = df[df["phoneme"] == ph]
        mid = sub[MID_COLS].mean().values.astype(float)
        traj = sub[TRAJ_COLS].mean().values.astype(float)
        if np.isfinite(mid).all() and np.isfinite(traj).all():
            mid_centroids.append(mid)
            traj_centroids.append(traj)
            used.append(ph)

    if len(used) < 3:
        return {
            "midpoint": "not enough complete vowel centroids",
            "trajectory": "not enough complete vowel centroids",
            "changed": "yes",
            "metric": np.nan,
        }

    d_mid = squareform(pdist(np.vstack(mid_centroids), metric="euclidean"))
    d_traj = squareform(pdist(np.vstack(traj_centroids), metric="euclidean"))
    iu = np.triu_indices(len(used), k=1)
    rho, p = spearmanr(d_mid[iu], d_traj[iu])

    if not np.isfinite(rho):
        changed = "yes"
        traj_txt = "trajectory distance ranking could not be compared"
    elif rho >= 0.85:
        changed = "No"
        traj_txt = f"very similar distance ranking (Spearman rho={rho:.3f})"
    elif rho >= 0.60:
        changed = "partly"
        traj_txt = f"partly similar distance ranking (Spearman rho={rho:.3f})"
    else:
        changed = "yes"
        traj_txt = f"substantially different distance ranking (Spearman rho={rho:.3f})"

    return {
        "midpoint": f"baseline midpoint vowel-space structure across {len(used)} vowels",
        "trajectory": traj_txt,
        "changed": changed,
        "metric": rho,
        "p_value": p,
    }


# ---------------------------------------------------------------------
# Analysis 2: most variable vowels
# ---------------------------------------------------------------------

def variable_vowels(df: pd.DataFrame, vowels: list[str], top_n: int) -> tuple[list[str], list[str], pd.DataFrame]:
    rows = []
    for ph in vowels:
        sub = df[df["phoneme"] == ph]
        if len(sub) < 3:
            continue
        mid_var = float(sub[MID_COLS].var(ddof=1).sum())
        traj_var = float(sub[TRAJ_COLS].var(ddof=1).sum())
        rows.append({
            "phoneme": ph,
            "n": len(sub),
            "midpoint_total_variance": mid_var,
            "trajectory_total_variance": traj_var,
        })

    tab = pd.DataFrame(rows)
    if tab.empty:
        return [], [], tab

    mid_top = tab.sort_values("midpoint_total_variance", ascending=False)["phoneme"].head(top_n).tolist()
    traj_top = tab.sort_values("trajectory_total_variance", ascending=False)["phoneme"].head(top_n).tolist()
    return mid_top, traj_top, tab


# ---------------------------------------------------------------------
# Analysis 3/4: L1/L2 significant F1/F2 contrasts
# ---------------------------------------------------------------------

def choose_group_test(g1: np.ndarray, g2: np.ndarray, alpha: float = ALPHA) -> tuple[str, float, float]:
    """Return test_name, statistic, p_value."""
    if len(g1) < 3 or len(g2) < 3:
        return "insufficient_n", np.nan, np.nan

    # Subsample for Shapiro if needed; deterministic because values are sorted.
    max_shapiro = 5000
    s1 = np.sort(g1)[:max_shapiro] if len(g1) > max_shapiro else g1
    s2 = np.sort(g2)[:max_shapiro] if len(g2) > max_shapiro else g2

    try:
        normal1 = shapiro(s1).pvalue > alpha
        normal2 = shapiro(s2).pvalue > alpha
    except Exception:
        normal1 = normal2 = False

    try:
        equal_var = levene(g1, g2, center="median").pvalue > alpha
    except Exception:
        equal_var = False

    if normal1 and normal2:
        stat, p = ttest_ind(g1, g2, equal_var=equal_var, nan_policy="omit")
        return ("t-test (Student)" if equal_var else "t-test (Welch)"), float(stat), float(p)

    stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
    return "Mann-Whitney U", float(stat), float(p)


def l1l2_tests(df: pd.DataFrame, vowels: list[str]) -> pd.DataFrame:
    specs = [
        ("midpoint", "F1", "F1_lob_mid"),
        ("midpoint", "F2", "F2_lob_mid"),
        ("trajectory", "F1", "F1_lob_traj_mean"),
        ("trajectory", "F2", "F2_lob_traj_mean"),
    ]
    rows = []
    for feature_type, formant, col in specs:
        for ph in vowels:
            sub = df[df["phoneme"] == ph]
            g1 = sub[sub["L1_norm"] == "L1"][col].dropna().to_numpy(float)
            g2 = sub[sub["L1_norm"] == "L2"][col].dropna().to_numpy(float)
            test_name, stat, p = choose_group_test(g1, g2)
            rows.append({
                "feature_type": feature_type,
                "formant": formant,
                "phoneme": ph,
                "n_L1": len(g1),
                "n_L2": len(g2),
                "mean_L1": float(np.mean(g1)) if len(g1) else np.nan,
                "mean_L2": float(np.mean(g2)) if len(g2) else np.nan,
                "estimate_L2_minus_L1": (float(np.mean(g2) - np.mean(g1)) if len(g1) and len(g2) else np.nan),
                "test": test_name,
                "stat": stat,
                "p_raw": p,
            })

    tab = pd.DataFrame(rows)
    # BH separately for midpoint and trajectory across all vowel × formant tests.
    tab["p_adj_BH"] = np.nan
    for feature_type in ["midpoint", "trajectory"]:
        idx = tab["feature_type"] == feature_type
        tab.loc[idx, "p_adj_BH"] = bh_correction(tab.loc[idx, "p_raw"].to_numpy(float))
    tab["significant_BH"] = tab["p_adj_BH"] < ALPHA
    return tab


def significant_vowels(test_tab: pd.DataFrame, feature_type: str, formant: str) -> list[str]:
    sub = test_tab[
        (test_tab["feature_type"] == feature_type)
        & (test_tab["formant"] == formant)
        & (test_tab["significant_BH"])
    ]
    return sub.sort_values("phoneme")["phoneme"].tolist()


# ---------------------------------------------------------------------
# Analysis 5: phoneme identification accuracy
# ---------------------------------------------------------------------

def loso_nearest_centroid(df: pd.DataFrame, feature_cols: list[str], vowels: list[str]) -> pd.DataFrame:
    rows = []
    speakers = sorted(df["speaker"].dropna().unique())
    for held_out in speakers:
        train = df[(df["speaker"] != held_out) & df["phoneme"].isin(vowels)].dropna(subset=feature_cols)
        test = df[(df["speaker"] == held_out) & df["phoneme"].isin(vowels)].dropna(subset=feature_cols)
        if train.empty or test.empty:
            continue

        centroids = {}
        for ph, sub in train.groupby("phoneme"):
            if len(sub) > 0:
                centroids[ph] = sub[feature_cols].mean().to_numpy(float)
        if not centroids:
            continue
        labels = list(centroids.keys())
        arr = np.vstack([centroids[p] for p in labels])

        for idx, row in test.iterrows():
            x = row[feature_cols].to_numpy(float)
            d = np.sqrt(((arr - x) ** 2).sum(axis=1))
            pred = labels[int(np.argmin(d))]
            rows.append({
                "index": idx,
                "speaker": row["speaker"],
                "L1_norm": row["L1_norm"],
                "true": row["phoneme"],
                "pred": pred,
                "correct": pred == row["phoneme"],
            })
    return pd.DataFrame(rows)


def accuracy_summary(preds: pd.DataFrame) -> dict:
    if preds.empty:
        return {"overall": np.nan, "L1": np.nan, "L2": np.nan, "n": 0}
    return {
        "overall": float(preds["correct"].mean()),
        "L1": float(preds.loc[preds["L1_norm"] == "L1", "correct"].mean()),
        "L2": float(preds.loc[preds["L1_norm"] == "L2", "correct"].mean()),
        "n": int(len(preds)),
    }


# ---------------------------------------------------------------------
# Main summary construction
# ---------------------------------------------------------------------

def build_summary(df: pd.DataFrame, vowels: list[str], top_n: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_rows = []
    details: dict[str, pd.DataFrame] = {}

    # 1. Vowel-space structure
    dstruct = distance_structure_similarity(df, vowels)
    summary_rows.append({
        "Analysis": "Vowel-space structure",
        "Midpoint conclusion": dstruct["midpoint"],
        "Trajectory-based conclusion": dstruct["trajectory"],
        "Did the conclusion change?": dstruct["changed"],
    })

    # 2. Most variable vowels
    mid_top, traj_top, var_tab = variable_vowels(df, vowels, top_n=top_n)
    details["variable_vowels"] = var_tab
    summary_rows.append({
        "Analysis": f"Most variable vowels (top {top_n})",
        "Midpoint conclusion": fmt_list(mid_top),
        "Trajectory-based conclusion": fmt_list(traj_top),
        "Did the conclusion change?": label_change(mid_top, traj_top, kind="set"),
    })

    # 3/4. L1/L2 significant contrasts
    tests = l1l2_tests(df, vowels)
    details["l1l2_tests"] = tests
    for formant in ["F1", "F2"]:
        mid_sig = significant_vowels(tests, "midpoint", formant)
        traj_sig = significant_vowels(tests, "trajectory", formant)
        summary_rows.append({
            "Analysis": f"L1/L2 significant {formant} contrasts after BH-FDR",
            "Midpoint conclusion": fmt_list(mid_sig),
            "Trajectory-based conclusion": fmt_list(traj_sig),
            "Did the conclusion change?": label_change(mid_sig, traj_sig, kind="set"),
        })

    # 5. Phoneme identification
    pred_mid = loso_nearest_centroid(df, MID_COLS, vowels)
    pred_traj = loso_nearest_centroid(df, TRAJ_COLS, vowels)
    acc_mid = accuracy_summary(pred_mid)
    acc_traj = accuracy_summary(pred_traj)
    acc_tab = pd.DataFrame([
        {"feature_type": "midpoint", **acc_mid},
        {"feature_type": "trajectory", **acc_traj},
    ])
    details["classifier_accuracy"] = acc_tab

    summary_rows.append({
        "Analysis": "Acoustic phoneme identification accuracy",
        "Midpoint conclusion": f"overall={acc_mid['overall']:.4f}; L1={acc_mid['L1']:.4f}; L2={acc_mid['L2']:.4f}",
        "Trajectory-based conclusion": f"overall={acc_traj['overall']:.4f}; L1={acc_traj['L1']:.4f}; L2={acc_traj['L2']:.4f}",
        "Did the conclusion change?": label_change(acc_mid["overall"], acc_traj["overall"], kind="numeric"),
    })

    return pd.DataFrame(summary_rows), details


def main() -> None:
    parser = argparse.ArgumentParser(description="Midpoint vs 25%/75% trajectory sensitivity analysis.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vowels", type=str, default=None,
                        help="Comma-separated target vowels. Default includes a,e,i,o,u,y,œ,ø,ɑ,ə,ɛ.")
    parser.add_argument("--long-threshold-ms", type=float, default=80.0)
    parser.add_argument("--top-n-variable", type=int, default=4)
    args = parser.parse_args()

    vowels = parse_vowels(args.vowels)
    tables_dir = args.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_long_vowels(args.input, vowels, args.long_threshold_ms)
    if df.empty:
        raise ValueError("No complete long-vowel rows remained after filtering. Check trajectory columns and vowel set.")

    present = sorted(df["phoneme"].unique().tolist())
    print(f"Loaded complete long-vowel subset: {len(df):,} tokens, {df['speaker'].nunique()} speakers")
    print(f"Target vowels present: {present}")

    summary, details = build_summary(df, vowels, top_n=args.top_n_variable)

    summary_path = tables_dir / "trajectory_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Detail outputs are useful for auditability; the summary table remains the main report table.
    details["l1l2_tests"].to_csv(tables_dir / "trajectory_l1l2_tests.csv", index=False)
    details["variable_vowels"].to_csv(tables_dir / "trajectory_variable_vowels.csv", index=False)
    details["classifier_accuracy"].to_csv(tables_dir / "trajectory_classifier_accuracy.csv", index=False)

    print("\nTrajectory sensitivity summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved main table -> {summary_path}")
    print(f"Saved detail tables -> {tables_dir}")


if __name__ == "__main__":
    main()
