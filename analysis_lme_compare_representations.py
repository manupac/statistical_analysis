"""
Compare acoustic and neural LME outputs across representation types.
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*singular.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*boundary.*")
warnings.filterwarnings("ignore", message=".*Hessian.*")
warnings.filterwarnings("ignore", message=".*gradient.*", category=UserWarning)

DATA_DIR = Path("data/pipeline")
ACOUSTIC_DIR = Path("results/lme_acoustic")
NEURAL_DIR = Path("results/lme_neural")
OUT_DIR = Path("results/compare_representations")
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"

LAYER = 20
PCA_DIM = 5
MODEL_ORDER = ["M0", "M1", "M2", "M3", "M4"]
TARGET_VOWELS = {"a", "e", "i", "o", "u", "y", "œ", "ø", "ɑ", "ə", "ɛ"}

REP_COLOURS = {
    "Acoustic F1": "#1f78b4",
    "Acoustic F2": "#a6cee3",
    "Whisper PC": "#e31a1c",
    "XLS-R PC": "#fb9a99",
}


# ---------------------------------------------------------------------
# Generic model helpers
# ---------------------------------------------------------------------

def fit_mixedlm(formula: str, df: pd.DataFrame, reml: bool = False):
    model = smf.mixedlm(formula, data=df, groups=df["speaker"])

    for method, maxiter in [("profile", 200), ("powell", 5000), ("lbfgs", 5000), ("nm", 5000)]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return model.fit(reml=reml, method=method, maxiter=maxiter, disp=False)
        except Exception:
            continue

    warnings.warn(f"Model fitting failed: {formula}")
    return None


def compute_r2(model_fit) -> tuple[float, float]:
    try:
        cov_re = model_fit.cov_re
        sigma2_u = 0.0 if cov_re is None or cov_re.size == 0 else float(np.trace(np.asarray(cov_re)))
        sigma2 = float(model_fit.scale)

        x = np.asarray(model_fit.model.exog, dtype=float)
        beta = np.asarray(model_fit.fe_params, dtype=float)
        fixed_pred = x @ beta
        var_fixed = float(np.var(fixed_pred, ddof=0))

        denominator = var_fixed + sigma2_u + sigma2
        if denominator <= 0 or not np.isfinite(denominator):
            return np.nan, np.nan

        return var_fixed / denominator, (var_fixed + sigma2_u) / denominator
    except Exception:
        return np.nan, np.nan


# ---------------------------------------------------------------------
# Existing comparison outputs: R², Q8 ICC, Q9 interaction
# ---------------------------------------------------------------------

def fit_null_icc(df: pd.DataFrame, response_col: str, label: str) -> dict:
    dff = df.dropna(subset=[response_col]).copy()
    dff = dff.rename(columns={response_col: "y"})

    if dff["speaker"].nunique() < 3:
        return {
            "representation": label,
            "ICC": np.nan,
            "sigma2_u": np.nan,
            "sigma2": np.nan,
            "n_tokens": len(dff),
            "n_speakers": dff["speaker"].nunique(),
        }

    result = fit_mixedlm("y ~ 1", dff, reml=True)

    if result is None:
        return {
            "representation": label,
            "ICC": np.nan,
            "sigma2_u": np.nan,
            "sigma2": np.nan,
            "n_tokens": len(dff),
            "n_speakers": dff["speaker"].nunique(),
        }

    sigma2_u = float(result.cov_re.iloc[0, 0])
    sigma2 = float(result.scale)
    icc = sigma2_u / (sigma2_u + sigma2) if (sigma2_u + sigma2) > 0 else np.nan

    return {
        "representation": label,
        "ICC": round(icc, 4),
        "sigma2_u": round(sigma2_u, 4),
        "sigma2": round(sigma2, 4),
        "n_tokens": len(dff),
        "n_speakers": dff["speaker"].nunique(),
    }


def load_acoustic_a() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "features_acoustic_norm.csv")
    df = df[~df["phoneme"].str.contains("ding", na=False)].copy()
    return df[df["phoneme"] == "a"].copy()


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(DATA_DIR / "phoneme_tokens.csv")
    manifest["manifest_row"] = manifest.index
    manifest = manifest[~manifest["phoneme"].str.contains("ding", na=False)].copy()
    return manifest


def align_pca_with_manifest(pca_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    original = manifest.copy()
    filtered = manifest.copy().reset_index(drop=True)
    filtered["filtered_row"] = filtered.index

    meta_cols = ["speaker", "phoneme", "L1", "gender"]

    merged_original = pca_df.merge(
        original[["manifest_row"] + meta_cols],
        left_on="token_index",
        right_on="manifest_row",
        how="left",
    )
    merged_filtered = pca_df.merge(
        filtered[["filtered_row", "manifest_row"] + meta_cols],
        left_on="token_index",
        right_on="filtered_row",
        how="left",
    )

    original_score = int(merged_original["speaker"].notna().sum())
    filtered_score = int(merged_filtered["speaker"].notna().sum())

    return merged_original if original_score >= filtered_score else merged_filtered


def load_whisper_a() -> pd.DataFrame:
    df = load_neural_pc_dataframe("whisper")
    return df[df["phoneme"] == "a"].copy()


def collect_r2() -> pd.DataFrame:
    rows = []

    acoustic_path = ACOUSTIC_DIR / "tables" / "r2_all_formants.csv"
    acoustic = pd.read_csv(acoustic_path)

    for _, row in acoustic.iterrows():
        rows.append({
            "representation": f"Acoustic {row['formant']}",
            "model": row["model"],
            "R2_marginal": row.get("R2_marginal", np.nan),
            "R2_conditional": row.get("R2_conditional", np.nan),
            "AIC": row.get("AIC", np.nan),
            "BIC": row.get("BIC", np.nan),
        })

    for representation, subdir in [("Whisper PC", "whisper"), ("XLS-R PC", "xlsr")]:
        summary_path = NEURAL_DIR / subdir / "tables" / "r2_summary.csv"
        summary = pd.read_csv(summary_path)

        for _, row in summary.iterrows():
            rows.append({
                "representation": representation,
                "model": row["model"],
                "R2_marginal": row.get("R2_marginal", np.nan),
                "R2_conditional": row.get("R2_conditional", np.nan),
                "AIC": np.nan,
                "BIC": np.nan,
            })

    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(["representation", "model"]).reset_index(drop=True)


def selected_model_from_lrt(lrt_path: Path, r2_sub: pd.DataFrame, label: str) -> dict:
    if not lrt_path.exists():
        valid = r2_sub.dropna(subset=["R2_marginal"])
        if valid.empty:
            return {}
        row = valid.loc[valid["R2_marginal"].idxmax()]
        return {
            "representation": label,
            "selected_model": row["model"],
            "R2_marginal": row["R2_marginal"],
            "R2_conditional": row["R2_conditional"],
            "selection_method": "max_R2_fallback",
        }

    lrt = pd.read_csv(lrt_path)
    selected = "M0"
    model_map = {
        "M0 → M1": "M1",
        "M0→M1": "M1",
        "M1 → M2": "M2",
        "M1→M2": "M2",
        "M2 → M3": "M3",
        "M2→M3": "M3",
        "M3 → M4": "M4",
        "M3→M4": "M4",
    }

    for _, row in lrt.iterrows():
        comparison = str(row.get("comparison", "")).strip()
        candidate = model_map.get(comparison)
        p_lrt = row.get("p_lrt", np.nan)
        aic_fuller = row.get("AIC_fuller", np.nan)
        aic_simpler = row.get("AIC_simpler", np.nan)
        converged = row.get("converged", True)

        if (
            candidate
            and bool(converged)
            and pd.notna(p_lrt)
            and p_lrt < 0.05
            and pd.notna(aic_fuller)
            and pd.notna(aic_simpler)
            and aic_fuller < aic_simpler
        ):
            selected = candidate

    match = r2_sub[r2_sub["model"] == selected]
    if match.empty:
        return {}

    row = match.iloc[0]
    return {
        "representation": label,
        "selected_model": selected,
        "R2_marginal": row["R2_marginal"],
        "R2_conditional": row["R2_conditional"],
        "selection_method": "LRT+AIC",
    }


def extract_selected_model_r2(r2_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for formant in ["F1", "F2"]:
        label = f"Acoustic {formant}"
        sub = r2_df[r2_df["representation"] == label]
        row = selected_model_from_lrt(
            ACOUSTIC_DIR / "tables" / f"model_comparison_{formant}.csv",
            sub,
            label,
        )
        if row:
            rows.append(row)

    for label, subdir in [("Whisper PC", "whisper"), ("XLS-R PC", "xlsr")]:
        sub = r2_df[r2_df["representation"] == label]
        row = selected_model_from_lrt(
            NEURAL_DIR / subdir / "tables" / "model_comparison_PC1.csv",
            sub,
            label,
        )
        if row:
            rows.append(row)

    return pd.DataFrame(rows)


def collect_interaction_significance() -> pd.DataFrame:
    rows = []
    targets = [
        ("Acoustic F1", ACOUSTIC_DIR / "tables" / "coefficients_F1.csv"),
        ("Acoustic F2", ACOUSTIC_DIR / "tables" / "coefficients_F2.csv"),
    ]

    for pc in range(1, PCA_DIM + 1):
        targets.append((
            f"Whisper PC{pc}",
            NEURAL_DIR / "whisper" / "tables" / f"coefficients_PC{pc}.csv",
        ))

    for label, path in targets:
        if not path.exists():
            rows.append({
                "representation": label,
                "interaction_term": "L2:Male",
                "estimate": np.nan,
                "SE": np.nan,
                "p_value": np.nan,
                "significant_05": np.nan,
                "available": False,
            })
            continue

        coef = pd.read_csv(path)
        mask = coef["term"].str.contains("L2", na=False) & coef["term"].str.contains("Male", na=False)

        if mask.any():
            row = coef[mask].iloc[0]
            p_value = float(row["p_value"])
            rows.append({
                "representation": label,
                "interaction_term": row["term"],
                "estimate": round(float(row["estimate"]), 6),
                "SE": round(float(row["SE"]), 6) if pd.notna(row.get("SE", np.nan)) else np.nan,
                "p_value": round(p_value, 6),
                "significant_05": bool(p_value < 0.05),
                "available": True,
            })
        else:
            rows.append({
                "representation": label,
                "interaction_term": "L2:Male (not in selected model)",
                "estimate": np.nan,
                "SE": np.nan,
                "p_value": np.nan,
                "significant_05": False,
                "available": True,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Q10-specific correction: L2 contribution after controlling for gender
# ---------------------------------------------------------------------

def add_l1_gender_columns(df: pd.DataFrame) -> pd.DataFrame:
    l1_map = {
        "fr": "L1", "fra": "L1", "l1": "L1", "french": "L1", "native": "L1",
        "ru": "L2", "rus": "L2", "l2": "L2", "russian": "L2",
    }
    gender_map = {"f": "F", "female": "F", "fem": "F", "m": "M", "male": "M"}

    df = df.copy()
    df["L1_norm"] = df["L1"].astype(str).str.lower().str.strip().map(l1_map)
    df["gender_norm"] = df["gender"].astype(str).str.lower().str.strip().map(gender_map)
    df["L2"] = (df["L1_norm"] == "L2").astype(int)
    df["Male"] = (df["gender_norm"] == "M").astype(int)
    return df.dropna(subset=["speaker", "L1_norm", "gender_norm", "L2", "Male"])


def load_acoustic_lme_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "features_acoustic_norm.csv")
    df = df[~df["phoneme"].str.contains("ding", na=False)].copy()
    df = df[df["phoneme"].isin(TARGET_VOWELS)].copy()
    return add_l1_gender_columns(df).reset_index(drop=True)


def load_neural_pc_dataframe(model: str) -> pd.DataFrame:
    npz_name = "features_whisper_pca.npz" if model == "whisper" else "features_xlsr_pca.npz"
    npz = np.load(DATA_DIR / npz_name, allow_pickle=True)
    key = f"hiddens_layer{LAYER}_pca{PCA_DIM}"

    if key not in npz:
        raise KeyError(f"{key} not found in {npz_name}")

    pc_cols = [f"PC{i}" for i in range(1, PCA_DIM + 1)]
    pca_df = pd.DataFrame(npz[key], columns=pc_cols)
    pca_df["token_index"] = npz["token_index"].astype(int)

    merged = align_pca_with_manifest(pca_df, load_manifest())
    merged = merged[merged["phoneme"].isin(TARGET_VOWELS)].copy()
    merged = add_l1_gender_columns(merged)
    return merged.reset_index(drop=True)


def l2_unique_r2_for_response(df: pd.DataFrame, response_col: str) -> dict:
    dff = df.dropna(subset=[response_col]).rename(columns={response_col: "y"}).copy()

    gender_only = fit_mixedlm("y ~ Male", dff, reml=False)
    gender_plus_l2 = fit_mixedlm("y ~ Male + L2", dff, reml=False)

    if gender_only is None or gender_plus_l2 is None:
        return {
            "R2_marginal_gender_only": np.nan,
            "R2_marginal_gender_plus_L2": np.nan,
            "delta_L2_given_gender": np.nan,
        }

    r2_gender, _ = compute_r2(gender_only)
    r2_gender_l2, _ = compute_r2(gender_plus_l2)

    return {
        "R2_marginal_gender_only": r2_gender,
        "R2_marginal_gender_plus_L2": r2_gender_l2,
        "delta_L2_given_gender": r2_gender_l2 - r2_gender,
    }


def extract_l2_unique_r2_gain_for_q10() -> pd.DataFrame:
    rows = []

    acoustic = load_acoustic_lme_data()
    for label, response_col in [("Acoustic F1", "F1_lob_mid"), ("Acoustic F2", "F2_lob_mid")]:
        values = l2_unique_r2_for_response(acoustic, response_col)
        rows.append({
            "representation": label,
            "n_components": 1,
            "R2_marginal_gender_only": round(values["R2_marginal_gender_only"], 6),
            "R2_marginal_gender_plus_L2": round(values["R2_marginal_gender_plus_L2"], 6),
            "delta_L2_given_gender": round(values["delta_L2_given_gender"], 6),
        })

    for label, model in [("Whisper PC", "whisper"), ("XLS-R PC", "xlsr")]:
        neural = load_neural_pc_dataframe(model)
        pc_rows = []

        for pc in range(1, PCA_DIM + 1):
            pc_rows.append(l2_unique_r2_for_response(neural, f"PC{pc}"))

        pc_df = pd.DataFrame(pc_rows)
        rows.append({
            "representation": label,
            "n_components": PCA_DIM,
            "R2_marginal_gender_only": round(float(pc_df["R2_marginal_gender_only"].mean()), 6),
            "R2_marginal_gender_plus_L2": round(float(pc_df["R2_marginal_gender_plus_L2"].mean()), 6),
            "delta_L2_given_gender": round(float(pc_df["delta_L2_given_gender"].mean()), 6),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

def plot_r2_progression(r2_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODEL_ORDER))

    for representation in ["Acoustic F1", "Acoustic F2", "Whisper PC", "XLS-R PC"]:
        sub = r2_df[r2_df["representation"] == representation].set_index("model")
        values = [sub.loc[model, "R2_marginal"] if model in sub.index else np.nan for model in MODEL_ORDER]

        ax.plot(
            x,
            values,
            marker="o",
            linewidth=1.8,
            markersize=6,
            label=representation,
            color=REP_COLOURS.get(representation),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_xlabel("Model")
    ax.set_ylabel("Marginal R²")
    ax.set_title("Marginal R² progression across representation types")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "r2_model_progression.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting R² tables ...")
    r2_unified = collect_r2()
    r2_unified.to_csv(TABLE_DIR / "r2_unified.csv", index=False)

    print("Extracting selected-model R² ...")
    r2_selected = extract_selected_model_r2(r2_unified)
    r2_selected.to_csv(TABLE_DIR / "r2_selected_model.csv", index=False)

    print("Computing gender-controlled L2 marginal R² gain for Q10 ...")
    q10_r2_gain = extract_l2_unique_r2_gain_for_q10()
    q10_r2_gain.to_csv(TABLE_DIR / "l1l2_r2_gain_for_q10.csv", index=False)

    print("Extracting L1×Gender interaction terms ...")
    interaction_df = collect_interaction_significance()
    interaction_df.to_csv(TABLE_DIR / "interaction_significance.csv", index=False)

    print("Computing /a/-specific ICC comparison ...")
    icc_rows = []
    icc_rows.append(fit_null_icc(load_acoustic_a(), "F1_lob_mid", "Acoustic F1 (/a/)"))
    icc_rows.append(fit_null_icc(load_whisper_a(), "PC1", "Whisper PC1 (/a/)"))
    pd.DataFrame(icc_rows).to_csv(TABLE_DIR / "icc_a_comparison.csv", index=False)

    print("Plotting R² progression ...")
    plot_r2_progression(r2_unified)

    print(f"Done. Tables saved to {TABLE_DIR}; figure saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
