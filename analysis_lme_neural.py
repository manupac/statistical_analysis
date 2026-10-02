"""
Fit linear mixed-effects models for neural representations.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    import statsmodels.formula.api as smf
except ImportError as exc:
    raise ImportError("statsmodels is required: pip install statsmodels") from exc


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*singular.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*boundary.*")
warnings.filterwarnings("ignore", message=".*Hessian.*")
warnings.filterwarnings("ignore", message=".*gradient.*", category=UserWarning)

DATA_DIR = "data/pipeline"
OUT_DIR = "results/lme_neural"
LAYER = 20
PCA_DIM = 5
ALPHA = 0.05

REPRESENTATIONS = [
    ("whisper", "features_whisper_pca.npz"),
    ("xlsr", "features_xlsr_pca.npz"),
]

TARGET_VOWELS = {"a", "e", "i", "o", "u", "y", "œ", "ø", "ɑ", "ə", "ɛ"}

VOWEL_HEIGHT = {
    "i": "high",
    "u": "high",
    "y": "high",
    "e": "mid",
    "o": "mid",
    "œ": "mid",
    "ø": "mid",
    "ə": "mid",
    "ɛ": "mid",
    "a": "low",
    "ɑ": "low",
}


def load_manifest() -> pd.DataFrame:
    tokens = pd.read_csv(os.path.join(DATA_DIR, "phoneme_tokens.csv"))
    tokens["manifest_row"] = tokens.index

    tokens = tokens[~tokens["phoneme"].str.contains("ding", na=False)].copy()

    l1_map = {
        "fr": "L1", "fra": "L1", "l1": "L1", "french": "L1", "native": "L1",
        "ru": "L2", "rus": "L2", "l2": "L2", "russian": "L2",
    }
    gender_map = {"f": "F", "female": "F", "fem": "F", "m": "M", "male": "M"}

    tokens["L1_norm"] = tokens["L1"].astype(str).str.lower().str.strip().map(l1_map)
    tokens["gender_norm"] = tokens["gender"].astype(str).str.lower().str.strip().map(gender_map)
    tokens["L2"] = (tokens["L1_norm"] == "L2").astype(int)
    tokens["Male"] = (tokens["gender_norm"] == "M").astype(int)

    tokens["is_vowel"] = tokens["phoneme"].isin(TARGET_VOWELS)
    tokens["vowel_height"] = tokens["phoneme"].map(VOWEL_HEIGHT)
    tokens.loc[~tokens["is_vowel"], "vowel_height"] = np.nan
    tokens["vowel_height"] = pd.Categorical(
        tokens["vowel_height"],
        categories=["mid", "high", "low"],
        ordered=False,
    )

    present_vowels = sorted(tokens.loc[tokens["is_vowel"], "phoneme"].unique())
    tokens["phoneme"] = pd.Categorical(
        tokens["phoneme"].where(tokens["is_vowel"]),
        categories=present_vowels,
    )

    required = ["manifest_row", "speaker", "L2", "Male", "is_vowel"]
    return tokens.dropna(subset=required).copy()


def choose_alignment(pca_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["speaker", "phoneme", "L2", "Male", "vowel_height", "is_vowel"]

    original = pca_df.merge(
        manifest[["manifest_row"] + meta_cols],
        left_on="token_index",
        right_on="manifest_row",
        how="left",
    )

    filtered = manifest.copy().reset_index(drop=True)
    filtered["filtered_row"] = filtered.index
    filtered_merge = pca_df.merge(
        filtered[["filtered_row", "manifest_row"] + meta_cols],
        left_on="token_index",
        right_on="filtered_row",
        how="left",
    )

    original_score = original["speaker"].notna().sum()
    filtered_score = filtered_merge["speaker"].notna().sum()

    if filtered_score > original_score:
        print("  token_index alignment: artifact_filtered_row")
        return filtered_merge

    print("  token_index alignment: original_manifest_row")
    return original


def load_pca_features(npz_name: str, manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    npz_path = os.path.join(DATA_DIR, npz_name)
    npz = np.load(npz_path, allow_pickle=True)

    key = f"hiddens_layer{LAYER}_pca{PCA_DIM}"
    if key not in npz:
        raise KeyError(f"{key} not found in {npz_name}")

    pc_cols = [f"PC{i + 1}" for i in range(PCA_DIM)]
    pca_df = pd.DataFrame(npz[key], columns=pc_cols)
    pca_df["token_index"] = npz["token_index"].astype(int)

    merged = choose_alignment(pca_df, manifest)
    merged = merged[merged["is_vowel"].fillna(False)].copy()

    print(f"  PCA features loaded: {len(merged):,} target-vowel tokens")
    return merged, pc_cols


def fit_model(formula: str, df: pd.DataFrame, reml: bool = False):
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


def fit_random_slope(formula: str, df: pd.DataFrame, reml: bool = False):
    model = smf.mixedlm(formula, data=df, groups=df["speaker"], re_formula="~L2")

    for method, maxiter in [("powell", 5000), ("lbfgs", 5000), ("nm", 5000)]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return model.fit(reml=reml, method=method, maxiter=maxiter, disp=False)
        except Exception:
            continue

    warnings.warn("Random-slope model fitting failed")
    return None


def lrt(simple, complex_):
    chi2 = 2 * (complex_.llf - simple.llf)
    df = max(int(complex_.df_modelwc - simple.df_modelwc), 1)
    p_value = float(scipy_stats.chi2.sf(chi2, df))
    return float(chi2), df, p_value


def compute_icc(result):
    sigma2_u = float(result.cov_re.iloc[0, 0])
    sigma2 = float(result.scale)
    icc = sigma2_u / (sigma2_u + sigma2)
    return icc, sigma2_u, sigma2


def compute_r2(result):
    cov_re = result.cov_re
    sigma2_u = 0.0 if cov_re is None or cov_re.size == 0 else float(np.trace(np.asarray(cov_re)))
    sigma2 = float(result.scale)

    x = np.asarray(result.model.exog, dtype=float)
    beta = np.asarray(result.fe_params, dtype=float)
    fixed_pred = x @ beta
    var_fixed = float(np.var(fixed_pred, ddof=0))

    denom = var_fixed + sigma2_u + sigma2
    if denom <= 0 or not np.isfinite(denom):
        return np.nan, np.nan

    return round(var_fixed / denom, 4), round((var_fixed + sigma2_u) / denom, 4)


def coefficient_table(result: object) -> pd.DataFrame:
    params = result.params
    bse = result.bse
    pvalues = result.pvalues

    return pd.DataFrame({
        "term": params.index,
        "estimate": params.values.round(6),
        "SE": bse.values.round(6),
        "CI_lower_95": (params - 1.96 * bse).values.round(6),
        "CI_upper_95": (params + 1.96 * bse).values.round(6),
        "p_value": pvalues.values.round(6),
        "significant": pvalues.values < 0.05,
    })


def run_model_sequence_one_pc(df: pd.DataFrame, pc_col: str, table_dir: str):
    print(f"\n  --- {pc_col} (n={df[pc_col].notna().sum():,}) ---")

    dff = df.dropna(subset=[pc_col]).rename(columns={pc_col: "y"}).copy()

    formulas = {
        "M0": "y ~ 1",
        "M1": "y ~ L2 + Male",
        "M2": "y ~ L2 + Male + L2:Male",
        "M3": "y ~ L2 + Male + L2:Male + C(vowel_height)",
        "M4": "y ~ L2 + Male + L2:Male + C(vowel_height)",
    }

    m0_reml = fit_model(formulas["M0"], dff, reml=True)
    models_ml = {
        "M0": fit_model(formulas["M0"], dff, reml=False),
        "M1": fit_model(formulas["M1"], dff, reml=False),
        "M2": fit_model(formulas["M2"], dff, reml=False),
        "M3": fit_model(formulas["M3"], dff, reml=False),
    }
    m4 = fit_random_slope(formulas["M4"], dff, reml=False)
    models_ml["M4"] = m4 if m4 is not None and m4.converged else None

    if m0_reml is None or models_ml["M0"] is None:
        print(f"    ERROR: null model failed for {pc_col}")
        return None

    icc, sigma2_u, sigma2 = compute_icc(m0_reml)
    print(f"    ICC = {icc:.4f}")

    comparisons = [
        ("M0 → M1", "M0", "M1", "add L2 + Male"),
        ("M1 → M2", "M1", "M2", "add L2:Male"),
        ("M2 → M3", "M2", "M3", "add vowel_height"),
        ("M3 → M4", "M3", "M4", "add random slope L2|speaker"),
    ]

    lrt_rows = []
    selected = "M0"

    for comparison, simple_name, complex_name, description in comparisons:
        simple = models_ml[simple_name]
        complex_ = models_ml[complex_name]

        if simple is None or complex_ is None:
            lrt_rows.append({
                "comparison": comparison,
                "description": description,
                "chi2": np.nan,
                "df": np.nan,
                "p_lrt": np.nan,
                "AIC_simpler": simple.aic if simple is not None else np.nan,
                "AIC_fuller": np.nan,
                "BIC_simpler": simple.bic if simple is not None else np.nan,
                "BIC_fuller": np.nan,
                "converged": False,
            })
            break

        chi2, df_lrt, p_value = lrt(simple, complex_)
        delta_aic = complex_.aic - simple.aic

        lrt_rows.append({
            "comparison": comparison,
            "description": description,
            "chi2": round(chi2, 4),
            "df": df_lrt,
            "p_lrt": round(p_value, 6),
            "AIC_simpler": round(simple.aic, 2),
            "AIC_fuller": round(complex_.aic, 2),
            "BIC_simpler": round(simple.bic, 2),
            "BIC_fuller": round(complex_.bic, 2),
            "converged": True,
        })

        print(f"    LRT {comparison}: p={p_value:.4f}, ΔAIC={delta_aic:+.1f}")

        if p_value < ALPHA and complex_.aic < simple.aic:
            selected = complex_name
        else:
            break

    print(f"    Selected: {selected}")

    selected_formula = formulas[selected]
    if selected == "M4":
        selected_reml = fit_random_slope(selected_formula, dff, reml=True)
    else:
        selected_reml = fit_model(selected_formula, dff, reml=True)

    if selected_reml is None:
        selected_reml = models_ml[selected]

    r2_rows = []
    for model_name in ["M0", "M1", "M2", "M3", "M4"]:
        model = models_ml[model_name]
        if model is None:
            r2_rows.append({
                "model": model_name,
                "R2_marginal": np.nan,
                "R2_conditional": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
            })
            continue

        r2_m, r2_c = compute_r2(model)
        r2_rows.append({
            "model": model_name,
            "R2_marginal": r2_m,
            "R2_conditional": r2_c,
            "AIC": round(model.aic, 2),
            "BIC": round(model.bic, 2),
        })

    icc_df = pd.DataFrame([{
        "pc": pc_col,
        "ICC": round(icc, 4),
        "sigma2_u": round(sigma2_u, 4),
        "sigma2": round(sigma2, 4),
    }])
    r2_df = pd.DataFrame(r2_rows)
    lrt_df = pd.DataFrame(lrt_rows)
    coef_df = coefficient_table(selected_reml)

    lrt_df.to_csv(os.path.join(table_dir, f"model_comparison_{pc_col}.csv"), index=False)
    coef_df.to_csv(os.path.join(table_dir, f"coefficients_{pc_col}.csv"), index=False)

    return icc_df, r2_df


def run_representation(rep_label: str, npz_name: str, manifest: pd.DataFrame):
    print(f"\n{'=' * 65}")
    print(f"Representation: {rep_label.upper()} | layer {LAYER} | PCA d={PCA_DIM}")
    print(f"{'=' * 65}")

    table_dir = os.path.join(OUT_DIR, rep_label, "tables")
    os.makedirs(table_dir, exist_ok=True)

    df, pc_cols = load_pca_features(npz_name, manifest)

    icc_tables = []
    r2_tables = []

    for pc_col in pc_cols:
        output = run_model_sequence_one_pc(df, pc_col, table_dir)
        if output is None:
            continue

        icc_df, r2_df = output
        icc_tables.append(icc_df)

        r2_df["pc"] = pc_col
        r2_df["rep"] = rep_label
        r2_tables.append(r2_df)

    if icc_tables:
        pd.concat(icc_tables).to_csv(
            os.path.join(table_dir, "icc_all_pcs.csv"),
            index=False,
        )

    if r2_tables:
        r2_all = pd.concat(r2_tables)
        r2_summary = (
            r2_all.groupby("model")[["R2_marginal", "R2_conditional"]]
            .mean()
            .round(4)
            .reset_index()
        )
        r2_summary["rep"] = rep_label
        r2_summary.to_csv(
            os.path.join(table_dir, "r2_summary.csv"),
            index=False,
        )

        print(f"\nR² summary for {rep_label}:")
        print(r2_summary[["model", "R2_marginal", "R2_conditional"]].to_string(index=False))

    print(f"\n{rep_label.upper()} outputs saved to {table_dir}")


def main() -> None:
    print("Loading phoneme manifest ...")
    manifest = load_manifest()

    for rep_label, npz_name in REPRESENTATIONS:
        run_representation(rep_label, npz_name, manifest)

    print(f"\nDone. Outputs saved under {OUT_DIR}/")


if __name__ == "__main__":
    main()
