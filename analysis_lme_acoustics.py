"""
Fit acoustic linear mixed-effects models for F1 and F2.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    import statsmodels.formula.api as smf
except ImportError as exc:
    raise ImportError("statsmodels is required. Install with: pip install statsmodels") from exc


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*singular.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*boundary.*")
warnings.filterwarnings("ignore", message=".*Hessian.*")
warnings.filterwarnings("ignore", message=".*gradient.*", category=UserWarning)

DATA_DIR = "data/pipeline"
OUT_DIR = "results/lme_acoustic"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
ALPHA = 0.05

TARGET_VOWELS = {"a", "e", "i", "o", "u", "y", "œ", "ø", "ɑ", "ə", "ɛ"}

VOWEL_HEIGHT = {
    "i": "high",
    "y": "high",
    "u": "high",
    "e": "mid",
    "ø": "mid",
    "œ": "mid",
    "o": "mid",
    "ɛ": "mid",
    "ə": "mid",
    "a": "low",
    "ɑ": "low",
}

FORMANTS = [("F1", "F1_lob_mid"), ("F2", "F2_lob_mid")]


def load_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "features_acoustic_norm.csv")
    df = pd.read_csv(path)

    df = df[~df["phoneme"].str.contains("ding", na=False)].copy()
    df = df[df["phoneme"].isin(TARGET_VOWELS)].copy()

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
    df["L2"] = (df["L1_norm"] == "L2").astype(int)
    df["Male"] = (df["gender_norm"] == "M").astype(int)
    df["vowel_height"] = df["phoneme"].map(VOWEL_HEIGHT)

    required = ["speaker", "L1_norm", "gender_norm", "L2", "Male", "vowel_height"]
    df = df.dropna(subset=required).reset_index(drop=True)

    present_vowels = [v for v in sorted(df["phoneme"].unique())]
    df["phoneme"] = pd.Categorical(df["phoneme"], categories=present_vowels)
    df["vowel_height"] = pd.Categorical(
        df["vowel_height"],
        categories=["mid", "high", "low"],
        ordered=False,
    )

    print(f"Loaded {len(df):,} vowel tokens from {df['speaker'].nunique()} speakers.")
    print(f"Vowels present: {present_vowels}")
    return df


def fit_model(formula: str, df: pd.DataFrame, reml: bool = False):
    model = smf.mixedlm(formula, data=df, groups=df["speaker"])

    for method, maxiter in [("profile", 200), ("powell", 5000), ("lbfgs", 5000)]:
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

    warnings.warn("Random-slope model fitting failed.")
    return None


def likelihood_ratio_test(null_model, full_model):
    chi2 = 2 * (full_model.llf - null_model.llf)
    df = max(int(full_model.df_modelwc - null_model.df_modelwc), 1)
    p_value = float(scipy_stats.chi2.sf(chi2, df))
    return float(chi2), df, p_value


def compute_icc(null_reml):
    sigma2_u = float(null_reml.cov_re.iloc[0, 0])
    sigma2 = float(null_reml.scale)
    icc = sigma2_u / (sigma2_u + sigma2)
    return icc, sigma2_u, sigma2


def compute_r2(model_fit):
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

        r2_marginal = var_fixed / denominator
        r2_conditional = (var_fixed + sigma2_u) / denominator
        return round(r2_marginal, 4), round(r2_conditional, 4)
    except Exception as exc:
        warnings.warn(f"Could not compute R²: {exc}")
        return np.nan, np.nan


def coefficient_table(model_fit) -> pd.DataFrame:
    params = model_fit.params
    standard_errors = model_fit.bse
    p_values = model_fit.pvalues

    return pd.DataFrame({
        "term": params.index,
        "estimate": params.values.round(4),
        "SE": standard_errors.values.round(4),
        "CI_lower_95": (params - 1.96 * standard_errors).values.round(4),
        "CI_upper_95": (params + 1.96 * standard_errors).values.round(4),
        "p_value": p_values.values.round(6),
        "significant": (p_values.values < 0.05),
    })


def run_model_sequence(df: pd.DataFrame, response_col: str, label: str):
    dff = df.dropna(subset=[response_col]).copy()
    dff = dff.rename(columns={response_col: "y"})

    formulas = {
        "M0": "y ~ 1",
        "M1": "y ~ L2 + Male",
        "M2": "y ~ L2 + Male + L2:Male",
        "M3": "y ~ L2 + Male + L2:Male + C(vowel_height)",
        "M4": "y ~ L2 + Male + L2:Male + C(vowel_height)",
    }

    print(f"\nFitting acoustic LME for {label} (n={len(dff):,})")

    m0_reml = fit_model(formulas["M0"], dff, reml=True)
    m0_ml = fit_model(formulas["M0"], dff, reml=False)

    if m0_reml is None or m0_ml is None:
        raise RuntimeError(f"Null model failed for {label}")

    models_ml = {
        "M0": m0_ml,
        "M1": fit_model(formulas["M1"], dff, reml=False),
        "M2": fit_model(formulas["M2"], dff, reml=False),
        "M3": fit_model(formulas["M3"], dff, reml=False),
    }

    m4_ml = fit_random_slope(formulas["M4"], dff, reml=False)
    models_ml["M4"] = m4_ml if m4_ml is not None and m4_ml.converged else None

    comparisons = [
        ("M0 → M1", "M0", "M1", "add L2 + Male"),
        ("M1 → M2", "M1", "M2", "add L2:Male"),
        ("M2 → M3", "M2", "M3", "add vowel_height"),
        ("M3 → M4", "M3", "M4", "add random slope L2|speaker"),
    ]

    lrt_rows = []
    selected_model = "M0"
    selected_formula = formulas["M0"]
    selected_random_slope = False

    for comparison, simpler_name, fuller_name, description in comparisons:
        simpler = models_ml.get(simpler_name)
        fuller = models_ml.get(fuller_name)

        if simpler is None or fuller is None:
            lrt_rows.append({
                "comparison": comparison,
                "description": description,
                "chi2": np.nan,
                "df": np.nan,
                "p_lrt": np.nan,
                "AIC_simpler": simpler.aic if simpler is not None else np.nan,
                "AIC_fuller": np.nan,
                "BIC_simpler": simpler.bic if simpler is not None else np.nan,
                "BIC_fuller": np.nan,
                "converged": False,
            })
            continue

        chi2, df_lrt, p_lrt = likelihood_ratio_test(simpler, fuller)
        aic_improved = fuller.aic < simpler.aic

        lrt_rows.append({
            "comparison": comparison,
            "description": description,
            "chi2": round(chi2, 4),
            "df": df_lrt,
            "p_lrt": round(p_lrt, 6),
            "AIC_simpler": round(simpler.aic, 2),
            "AIC_fuller": round(fuller.aic, 2),
            "BIC_simpler": round(simpler.bic, 2),
            "BIC_fuller": round(fuller.bic, 2),
            "converged": True,
        })

        print(f"  {comparison}: p={p_lrt:.4f}, ΔAIC={fuller.aic - simpler.aic:+.2f}")

        if p_lrt < ALPHA and aic_improved:
            selected_model = fuller_name
            selected_formula = formulas[fuller_name]
            selected_random_slope = fuller_name == "M4"

    print(f"  Selected model for {label}: {selected_model}")

    if selected_random_slope:
        selected_reml = fit_random_slope(selected_formula, dff, reml=True)
    else:
        selected_reml = fit_model(selected_formula, dff, reml=True)

    if selected_reml is None:
        selected_reml = models_ml[selected_model]

    icc, sigma2_u, sigma2 = compute_icc(m0_reml)
    icc_row = {
        "formant": label,
        "ICC": round(icc, 4),
        "sigma2_u": round(sigma2_u, 4),
        "sigma2": round(sigma2, 4),
    }

    r2_rows = []
    for model_name in ["M0", "M1", "M2", "M3", "M4"]:
        model_fit = models_ml.get(model_name)
        if model_fit is None:
            r2_rows.append({
                "model": model_name,
                "R2_marginal": np.nan,
                "R2_conditional": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
            })
            continue

        r2_marginal, r2_conditional = compute_r2(model_fit)
        r2_rows.append({
            "model": model_name,
            "R2_marginal": r2_marginal,
            "R2_conditional": r2_conditional,
            "AIC": round(model_fit.aic, 2),
            "BIC": round(model_fit.bic, 2),
        })

    model_comparison = pd.DataFrame(lrt_rows)
    r2_table = pd.DataFrame(r2_rows)
    coefficients = coefficient_table(selected_reml)

    model_comparison.to_csv(
        os.path.join(TABLE_DIR, f"model_comparison_{label}.csv"),
        index=False,
    )
    coefficients.to_csv(
        os.path.join(TABLE_DIR, f"coefficients_{label}.csv"),
        index=False,
    )

    r2_table["formant"] = label
    return icc_row, r2_table


def main() -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)

    df = load_data()
    icc_rows = []
    r2_tables = []

    for label, col in FORMANTS:
        icc_row, r2_table = run_model_sequence(df, col, label)
        icc_rows.append(icc_row)
        r2_tables.append(r2_table)

    pd.DataFrame(icc_rows).to_csv(os.path.join(TABLE_DIR, "icc.csv"), index=False)
    pd.concat(r2_tables, ignore_index=True).to_csv(
        os.path.join(TABLE_DIR, "r2_all_formants.csv"),
        index=False,
    )

    print(f"\nDone. Tables saved to {TABLE_DIR}")


if __name__ == "__main__":
    main()
