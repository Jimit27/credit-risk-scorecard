"""Serving-time feature derivation.

The gold layer is built in Spark, but a scoring API or a Streamlit form has one
applicant and no cluster. So the same derivations exist here in pandas.

Two implementations of one transformation is exactly how training/serving skew
gets in - the batch job and the live endpoint drift apart, and the model quietly
scores different features from the ones it was trained on. The mitigation is
``tests/test_serving_parity.py``, which re-derives a sample of the Spark-built
gold table through this module and asserts the two agree to floating-point
tolerance. If someone edits one and not the other, CI fails.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# The raw fields an application form actually collects, before any engineering.
RAW_INPUTS = [
    "revolving_utilisation",
    "age",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "real_estate_lines",
    "dependents",
    "times_30_59_dpd",
    "times_60_89_dpd",
    "times_90_dpd",
]


def age_band(age: float) -> str:
    """The age band used to impute income. Mirrors the gold layer exactly."""
    if age is None or pd.isna(age):
        return "unknown"
    for limit, label in ((30, "18-29"), (40, "30-39"), (50, "40-49"), (60, "50-59"), (70, "60-69")):
        if age < limit:
            return label
    return "70+"


def load_imputation_constants(cfg=None) -> dict:
    """The medians the batch job used, as written by the gold build.

    A scoring request is one applicant; there is no population to take a median
    from. Recomputing an imputation at serving time is impossible, and guessing
    one is how a live endpoint quietly feeds the model different values from the
    ones it was trained on - so the batch job's decisions are persisted and
    replayed here.
    """
    from creditrisk.config import load_config

    cfg = cfg or load_config()
    path = cfg.path("paths.reports") / "imputation_constants.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _impute(frame: pd.DataFrame, constants: dict) -> pd.DataFrame:
    """Fill missing raw inputs with the constants the gold build recorded."""
    if not constants:
        return frame

    by_band = constants.get("monthly_income_by_age_band", {})
    global_median = constants.get("monthly_income_global_median")
    if global_median is not None and frame["monthly_income"].isna().any():
        bands = frame["age"].map(age_band)
        band_income = bands.map(lambda b: by_band.get(b, global_median))
        frame["monthly_income"] = frame["monthly_income"].fillna(band_income)

    for column, key in (
        ("age", "age_median"),
        ("debt_ratio", "debt_ratio_median"),
        ("dependents", "dependents_fill"),
    ):
        if key in constants:
            frame[column] = frame[column].fillna(constants[key])
    return frame


def derive_features(raw: pd.DataFrame | dict, constants: dict | None = None) -> pd.DataFrame:
    """Reproduce the gold-layer feature engineering for one or more applicants.

    Mirrors ``creditrisk.features.build_gold``, including its imputation. Any
    change to one belongs in both, and ``tests/test_serving_parity.py`` exists
    to make that non-optional.
    """
    frame = pd.DataFrame([raw]) if isinstance(raw, dict) else raw.copy()

    for column in RAW_INPUTS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if constants is None:
        constants = load_imputation_constants()
    frame = _impute(frame, constants)

    d30 = frame["times_30_59_dpd"].fillna(0.0)
    d60 = frame["times_60_89_dpd"].fillna(0.0)
    d90 = frame["times_90_dpd"].fillna(0.0)

    out = frame.copy()
    out["total_delinquencies"] = d30 + d60 + d90
    out["worst_delinquency_severity"] = np.select(
        [d90 > 0, d60 > 0, d30 > 0], [3.0, 2.0, 1.0], default=0.0
    )
    lines = out["open_credit_lines"].astype(float).clip(lower=1.0)
    out["delinquencies_per_credit_line"] = out["total_delinquencies"] / lines
    out["unsecured_lines"] = (out["open_credit_lines"] - out["real_estate_lines"]).clip(lower=0.0)
    out["real_estate_share_of_lines"] = out["real_estate_lines"] / lines

    # Last-resort fills for a caller with no persisted constants available.
    # _impute has already applied the batch job's medians where it could.
    out["dependents"] = out["dependents"].fillna(0.0)
    out["debt_ratio"] = out["debt_ratio"].fillna(constants.get("debt_ratio_median", 0.0))
    out["monthly_debt_service"] = out["monthly_income"] * out["debt_ratio"]
    out["disposable_income"] = out["monthly_income"] - out["monthly_debt_service"]
    out["income_per_dependent"] = out["monthly_income"] / (1.0 + out["dependents"])
    out["log_monthly_income"] = np.log1p(out["monthly_income"].clip(lower=0.0))
    out["utilisation_x_delinquency"] = out["revolving_utilisation"] * (1.0 + out["total_delinquencies"])

    # Fields the silver layer reconstructs. A scoring request arrives after
    # validation, so these default to "clean" unless explicitly supplied.
    for flag in ("flag_delinquency_sentinel", "flag_monthly_income_missing", "flag_debt_ratio_is_amount"):
        if flag not in out.columns:
            out[flag] = 0.0
    if "monthly_debt_amount" not in out.columns:
        out["monthly_debt_amount"] = 0.0
    out["monthly_debt_amount"] = out["monthly_debt_amount"].fillna(0.0)

    return out


# Illustrative profiles for the demo app, so a reviewer can see the model
# respond without inventing plausible numbers themselves. Each lands in a
# different policy grade - the labels below match the grades they actually
# produce against the shipped model.
EXAMPLE_APPLICANTS: dict[str, dict[str, float]] = {
    "Grade A - homeowner, clean file": {
        "revolving_utilisation": 0.03,
        "age": 66,
        "debt_ratio": 0.12,
        "monthly_income": 8000,
        "open_credit_lines": 12,
        "real_estate_lines": 2,
        "dependents": 0,
        "times_30_59_dpd": 0,
        "times_60_89_dpd": 0,
        "times_90_dpd": 0,
    },
    "Grade B - solid file, low usage": {
        "revolving_utilisation": 0.12,
        "age": 52,
        "debt_ratio": 0.24,
        "monthly_income": 5200,
        "open_credit_lines": 9,
        "real_estate_lines": 2,
        "dependents": 1,
        "times_30_59_dpd": 0,
        "times_60_89_dpd": 0,
        "times_90_dpd": 0,
    },
    "Grade C - mid file, moderate usage": {
        "revolving_utilisation": 0.34,
        "age": 38,
        "debt_ratio": 0.33,
        "monthly_income": 3400,
        "open_credit_lines": 6,
        "real_estate_lines": 1,
        "dependents": 2,
        "times_30_59_dpd": 0,
        "times_60_89_dpd": 0,
        "times_90_dpd": 0,
    },
    "Grade D - stretched, one late payment": {
        "revolving_utilisation": 0.30,
        "age": 44,
        "debt_ratio": 0.30,
        "monthly_income": 3300,
        "open_credit_lines": 6,
        "real_estate_lines": 0,
        "dependents": 2,
        "times_30_59_dpd": 1,
        "times_60_89_dpd": 0,
        "times_90_dpd": 0,
    },
    "Grade E - maxed out, recent arrears": {
        "revolving_utilisation": 0.97,
        "age": 27,
        "debt_ratio": 0.71,
        "monthly_income": 1850,
        "open_credit_lines": 7,
        "real_estate_lines": 0,
        "dependents": 1,
        "times_30_59_dpd": 3,
        "times_60_89_dpd": 1,
        "times_90_dpd": 2,
    },
}
