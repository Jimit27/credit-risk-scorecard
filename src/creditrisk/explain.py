"""Explainability: a points table, per-applicant attribution, and reason codes.

A lender that declines an application generally has to say why, in specific
terms - the UK FCA's consumer-credit rules and the US FCRA adverse-action
notice both push in that direction. "The gradient boosting said no" is not an
answer, so this module produces two things.

**The points table.** For the WoE logistic champion, every bin of every feature
carries a fixed number of points. The whole model collapses into one printable
table, and an applicant's score is the sum of their rows. This is the artefact
a credit committee signs off, and it is exact rather than an approximation.

**Reason codes.** The standard method is distance-to-maximum: for each feature,
compare the points the applicant earned against the most anyone could earn on
that feature, and rank the shortfalls. The largest shortfalls are the reasons
the applicant scored where they did.

SHAP is available for the gradient-boosted challenger, where no closed-form
points table exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from creditrisk.model import ScorecardModel
from creditrisk.woe import MISSING_LABEL

# Plain-English descriptions used when a reason code is shown to a human.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "revolving_utilisation": "Proportion of available revolving credit already in use",
    "utilisation_x_delinquency": "Revolving credit usage, weighted by past missed payments",
    "age": "Age of the applicant",
    "debt_ratio": "Monthly debt repayments as a share of monthly income",
    "monthly_income": "Reported monthly income",
    "log_monthly_income": "Reported monthly income",
    "open_credit_lines": "Number of open credit lines and loans",
    "real_estate_lines": "Number of mortgage and property-secured lines",
    "real_estate_share_of_lines": "Share of credit lines that are property-secured",
    "unsecured_lines": "Number of unsecured credit lines",
    "dependents": "Number of financial dependants",
    "times_30_59_dpd": "Times 30-59 days late in the last two years",
    "times_60_89_dpd": "Times 60-89 days late in the last two years",
    "times_90_dpd": "Times 90+ days late in the last two years",
    "total_delinquencies": "Total missed payments in the last two years",
    "worst_delinquency_severity": "Severity of the worst missed payment on record",
    "delinquencies_per_credit_line": "Missed payments per open credit line",
    "monthly_debt_service": "Monthly debt repayment amount",
    "disposable_income": "Income left after debt repayments",
    "income_per_dependent": "Income available per household member",
    "flag_delinquency_sentinel": "Bureau delinquency record is coded rather than counted",
    "flag_monthly_income_missing": "No income figure was supplied",
    "flag_debt_ratio_is_amount": "Debt figure supplied without a matching income",
}


def _require_logistic(model: ScorecardModel) -> None:
    if model.kind != "logistic_woe" or model.woe is None:
        raise TypeError("A points table is only defined for the WoE logistic scorecard")


def points_table(model: ScorecardModel) -> pd.DataFrame:
    """The full scorecard: points awarded for every bin of every feature.

    Derivation. With ``score = offset + factor * ln(odds_good)`` and the
    logistic modelling log-odds of *bad*::

        ln(odds_good) = -(intercept + sum_i beta_i * woe_i)

    so the constant term is shared equally across the ``n`` features and each
    bin contributes ``-factor * beta_i * woe_ib``.
    """
    _require_logistic(model)
    assert model.woe is not None

    coefficients = dict(zip(model.features, np.asarray(model.estimator.coef_).ravel(), strict=True))
    intercept = float(np.asarray(model.estimator.intercept_).ravel()[0])
    factor = model.scaling.factor
    n_features = max(len(model.features), 1)
    shared_constant = (model.scaling.offset - factor * intercept) / n_features

    rows: list[dict[str, Any]] = []
    for feature in model.features:
        binning = model.woe.bins_[feature]
        beta = float(coefficients[feature])
        for _, stat in binning.stats.iterrows():
            points = shared_constant - factor * beta * float(stat["woe"])
            rows.append(
                {
                    "feature": feature,
                    "description": FEATURE_DESCRIPTIONS.get(feature, feature),
                    "bin": stat["bin"],
                    "count": int(stat["count"]),
                    "bad_rate": float(stat["bad_rate"]),
                    "woe": float(stat["woe"]),
                    "points": round(points, 2),
                }
            )

    table = pd.DataFrame(rows)
    # Points relative to the best achievable bin - the quantity reason codes rank.
    table["max_points_for_feature"] = table.groupby("feature")["points"].transform("max")
    table["points_lost"] = (table["max_points_for_feature"] - table["points"]).round(2)
    return table


def _bin_label_for(model: ScorecardModel, feature: str, value: float) -> str:
    binning = model.woe.bins_[feature]  # type: ignore[union-attr]
    # pd.isna, not isinstance(value, float) and np.isnan: np.float32 is not a
    # float subclass and pd.NA raises on np.isnan, so the narrower check let a
    # missing value fall through to np.searchsorted - which returns the top
    # index for NaN and silently awarded the applicant the *best* bin.
    if value is None or pd.isna(value):
        return MISSING_LABEL
    index = int(np.searchsorted(binning.edges, float(value), side="left"))
    labels = binning.stats[binning.stats["bin"] != MISSING_LABEL]["bin"].tolist()
    return labels[index] if index < len(labels) else labels[-1]


def explain_applicant(model: ScorecardModel, applicant: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Per-feature points for one applicant, ranked by points lost."""
    _require_logistic(model)
    row = applicant.iloc[0] if isinstance(applicant, pd.DataFrame) else applicant
    table = points_table(model)

    records = []
    for feature in model.features:
        value = row.get(feature, np.nan)
        label = _bin_label_for(model, feature, value)
        match = table[(table["feature"] == feature) & (table["bin"] == label)]
        if match.empty:
            continue
        entry = match.iloc[0]
        records.append(
            {
                "feature": feature,
                "description": entry["description"],
                "value": value,
                "bin": label,
                "points": float(entry["points"]),
                "points_lost": float(entry["points_lost"]),
                "bin_bad_rate": float(entry["bad_rate"]),
            }
        )
    return pd.DataFrame(records).sort_values("points_lost", ascending=False).reset_index(drop=True)


def reason_codes(model: ScorecardModel, applicant: pd.Series | pd.DataFrame, top_n: int = 4) -> list[dict[str, Any]]:
    """The top adverse-action reasons for an applicant's score.

    Only features where the applicant actually gave up points are returned; a
    reason code that cost nothing is not a reason.
    """
    contributions = explain_applicant(model, applicant)
    material = contributions[contributions["points_lost"] > 0.5].head(top_n)
    return [
        {
            "rank": position + 1,
            "feature": entry["feature"],
            "reason": entry["description"],
            "applicant_value": entry["value"],
            "bin": entry["bin"],
            "points_lost": round(entry["points_lost"], 1),
            "bin_default_rate": round(entry["bin_bad_rate"], 4),
        }
        for position, (_, entry) in enumerate(material.iterrows())
    ]


def shap_values(model: ScorecardModel, X: pd.DataFrame, max_rows: int = 2000) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP values for the gradient-boosted challenger.

    Returns the value matrix and the sample it was computed on. Raises if
    ``shap`` is not installed, which keeps it an optional dependency.
    """
    import shap  # imported lazily: optional dependency

    if model.kind != "gbm":
        raise TypeError("SHAP is provided for the GBM challenger; use points_table() for the scorecard")

    sample = X[model.features].head(max_rows)
    explainer = shap.TreeExplainer(model.estimator)
    values = explainer.shap_values(sample)
    return values, sample


def global_importance(model: ScorecardModel) -> pd.DataFrame:
    """Feature strength for the scorecard, in points.

    Two columns, because the naive one is misleading on its own.
    ``points_range`` is the widest swing a feature *can* produce; but a bin
    holding 0.2% of the book can carry a 50-point range and never move a real
    decision. ``population_weighted_spread`` is the mean absolute deviation of
    points from the feature's population-weighted average, so it answers the
    question that actually matters: how much does this feature move scores
    across the book we see? Ranking is by the weighted measure.
    """
    table = points_table(model)

    rows = []
    for (feature, description), group in table.groupby(["feature", "description"]):
        weights = group["count"].to_numpy(dtype=float)
        points = group["points"].to_numpy(dtype=float)
        total = weights.sum()
        if total <= 0:
            continue
        mean_points = float(np.average(points, weights=weights))
        rows.append(
            {
                "feature": feature,
                "description": description,
                "min": float(points.min()),
                "max": float(points.max()),
                "points_range": round(float(points.max() - points.min()), 2),
                "population_weighted_spread": round(float(np.average(np.abs(points - mean_points), weights=weights)), 2),
                "largest_bin_share": round(float(weights.max() / total), 3),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("population_weighted_spread", ascending=False)
        .reset_index(drop=True)
    )
