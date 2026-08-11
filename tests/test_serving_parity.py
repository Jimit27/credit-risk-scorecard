"""Guard against training/serving skew.

The gold layer is built in Spark; the scoring app derives the same features in
pandas. Two implementations of one transformation is how a model quietly starts
scoring different features from the ones it was trained on.

This test re-derives a sample of the Spark-built gold table through the
serving-time code and asserts the two agree. If someone edits ``features.py``
and forgets ``serving.py``, this fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from creditrisk.features import MODEL_FEATURES
from creditrisk.serving import RAW_INPUTS, derive_features

# Fields the silver layer reconstructs from the raw file and a scoring request
# would carry as validated inputs; they are not re-derived at serving time.
#
# monthly_debt_amount is here because it cannot be recovered from the ratio: it
# exists precisely for the applicants whose debt figure arrived *without* an
# income to divide by. A real API would receive it as its own field; the demo
# form has no way to collect it, so serving defaults it to zero.
SERVING_EXEMPT = {
    "flag_delinquency_sentinel",
    "flag_monthly_income_missing",
    "flag_debt_ratio_is_amount",
    "monthly_debt_amount",
}


def test_serving_derivation_matches_the_spark_gold_table(gold):
    sample = gold.sample(n=min(3000, len(gold)), random_state=0).reset_index(drop=True)

    # Feed the serving code only the raw inputs, exactly as an API would receive them.
    derived = derive_features(sample[RAW_INPUTS])

    mismatches = []
    for feature in MODEL_FEATURES:
        if feature in SERVING_EXEMPT or feature not in gold.columns or feature not in derived.columns:
            continue
        expected = sample[feature].to_numpy(dtype=float)
        actual = derived[feature].to_numpy(dtype=float)
        both_nan = np.isnan(expected) & np.isnan(actual)
        if not np.allclose(expected[~both_nan], actual[~both_nan], rtol=1e-6, atol=1e-6, equal_nan=True):
            worst = np.nanmax(np.abs(expected - actual))
            mismatches.append(f"{feature} (max difference {worst:.6g})")

    assert not mismatches, "serving derivation has drifted from the Spark gold layer: " + ", ".join(mismatches)


def test_serving_handles_a_single_applicant_dict():
    applicant = {
        "revolving_utilisation": 0.4,
        "age": 40,
        "debt_ratio": 0.3,
        "monthly_income": 3000,
        "open_credit_lines": 5,
        "real_estate_lines": 1,
        "dependents": 1,
        "times_30_59_dpd": 1,
        "times_60_89_dpd": 0,
        "times_90_dpd": 0,
    }
    frame = derive_features(applicant)
    assert len(frame) == 1
    assert frame["total_delinquencies"].iloc[0] == 1.0
    assert frame["worst_delinquency_severity"].iloc[0] == 1.0
    assert frame["monthly_debt_service"].iloc[0] == pytest.approx(900.0)
    assert frame["disposable_income"].iloc[0] == pytest.approx(2100.0)


def test_missing_raw_fields_do_not_crash_the_derivation():
    frame = derive_features({"age": 30, "monthly_income": 2000})
    for feature in MODEL_FEATURES:
        assert feature in frame.columns


def test_worst_severity_uses_the_worst_arrears_not_the_count():
    """Three 30-day misses must rank below one 90-day miss."""
    frequent_but_mild = derive_features(
        {"times_30_59_dpd": 3, "times_60_89_dpd": 0, "times_90_dpd": 0, "open_credit_lines": 4}
    )
    single_severe = derive_features(
        {"times_30_59_dpd": 0, "times_60_89_dpd": 0, "times_90_dpd": 1, "open_credit_lines": 4}
    )
    assert frequent_but_mild["worst_delinquency_severity"].iloc[0] == 1.0
    assert single_severe["worst_delinquency_severity"].iloc[0] == 3.0
    assert frequent_but_mild["total_delinquencies"].iloc[0] > single_severe["total_delinquencies"].iloc[0]


def test_serving_replays_the_batch_jobs_imputation(cfg):
    """A live request with no income must get the value the model was trained on.

    Without the persisted constants, serving would leave income as NaN, the WoE
    transform would route it to a neutral bin, and the endpoint would silently
    score a different feature from the one the batch job produced.
    """
    from creditrisk.serving import load_imputation_constants

    constants = load_imputation_constants(cfg)
    if not constants:
        pytest.skip("Imputation constants not built. Run: python -m creditrisk.pipeline gold")

    applicant = {"age": 45, "revolving_utilisation": 0.3, "open_credit_lines": 5}  # no income supplied
    derived = derive_features(applicant, constants)

    expected_income = constants["monthly_income_by_age_band"].get(
        "40-49", constants["monthly_income_global_median"]
    )
    assert derived["monthly_income"].iloc[0] == pytest.approx(expected_income)
    assert np.isfinite(derived["log_monthly_income"].iloc[0])
    assert np.isfinite(derived["disposable_income"].iloc[0])


def test_ambiguous_debt_ratio_is_not_imputed_to_zero(cfg, gold):
    """Filling an uninterpretable debt figure with 0 would be the most optimistic
    value available. The gold layer must use the median instead."""
    from creditrisk.serving import load_imputation_constants

    constants = load_imputation_constants(cfg)
    if not constants:
        pytest.skip("Imputation constants not built.")
    assert constants["debt_ratio_median"] > 0.0

    flagged = gold[gold["flag_debt_ratio_is_amount"] == 1]
    if len(flagged):
        assert (flagged["debt_ratio"] > 0).all(), "ambiguous-debt applicants were told they have zero debt"
