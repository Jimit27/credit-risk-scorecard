"""Tests for Weight-of-Evidence binning.

These assert the properties a credit committee relies on - monotonic bad rate,
a minimum bin size, missing handled as its own bin - rather than checking that
the code returns some numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk.woe import MISSING_LABEL, WoETransformer, fit_feature_bins, iv_strength


def test_bad_rate_is_monotonic_across_bins(synthetic_credit_data):
    X, y = synthetic_credit_data
    binning = fit_feature_bins(X["utilisation"], y, max_bins=8, min_bin_fraction=0.05)

    rates = binning.stats[binning.stats["bin"] != MISSING_LABEL]["bad_rate"].to_numpy()
    assert len(rates) >= 3, "binning collapsed to too few bins to be useful"
    increasing = np.all(np.diff(rates) >= -1e-12)
    decreasing = np.all(np.diff(rates) <= 1e-12)
    assert increasing or decreasing, f"bad rate is not monotonic across bins: {rates}"


def test_woe_moves_opposite_to_bad_rate(synthetic_credit_data):
    """Positive WoE must mean a safer bin - the sign convention the scorecard depends on."""
    X, y = synthetic_credit_data
    binning = fit_feature_bins(X["utilisation"], y, max_bins=6)
    finite = binning.stats[binning.stats["bin"] != MISSING_LABEL]
    assert np.corrcoef(finite["bad_rate"], finite["woe"])[0, 1] < -0.9


def test_no_bin_is_smaller_than_the_configured_floor(synthetic_credit_data):
    X, y = synthetic_credit_data
    binning = fit_feature_bins(X["utilisation"], y, max_bins=10, min_bin_fraction=0.10)
    finite = binning.stats[binning.stats["bin"] != MISSING_LABEL]
    assert (finite["count"] >= 0.10 * len(X) * 0.999).all()


def test_missing_values_get_their_own_bin():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan] * 60)
    # Missing rows are deliberately much riskier than present rows.
    y = pd.Series(([0, 0, 0, 1] + [1, 1, 1, 0]) * 60)
    binning = fit_feature_bins(values, y, max_bins=4, min_bin_fraction=0.05)

    missing = binning.stats[binning.stats["bin"] == MISSING_LABEL]
    assert len(missing) == 1
    assert missing.iloc[0]["count"] == 240
    # And the transform must route NaN to that bin's WoE.
    assert binning.transform(pd.Series([np.nan]))[0] == pytest.approx(binning.missing_woe)


def test_unseen_missing_at_serving_time_is_neutral_not_an_error():
    """A feature with no missing values in training must still score a null."""
    values = pd.Series(np.linspace(0, 1, 400))
    y = pd.Series((np.linspace(0, 1, 400) > 0.5).astype(int))
    binning = fit_feature_bins(values, y, max_bins=4)
    assert binning.missing_woe == 0.0
    assert np.isfinite(binning.transform(pd.Series([np.nan]))[0])


def test_information_value_ranks_a_real_signal_above_noise(synthetic_credit_data):
    X, y = synthetic_credit_data
    rng = np.random.default_rng(1)
    X = X.assign(noise=rng.normal(size=len(X)))

    transformer = WoETransformer(max_bins=8).fit(X, y)
    iv = transformer.information_values()
    assert iv["utilisation"] > iv["noise"]
    assert iv["noise"] < 0.02, "pure noise should fall below the selection floor"


def test_transform_rejects_a_feature_it_has_no_bins_for(synthetic_credit_data):
    X, y = synthetic_credit_data
    transformer = WoETransformer().fit(X[["utilisation"]], y)
    with pytest.raises(KeyError):
        transformer.transform(X[["utilisation", "income"]])


def test_transform_accepts_a_subset_of_fitted_features(synthetic_credit_data):
    """Feature selection transforms fewer columns than were fitted; that must work."""
    X, y = synthetic_credit_data
    transformer = WoETransformer().fit(X, y)
    out = transformer.transform(X[["income"]])
    assert list(out.columns) == ["woe_income"]


def test_single_class_target_is_rejected():
    values = pd.Series(np.arange(100, dtype=float))
    with pytest.raises(ValueError, match="only one class"):
        fit_feature_bins(values, pd.Series(np.zeros(100, dtype=int)))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.01, "unpredictive"), (0.05, "weak"), (0.2, "medium"), (0.4, "strong")],
)
def test_iv_strength_labels(value, expected):
    assert iv_strength(value) == expected
