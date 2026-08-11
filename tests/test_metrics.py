"""Tests for the credit-risk metric implementations.

Each metric is checked against a case where the correct answer is known
analytically, so a silent regression in the maths cannot hide behind a
plausible-looking number.
"""

from __future__ import annotations

import numpy as np
import pytest

from creditrisk.metrics import (
    approval_curve,
    calibration_table,
    evaluate,
    gains_table,
    gini,
    ks_statistic,
    population_stability_index,
    psi_verdict,
)


def test_gini_of_a_perfect_ranking_is_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert gini(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(1.0)


def test_gini_of_a_reversed_ranking_is_minus_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert gini(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == pytest.approx(-1.0)


def test_ks_is_one_when_the_distributions_do_not_overlap():
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    assert ks_statistic(y, scores) == pytest.approx(1.0)


def test_ks_is_zero_for_a_constant_score():
    y = np.array([0, 1] * 50)
    assert ks_statistic(y, np.full(100, 0.5)) == pytest.approx(0.0, abs=1e-9)


def test_psi_of_a_distribution_against_itself_is_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=5000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=8000)
    small = population_stability_index(reference, rng.normal(loc=0.15, size=8000))
    large = population_stability_index(reference, rng.normal(loc=1.20, size=8000))
    assert small < large
    assert psi_verdict(large) == "revalidate"


def test_psi_survives_an_empty_bin():
    """A bin with no mass must not produce an infinity that swamps the score."""
    rng = np.random.default_rng(0)
    reference = rng.normal(size=3000)
    disjoint = rng.normal(loc=25.0, size=3000)
    value = population_stability_index(reference, disjoint)
    assert np.isfinite(value) and value > 1.0


@pytest.mark.parametrize(
    ("value", "expected"), [(0.05, "stable"), (0.15, "investigate"), (0.40, "revalidate")]
)
def test_psi_verdict_thresholds(value, expected):
    assert psi_verdict(value) == expected


def test_calibration_table_recovers_a_known_default_rate():
    rng = np.random.default_rng(3)
    probability = rng.uniform(0.01, 0.5, 20000)
    y = (rng.uniform(size=20000) < probability).astype(int)
    table = calibration_table(y, probability, bins=10)

    assert len(table) == 10
    # A perfectly calibrated model: predicted and observed agree within noise.
    assert np.abs(table["difference"]).max() < 0.05


def test_gains_table_captures_every_default_by_the_final_decile():
    rng = np.random.default_rng(4)
    probability = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < probability).astype(int)
    table = gains_table(y, probability, bins=10)

    assert table["cumulative_bad_capture"].iloc[-1] == pytest.approx(1.0)
    assert table["cumulative_bad_capture"].is_monotonic_increasing
    # A model that ranks at all captures more than its share in the first decile.
    assert table["lift"].iloc[0] > 1.0


def test_approval_curve_prefers_a_cutoff_when_losses_dominate():
    """With a large loss-given-default, approving everyone must not be optimal."""
    rng = np.random.default_rng(5)
    probability = rng.uniform(0.0, 0.6, 4000)
    y = (rng.uniform(size=4000) < probability).astype(int)

    curve = approval_curve(y, probability, revenue_per_good=100.0, loss_per_bad=5000.0)
    best = curve.loc[curve["expected_profit"].idxmax()]
    assert best["approval_rate"] < 0.95
    assert curve["bad_rate"].iloc[-1] > curve["bad_rate"].iloc[0]


def test_evaluate_returns_the_full_metric_set():
    rng = np.random.default_rng(6)
    probability = rng.uniform(0.01, 0.9, 2000)
    y = (rng.uniform(size=2000) < probability).astype(int)
    result = evaluate(y, probability)

    assert set(result) == {
        "auc", "gini", "ks", "brier", "log_loss",
        "observed_default_rate", "mean_predicted_rate", "n",
    }
    assert result["gini"] == pytest.approx(2 * result["auc"] - 1)
    assert 0.0 <= result["brier"] <= 1.0


def test_psi_detects_drift_in_a_zero_inflated_counter():
    """The failure mode that matters: arrears counters are ~95% zero.

    Quantile edges collapse on a variable that concentrated, and an
    implementation that returns 0.0 in that case reports "stable" on precisely
    the features whose drift a lender most needs to see.
    """
    reference = np.zeros(20000)
    reference[:1000] = 1.0
    reference[:200] = 2.0

    unchanged = reference.copy()
    assert population_stability_index(reference, unchanged) == pytest.approx(0.0, abs=1e-9)

    drifted = reference.copy()
    drifted[: len(drifted) // 2] = 5.0  # half the book suddenly has 5 arrears
    value = population_stability_index(reference, drifted)
    assert value > 0.25, f"drift in a zero-inflated counter went undetected (PSI {value})"
    assert psi_verdict(value) == "revalidate"


def test_approval_curve_reaches_full_approval():
    """The last row must be approve-everyone, or the baseline it defines is wrong."""
    rng = np.random.default_rng(7)
    probability = rng.uniform(0.0, 0.5, 1000)
    y = (rng.uniform(size=1000) < probability).astype(int)

    curve = approval_curve(y, probability, revenue_per_good=100.0, loss_per_bad=1000.0)
    assert curve["approval_rate"].iloc[-1] == pytest.approx(1.0)
    assert curve["accepted"].iloc[-1] == len(y)

    bads = int(y.sum())
    expected = (len(y) - bads) * 100.0 - bads * 1000.0
    assert curve["expected_profit"].iloc[-1] == pytest.approx(expected)
