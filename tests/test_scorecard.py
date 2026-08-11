"""Tests for the PD-to-score mapping and the policy grades."""

from __future__ import annotations

import numpy as np
import pytest

from creditrisk.scorecard import ScoreScaling, assign_band

BANDS = [
    {"name": "A", "min_score": 640, "label": "Very low risk"},
    {"name": "B", "min_score": 600, "label": "Low risk"},
    {"name": "C", "min_score": 570, "label": "Moderate risk"},
    {"name": "D", "min_score": 540, "label": "Elevated risk"},
    {"name": "E", "min_score": 0, "label": "High risk"},
]


def test_base_odds_map_to_the_base_score():
    scaling = ScoreScaling(base_score=600, base_odds=50, pdo=20)
    # 50:1 good:bad odds means PD = 1/51.
    assert scaling.to_score(1 / 51) == pytest.approx(600.0)


def test_pdo_doubles_the_odds():
    """The defining property: +PDO points means half the risk."""
    scaling = ScoreScaling(base_score=600, base_odds=50, pdo=20)
    at_600 = scaling.to_score(1 / 51)
    at_double = scaling.to_score(1 / 101)  # odds of 100:1
    assert at_double - at_600 == pytest.approx(20.0)


def test_score_falls_as_risk_rises():
    scaling = ScoreScaling()
    scores = scaling.to_score(np.array([0.01, 0.05, 0.20, 0.50]))
    assert np.all(np.diff(scores) < 0)


def test_score_and_probability_round_trip():
    scaling = ScoreScaling(base_score=600, base_odds=50, pdo=20)
    probabilities = np.array([0.001, 0.02, 0.1, 0.35, 0.8])
    assert np.allclose(scaling.to_probability(scaling.to_score(probabilities)), probabilities)


def test_extreme_probabilities_do_not_produce_infinities():
    scaling = ScoreScaling()
    assert np.all(np.isfinite(scaling.to_score(np.array([0.0, 1.0]))))


@pytest.mark.parametrize(
    ("score", "expected"),
    [(900, "A"), (640, "A"), (639, "B"), (600, "B"), (571, "C"), (540, "D"), (539, "E"), (300, "E")],
)
def test_band_boundaries_are_inclusive_at_the_minimum(score, expected):
    assert assign_band(score, BANDS)[0] == expected


def test_assign_band_is_vectorised():
    result = assign_band(np.array([700, 610, 575, 545, 400]), BANDS)
    assert list(result) == ["A", "B", "C", "D", "E"]


def test_band_assignment_does_not_depend_on_config_ordering():
    shuffled = [BANDS[2], BANDS[0], BANDS[4], BANDS[1], BANDS[3]]
    assert list(assign_band(np.array([700, 610, 575, 545, 400]), shuffled)) == ["A", "B", "C", "D", "E"]
