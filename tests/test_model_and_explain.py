"""Tests on the trained artefact itself.

These are the checks that would sit in a model-risk sign-off: the score is
monotonic in the things it should be monotonic in, the points table reconstructs
the score exactly, and the reason codes point at the factors that actually cost
the applicant points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk.explain import explain_applicant, global_importance, points_table, reason_codes
from creditrisk.serving import EXAMPLE_APPLICANTS, derive_features


def test_riskier_applicants_score_lower(model):
    """Ordering the demo profiles by grade must order them by score."""
    scored = {
        name: model.score(derive_features(profile)[model.features])[0]
        for name, profile in EXAMPLE_APPLICANTS.items()
    }
    ordered = [scored[name] for name in EXAMPLE_APPLICANTS]
    assert ordered == sorted(ordered, reverse=True), scored


def test_more_arrears_never_improves_the_score(model):
    """A monotonicity check a credit committee would insist on."""
    base = dict(EXAMPLE_APPLICANTS["Grade C - mid file, moderate usage"])
    scores = []
    for arrears in range(0, 5):
        applicant = {**base, "times_90_dpd": arrears}
        scores.append(model.score(derive_features(applicant)[model.features])[0])
    assert np.all(np.diff(scores) <= 1e-6), scores


def test_higher_utilisation_never_improves_the_score(model):
    base = dict(EXAMPLE_APPLICANTS["Grade C - mid file, moderate usage"])
    scores = [
        model.score(derive_features({**base, "revolving_utilisation": u})[model.features])[0]
        for u in (0.05, 0.25, 0.5, 0.75, 0.98)
    ]
    assert np.all(np.diff(scores) <= 1e-6), scores


def test_points_table_reconstructs_the_shipped_score(model, gold):
    """The published table must be the model, not an approximation of it.

    This compares against ``model.score`` - the number the applicant is actually
    given - rather than against an intermediate quantity. Comparing to the
    pre-calibration log-odds would pass by construction and prove nothing, which
    is exactly the trap a reviewer would look for.
    """
    if model.kind != "logistic_woe":
        pytest.skip("points table applies to the WoE logistic champion")

    sample = gold[gold["split"] == "test"].sample(n=300, random_state=0).reset_index(drop=True)
    scores = model.score(sample[model.features])

    differences = []
    for position in range(len(sample)):
        row_sum = explain_applicant(model, sample.iloc[position])["points"].sum()
        differences.append(abs(row_sum - scores[position]))

    # The only slack is the 2-decimal rounding of the published points column.
    assert max(differences) < 0.1, f"points do not sum to the score: max difference {max(differences):.4f}"


def test_demo_profiles_also_reconstruct(model):
    if model.kind != "logistic_woe":
        pytest.skip("points table applies to the WoE logistic champion")
    for profile in EXAMPLE_APPLICANTS.values():
        applicant = derive_features(profile)
        row_sum = explain_applicant(model, applicant.iloc[0])["points"].sum()
        assert row_sum == pytest.approx(model.score(applicant[model.features])[0], abs=0.1)


def test_missing_value_does_not_earn_the_best_bin(model):
    """A NaN must route to the Missing bin, whatever float type it arrives as."""
    from creditrisk.explain import _bin_label_for
    from creditrisk.woe import MISSING_LABEL

    if model.kind != "logistic_woe":
        pytest.skip("points table applies to the WoE logistic champion")

    feature = model.features[0]
    for missing in (np.nan, np.float32("nan"), np.float64("nan"), None, pd.NA):
        assert _bin_label_for(model, feature, missing) == MISSING_LABEL, f"{type(missing)} leaked past the NaN check"


def test_calibrated_probability_stays_in_range(model, gold):
    sample = gold.sample(n=min(2000, len(gold)), random_state=1)
    probability = model.predict_proba(sample[model.features])
    assert np.all((probability > 0) & (probability < 1))


def test_decide_returns_score_band_and_probability(model):
    applicant = derive_features(EXAMPLE_APPLICANTS["Grade E - maxed out, recent arrears"])
    decision = model.decide(applicant[model.features])
    assert list(decision.columns) == ["probability_of_default", "score", "band"]
    assert decision["band"].iloc[0] in {b["name"] for b in model.bands}


def test_reason_codes_are_ranked_and_material(model):
    applicant = derive_features(EXAMPLE_APPLICANTS["Grade E - maxed out, recent arrears"])
    codes = reason_codes(model, applicant.iloc[0], top_n=4)

    assert 0 < len(codes) <= 4
    lost = [c["points_lost"] for c in codes]
    assert lost == sorted(lost, reverse=True), "reason codes must be ranked by points lost"
    assert all(c["points_lost"] > 0 for c in codes), "a factor that cost nothing is not a reason"


def test_a_strong_applicant_has_few_reason_codes(model):
    applicant = derive_features(EXAMPLE_APPLICANTS["Grade A - homeowner, clean file"])
    strong = reason_codes(model, applicant.iloc[0], top_n=5)
    weak = reason_codes(
        model, derive_features(EXAMPLE_APPLICANTS["Grade E - maxed out, recent arrears"]).iloc[0], top_n=5
    )
    assert sum(c["points_lost"] for c in strong) < sum(c["points_lost"] for c in weak)


def test_missing_features_raise_rather_than_score_silently(model):
    with pytest.raises(KeyError):
        model.predict_proba(pd.DataFrame({"age": [40]}))


def test_global_importance_ranks_by_population_weighted_spread(model):
    """Ranking on raw points range would promote features almost nobody sits in.

    A bin holding 0.2% of the book can carry a 50-point range and never move a
    real decision, so the ranking has to weight by how many applicants actually
    land in each bin.
    """
    if model.kind != "logistic_woe":
        pytest.skip("points table applies to the WoE logistic champion")
    importance = global_importance(model)
    assert importance["population_weighted_spread"].is_monotonic_decreasing
    assert len(importance) == len(model.features)
    assert (importance["population_weighted_spread"] <= importance["points_range"] + 1e-9).all()


def test_every_feature_has_a_human_description(model):
    if model.kind != "logistic_woe":
        pytest.skip("points table applies to the WoE logistic champion")
    table = points_table(model)
    assert (table["description"] != table["feature"]).all(), "a feature is missing a plain-English description"
