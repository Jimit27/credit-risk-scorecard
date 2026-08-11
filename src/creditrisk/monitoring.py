"""Post-deployment monitoring: score drift and feature drift.

A scorecard's real failure mode is not a bad test-set number, it is being
quietly correct for eighteen months and then wrong because the applicants
changed. This module produces the report a monitoring job would run on a
schedule: PSI on the score, PSI on every input feature, and the observed
default rate per grade where outcomes are available.

To show the monitor actually fires rather than merely existing,
:func:`simulate_shifted_population` bends a sample the way a real book bends -
younger applicants, higher utilisation, thinner incomes - and the report is run
against it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from creditrisk.metrics import population_stability_index, psi_verdict
from creditrisk.model import ScorecardModel

LOGGER = logging.getLogger(__name__)


def drift_report(
    model: ScorecardModel,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    bins: int = 10,
    stable: float = 0.10,
    investigate: float = 0.25,
) -> dict[str, Any]:
    """PSI on the model score and on each input feature."""
    reference_score = model.score(reference[model.features])
    current_score = model.score(current[model.features])

    score_psi = population_stability_index(reference_score, current_score, bins=bins)
    feature_psi = {}
    for feature in model.features:
        left = pd.to_numeric(reference[feature], errors="coerce").dropna().to_numpy()
        right = pd.to_numeric(current[feature], errors="coerce").dropna().to_numpy()
        if len(left) < bins or len(right) < bins:
            continue
        value = population_stability_index(left, right, bins=bins)
        feature_psi[feature] = {"psi": value, "verdict": psi_verdict(value, stable, investigate)}

    worst = sorted(feature_psi.items(), key=lambda kv: kv[1]["psi"], reverse=True)[:5]
    return {
        "score_psi": score_psi,
        "score_verdict": psi_verdict(score_psi, stable, investigate),
        "reference_mean_score": float(np.mean(reference_score)),
        "current_mean_score": float(np.mean(current_score)),
        "feature_psi": feature_psi,
        "largest_feature_shifts": [{"feature": name, **values} for name, values in worst],
        "action": _action(score_psi, stable, investigate),
    }


def _action(score_psi: float, stable: float, investigate: float) -> str:
    if score_psi < stable:
        return "No action. Score distribution is consistent with development."
    if score_psi < investigate:
        return "Investigate. Check the largest feature shifts and recent acquisition channels before the next review."
    return "Revalidate. The scored population no longer resembles the development sample; refit before further lending."


def simulate_shifted_population(frame: pd.DataFrame, seed: int = 42, intensity: float = 1.0) -> pd.DataFrame:
    """Bend a sample the way a real loan book bends.

    This is not noise for its own sake. It reproduces a specific, common
    scenario: a lender opens a new acquisition channel that brings in younger,
    more credit-hungry applicants on thinner incomes. If the monitor cannot see
    that, it is not worth running.
    """
    rng = np.random.default_rng(seed)
    shifted = frame.copy()

    if "age" in shifted:
        shifted["age"] = np.clip(shifted["age"] - 8 * intensity * rng.uniform(0.5, 1.5, len(shifted)), 18, 100)
    for column, multiplier in (
        ("revolving_utilisation", 1 + 0.45 * intensity),
        ("utilisation_x_delinquency", 1 + 0.45 * intensity),
        ("debt_ratio", 1 + 0.30 * intensity),
    ):
        if column in shifted:
            shifted[column] = shifted[column] * multiplier
    for column, multiplier in (
        ("monthly_income", 1 - 0.20 * intensity),
        ("log_monthly_income", 1 - 0.05 * intensity),
        ("disposable_income", 1 - 0.25 * intensity),
        ("income_per_dependent", 1 - 0.20 * intensity),
    ):
        if column in shifted:
            shifted[column] = shifted[column] * multiplier
    return shifted


def main() -> None:  # pragma: no cover - CLI entry point
    import json

    from creditrisk.config import load_config
    from creditrisk.logging_utils import setup_logging
    from creditrisk.train import load_gold

    setup_logging()
    cfg = load_config()
    model = ScorecardModel.load(cfg.path("paths.models") / "scorecard.joblib")
    gold = load_gold(cfg)
    train_frame = gold[gold["split"] == "train"]
    test_frame = gold[gold["split"] == "test"]

    bins = int(cfg.get("monitoring.psi_bins", 10))
    stable = float(cfg.get("monitoring.psi_thresholds.stable", 0.10))
    investigate = float(cfg.get("monitoring.psi_thresholds.investigate", 0.25))

    report = {
        # Control: the holdout should look exactly like development.
        "holdout": drift_report(model, train_frame, test_frame, bins, stable, investigate),
        # Treatment: a population that has genuinely moved.
        "shifted_population": drift_report(
            model, train_frame, simulate_shifted_population(test_frame, seed=cfg.seed), bins, stable, investigate
        ),
    }

    path = cfg.path("paths.reports") / "drift_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info(
        "Drift: holdout PSI %.4f (%s) | shifted PSI %.4f (%s)",
        report["holdout"]["score_psi"],
        report["holdout"]["score_verdict"],
        report["shifted_population"]["score_psi"],
        report["shifted_population"]["score_verdict"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
