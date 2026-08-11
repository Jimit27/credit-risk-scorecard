"""Turning a probability of default into a score a human can act on.

Lenders do not operate on probabilities; they operate on scores and cut-offs.
The industry standard mapping is points-to-double-the-odds:

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(odds_of_good)

With the defaults in ``conf/config.yaml`` - 600 points at 50:1 odds, PDO 20 -
every extra 20 points means an applicant is half as likely to default. That is
the property that lets a credit policy be written in plain English, and it is
why the score is derived from the calibrated PD rather than from raw model
output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreScaling:
    """Points-to-double-the-odds scaling parameters."""

    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def to_score(self, probability_of_default: np.ndarray | float) -> np.ndarray:
        """Map PD onto the score scale."""
        pd_clipped = np.clip(np.asarray(probability_of_default, dtype=float), 1e-6, 1 - 1e-6)
        odds_good = (1.0 - pd_clipped) / pd_clipped
        return self.offset + self.factor * np.log(odds_good)

    def to_probability(self, score: np.ndarray | float) -> np.ndarray:
        """Inverse mapping, for reading a policy cut-off back as a PD."""
        odds_good = np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor)
        return 1.0 / (1.0 + odds_good)


def assign_band(score: np.ndarray | float, bands: list[dict]) -> np.ndarray:
    """Label each score with its risk band.

    ``bands`` is the list from the config, ordered from safest downwards; the
    first band whose ``min_score`` the applicant clears wins.
    """
    scores = np.atleast_1d(np.asarray(score, dtype=float))
    ordered = sorted(bands, key=lambda b: float(b["min_score"]), reverse=True)
    out = np.array([ordered[-1]["name"]] * len(scores), dtype=object)
    assigned = np.zeros(len(scores), dtype=bool)
    for band in ordered:
        mask = (~assigned) & (scores >= float(band["min_score"]))
        out[mask] = band["name"]
        assigned |= mask
    return out


def band_lookup(bands: list[dict]) -> dict[str, str]:
    """Band name to its human-readable label."""
    return {str(b["name"]): str(b.get("label", b["name"])) for b in bands}
