"""The deployable scorecard bundle.

Everything needed to turn a raw applicant record into a decision travels in one
picklable object: the WoE bins, the selected feature list, the fitted
estimator, the post-hoc calibrator and the score scaling. The Streamlit app and
the batch scorer both load this and nothing else, which removes the classic
training/serving skew where the app quietly re-implements the preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from creditrisk.scorecard import ScoreScaling, assign_band
from creditrisk.woe import WoETransformer


@dataclass
class ScorecardModel:
    """A calibrated PD model plus its scaling and policy bands."""

    kind: str  # "logistic_woe" or "gbm"
    estimator: Any
    features: list[str]
    scaling: ScoreScaling
    bands: list[dict]
    woe: WoETransformer | None = None
    calibrator: Any | None = None
    metadata: dict[str, Any] | None = None

    # --- preprocessing -------------------------------------------------------
    def _design_matrix(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in raw.columns]
        if missing:
            raise KeyError(f"Missing required features: {missing}")
        frame = raw[self.features]
        if self.woe is not None:
            return self.woe.transform(frame)
        return frame.astype(float)

    # --- scoring -------------------------------------------------------------
    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default."""
        X = self._design_matrix(raw)
        raw_probability = self.estimator.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            calibrated = self.calibrator.predict(raw_probability)
            return np.clip(calibrated, 1e-6, 1 - 1e-6)
        return raw_probability

    def score(self, raw: pd.DataFrame) -> np.ndarray:
        """Points-to-double-the-odds score.

        For the WoE scorecard the score is derived from the model's own
        log-odds, **not** from the calibrated probability. That is not a detail:
        it is what makes the published points table exact. The score is then
        literally the sum of one row per feature from
        :func:`creditrisk.explain.points_table`, and a credit officer can
        reproduce any decision on paper.

        Deriving the score from the calibrated PD instead would break that. The
        calibrator is isotonic - a monotone step function, not an affine one -
        so passing it through the PDO transform preserves the *ranking* but
        destroys the additivity, and the points would no longer sum to the
        score. The two roles are kept separate: the score is the model, the
        calibrated PD is the price.
        """
        if self.kind == "logistic_woe":
            X = self._design_matrix(raw)
            log_odds_bad = np.asarray(self.estimator.decision_function(X), dtype=float)
            return self.scaling.offset - self.scaling.factor * log_odds_bad
        # No closed-form points table exists for the challenger, so its score is
        # the PDO transform of its calibrated probability.
        return self.scaling.to_score(self.predict_proba(raw))

    def decide(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Calibrated PD, scorecard points, and policy grade."""
        pd_hat = self.predict_proba(raw)
        scores = self.score(raw)
        return pd.DataFrame(
            {
                "probability_of_default": pd_hat,
                "score": np.round(scores).astype(int),
                "band": assign_band(scores, self.bands),
            },
            index=raw.index,
        )

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> ScorecardModel:
        return joblib.load(Path(path))
