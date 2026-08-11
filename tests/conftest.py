"""Shared fixtures.

Most tests are pure-logic and run in a second on any machine. The few that need
the built artefacts (the gold table, the trained model) skip cleanly when the
pipeline has not been run, so ``pytest`` is never a wall for a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditrisk.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def gold(cfg):
    """The built gold table, or a skip if the pipeline has not been run."""
    path = cfg.path("paths.gold") / "features"
    if not path.exists():
        pytest.skip("Gold table not built. Run: python -m creditrisk.pipeline bronze silver gold")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def model(cfg):
    """The trained champion, or a skip if training has not been run."""
    from creditrisk.model import ScorecardModel

    path = cfg.path("paths.models") / "scorecard.joblib"
    if not path.exists():
        pytest.skip("Model not trained. Run: python -m creditrisk.pipeline train")
    return ScorecardModel.load(path)


@pytest.fixture()
def synthetic_credit_data() -> tuple[pd.DataFrame, pd.Series]:
    """A small, self-contained dataset with a known monotonic relationship.

    Used by the WoE and metric tests so they assert on behaviour that is true by
    construction rather than on whatever the real data happens to do.
    """
    rng = np.random.default_rng(0)
    n = 4000
    utilisation = rng.uniform(0, 1, n)
    income = rng.lognormal(8, 0.5, n)
    noise = rng.normal(0, 0.4, n)
    # Risk rises with utilisation and falls with income - strictly monotonic.
    logit = -3.0 + 3.5 * utilisation - 0.6 * np.log(income / 3000) + noise
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame({"utilisation": utilisation, "income": income})
    return X, pd.Series(y, name="default")
