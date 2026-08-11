"""The simulation must produce byte-identical output on every run.

This exists because it did not. ``RECURRING_CATEGORIES`` was a set, and it is
iterated while drawing each applicant's fixed merchants. Python randomises
string hashing per process, so the iteration order differed between runs, the
random draws desynchronised, and the ledger came out different every time -
same transaction count, different contents. The reported unseen-merchant
accuracy drifted by roughly a point between runs of identical committed code,
which quietly invalidated every Open Banking figure in the documentation.

An in-process check cannot catch this: one process has one hash seed. The test
therefore shells out with ``PYTHONHASHSEED`` set to different values, which is
the only way to reproduce the original defect.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from creditrisk import openbanking

REPO_ROOT = Path(__file__).resolve().parents[1]

# Generate a small ledger and print a digest of it. Small enough to run twice
# inside a test, large enough that a desynchronised RNG shows up.
PROBE = """
import hashlib
import pandas as pd
from creditrisk.openbanking import LedgerSpec, generate_ledger

ledger = generate_ledger(LedgerSpec(n_applicants=40, months=3, seed=7))
ledger = ledger.sort_values(list(ledger.columns)).reset_index(drop=True)
digest = hashlib.sha256(
    pd.util.hash_pandas_object(ledger, index=False).values.tobytes()
).hexdigest()
print(f"{len(ledger)}:{digest}")
"""


def _probe(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"Probe failed under PYTHONHASHSEED={hash_seed}:\n{result.stderr}")
    return result.stdout.strip()


def test_category_constants_have_a_stable_order() -> None:
    """Anything iterated during simulation must be an ordered type.

    A set here is not a style question - it silently breaks reproducibility.
    """
    for name in ("RECURRING_CATEGORIES", "ESSENTIAL_CATEGORIES", "DISCRETIONARY_CATEGORIES"):
        value = getattr(openbanking, name)
        assert isinstance(value, tuple), f"{name} must be a tuple, not {type(value).__name__}"
        assert len(value) == len(set(value)), f"{name} contains duplicates"


def test_ledger_is_identical_under_different_hash_seeds() -> None:
    """The exact defect: same seed, different PYTHONHASHSEED, different ledger."""
    first = _probe("1")
    second = _probe("2")
    assert first == second, (
        "The simulated ledger changed when PYTHONHASHSEED changed. Something in "
        "the generation path is iterating an unordered collection.\n"
        f"  PYTHONHASHSEED=1 -> {first}\n"
        f"  PYTHONHASHSEED=2 -> {second}"
    )
