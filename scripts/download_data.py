"""Fetch the raw Give Me Some Credit extract.

The dataset is from the 2011 Kaggle "Give Me Some Credit" competition: 150,000
real consumer credit files with a two-year forward default flag. It is not
committed to this repository - 7 MB of competition data does not belong in git
history - so this script pulls it from a public mirror into ``data/raw/``.

    python scripts/download_data.py

If the mirror has moved, download ``cs-training.csv`` from the competition page
at https://www.kaggle.com/c/GiveMeSomeCredit/data and drop it in ``data/raw/``.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MIRROR = (
    "https://raw.githubusercontent.com/JLZml/Credit-Scoring-Data-Sets/master/"
    "3.%20Kaggle/Give%20Me%20Some%20Credit/cs-training.csv"
)
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "raw" / "cs-training.csv"

EXPECTED_ROWS = 150_000
EXPECTED_HEADER = "SeriousDlqin2yrs"


def main() -> int:
    if DESTINATION.exists():
        print(f"Already present: {DESTINATION}")
        return 0

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {MIRROR}")
    try:
        urllib.request.urlretrieve(MIRROR, DESTINATION)
    except Exception as error:  # noqa: BLE001 - the message matters more than the type
        print(f"Download failed: {error}\nFetch cs-training.csv manually into {DESTINATION.parent}", file=sys.stderr)
        return 1

    # Verify what landed is the file we expected, not an HTML error page.
    with open(DESTINATION, encoding="utf-8") as handle:
        header = handle.readline()
        rows = sum(1 for _ in handle)

    if EXPECTED_HEADER not in header or rows != EXPECTED_ROWS:
        print(f"Downloaded file looks wrong: {rows} rows, header {header[:80]!r}", file=sys.stderr)
        DESTINATION.unlink(missing_ok=True)
        return 1

    digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    print(f"Saved {DESTINATION} ({rows:,} rows)\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
