"""Consistent logging across CLI entry points and notebooks."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with a format that reads well in a job log."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(level)

    # Spark is extremely chatty at INFO and drowns the pipeline's own output.
    for noisy in ("py4j", "pyspark"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
