"""Pipeline orchestrator.

``python -m creditrisk.pipeline all`` runs the whole thing end to end: bronze,
silver, gold, training, monitoring, figures and the Open Banking track. Each
stage is also runnable on its own so a failure does not force a full rebuild.

Only the bronze/silver/gold stages need PySpark. Everything downstream reads
Parquet with pandas, which is what keeps CI and the Streamlit app free of a
Spark dependency.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable

from creditrisk.config import Config, load_config
from creditrisk.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)

SPARK_STAGES = {"bronze", "silver", "gold"}


def stage_bronze(cfg: Config) -> None:
    from creditrisk.ingest import build_bronze

    build_bronze(cfg)


def stage_silver(cfg: Config) -> None:
    from creditrisk.clean import build_silver

    _, report = build_silver(cfg)
    LOGGER.info("Data-quality rules applied: %s", json.dumps(report["rules"])[:400])


def stage_gold(cfg: Config) -> None:
    from creditrisk.features import build_gold

    build_gold(cfg)


def stage_train(cfg: Config) -> None:
    from creditrisk.train import train

    report = train(cfg)
    champion = report["models"][report["champion"]]["test"]
    LOGGER.info("Champion test Gini %.4f | KS %.4f", champion["gini"], champion["ks"])


def stage_monitor(cfg: Config) -> None:
    from creditrisk.monitoring import main as monitor_main

    monitor_main()


def stage_openbanking(cfg: Config) -> None:
    from creditrisk.openbanking import main as openbanking_main

    openbanking_main()


def stage_figures(cfg: Config) -> None:
    from creditrisk.plots import build_all

    build_all(cfg)


STAGES: dict[str, Callable[[Config], None]] = {
    "bronze": stage_bronze,
    "silver": stage_silver,
    "gold": stage_gold,
    "train": stage_train,
    "monitor": stage_monitor,
    "openbanking": stage_openbanking,
    "figures": stage_figures,
}

DEFAULT_ORDER = ["bronze", "silver", "gold", "train", "monitor", "openbanking", "figures"]


def run(stages: list[str], cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    for name in stages:
        if name not in STAGES:
            raise SystemExit(f"Unknown stage '{name}'. Choose from: {', '.join(DEFAULT_ORDER)}")
        started = time.perf_counter()
        LOGGER.info("--- stage: %s ---", name)
        STAGES[name](cfg)
        LOGGER.info("--- stage %s finished in %.1fs ---", name, time.perf_counter() - started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Credit risk scorecard pipeline")
    parser.add_argument(
        "stages",
        nargs="*",
        default=["all"],
        help=f"Stages to run: all, or any of {', '.join(DEFAULT_ORDER)}",
    )
    parser.add_argument("--config", default=None, help="Path to an alternative config file")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    stages = DEFAULT_ORDER if args.stages == ["all"] or "all" in args.stages else args.stages

    if set(stages) & SPARK_STAGES:
        try:
            import pyspark  # noqa: F401
        except ImportError:
            LOGGER.error(
                "Stages %s need PySpark. Install it with 'pip install -r requirements-spark.txt', "
                "or run the downstream stages only: python -m creditrisk.pipeline train figures",
                sorted(set(stages) & SPARK_STAGES),
            )
            return 1

    run(stages, cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
