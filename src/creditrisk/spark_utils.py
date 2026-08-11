"""Spark session management and storage-layer helpers.

The pipeline is written once and runs in two places:

* **Locally** - a ``local[*]`` session writing Parquet under ``data/``.
* **On Databricks** - the notebook attaches to an existing cluster session and
  the same functions write Delta tables into Unity Catalog.

Nothing in ``clean.py`` or ``features.py`` knows which of those it is running
in; that is the whole point of routing every read and write through here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession


def on_databricks() -> bool:
    """True when running inside a Databricks runtime."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def table_format() -> str:
    """Delta on Databricks, Parquet locally.

    Delta is not available in a bare open-source PySpark install without the
    ``delta-spark`` package, so local runs use Parquet. The layer semantics -
    bronze / silver / gold - are identical either way.
    """
    return os.environ.get("CREDITRISK_TABLE_FORMAT", "delta" if on_databricks() else "parquet")


def get_spark(app_name: str = "credit-risk-scorecard", memory: str = "4g") -> SparkSession:
    """Return the active Spark session, creating a local one if needed."""
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is not None:
        return active

    return (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def write_table(df: DataFrame, destination: str | Path, partition_by: list[str] | None = None) -> None:
    """Write a medallion-layer table, overwriting any previous run."""
    writer = df.write.mode("overwrite").format(table_format())
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(str(destination))


def read_table(spark: SparkSession, source: str | Path) -> DataFrame:
    """Read a medallion-layer table written by :func:`write_table`."""
    return spark.read.format(table_format()).load(str(source))
