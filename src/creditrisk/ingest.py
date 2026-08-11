"""Bronze layer: land the raw extract exactly as supplied.

The bronze rule is that nothing is corrected here. Column names are made
SQL-safe and provenance columns are attached, but no value is altered - so if
a downstream number looks wrong, bronze is the ground truth you diff against.

Everything is read as a string on purpose. The extract encodes missing income
as the literal text ``NA``, which means a schema-inferring reader either types
the column as a string anyway or, worse, silently coerces it. Casting is a
*decision*, and decisions belong in silver where they can be counted and
reported rather than in a reader option nobody reads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from creditrisk.config import Config
from creditrisk.spark_utils import get_spark, write_table

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

LOGGER = logging.getLogger(__name__)

# Source column -> SQL-safe name. The hyphens in the original headers are not
# legal in a Spark/SQL identifier without backticks, which is a footgun waiting
# to happen in every downstream query.
COLUMN_RENAMES: dict[str, str] = {
    "SeriousDlqin2yrs": "target_default_90dpd_2yr",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilisation",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "times_30_59_dpd",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "times_90_dpd",
    "NumberRealEstateLoansOrLines": "real_estate_lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "times_60_89_dpd",
    "NumberOfDependents": "dependents",
}


def build_bronze(cfg: Config) -> DataFrame:
    """Read the raw CSV and land it in the bronze layer."""
    from pyspark.sql import functions as F

    spark = get_spark()
    source = cfg.path("paths.raw")
    LOGGER.info("Reading raw extract from %s", source)

    df = spark.read.csv(str(source), header=True, inferSchema=False)

    # The extract ships with an unnamed integer index in column 0. That index
    # is the only stable applicant key we have, so it is promoted rather than
    # discarded.
    index_col = df.columns[0]
    df = df.withColumnRenamed(index_col, "applicant_id")

    for source_name, target_name in COLUMN_RENAMES.items():
        if source_name in df.columns:
            df = df.withColumnRenamed(source_name, target_name)

    df = (
        df.withColumn("ingested_at", F.current_timestamp())
        .withColumn("source_file", F.lit(source.name))
    )

    destination = cfg.path("paths.bronze") / "applications"
    write_table(df, destination)
    LOGGER.info("Bronze written to %s (%s rows)", destination, df.count())
    return df


def main() -> None:  # pragma: no cover - CLI entry point
    from creditrisk.config import load_config
    from creditrisk.logging_utils import setup_logging

    setup_logging()
    build_bronze(load_config())


if __name__ == "__main__":  # pragma: no cover
    main()
