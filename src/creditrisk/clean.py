"""Silver layer: apply documented, auditable data-quality rules.

Every rule in this module exists because of a specific, evidenced defect in the
source extract. The rules are declared in ``conf/config.yaml`` and each one
emits a counter into ``reports/data_quality.json``, so a reviewer can see how
many records each correction touched instead of taking it on trust.

The defects being corrected:

1. **Administrative sentinels.** The three delinquency counters use 96 and 98
   as codes, not counts. Left untreated they tell the model that 225 people
   missed ninety-eight payments and still mostly repaid.
2. **Impossible ratios.** ``revolving_utilisation`` reaches 50,708 and
   ``debt_ratio`` reaches 329,664. Both are bounded quantities by definition.
3. **Debt ratio switching units.** Where ``monthly_income`` is null,
   ``debt_ratio`` frequently holds an absolute monetary amount rather than a
   ratio - the same column carrying two different meanings.
4. **Missingness that means something.** 19.8% of incomes are absent, and that
   absence is itself predictive, so it is flagged before it is imputed.
5. **Exact duplicates.** 609 rows are byte-identical to another row.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from creditrisk.config import Config
from creditrisk.spark_utils import get_spark, read_table, write_table

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

LOGGER = logging.getLogger(__name__)

# Column names as they exist after the bronze rename.
DELINQUENCY_COLUMNS = ["times_30_59_dpd", "times_60_89_dpd", "times_90_dpd"]

WINSORISE_MAP = {
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilisation",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfDependents": "dependents",
}

# Bronze lands everything as text. This is the one place the project commits to
# a type for each field, and every value that will not cast is counted.
NUMERIC_SCHEMA: dict[str, str] = {
    "applicant_id": "int",
    "target_default_90dpd_2yr": "int",
    "revolving_utilisation": "double",
    "age": "double",
    "times_30_59_dpd": "double",
    "debt_ratio": "double",
    "monthly_income": "double",
    "open_credit_lines": "double",
    "times_90_dpd": "double",
    "real_estate_lines": "double",
    "times_60_89_dpd": "double",
    "dependents": "double",
}


def _cast_with_audit(df: DataFrame, report: dict[str, Any]) -> DataFrame:
    """Cast bronze strings to their declared types, counting every failure.

    ``try_cast`` returns null instead of throwing, so a single unparseable
    value cannot abort a 150,000-row job - but the count of what it swallowed
    is written into the data-quality report rather than lost.
    """
    from pyspark.sql import functions as F

    unparseable: dict[str, int] = {}
    for column, dtype in NUMERIC_SCHEMA.items():
        if column not in df.columns:
            continue
        raw = F.trim(F.col(column))
        # TRY_CAST via SQL rather than the Column method: the SQL form is
        # available across Spark 3.2+ and every Databricks runtime this is
        # likely to meet, the Column method only from PySpark 4.
        typed = F.expr(f"try_cast(trim(`{column}`) as {dtype})")
        # A value is "unparseable" only if it was present but would not cast -
        # a genuinely empty cell is missing data, not a parse failure.
        failed = raw.isNotNull() & (raw != "") & typed.isNull()
        count = df.filter(failed).count()
        if count:
            unparseable[column] = count
        df = df.withColumn(column, typed)

    report["rules"]["unparseable_values_nulled"] = unparseable
    return df


def build_silver(cfg: Config) -> tuple[DataFrame, dict[str, Any]]:
    """Clean the bronze table and return it alongside a data-quality report."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    spark = get_spark()
    df = read_table(spark, cfg.path("paths.bronze") / "applications")

    report: dict[str, Any] = {"rows_in": df.count(), "rules": {}}
    df = _cast_with_audit(df, report)

    # --- Rule 1: exact duplicates -------------------------------------------
    if cfg.get("cleaning.drop_exact_duplicates", True):
        feature_cols = [c for c in df.columns if c not in {"applicant_id", "ingested_at", "source_file"}]
        before = df.count()
        # dropDuplicates does not define *which* row of a duplicate group
        # survives, and the train/test split is hashed from applicant_id - so a
        # non-deterministic survivor would let 609 records drift across the
        # split boundary between rebuilds. Keep the lowest id, always.
        window = Window.partitionBy(*feature_cols).orderBy(F.col("applicant_id").asc())
        df = df.withColumn("_dup_rank", F.row_number().over(window)).filter(F.col("_dup_rank") == 1).drop("_dup_rank")
        removed = before - df.count()
        report["rules"]["exact_duplicates_removed"] = removed
        LOGGER.info("Dropped %s exact duplicate applications", removed)

    # --- Rule 2: delinquency sentinel codes ---------------------------------
    sentinels = [int(v) for v in cfg.get("cleaning.delinquency_sentinels", [96, 98])]
    sentinel_hit = F.lit(False)
    for column in DELINQUENCY_COLUMNS:
        sentinel_hit = sentinel_hit | F.col(column).isin(sentinels)

    # The flag is retained deliberately: an applicant whose bureau record is
    # coded rather than counted is a different kind of applicant, and the model
    # is entitled to know that.
    df = df.withColumn("flag_delinquency_sentinel", sentinel_hit.cast("int"))
    report["rules"]["delinquency_sentinel_rows"] = df.filter(F.col("flag_delinquency_sentinel") == 1).count()

    for column in DELINQUENCY_COLUMNS:
        df = df.withColumn(
            column,
            F.when(F.col(column).isin(sentinels), F.lit(None).cast("double")).otherwise(F.col(column).cast("double")),
        )

    # --- Rule 3: implausible ages -------------------------------------------
    age_min = int(cfg.get("cleaning.age.min_valid", 18))
    age_max = int(cfg.get("cleaning.age.max_valid", 100))
    invalid_age = (F.col("age") < age_min) | (F.col("age") > age_max)
    report["rules"]["invalid_age_rows"] = df.filter(invalid_age).count()
    df = df.withColumn("age", F.when(invalid_age, F.lit(None).cast("double")).otherwise(F.col("age").cast("double")))

    # --- Rule 4: debt_ratio carrying an absolute amount ----------------------
    # Where income is missing, a "ratio" above 100 is not a ratio. Preserve the
    # value in its own column instead of silently winsorising away a number
    # that actually means something.
    ambiguous_debt = F.col("monthly_income").isNull() & (F.col("debt_ratio") > 100)
    report["rules"]["debt_ratio_unit_ambiguous_rows"] = df.filter(ambiguous_debt).count()
    df = (
        df.withColumn("flag_debt_ratio_is_amount", ambiguous_debt.cast("int"))
        .withColumn("monthly_debt_amount", F.when(ambiguous_debt, F.col("debt_ratio")).otherwise(F.lit(None)))
        .withColumn("debt_ratio", F.when(ambiguous_debt, F.lit(None).cast("double")).otherwise(F.col("debt_ratio")))
    )

    # --- Rule 5: missingness flags, recorded before imputation --------------
    for column in ("monthly_income", "dependents"):
        flag = f"flag_{column}_missing"
        df = df.withColumn(flag, F.col(column).isNull().cast("int"))
        report["rules"][f"{column}_missing_rows"] = df.filter(F.col(flag) == 1).count()

    # --- Rule 6: winsorise bounded quantities -------------------------------
    winsor_cfg = cfg.get("cleaning.winsorise", {}) or {}
    capped: dict[str, int] = {}
    for source_name, bounds in winsor_cfg.items():
        column = WINSORISE_MAP.get(source_name, source_name)
        if column not in df.columns:
            continue
        low, high = float(bounds[0]), float(bounds[1])
        out_of_range = F.col(column).isNotNull() & ((F.col(column) < low) | (F.col(column) > high))
        capped[column] = df.filter(out_of_range).count()
        df = df.withColumn(
            column,
            F.when(F.col(column).isNull(), F.lit(None).cast("double"))
            .otherwise(F.least(F.greatest(F.col(column).cast("double"), F.lit(low)), F.lit(high))),
        )
    report["rules"]["values_winsorised"] = capped

    df = df.withColumn("cleaned_at", F.current_timestamp())
    report["rows_out"] = df.count()

    destination = cfg.path("paths.silver") / "applications"
    write_table(df, destination)

    report_path = cfg.path("paths.reports") / "data_quality.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Silver written to %s (%s rows)", destination, report["rows_out"])
    return df, report


def main() -> None:  # pragma: no cover - CLI entry point
    from creditrisk.config import load_config
    from creditrisk.logging_utils import setup_logging

    setup_logging()
    build_silver(load_config())


if __name__ == "__main__":  # pragma: no cover
    main()
