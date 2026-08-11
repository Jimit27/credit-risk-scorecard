"""Gold layer: model-ready features with a credit-domain rationale.

Two rules govern this module.

*Every feature has to be explainable to a credit officer.* A scorecard that
declines someone has to be able to say why in a sentence, so ratios and counts
are preferred over anything a human cannot narrate.

*The split is assigned here, deterministically, from a hash of the applicant
id.* Re-running the pipeline cannot move an applicant between train and test,
which is the cheapest possible insurance against the leakage that quietly
inflates the metrics in a large share of portfolio credit projects.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from creditrisk.config import Config
from creditrisk.spark_utils import get_spark, read_table, write_table

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame

LOGGER = logging.getLogger(__name__)

# The feature set handed to the models. Kept explicit so that a column added to
# gold does not silently enter the model without a decision being made.
MODEL_FEATURES: list[str] = [
    "revolving_utilisation",
    "age",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "real_estate_lines",
    "dependents",
    "times_30_59_dpd",
    "times_60_89_dpd",
    "times_90_dpd",
    "monthly_debt_amount",
    "total_delinquencies",
    "worst_delinquency_severity",
    "delinquencies_per_credit_line",
    "unsecured_lines",
    "real_estate_share_of_lines",
    "monthly_debt_service",
    "disposable_income",
    "income_per_dependent",
    "log_monthly_income",
    "utilisation_x_delinquency",
    "flag_delinquency_sentinel",
    "flag_monthly_income_missing",
    "flag_debt_ratio_is_amount",
]

TARGET = "target_default_90dpd_2yr"


def build_gold(cfg: Config) -> DataFrame:
    """Engineer features, impute, assign the split, and write the gold table."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    spark = get_spark()
    df = read_table(spark, cfg.path("paths.silver") / "applications")

    # --- Delinquency profile -------------------------------------------------
    # Nulled sentinels are treated as zero for the *aggregate* counts; the
    # sentinel flag from silver carries the "we do not actually know" signal.
    d30 = F.coalesce(F.col("times_30_59_dpd"), F.lit(0.0))
    d60 = F.coalesce(F.col("times_60_89_dpd"), F.lit(0.0))
    d90 = F.coalesce(F.col("times_90_dpd"), F.lit(0.0))

    df = (
        df.withColumn("total_delinquencies", d30 + d60 + d90)
        # Severity, not frequency: one 90-day miss is worse news than three
        # 30-day misses, and a single summed count loses that ordering.
        .withColumn(
            "worst_delinquency_severity",
            F.when(d90 > 0, F.lit(3.0)).when(d60 > 0, F.lit(2.0)).when(d30 > 0, F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "delinquencies_per_credit_line",
            F.col("total_delinquencies") / F.greatest(F.col("open_credit_lines").cast("double"), F.lit(1.0)),
        )
    )

    # --- Credit mix ----------------------------------------------------------
    df = (
        df.withColumn(
            "unsecured_lines",
            F.greatest(F.col("open_credit_lines").cast("double") - F.col("real_estate_lines").cast("double"), F.lit(0.0)),
        )
        .withColumn(
            "real_estate_share_of_lines",
            F.col("real_estate_lines").cast("double") / F.greatest(F.col("open_credit_lines").cast("double"), F.lit(1.0)),
        )
    )

    # --- Income imputation ---------------------------------------------------
    # Median income within an age band, because income and age are strongly
    # related and a single global median would flatten that structure. The
    # missingness flag survives into the model regardless.
    df = df.withColumn(
        "age_band",
        F.when(F.col("age").isNull(), F.lit("unknown"))
        .when(F.col("age") < 30, F.lit("18-29"))
        .when(F.col("age") < 40, F.lit("30-39"))
        .when(F.col("age") < 50, F.lit("40-49"))
        .when(F.col("age") < 60, F.lit("50-59"))
        .when(F.col("age") < 70, F.lit("60-69"))
        .otherwise(F.lit("70+")),
    )

    band_median = Window.partitionBy("age_band")
    global_median = df.approxQuantile("monthly_income", [0.5], 0.001)
    fallback_income = float(global_median[0]) if global_median else 5000.0

    df = df.withColumn(
        "monthly_income",
        F.coalesce(
            F.col("monthly_income"),
            F.percentile_approx(F.col("monthly_income"), 0.5).over(band_median),
            F.lit(fallback_income),
        ),
    )

    median_age = df.approxQuantile("age", [0.5], 0.001)
    df = df.withColumn("age", F.coalesce(F.col("age"), F.lit(float(median_age[0]) if median_age else 52.0)))
    df = df.withColumn("dependents", F.coalesce(F.col("dependents"), F.lit(0.0)))

    # debt_ratio is null for the 22,691 applicants whose figure was an absolute
    # amount rather than a ratio (silver rule 4). Filling those with 0 would be
    # the single most optimistic value available - it would tell the model that
    # applicants whose debt figure was *too large to interpret* have no debt at
    # all, and hand them a full disposable income. The median is the neutral
    # choice, and the amount itself is preserved in monthly_debt_amount.
    median_debt_ratio = df.approxQuantile("debt_ratio", [0.5], 0.001)
    imputed_debt_ratio = float(median_debt_ratio[0]) if median_debt_ratio else 0.35
    df = df.withColumn("debt_ratio", F.coalesce(F.col("debt_ratio"), F.lit(imputed_debt_ratio)))
    df = df.withColumn("monthly_debt_amount", F.coalesce(F.col("monthly_debt_amount"), F.lit(0.0)))

    # --- Affordability -------------------------------------------------------
    df = (
        df.withColumn("monthly_debt_service", F.col("monthly_income") * F.col("debt_ratio"))
        .withColumn("disposable_income", F.col("monthly_income") - F.col("monthly_debt_service"))
        .withColumn("income_per_dependent", F.col("monthly_income") / (F.lit(1.0) + F.col("dependents")))
        .withColumn("log_monthly_income", F.log1p(F.greatest(F.col("monthly_income"), F.lit(0.0))))
        # An interaction a credit officer would recognise: high utilisation is
        # a very different signal from someone with a clean history than from
        # someone already missing payments.
        .withColumn("utilisation_x_delinquency", F.col("revolving_utilisation") * (F.lit(1.0) + F.col("total_delinquencies")))
    )

    # --- Deterministic, hash-based split ------------------------------------
    train_frac = float(cfg.get("data.split.train", 0.6))
    valid_frac = float(cfg.get("data.split.valid", 0.2))
    bucket = F.pmod(F.hash(F.concat_ws("|", F.col("applicant_id").cast("string"), F.lit(cfg.seed))), F.lit(1000))
    df = df.withColumn(
        "split",
        F.when(bucket < F.lit(int(train_frac * 1000)), F.lit("train"))
        .when(bucket < F.lit(int((train_frac + valid_frac) * 1000)), F.lit("valid"))
        .otherwise(F.lit("test")),
    )

    # Persist the imputation constants. The scoring path has one applicant and
    # no population to take a median from, so it has to be told what the batch
    # job decided - otherwise a live request with a missing income silently gets
    # a different feature value from the one the model was trained on.
    income_by_band = {
        row["age_band"]: float(row["median_income"])
        for row in df.groupBy("age_band")
        .agg(F.percentile_approx("monthly_income", 0.5).alias("median_income"))
        .collect()
        if row["median_income"] is not None
    }
    constants = {
        "monthly_income_by_age_band": income_by_band,
        "monthly_income_global_median": fallback_income,
        "age_median": float(median_age[0]) if median_age else 52.0,
        "debt_ratio_median": imputed_debt_ratio,
        "dependents_fill": 0.0,
        "monthly_debt_amount_fill": 0.0,
    }
    constants_path = cfg.path("paths.reports") / "imputation_constants.json"
    constants_path.parent.mkdir(parents=True, exist_ok=True)
    constants_path.write_text(json.dumps(constants, indent=2), encoding="utf-8")

    keep = ["applicant_id", TARGET, "split", "age_band", *MODEL_FEATURES]
    df = df.select(*[c for c in keep if c in df.columns])

    destination = cfg.path("paths.gold") / "features"
    write_table(df, destination, partition_by=["split"])
    LOGGER.info("Gold written to %s (%s rows, %s features)", destination, df.count(), len(MODEL_FEATURES))
    return df


def main() -> None:  # pragma: no cover - CLI entry point
    from creditrisk.config import load_config
    from creditrisk.logging_utils import setup_logging

    setup_logging()
    build_gold(load_config())


if __name__ == "__main__":  # pragma: no cover
    main()
