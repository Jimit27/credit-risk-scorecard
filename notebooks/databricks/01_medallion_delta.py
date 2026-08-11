# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Medallion layers as Delta tables in Unity Catalog
# MAGIC
# MAGIC The transformation logic is **not** re-written here. This notebook installs the
# MAGIC `creditrisk` package from the repo and calls the same `build_bronze` /
# MAGIC `build_silver` / `build_gold` functions the local pipeline uses; the only thing
# MAGIC that changes is where the tables land and in what format.
# MAGIC
# MAGIC That is the point of routing every read and write through
# MAGIC `creditrisk.spark_utils` - `table_format()` returns `delta` here and `parquet`
# MAGIC locally, and no transformation code has to know the difference.
# MAGIC
# MAGIC **Cluster**: Databricks Runtime 14.3 LTS or later. Free Edition is sufficient.

# COMMAND ----------

# MAGIC %pip install --quiet pyyaml joblib
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

# Point this at wherever the repo is checked out in your workspace.
REPO_PATH = "/Workspace/Repos/<your-user>/credit-risk-scorecard"
sys.path.insert(0, f"{REPO_PATH}/src")

CATALOG = "credit_risk"
SCHEMA = "scorecard"
VOLUME = "raw"

os.environ["CREDITRISK_TABLE_FORMAT"] = "delta"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Governance objects
# MAGIC
# MAGIC A managed volume holds the raw CSV; the three layers are managed Delta tables in
# MAGIC the same schema, so lineage, permissions and history are all handled by Unity
# MAGIC Catalog rather than by convention.

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/cs-training.csv"
print("Upload cs-training.csv to:", RAW_PATH)

# COMMAND ----------

from creditrisk.config import load_config
from creditrisk.logging_utils import setup_logging

setup_logging()
cfg = load_config(f"{REPO_PATH}/conf/config.yaml")

# Redirect the configured paths at Unity Catalog table names.
cfg.raw["paths"]["raw"] = RAW_PATH
cfg.raw["paths"]["bronze"] = f"{CATALOG}.{SCHEMA}"
cfg.raw["paths"]["silver"] = f"{CATALOG}.{SCHEMA}"
cfg.raw["paths"]["gold"] = f"{CATALOG}.{SCHEMA}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze - land the extract untouched
# MAGIC
# MAGIC Everything is read as a string. The extract writes missing income as the literal
# MAGIC text `NA`, and casting is a decision that belongs in silver where it can be
# MAGIC counted, not in a reader option nobody reads.

# COMMAND ----------

from creditrisk.ingest import build_bronze

bronze = build_bronze(cfg)
bronze.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.bronze_applications")
display(spark.table(f"{CATALOG}.{SCHEMA}.bronze_applications").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver - documented data-quality rules, each one counted
# MAGIC
# MAGIC The rules correct the specific defects evidenced in `docs/data_quality.md`:
# MAGIC administrative sentinel codes in the delinquency counters, ratios that exceed
# MAGIC their own definition, a debt column that switches units when income is absent,
# MAGIC and 609 byte-identical duplicate rows.

# COMMAND ----------

from creditrisk.clean import build_silver

silver, quality_report = build_silver(cfg)
silver.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.silver_applications")
display(spark.createDataFrame([(k, str(v)) for k, v in quality_report["rules"].items()], ["rule", "records_affected"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold - model-ready features and a deterministic split
# MAGIC
# MAGIC The train/validation/test assignment is hashed from the applicant id, so
# MAGIC re-running the pipeline cannot move an applicant across the split boundary.

# COMMAND ----------

from creditrisk.features import build_gold

gold = build_gold(cfg)
(
    gold.write.mode("overwrite")
    .partitionBy("split")
    .option("delta.autoOptimize.optimizeWrite", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_features")
)

# COMMAND ----------

# Compaction and data skipping. Applications are read back by split and scored by
# applicant, which is what these are ordered for.
spark.sql(f"OPTIMIZE {CATALOG}.{SCHEMA}.gold_features ZORDER BY (applicant_id)")

spark.sql(
    f"""
    ALTER TABLE {CATALOG}.{SCHEMA}.gold_features
    SET TBLPROPERTIES (
      'comment' = 'Model-ready credit application features. Deterministic hash split. Built by creditrisk.features.',
      'quality' = 'gold'
    )
    """
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Expectations
# MAGIC
# MAGIC Cheap assertions that fail the job rather than quietly poisoning training.

# COMMAND ----------

checks = spark.sql(
    f"""
    SELECT
      COUNT(*)                                                   AS rows,
      COUNT(DISTINCT applicant_id)                               AS distinct_applicants,
      SUM(CASE WHEN target_default_90dpd_2yr IS NULL THEN 1 END) AS null_targets,
      COUNT(DISTINCT split)                                      AS splits,
      ROUND(AVG(target_default_90dpd_2yr), 5)                    AS default_rate
    FROM {CATALOG}.{SCHEMA}.gold_features
    """
).first()

assert checks["rows"] == checks["distinct_applicants"], "applicant_id is not unique in gold"
assert checks["null_targets"] is None, "gold contains rows with no outcome"
assert checks["splits"] == 3, "expected exactly three split values"
assert 0.03 < checks["default_rate"] < 0.12, f"default rate {checks['default_rate']} is outside the plausible range"
print(dict(checks.asDict()))
