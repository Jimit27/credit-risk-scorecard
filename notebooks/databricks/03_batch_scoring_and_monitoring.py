# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Scheduled batch scoring and drift monitoring
# MAGIC
# MAGIC The job a scorecard actually spends its life doing. Two things happen here:
# MAGIC
# MAGIC 1. **Score** the current application book from the `@champion` alias and append
# MAGIC    the decisions to a Delta table, so every decision the model ever made is
# MAGIC    reproducible from the version that made it.
# MAGIC 2. **Monitor** the scored population against the development sample. A model's
# MAGIC    real failure mode is not a bad test metric - it is being quietly correct for
# MAGIC    eighteen months and then wrong because the applicants changed.
# MAGIC
# MAGIC Schedule as a Databricks Job, daily.

# COMMAND ----------

# MAGIC %pip install --quiet pyyaml joblib
# MAGIC %restart_python

# COMMAND ----------

import sys

REPO_PATH = "/Workspace/Repos/<your-user>/credit-risk-scorecard"
sys.path.insert(0, f"{REPO_PATH}/src")

CATALOG, SCHEMA = "credit_risk", "scorecard"
MODEL_URI = f"models:/{CATALOG}.{SCHEMA}.pd_scorecard@champion"

# COMMAND ----------

import mlflow
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score the book

# COMMAND ----------

applications = spark.table(f"{CATALOG}.{SCHEMA}.gold_features")

model_version = mlflow.models.get_model_info(MODEL_URI).registered_model_version
scorer = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="struct<probability_of_default:double,score:long,band:string>")

from creditrisk.features import MODEL_FEATURES

feature_columns = [c for c in MODEL_FEATURES if c in applications.columns]

scored = (
    applications.withColumn("decision", scorer(*[F.col(c) for c in feature_columns]))
    .select(
        "applicant_id",
        F.col("decision.probability_of_default").alias("probability_of_default"),
        F.col("decision.score").alias("score"),
        F.col("decision.band").alias("band"),
    )
    .withColumn("scored_at", F.current_timestamp())
    .withColumn("model_version", F.lit(model_version))
)

(
    scored.write.mode("append")
    .option("mergeSchema", "false")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.decisions")
)
display(scored.groupBy("band").count().orderBy("band"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drift check
# MAGIC
# MAGIC PSI on the score, and on every input feature. The thresholds are the industry
# MAGIC convention: below 0.10 stable, 0.10-0.25 investigate, above 0.25 revalidate.

# COMMAND ----------

from creditrisk.config import load_config
from creditrisk.monitoring import drift_report

cfg = load_config(f"{REPO_PATH}/conf/config.yaml")
model = mlflow.pyfunc.load_model(MODEL_URI)._model_impl.python_model.model  # the ScorecardModel bundle

reference = spark.table(f"{CATALOG}.{SCHEMA}.gold_features").filter("split = 'train'").toPandas()
current = applications.toPandas()

report = drift_report(
    model,
    reference,
    current,
    bins=int(cfg.get("monitoring.psi_bins", 10)),
    stable=float(cfg.get("monitoring.psi_thresholds.stable", 0.10)),
    investigate=float(cfg.get("monitoring.psi_thresholds.investigate", 0.25)),
)
print(report["action"])

# COMMAND ----------

import json

monitoring_row = spark.createDataFrame(
    [
        (
            float(report["score_psi"]),
            report["score_verdict"],
            float(report["reference_mean_score"]),
            float(report["current_mean_score"]),
            json.dumps(report["largest_feature_shifts"]),
            str(model_version),
        )
    ],
    "score_psi double, verdict string, reference_mean_score double, current_mean_score double, "
    "largest_feature_shifts string, model_version string",
).withColumn("checked_at", F.current_timestamp())

monitoring_row.write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.monitoring")

# Fail the job on a material shift. A monitoring table nobody reads is not
# monitoring - the job has to be able to stop the line.
if report["score_verdict"] == "revalidate":
    raise ValueError(f"Population drift requires revalidation: PSI {report['score_psi']:.3f}. {report['action']}")
