# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Train, log to MLflow, register in Unity Catalog
# MAGIC
# MAGIC Training runs on the driver, not on Spark. That is deliberate: 150,000 rows and
# MAGIC 13 features is a single-node problem, and distributing it would add shuffle cost
# MAGIC for no benefit. Spark's job was the ETL; this is where it hands over.
# MAGIC
# MAGIC What Databricks adds here is the experiment tracking and the registry - the
# MAGIC audit trail a model-risk function asks for: which data, which parameters, which
# MAGIC metrics, which artefact, and who promoted it.

# COMMAND ----------

# MAGIC %pip install --quiet pyyaml joblib xgboost
# MAGIC %restart_python

# COMMAND ----------

import sys

REPO_PATH = "/Workspace/Repos/<your-user>/credit-risk-scorecard"
sys.path.insert(0, f"{REPO_PATH}/src")

CATALOG, SCHEMA = "credit_risk", "scorecard"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.pd_scorecard"

# COMMAND ----------

import mlflow
import pandas as pd

from creditrisk.config import load_config
from creditrisk.logging_utils import setup_logging

setup_logging()
mlflow.set_registry_uri("databricks-uc")
cfg = load_config(f"{REPO_PATH}/conf/config.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull gold out of Delta
# MAGIC
# MAGIC `creditrisk.train.load_gold` reads Parquet from the local lake. On Databricks the
# MAGIC gold table is a Delta table, so it is read here and handed to the same training
# MAGIC code, which never cared where the frame came from.

# COMMAND ----------

gold = spark.table(f"{CATALOG}.{SCHEMA}.gold_features").toPandas()
gold["split"] = gold["split"].astype(str)
print(gold.groupby("split").size().to_dict())

# COMMAND ----------

import creditrisk.train as train_module

# Substitute the Delta-backed frame for the Parquet reader.
train_module.load_gold = lambda _cfg: gold

# COMMAND ----------

# MAGIC %md
# MAGIC ## The pyfunc wrapper
# MAGIC
# MAGIC Defined before the training run so the logged model can reference it. It returns
# MAGIC the full decision - PD, points and policy grade - rather than a bare probability,
# MAGIC because that is what a downstream consumer actually needs.

# COMMAND ----------


class ScorecardWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from creditrisk.model import ScorecardModel

        self.model = ScorecardModel.load(context.artifacts["bundle"])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return self.model.decide(model_input[self.model.features])


# COMMAND ----------

with mlflow.start_run(run_name="woe-scorecard") as run:
    report = train_module.train(cfg)
    champion_name = report["champion"]
    champion = report["models"][champion_name]

    mlflow.log_params(
        {
            "champion": champion_name,
            "features_selected": report["dataset"]["features_selected"],
            "woe_max_bins": cfg.get("woe.max_bins"),
            "woe_min_iv": cfg.get("woe.min_information_value"),
            "max_pairwise_correlation": cfg.get("woe.max_pairwise_correlation"),
            "calibration": cfg.get("model.calibration.method"),
            "pdo": cfg.get("scorecard.pdo"),
        }
    )
    for split_name, metrics in champion.items():
        mlflow.log_metrics({f"{split_name}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
    mlflow.log_metric("psi_train_vs_test", report["stability"]["psi_train_vs_test"])
    mlflow.log_metric("profit_optimal_approval_rate", report["business"]["profit_maximising_approval_rate"])

    reports_dir = cfg.path("paths.reports")
    for artefact in ("metrics.json", "woe_bin_table.csv", "information_values.csv", "band_table.csv", "data_quality.json"):
        path = reports_dir / artefact
        if path.exists():
            mlflow.log_artifact(str(path))

    # The registered object is the whole bundle - WoE bins, estimator, calibrator,
    # scaling - so serving cannot drift from training.
    from creditrisk.model import ScorecardModel

    model = ScorecardModel.load(cfg.path("paths.models") / "scorecard.joblib")
    example = gold[gold["split"] == "test"][model.features].head(5)
    signature = mlflow.models.infer_signature(example, model.predict_proba(example))

    mlflow.pyfunc.log_model(
        artifact_path="scorecard",
        python_model=ScorecardWrapper(),
        artifacts={"bundle": str(cfg.path("paths.models") / "scorecard.joblib")},
        signature=signature,
        input_example=example,
        registered_model_name=MODEL_NAME,
        pip_requirements=["scikit-learn", "pandas", "numpy", "joblib", "pyyaml"],
    )
    print("run:", run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promotion gate
# MAGIC
# MAGIC An alias is only moved if the model clears the thresholds a credit committee set
# MAGIC in advance. Automating the gate is the difference between a registry and a
# MAGIC filing cabinet.

# COMMAND ----------

from mlflow.tracking import MlflowClient

MIN_GINI, MIN_KS, MAX_BRIER = 0.65, 0.50, 0.06

test_metrics = report["models"][report["champion"]]["test"]
gates = {
    "gini": test_metrics["gini"] >= MIN_GINI,
    "ks": test_metrics["ks"] >= MIN_KS,
    "brier": test_metrics["brier"] <= MAX_BRIER,
    "bands_monotonic": report.get("bands_monotonic", False),
    "population_stable": report["stability"]["verdict"] == "stable",
}
print(gates)

if all(gates.values()):
    client = MlflowClient()
    latest = client.get_registered_model(MODEL_NAME).latest_versions[0]
    client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
    print(f"Promoted version {latest.version} to @champion")
else:
    failed = [name for name, passed in gates.items() if not passed]
    raise ValueError(f"Model failed the promotion gate on: {failed}")
