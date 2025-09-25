# Databricks notebook source

# from churn.utils import is_databricks
import os

import mlflow

# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
from dotenv import load_dotenv

from churn.config import ProjectConfig, Tags
from churn.models.XGBmodel import XGBModel

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

# if not is_databricks():
load_dotenv()
profile = os.environ.get("PROFILE", "DEFAULT")
mlflow.set_tracking_uri(f"databricks://{profile}")
mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.remote(cluster_id="0911-113128-1v3eeo04").getOrCreate()
tags = Tags(**{"git_sha": "TEMP", "branch": "week2", "model": "xgboost"})

# COMMAND ----------
# Initialize model with the config path
XGB_model = XGBModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
XGB_model.load_data()
XGB_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
XGB_model.train()
XGB_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Users/rosaverhoeven@live.nl/churn"], filter_string="tags.branch='week2'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/xgb-pipeline-model")

# COMMAND ----------
# Retrieve dataset for the current run
XGB_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
XGB_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
XGB_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = XGB_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
