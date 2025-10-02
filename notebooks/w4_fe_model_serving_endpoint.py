# Databricks notebook source
# MAGIC %pip uninstall -y churn

# COMMAND ----------
# MAGIC import os
# MAGIC dbutils.fs.rm("dbfs:/tmp/churn-0.0.1-py3-none-any.whl")

# COMMAND ----------
# MAGIC %pip install /dbfs/tmp/churn-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import os
import time

import requests
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from loguru import logger
from pyspark.sql import SparkSession

from churn.config import ProjectConfig
from churn.serving.fe_model_serving import FeatureLookupServing

# COMMAND ----------


spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "churn-model-serving-fe"

# COMMAND ----------
# Initialize Feature Lookup Serving Manager
feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{schema_name}.churn_model",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.churn_features",
)

# COMMAND ----------
# Create online store
fe = FeatureEngineeringClient()
online_store_name = "churn-predictions"
if fe.get_online_store(name=online_store_name) is None:
    fe.create_online_store(name=online_store_name, capacity="CU_1")
    online_store = fe.get_online_store(name=online_store_name)
else:
    online_store = fe.get_online_store(name=online_store_name)
# COMMAND ----------

# Create online table
feature_table_name = f"{catalog_name}.{schema_name}.churn_features"
online_table_name = f"{catalog_name}.{schema_name}.churn_features_online"

# Specify primary keys and scheduling policy
spec = OnlineTableSpec(
    primary_key_columns=["customerID"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=True,
)

# Create OnlineTable object
online_table = OnlineTable(name=online_table_name, spec=spec)

# Create the online table in the catalog
try:
    w.online_tables.create(online_table)
    print(f"✅ Online table '{online_table_name}' succesfully created.")
except Exception as e:
    if "already exists" in str(e):
        print(f"ℹ️ Online table '{online_table_name}' already exists.")
    else:
        print(f"❌ Error: {e}")
        raise e

# COMMAND ----------
# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body
required_columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_services",
    "is_vulnerable",
    "has_excessive_costs",
]


spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()

# Add column to test set (not automatically applied like in training set)
train_set["has_excessive_costs"] = (train_set["MonthlyCharges"] * train_set["tenure"]) < train_set["TotalCharges"]

# Add column "num_services" to test set (not automatically applied like in training set)
service_cols = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
train_set["num_services"] = train_set[service_cols].apply(lambda row: sum(row == "Yes"), axis=1)

# Add column "is_vulnerable" to test set (not automatically applied like in training set)
train_set["is_vulnerable"] = (
    (train_set["SeniorCitizen"] == 1) & (train_set["Partner"] == "No") & (train_set["Dependents"] == "No")
)

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])


# COMMAND ----------
# Call the endpoint with one sample record
def call_endpoint(record: dict) -> tuple[int, str]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
