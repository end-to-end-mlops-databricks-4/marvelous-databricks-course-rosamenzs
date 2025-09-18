"""Script to preprocess data."""

import yaml
from databricks.connect import DatabricksSession
from loguru import logger

from churn import PROJECT_DIR
from churn.config import ProjectConfig
from churn.data_processor import \
    DataProcessor  # , generate_synthetic_data, generate_test_data

config = ProjectConfig.from_yaml(config_path=(PROJECT_DIR / "project_config.yml").resolve(), env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# spark = SparkSession.builder.local().getOrCreate()
spark = DatabricksSession.builder.remote(cluster_id="0911-113128-1v3eeo04").getOrCreate()

# Load the churn dataset
df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/data.csv", header=True, inferSchema=True
).toPandas()

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
