"""LR model with feature lookup implementation."""

from functools import reduce

import mlflow

# New packages for feature store
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient

# from databricks.connect import DatabricksSession
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn.config import ProjectConfig, Tags


class FeatureLookupLRModel:
    """A model class for churn prediction using LR with feature lookup."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        # def __init__(self, config: ProjectConfig, tags: Tags, spark: DatabricksSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters_LR
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.churn_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.get_churn_features"

        # Mlflow configuration
        self.experiment_name = self.config.experiment_name_LR_feature_lookup
        self.model_name = f"{self.catalog_name}.{self.schema_name}.churn_model"
        self.tags = tags.dict()

        logger.info("✅ Model initialized with configuration.")

    def create_feature_table(self) -> None:
        """Create or update the churn_features table and populate it with engineered features."""
        # Determine table structure
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name} (
            customerID STRING NOT NULL,
            num_services INT,
            is_vulnerable BOOLEAN
        );
        """)

        # Determine primary key
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT churn_pk PRIMARY KEY(customerID);")

        # Activate Delta Lake change data feed
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        # Load data from train and test sets, combine
        train_df = self.spark.read.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        test_df = self.spark.read.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        df = train_df.unionByName(test_df)

        # Feature engineering steps
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

        # Calculate number of services subscribed
        # df = df.withColumn("num_services", sum([F.when(F.col(c) == "Yes", 1).otherwise(0) for c in service_cols]))
        df = df.withColumn(
            "num_services",
            reduce(lambda a, b: a + b, [F.when(F.col(c) == "Yes", 1).otherwise(0) for c in service_cols]),
        )

        # Identify vulnerable customers (senior citizens without partner or dependents)
        df = df.withColumn(
            "is_vulnerable", (F.col("SeniorCitizen") == 1) & (F.col("Partner") == "No") & (F.col("Dependents") == "No")
        )

        # Write to feature table
        df.select("customerID", "num_services", "is_vulnerable").write.mode("overwrite").saveAsTable(
            self.feature_table_name
        )

        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a SQL feature function to detect excessive costs."""
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(MonthlyCharges DOUBLE, tenure INT, TotalCharges DOUBLE)
        RETURNS BOOLEAN
        LANGUAGE PYTHON
        AS $$
            return (MonthlyCharges * tenure) < TotalCharges
        $$;
        """)

        logger.info("✅ Feature function 'has_excessive_costs' defined in Unity Catalog.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")

        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = self.train_set.withColumn("tenure", self.train_set["tenure"].cast("int"))
        self.train_set = self.train_set.withColumn("MonthlyCharges", self.train_set["MonthlyCharges"].cast("double"))
        self.train_set = self.train_set.withColumn("TotalCharges", self.train_set["TotalCharges"].cast("double"))
        self.train_set = self.train_set.withColumn("customerID", self.train_set["customerID"].cast("string"))

        logger.info("✅ Data successfully loaded.")
        logger.info("Test")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables and applying feauture function."""
        logger.info("Starting feature engineering...")

        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["num_services", "is_vulnerable"],
                    lookup_key=["customerID"],
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="has_excessive_costs",
                    input_bindings={
                        "MonthlyCharges": "MonthlyCharges",
                        "tenure": "tenure",
                        "TotalCharges": "TotalCharges",
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        # logger.info(f"Training DF columns: {self.training_set.load_df().toPandas().columns.tolist()}")

        # Loads as Pandas DataFrame
        self.training_df = self.training_set.load_df().toPandas()

        # Add column to test set (not automatically applied like in training set)
        self.test_set["has_excessive_costs"] = (
            self.test_set["MonthlyCharges"] * self.test_set["tenure"]
        ) < self.test_set["TotalCharges"]

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
        self.test_set["num_services"] = self.test_set[service_cols].apply(lambda row: sum(row == "Yes"), axis=1)

        # Add column "is_vulnerable" to test set (not automatically applied like in training set)
        self.test_set["is_vulnerable"] = (
            (self.test_set["SeniorCitizen"] == 1)
            & (self.test_set["Partner"] == "No")
            & (self.test_set["Dependents"] == "No")
        )

        # Determine which features are used for model training
        self.X_train = self.training_df[
            self.num_features + self.cat_features + ["num_services", "is_vulnerable", "has_excessive_costs"]
        ]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[
            self.num_features + self.cat_features + ["num_services", "is_vulnerable", "has_excessive_costs"]
        ]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classification_model", LogisticRegression(**self.parameters))]
        )

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            logger.info(f"📊 Precision: {precision}")
            logger.info(f"📊 Recall: {recall}")
            logger.info(f"📊 F1 Score: {f1}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LR with lookup")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)

            # self.fe.log_model(
            #    model=pipeline,
            #    flavor=mlflow.sklearn,
            #    artifact_path="lookup-lr-pipeline-model",
            #    training_set=self.training_set,
            #    signature=signature,
            # )
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="lookup-lr-pipeline-model",
                signature=signature,
                input_example=None,  # or provide an input example if available
            )

    def register_model(self) -> str:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lookup-lr-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )

        return latest_version
