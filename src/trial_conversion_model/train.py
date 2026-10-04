import json
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.xgboost
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import boto3

from trial_conversion_model.data import load_processed
from trial_conversion_model.features import TARGET

MODEL_DIR = Path("models")
TEST_SIZE = 0.25
RANDOM_STATE = 42
EXPERIMENT_NAME = "trial-conversion"

# Define params for training and MLflow tracking
PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "auc",
}

def train(model_dir: Path = MODEL_DIR) -> dict:
    """Train the trial conversion model from the processed training table."""
    load_dotenv()
    mlflow.set_experiment(EXPERIMENT_NAME)

    table = load_processed()
    X = table.drop(columns=[TARGET])
    y = table[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    with mlflow.start_run():
        # mlflow.xgboost.autolog()

        # Log model parameters to MLflow
        mlflow.log_params(PARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_features", X.shape[1])

        # Log the data to MLflow. 
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_dataset = mlflow.data.from_pandas(train_data, name="train_data", targets=TARGET)
        test_dataset = mlflow.data.from_pandas(test_data, name="test_data", targets=TARGET)
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(test_dataset, context="testing")

        model = XGBClassifier(**PARAMS)
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test_auc", auc)
        mlflow.xgboost.log_model(model, name="model")

    model_dir.mkdir(exist_ok=True)
    model.save_model(model_dir / "model.json")

    # Publish the deployable model to S3
    bucket_name = "trial-conversion-artifacts-adel"
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        f"{model_dir}/model.json",
        bucket_name,
        "models/model.json",
    )

    metrics = {
        "test_auc": round(float(auc), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": list(X.columns),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
