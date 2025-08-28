import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

RUN_MLFLOW = os.getenv("RUN_MLFLOW", "1")  # 1 = actif, 0 = désactivé

if RUN_MLFLOW == "1":
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    MLFLOW_S3_BUCKET = os.getenv("MLFLOW_S3_BUCKET")
    MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

    if MLFLOW_S3_BUCKET and MLFLOW_S3_ENDPOINT_URL:
        mlflow.set_tracking_uri(f"s3://{MLFLOW_S3_BUCKET}@{MLFLOW_S3_ENDPOINT_URL}")
        mlflow.set_experiment("louve_movies_monitoring")
