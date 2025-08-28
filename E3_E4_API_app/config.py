import os
from dotenv import load_dotenv
import mlflow

# Charge les variables depuis .env
load_dotenv()  # par défaut, cherche un fichier .env à la racine

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MLFLOW_S3_BUCKET = os.getenv("MLFLOW_S3_BUCKET")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

# Configure MLflow pour utiliser S3
mlflow.set_tracking_uri(f"s3://{MLFLOW_S3_BUCKET}@{MLFLOW_S3_ENDPOINT_URL}")
mlflow.set_experiment("louve_movies_monitoring")
