import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

RUN_MLFLOW = os.getenv("RUN_MLFLOW", "1")

if RUN_MLFLOW == "1":
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    MLFLOW_S3_BUCKET = os.getenv("MLFLOW_S3_BUCKET")
    MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

    # Backend pour stocker les runs localement
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Créer l'expérience et définir S3 comme stockage des artefacts
    mlflow.set_experiment(
        "louve_movies_monitoring",
        artifact_location=f"s3://{MLFLOW_S3_BUCKET}@{MLFLOW_S3_ENDPOINT_URL}"
    )
