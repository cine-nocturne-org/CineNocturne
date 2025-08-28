import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

RUN_MLFLOW = os.getenv("RUN_MLFLOW", "1")

if RUN_MLFLOW == "1":
    # Backend pour stocker les runs localement (toujours)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    exp_name = "louve_movies_monitoring"

    # Si S3 configuré correctement, utiliser S3 pour les artefacts
    bucket = os.getenv("MLFLOW_S3_BUCKET")
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if bucket and endpoint:
        artifact_location = f"s3://{bucket}@{endpoint}"
        try:
            mlflow.create_experiment(exp_name, artifact_location=artifact_location)
        except mlflow.exceptions.MlflowException:
            pass  # existe déjà

    # Toujours définir l'expérience
    mlflow.set_experiment(exp_name)
