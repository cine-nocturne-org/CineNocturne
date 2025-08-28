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

    # Créer l'expérience avec S3 comme storage des artefacts
    exp_name = "louve_movies_monitoring"
    artifact_location = f"s3://{MLFLOW_S3_BUCKET}@{MLFLOW_S3_ENDPOINT_URL}"

    try:
        exp_id = mlflow.create_experiment(exp_name, artifact_location=artifact_location)
    except mlflow.exceptions.MlflowException:
        # Si l'expérience existe déjà, on récupère son ID
        exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id

    mlflow.set_experiment(exp_name)
