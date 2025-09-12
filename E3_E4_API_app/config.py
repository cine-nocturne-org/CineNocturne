# E3_E4_API_app/config.py
import os
import pathlib

# URI DB pour MLflow (viens de Render ENV idéalement)
DB_URL = os.getenv(
    "MLFLOW_TRACKING_URI",
    "postgresql+psycopg2://<user>:<pass>@<host>/<db>"
)

MLFLOW_TRACKING_URI = DB_URL
MLFLOW_ARTIFACT_LOCATION = str(pathlib.Path(__file__).parent / "mlflow_artifacts")
pathlib.Path(MLFLOW_ARTIFACT_LOCATION).mkdir(exist_ok=True)

EXPERIMENT_NAME = "louve_movies_monitoring"

def setup_mlflow(experiment: str = EXPERIMENT_NAME):
    """À appeler uniquement dans les scripts qui en ont besoin (train, etc.)."""
    import mlflow  # import local
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Créer l’expériment si besoin (utile côté TRAIN, pas nécessaire côté API)
    try:
        mlflow.create_experiment(experiment, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    except Exception:
        pass
    mlflow.set_experiment(experiment)
    return mlflow
