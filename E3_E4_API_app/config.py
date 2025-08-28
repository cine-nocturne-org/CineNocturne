# config.py
import os
import mlflow

# ------------------------------
# PostgreSQL Render pour MLflow
# ------------------------------
DB_URL = "postgresql+psycopg2://mlflow_db_evzp_user:ZIG9DNZ2gWeqCOQ7SkjIM7cZI9tJcp53@dpg-d2o5rour433s73avu490-a.frankfurt-postgres.render.com/mlflow_db_evzp"

# Dossier local pour les artefacts (tu peux changer)
import pathlib
MLFLOW_ARTIFACT_LOCATION = str(pathlib.Path(__file__).parent / "mlflow_artifacts")

# Crée le dossier artefacts s'il n'existe pas
pathlib.Path(MLFLOW_ARTIFACT_LOCATION).mkdir(exist_ok=True)

# ------------------------------
# Configuration MLflow
# ------------------------------
mlflow.set_tracking_uri(DB_URL)

# Créer l'expérience si elle n'existe pas
EXPERIMENT_NAME = "louve_movies_monitoring"
try:
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION)
except mlflow.exceptions.MlflowException:
    # Si l'expérience existe déjà
    pass

mlflow.set_experiment(EXPERIMENT_NAME)
