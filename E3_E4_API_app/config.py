# config.py
import os
import pathlib
from sqlalchemy import create_engine, text

# ------------------------------
# MySQL BDD pour MLflow / monitoring
# ------------------------------
DB_URL = "mysql+pymysql://louve:%40Marley080922@mysql-louve.alwaysdata.net/louve_movies"

# Nom de la table pour les logs
MONITORING_TABLE = "monitoring"

# Dossier local pour les artefacts (tu peux changer)
MLFLOW_ARTIFACT_LOCATION = str(pathlib.Path(__file__).parent / "mlflow_artifacts")
pathlib.Path(MLFLOW_ARTIFACT_LOCATION).mkdir(exist_ok=True)

# ------------------------------
# Connexion BDD
# ------------------------------
engine = create_engine(DB_URL, pool_pre_ping=True)

# ------------------------------
# Création table monitoring si elle n'existe pas
# ------------------------------
with engine.begin() as conn:
    conn.execute(
        text(f"""
        CREATE TABLE IF NOT EXISTS {MONITORING_TABLE} (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            run_id VARCHAR(255),
            endpoint VARCHAR(255),
            input_title TEXT,
            params JSON,
            metrics JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
    )

# ------------------------------
# MLflow (local artefacts seulement)
# ------------------------------
import mlflow

mlflow.set_tracking_uri(f"file://{MLFLOW_ARTIFACT_LOCATION}")

EXPERIMENT_NAME = "louve_movies_monitoring"
try:
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION)
except mlflow.exceptions.MlflowException:
    pass

mlflow.set_experiment(EXPERIMENT_NAME)