import os
import sys
import base64
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

load_dotenv()

# ---- mocks AVANT d'importer l'API (évite joblib.load & appels MLflow) ----
joblib_mock = MagicMock()
joblib_mock.load = MagicMock(return_value=MagicMock())
sys.modules['joblib'] = joblib_mock

sys.modules['mlflow'] = MagicMock()
mlflow_tracking_mock = MagicMock()
mlflow_tracking_mock.MlflowClient = MagicMock()
sys.modules['mlflow.tracking'] = mlflow_tracking_mock
# --------------------------------------------------------------------------

import api_movie_v2  # après les mocks
client = TestClient(api_movie_v2.app)

def get_auth_headers():
    username = os.getenv("API_USERNAME")
    password = os.getenv("API_PASSWORD")
    if username and password:
        b64 = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {b64}"}
    token = os.getenv("API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    raise ValueError("Aucun identifiant ou token fourni dans les variables d'environnement")

@patch("api_movie_v2.engine.connect")
def test_get_unique_genres(mock_connect):
    # -- faux context manager: with engine.connect() as conn:
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn

    # la requête SELECT genres FROM movies renverra ces lignes
    # (on couvre la séparation par virgule et par pipe)
    mock_rows = [("Action,Comédie",), ("Horreur|Thriller",), ("  Drame  ",)]
    mock_result = MagicMock()
    mock_result.fe_
