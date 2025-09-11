import os
import sys
import base64
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# ---------- mocks AVANT d'importer l'API ----------
# Évite les loads d'artefacts au module import
joblib_mock = MagicMock()
joblib_mock.load = MagicMock(return_value=MagicMock())
sys.modules['joblib'] = joblib_mock

# Évite tout accès MLflow réel
sys.modules['mlflow'] = MagicMock()
mlflow_tracking_mock = MagicMock()
mlflow_tracking_mock.MlflowClient = MagicMock()
sys.modules['mlflow.tracking'] = mlflow_tracking_mock
# ---------- fin des mocks pré-import ----------

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
def test_fuzzy_match_found_db_only(mock_connect):
    # Mock du context manager: with engine.connect() as conn:
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn

    # Première requête FULLTEXT renvoie des lignes (movie_id, title)
    rows = [(1, "Zombieland"), (2, "Zombiever"), (3, "Inside Out")]
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows
    mock_conn.execute.return_value = mock_result

    r = client.get("/fuzzy_match/Zombieland", headers=get_auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert any(m["title"] == "Zombieland" and m["movie_id"] == 1 for m in data["matches"])

@patch("api_movie_v2.engine.connect")
def test_fuzzy_match_not_found_db_only(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn

    # Aucune des requêtes ne trouve de lignes => fetchall() vide
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_conn.execute.return_value = mock_result

    r = client.get("/fuzzy_match/DoesNotExist", headers=get_auth_headers())
    assert r.status_code == 404
