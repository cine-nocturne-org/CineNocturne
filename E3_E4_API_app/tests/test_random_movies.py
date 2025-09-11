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

import api_movie_v2
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
def test_get_random_movies_valid(mock_connect):
    # faux context manager: with engine.connect() as conn:
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn

    # la requête renvoie 1 ligne exploitable
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        ("Zombieland", "Synopsis", "url", "Action,Comédie", "netflix", 2009)
    ]
    mock_conn.execute.return_value = mock_result

    # neutraliser l'aléa de random.sample
    with patch("api_movie_v2.random.sample", lambda seq, k: seq[:k]):
        resp = client.get(
            "/random_movies/?genre=Action&platforms=netflix&limit=1",
            headers=get_auth_headers()
        )

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["title"] == "Zombieland"
    assert data[0]["releaseYear"] == 2009
    assert data[0]["platform"] == "netflix"
    assert data[0]["poster_url"] == "url"
    assert data[0]["synopsis"] == "Synopsis"

@patch("api_movie_v2.engine.connect")
def test_get_random_movies_no_result(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn

    # aucune ligne trouvée
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_conn.execute.return_value = mock_result

    resp = client.get(
        "/random_movies/?genre=Action&platforms=netflix&limit=1",
        headers=get_auth_headers()
    )
    assert resp.status_code == 404
