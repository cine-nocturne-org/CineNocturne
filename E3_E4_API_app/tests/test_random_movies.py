import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api_movie_v2 import app, USERNAME, PASSWORD
import os


client = TestClient(app)

def get_auth_headers():
    # Basic Auth
    import base64
    credentials = f"{os.getenv('API_USERNAME')}:{os.getenv('API_PASSWORD')}"
    b64_credentials = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {b64_credentials}"}

@patch("api_movie_v2.engine.connect")
def test_get_random_movies_valid(mock_connect):
    # mock du context manager 'with engine.connect() as conn'
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn

    # mock de execute() → renvoie un objet qui a fetchall()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        ("Zombieland", "Synopsis", "url", "Action,Comédie", "netflix")
    ]
    mock_conn.execute.return_value = mock_result

    response = client.get("/random_movies/?genre=Action&platforms=netflix&limit=1",
                          headers=get_auth_headers())
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data[0]["title"] == "Zombieland"


@patch("api_movie_v2.engine.connect")
def test_get_random_movies_no_result(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn

    # execute() → objet résultat dont fetchall() renvoie une liste vide
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_conn.execute.return_value = mock_result

    response = client.get(
        "/random_movies/?genre=Action&platforms=netflix&limit=1",
        headers=get_auth_headers()
    )
    assert response.status_code == 404



