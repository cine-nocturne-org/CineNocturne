import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api_movie_v2 import app, USERNAME, PASSWORD
import os

client = TestClient(app)

def get_auth_headers():
    # Basic Auth
    import base64
    credentials = f"{os.getenv('API_USERNAME')}:{os.getenv('API_PASSWORD')}"
    b64_credentials = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {b64_credentials}"}

@patch("api_movie_v2.engine")
def test_update_rating_valid(mock_engine):
    mock_conn = mock_engine.begin.return_value.__enter__.return_value
    mock_conn.execute.return_value.rowcount = 1
    payload = {"title": "Zombieland", "rating": 8.5}
    response = client.post("/update_rating", json=payload, headers=get_auth_headers())
    assert response.status_code == 200
    assert "message" in response.json()

@patch("api_movie_v2.engine")
def test_update_rating_not_found(mock_engine):
    # Simule rowcount = 0
    mock_conn = mock_engine.begin.return_value.__enter__.return_value
    mock_conn.execute.return_value.rowcount = 0

    payload = {"title": "TitreInexistant", "rating": 7}
    response = client.post("/update_rating", json=payload, headers=get_auth_headers())
    
    # Maintenant ton API doit renvoyer 404, sinon corriger l'API
    assert response.status_code == 404

def test_update_rating_invalid_rating():
    payload = {"title": "Zombieland", "rating": 15.0}
    response = client.post("/update_rating", json=payload, headers=get_auth_headers())
    assert response.status_code == 400


