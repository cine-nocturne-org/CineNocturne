import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api_movie_v2 import app, USERNAME, PASSWORD
import os


client = TestClient(app)

def get_auth_headers():
    import base64
    # Essayer Basic Auth
    username = os.getenv("API_USERNAME")
    password = os.getenv("API_PASSWORD")
    if username and password:
        credentials = f"{username}:{password}"
        b64_credentials = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {b64_credentials}"}
    
    # Sinon, essayer Bearer token
    token = os.getenv("API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    
    raise ValueError("Aucun identifiant ou token fourni dans les variables d'environnement")

@patch("api_movie_v2.engine")
def test_get_unique_genres(mock_engine):
    # Mock de la connexion et du fetch
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [("Action,Comédie",)]
    mock_conn.execute.return_value = mock_cursor

    response = client.get("/genres/", headers=get_auth_headers())
    assert response.status_code == 200
    data = response.json()
    # On vérifie que la réponse contient bien les genres séparés
    assert "Action" in data
    assert "Comédie" in data




