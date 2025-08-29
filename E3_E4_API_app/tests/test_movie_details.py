import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api_movie_v2 import app, USERNAME, PASSWORD
import os
from dotenv import load_dotenv
load_dotenv()

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
def test_get_movie_details_found(mock_engine):
    mock_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        (1, "Zombieland", "Action,Comédie", 2009, 7.5, "Synopsis", "url", "Netflix")
    ]
    mock_conn.execute.return_value = mock_cursor

    response = client.get("/movie-details/Zombieland", headers=get_auth_headers())
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Zombieland"
    assert isinstance(data["platforms"], list)

@patch("api_movie_v2.engine.connect")
def test_get_movie_details_not_found(mock_connect):
    # Création du mock du context manager
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn

    # Mock de l'objet résultat avec fetchall() qui renvoie une liste vide
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []  # aucun résultat
    mock_conn.execute.return_value = mock_result

    response = client.get("/movie-details/TitreInexistant123", headers=get_auth_headers())
    assert response.status_code == 404










