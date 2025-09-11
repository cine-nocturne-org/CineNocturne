import os
import base64
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

load_dotenv()

import api_movie_v2  # après load_dotenv
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

@patch("api_movie_v2.get_engine")
def test_get_unique_genres(mock_get_engine):
    # -- fabrique un faux engine/connexion/context manager
    mock_engine = MagicMock()
    mock_conn_cm = mock_engine.connect.return_value
    mock_conn = mock_conn_cm.__enter__.return_value

    # la requête SELECT genres FROM movies renverra ces lignes
    # (on couvre la séparation par virgule et par pipe)
    mock_rows = [("Action,Comédie",), ("Horreur|Thriller",), ("  Drame  ",)]
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = mock_rows
    mock_conn.execute.return_value = mock_cursor

    mock_get_engine.return_value = mock_engine

    # --- appel endpoint
    resp = client.get("/genres/", headers=get_auth_headers())
    assert resp.status_code == 200
    data = resp.json()

    # --- vérifications : les genres doivent être séparés/nettoyés
    assert "Action" in data
    assert "Comédie" in data   # accent conservé car l'API ne normalise pas ici
    assert "Horreur" in data
    assert "Thriller" in data
    assert "Drame" in data
