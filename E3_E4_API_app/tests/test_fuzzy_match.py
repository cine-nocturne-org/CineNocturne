import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api_movie_v2 import app, USERNAME, PASSWORD
import sys
import os

sys.modules['mlflow'] = MagicMock()

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

# ------------------------------
# Test /fuzzy_match trouvé
# ------------------------------
# @patch("api_movie_v2.movie_index_df")
# @patch("api_movie_v2.movies_dict")
# def test_fuzzy_match_found(mock_movies_dict, mock_movie_index_df):
#     mock_movie_index_df.query.return_value = [{"title": "Zombieland"}]
#     mock_movies_dict.__getitem__.return_value = {"movie_id": 1}
#     response = client.get("/fuzzy_match/Zombieland", headers=get_auth_headers())
#     assert response.status_code == 200
#     data = response.json()
#     assert "movie_id" in data

# ------------------------------
# Test /fuzzy_match non trouvé
# ------------------------------
@patch("api_movie_v2.movie_index_df")
@patch("api_movie_v2.movies_dict", new_callable=lambda: {"Zombieland": {"movie_id": 1}})
def test_fuzzy_match_found(mock_movies_dict, mock_movie_index_df):
    # Simuler un DataFrame avec une colonne 'title'
    mock_movie_index_df.__getitem__.return_value.tolist.return_value = ["Zombieland"]

    response = client.get("/fuzzy_match/Zombieland", headers=get_auth_headers())
    assert response.status_code == 200
    data = response.json()
    assert "matches" in data
    assert data["matches"][0]["title"] == "Zombieland"
    assert data["matches"][0]["movie_id"] == 1






