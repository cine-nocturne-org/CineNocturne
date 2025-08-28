import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from httpx import AsyncClient
from api_movie_v2 import app  # chemin vers ton app FastAPI

# Mock des prédictions (2 candidats → 2 proba)
mock_xgb_pred = np.array([[0.1, 0.9]])
mock_svd_matrix = np.array([[0.5, 0.5]])

# Utilitaire pour simuler un retour async
async def async_return(val):
    return val

@pytest.mark.asyncio
@patch("api_movie_v2.nn_full", new_callable=MagicMock)
@patch("api_movie_v2.svd_full", new_callable=MagicMock)
@patch("api_movie_v2.genres_encoded_matrix", new=np.array([[1,0],[0,1]]))
@patch("api_movie_v2.years_scaled", new=np.array([[2020],[2021]]))
@patch("api_movie_v2.tfidf_matrix", new=np.array([[0.1]*10, [0.2]*10]))  # 2 films, 10 features
@patch("api_movie_v2.xgb_model", new_callable=MagicMock)
@patch("api_movie_v2.titles", new=["Zombieland", "AutreFilm"])
@patch(
    "api_movie_v2.movies_dict",
    new={
        "Zombieland": {"title": "Zombieland", "genres": ["Action", "Comédie"], "poster_url": "zomb.jpg"},
        "AutreFilm": {"title": "AutreFilm", "genres": ["Action"], "poster_url": "autre.jpg"},
    }
)
@patch("api_movie_v2.genres_list_all", new=[["Action", "Comédie"], ["Action"]])
async def test_recommend_xgb_valid(mock_xgb, mock_svd, mock_nn):
    # Mock des méthodes awaitables si nécessaire
    mock_xgb.predict_proba.return_value = mock_xgb_pred
    mock_svd.transform.side_effect = lambda x: mock_svd_matrix
    mock_nn.kneighbors.side_effect = lambda x: (np.array([[0.0, 0.1]]), None)

    # On utilise AsyncClient avec ASGI
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/recommend_xgb_personalized/Zombieland")

    assert response.status_code == 200
    data = response.json()

    # Vérifie que c'est bien une liste non vide
    assert isinstance(data, list)
    assert len(data) > 0

    # Vérifie que "AutreFilm" est bien recommandé
    assert any(rec["title"] == "AutreFilm" for rec in data)


@pytest.mark.asyncio
async def test_recommend_xgb_invalid():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/recommend_xgb_personalized/FilmInexistant")
    assert response.status_code == 404
