# test_recommend_xgb.py
import pytest
import numpy as np
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
import prometheus_client
prometheus_client.REGISTRY = CollectorRegistry()
from httpx import AsyncClient

# ----- Import de l'app FastAPI -----
from api_movie_v2 import app

# ----- Mock des variables globales de l'endpoint -----
titles = ["Zombiland", "M3gan", "Kpop Demon Hunter"]
genres_list_all = [["Comédie", "Horreur"], ["Horreur", "Science-Fiction"], ["Action", "Musical"]]
tfidf_matrix = np.eye(3)  # matrice identité pour simplifier
svd_full = type("svd", (), {"transform": lambda self, x: np.ones((1, 2))})()
nn_full = type("nn", (), {"kneighbors": lambda self, x: (np.zeros((1, 3)), None)})()
genres_encoded_matrix = np.ones((3, 1))
years_scaled = np.ones(3)
movies_dict = {
    "Zombiland": {"title": "Zombiland", "genres": ["Comédie", "Horreur"], "synopsis": "Zombie apocalypse drôle", "user_rating": 8.5, "rating": 7.5, "poster_url": "url_zombiland"},
    "M3gan": {"title": "M3gan", "genres": ["Horreur", "Science-Fiction"], "synopsis": "Une poupée qui devient vivante", "user_rating": 7.0, "rating": 6.5, "poster_url": "url_m3gan"},
    "Kpop Demon Hunter": {"title": "Kpop Demon Hunter", "genres": ["Action", "Musical"], "synopsis": "Idols qui combattent des démons", "user_rating": 9.0, "rating": 8.0, "poster_url": "url_kpop"}
}

class MockXGB:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]] * len(X))  # score constant pour simplifier

xgb_model = MockXGB()

# ----- Création du client FastAPI -----
client = TestClient(app)

# ----- Test simple pour Zombiland -----
@pytest.mark.asyncio
async def test_recommend_zombiland():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/recommend_xgb_personalized/Zombiland?top_k=2")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

# ----- Test pour un film inexistant -----
def test_recommend_not_found():
    response = client.get("/recommend_xgb_personalized/Film Inconnu")
    assert response.status_code == 404
    assert response.json() == {"detail": "Film non trouvé"}



