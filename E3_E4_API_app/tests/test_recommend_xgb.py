import os
import sys
import base64
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# ---------- Mocks AVANT d'importer l'API ----------
# joblib.load → évite le chargement de vrais artefacts
joblib_mock = MagicMock()
joblib_mock.load = MagicMock(return_value=MagicMock())
sys.modules['joblib'] = joblib_mock

# mlflow → fournit un start_run utilisable comme context manager
mlflow_mock = MagicMock()
run_cm = MagicMock()
# objet retourné par __enter__ avec un .info.run_id exploitable
run_cm.__enter__.return_value = MagicMock(info=MagicMock(run_id="test_run_id"))
run_cm.__exit__.return_value = False
mlflow_mock.start_run.return_value = run_cm
mlflow_mock.active_run.return_value = None
# stubs pour les fonctions utilisées
mlflow_mock.set_experiment = MagicMock()
mlflow_mock.set_tags = MagicMock()
mlflow_mock.log_param = MagicMock()
mlflow_mock.log_metric = MagicMock()
mlflow_mock.log_table = MagicMock()
mlflow_mock.log_artifact = MagicMock()
mlflow_mock.log_dict = MagicMock()
sys.modules['mlflow'] = mlflow_mock

# client MLflow (pas utilisé ici, mais on sécurise)
mlflow_tracking_mock = MagicMock()
mlflow_tracking_mock.MlflowClient = MagicMock()
sys.modules['mlflow.tracking'] = mlflow_tracking_mock
# ---------- Fin des mocks pré-import ----------

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
    # L'endpoint /recommend_xgb_personalized n'est pas protégé, donc pas bloquant
    return {}

# Classe factice pour svd_full avec .transform(...)
class _FakeSVD:
    def transform(self, X):
        # renvoie un vecteur 2D constant, quelle que soit l'entrée (shape: (1, 2))
        X = np.asarray(X)
        n = X.shape[0] if X.ndim == 2 else 1
        return np.tile(np.array([[0.5, 0.5]]), (n, 1))

@patch("api_movie_v2.titles", new=["Zombieland", "AutreFilm"])
@patch(
    "api_movie_v2.movies_dict",
    new={
        "Zombieland": {"title": "Zombieland", "genres": "Action|Comédie", "poster_url": "zomb.jpg", "rating": 7.0},
        "AutreFilm": {"title": "AutreFilm",  "genres": "Action",         "poster_url": "autre.jpg", "rating": 6.0},
    }
)
@patch("api_movie_v2.genres_list_all", new=[["Action", "Comédie"], ["Action"]])
@patch("api_movie_v2.tfidf_matrix", new=np.array([[0.1]*10, [0.2]*10], dtype=float))  # 2 films x 10 feats
@patch("api_movie_v2.genres_encoded_matrix", new=np.array([[1, 0], [0, 1]], dtype=float))
@patch("api_movie_v2.years_scaled", new=np.array([[0.0], [1.0]], dtype=float))
@patch("api_movie_v2.svd_full", new=_FakeSVD())
@patch("api_movie_v2.nn_full")
@patch("api_movie_v2.xgb_model")
def test_recommend_xgb_valid(mock_xgb_model, mock_nn_full):
    # xgb renvoie proba positive élevée pour 1 candidat
    mock_xgb_model.predict_proba.return_value = np.array([[0.1, 0.9]])

    # kneighbors pour le candidat (on enlèvera la première distance self=0.0)
    # shape attendue: (1, n_neighbors). On met 2 voisins -> distances [0.0, 0.1]
    mock_nn_full.kneighbors.return_value = (np.array([[0.0, 0.1]]), None)

    response = client.get("/recommend_xgb_personalized/Zombieland", headers=get_auth_headers())
    assert response.status_code == 200
    payload = response.json()

    # structure
    assert isinstance(payload, dict)
    assert "run_id" in payload and isinstance(payload["run_id"], str)
    assert "recommendations" in payload and isinstance(payload["recommendations"], list)
    recos = payload["recommendations"]
    assert len(recos) > 0

    # "AutreFilm" doit être recommandé
    assert any(r["title"] == "AutreFilm" for r in recos)

@patch("api_movie_v2.titles", new=["Zombieland"])  # liste contrôlée sans "FilmInexistant"
def test_recommend_xgb_invalid():
    response = client.get("/recommend_xgb_personalized/FilmInexistant", headers=get_auth_headers())
    assert response.status_code == 404
