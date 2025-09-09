# E3_E4_API_app/tests/test_fuzzy_match.py
import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from dotenv import load_dotenv

load_dotenv()
# Mock mlflow pour éviter tout import/side effect
sys.modules['mlflow'] = MagicMock()

import api_movie_v2  # après le mock mlflow
client = TestClient(api_movie_v2.app)

def get_auth_headers():
    import base64
    username = os.getenv("API_USERNAME")
    password = os.getenv("API_PASSWORD")
    if username and password:
        credentials = f"{username}:{password}"
        b64 = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {b64}"}
    token = os.getenv("API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    raise ValueError("Aucun identifiant ou token fourni dans les variables d'environnement")


# --- Petits fakes pour simuler la BDD ---
class _FakeResult:
    def __init__(self, rows):
        self._rows = rows
    def fetchall(self):
        return self._rows

class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def execute(self, *a, **k):
        # On renvoie toujours les mêmes "résultats SQL"
        return _FakeResult(self._rows)

class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows
    def connect(self):
        return _FakeConn(self._rows)


# ------------------------------
# /fuzzy_match : trouvé
# ------------------------------
def test_fuzzy_match_found_db_only(monkeypatch):
    # Simule la BDD qui renvoie quelques titres (comme un FULLTEXT/LIKE)
    rows = [
        (1, "Zombieland"),
        (2, "Zombiever"),
        (3, "Inside Out"),
    ]
    monkeypatch.setattr(api_movie_v2, "engine", _FakeEngine(rows))

    r = client.get("/fuzzy_match/Zombieland", headers=get_auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert "matches" in data and isinstance(data["matches"], list)
    titles = {m["title"] for m in data["matches"]}
    assert "Zombieland" in titles
    assert any(m["title"] == "Zombieland" and m["movie_id"] == 1 for m in data["matches"])


# ------------------------------
# /fuzzy_match : non trouvé
# ------------------------------
def test_fuzzy_match_not_found_db_only(monkeypatch):
    # Simule la BDD vide -> l'endpoint doit renvoyer 404 "Aucune correspondance trouvée."
    monkeypatch.setattr(api_movie_v2, "engine", _FakeEngine([]))

    r = client.get("/fuzzy_match/QuelqueChoseQuiNExistePas", headers=get_auth_headers())
    assert r.status_code == 404
    assert "Aucune correspondance" in r.json().get("detail", "")
