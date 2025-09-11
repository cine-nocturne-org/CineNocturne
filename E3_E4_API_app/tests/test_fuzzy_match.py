import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from dotenv import load_dotenv

load_dotenv()

# --- Mock mlflow avant d'importer l'API ---
sys.modules['mlflow'] = MagicMock()
tracking_mock = MagicMock()
tracking_mock.MlflowClient = MagicMock()
sys.modules['mlflow.tracking'] = tracking_mock

import api_movie_v2  # après les mocks
client = TestClient(api_movie_v2.app)

def get_auth_headers():
    import base64
    username = os.getenv("API_USERNAME")
    password = os.getenv("API_PASSWORD")
    if username and password:
        b64 = base64.b64encode(f"{username}:{password}".encode()).decode()
        return {"Authorization": f"Basic {b64}"}
    token = os.getenv("API_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    raise ValueError("Aucun identifiant ou token fourni dans les variables d'environnement")

# --- Fakes DB ---
class _FakeResult:
    def __init__(self, rows): self._rows = rows
    def fetchall(self): return self._rows
    # compat fetchone/scalar si nécessaire ailleurs
    def fetchone(self): return self._rows[0] if self._rows else None

class _FakeConn:
    def __init__(self, rows): self._rows = rows
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def execute(self, *a, **k): return _FakeResult(self._rows)

class _FakeEngine:
    def __init__(self, rows): self._rows = rows
    def connect(self): return _FakeConn(self._rows)
    def begin(self): return _FakeConn(self._rows)

# --- Tests ---
def test_fuzzy_match_found_db_only(monkeypatch):
    # tuples (movie_id, title) renvoyés par la "BDD"
    rows = [(1, "Zombieland"), (2, "Zombiever"), (3, "Inside Out")]
    # IMPORTANT : patcher l'engine utilisé par get_engine()
    monkeypatch.setitem(api_movie_v2.STATE, "engine", _FakeEngine(rows))

    r = client.get("/fuzzy_match/Zombieland", headers=get_auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert any(m["title"] == "Zombieland" and m["movie_id"] == 1 for m in data["matches"])

def test_fuzzy_match_not_found_db_only(monkeypatch):
    # aucune ligne -> endpoint doit renvoyer 404
    monkeypatch.setitem(api_movie_v2.STATE, "engine", _FakeEngine([]))

    r = client.get("/fuzzy_match/DoesNotExist", headers=get_auth_headers())
    assert r.status_code == 404
