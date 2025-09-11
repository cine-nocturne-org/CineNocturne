import os
import sys
import base64
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# ---------- mocks AVANT d'importer l'API ----------
# Évite les loads d'artefacts au module import
joblib_mock = MagicMock()
joblib_mock.load = MagicMock(return_value=MagicMock())
sys.modules['joblib'] = joblib_mock

# Évite tout accès MLflow réel
sys.modules['mlflow'] = MagicMock()
mlflow_tracking_mock = MagicMock()
mlflow_tracking_mock.MlflowClient = MagicMock()
sys.modules['mlflow.tracking'] = mlflow_tracking_mock
# ---------- fin des mocks pré-import ----------

import api_movie_v2  # après les mocks
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

# --- fakes DB minimalistes ---
class _FakeResult:
    def __init__(self, rows): self._rows = rows
    def fetchall(self): return self._rows

class _FakeConn:
    def __init__(self, rows): self._rows = rows
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def execute(self, *a, **k): return _FakeResult(self._rows)

class _FakeEngine:
    def __init__(self, rows): self._rows = rows
    def connect(self): return _FakeConn(self._rows)
    def begin(self): return _FakeConn(self._rows)

# --- tests ---
def test_fuzzy_match_found_db_only(monkeypatch):
    rows = [(1, "Zombieland"), (2, "Zombiever"), (3, "Inside Out")]
    # Patch direct du moteur utilisé par l'API
    monkeypatch.setitem(api_movie_v2.STATE, "engine", _FakeEngine(rows))

    r = client.get("/fuzzy_match/Zombieland", headers=get_auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert any(m["title"] == "Zombieland" and m["movie_id"] == 1 for m in data["matches"])

def test_fuzzy_match_not_found_db_only(monkeypatch):
    monkeypatch.setitem(api_movie_v2.STATE, "engine", _FakeEngine([]))
    r = client.get("/fuzzy_match/DoesNotExist", headers=get_auth_headers())
    assert r.status_code == 404
