# E3_E4_API_app/tests/test_fuzzy_match.py
import os
import sys
import pytest
from unittest.mock import MagicMock

# 1) Env AVANT l'import de l'API
os.environ.setdefault("MYSQL_URL", "sqlite+pysqlite:///:memory:")  # engine OK en tests
os.environ.setdefault("API_TOKEN", "test-token")                   # passe l'auth Bearer

# 2) Mock mlflow AVANT l'import de l'API
sys.modules['mlflow'] = MagicMock()
tracking_mock = MagicMock()
tracking_mock.MlflowClient = MagicMock()
sys.modules['mlflow.tracking'] = tracking_mock

# 3) Import de l'API après les env + mocks
from fastapi.testclient import TestClient
import api_movie_v2
client = TestClient(api_movie_v2.app)

def get_auth_headers():
    return {"Authorization": "Bearer test-token"}  # simple et robuste

# --- fakes DB ---
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

def test_fuzzy_match_found_db_only(monkeypatch):
    rows = [(1, "Zombieland"), (2, "Zombiever"), (3, "Inside Out")]
    monkeypatch.setattr(api_movie_v2, "engine", _FakeEngine(rows))
    r = client.get("/fuzzy_match/Zombieland", headers=get_auth_headers())
    assert r.status_code == 200
    data = r.json()
    assert any(m["title"] == "Zombieland" and m["movie_id"] == 1 for m in data["matches"])

def test_fuzzy_match_not_found_db_only(monkeypatch):
    monkeypatch.setattr(api_movie_v2, "engine", _FakeEngine([]))
    r = client.get("/fuzzy_match/DoesNotExist", headers=get_auth_headers())
    assert r.status_code == 404
