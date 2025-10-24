from __future__ import annotations

import os

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from core.index import get_index, _INDEX_CACHE
from server import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.json() == {'status': 'ok'}


def test_ask_returns_answer_with_index(tmp_path) -> None:
    os.environ['SUPER_INDEX_DIR'] = str(tmp_path / 'indexes')
    _INDEX_CACHE.clear()
    index = get_index()
    index.add_document('Super AI coordinates modular behaviors for summaries.', {'title': 'Demo'}, tags=['demo'])

    client = TestClient(app)
    resp = client.post('/ask', json={'question': 'Summarize the demo behaviors', 'tags': 'demo'})
    assert resp.status_code == 200
    data = resp.json()
    assert data['answer']
    assert data['plan']


def test_plan_no_steps_returns_404(tmp_path) -> None:
    os.environ['SUPER_INDEX_DIR'] = str(tmp_path / 'indexes')
    _INDEX_CACHE.clear()
    client = TestClient(app)
    resp = client.post('/plan', json={'goal': 'summary'})
    assert resp.status_code == 404
