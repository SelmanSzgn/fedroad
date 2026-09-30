import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from fedroad import api
from fedroad.model import Model


@pytest.fixture
def client(tmp_path, monkeypatch):
    pt = tmp_path / "model.pt"
    torch.save(Model().state_dict(), pt)
    monkeypatch.setenv("MODEL_PATH", str(pt))
    api.model.cache_clear()
    yield TestClient(api.app)
    api.model.cache_clear()


def png():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 48, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return b.getvalue()


def test_health_and_predict(client):
    assert client.get("/health").json() == {"status": "ok"}
    files = {"file": ("x.png", png(), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    j = r.json()
    assert j["label"] in api.CLASSES
    assert sum(j["probs"].values()) == pytest.approx(1.0, abs=1e-5)


def test_bad_file_is_rejected(client):
    files = {"file": ("x.png", b"nope", "image/png")}
    assert client.post("/predict", files=files).status_code == 400


def test_health_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    api.model.cache_clear()
    assert TestClient(api.app).get("/health").status_code == 503
    api.model.cache_clear()
