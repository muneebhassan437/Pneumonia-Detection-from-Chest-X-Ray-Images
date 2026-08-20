import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import MODEL_PATH
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model weights not exported yet")
def test_predict_returns_label():
    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post("/predict", files={"file": ("test.jpg", buffer, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("Normal", "Pneumonia")
    assert 0.0 <= body["confidence"] <= 1.0
