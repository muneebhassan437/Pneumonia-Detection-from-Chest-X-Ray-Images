import io
from functools import lru_cache

import torch
from fastapi import HTTPException
from PIL import Image
from torchvision import transforms

from app.config import (
    CLASS_NAMES,
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_PATH,
)
from app.model import pnuemoniaCNN

_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


@lru_cache(maxsize=1)
def _load_model() -> pnuemoniaCNN:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model weights not found at {MODEL_PATH}. "
                "Run the weight-export cell in CNN_Classifier.ipynb first."
            ),
        )
    model = pnuemoniaCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def predict(image_bytes: bytes) -> dict:
    model = _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        log_probs = model(tensor)
        probs = torch.exp(log_probs)
        confidence, pred_idx = torch.max(probs, dim=1)

    return {
        "label": CLASS_NAMES[pred_idx.item()],
        "confidence": round(confidence.item(), 4),
        "probabilities": {
            CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs.squeeze(0).tolist())
        },
    }
