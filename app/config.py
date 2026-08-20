from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "pneumonia_cnn.pt"

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["Normal", "Pneumonia"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
