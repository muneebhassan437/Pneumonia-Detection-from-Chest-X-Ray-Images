# Pneumonia Detection from Chest X-Ray Images

A deep learning pipeline that classifies chest X-ray images as **Normal** or **Pneumonia**, built with PyTorch and served as a **FastAPI** inference API.

> **Disclaimer:** This project is for educational and research purposes only. It is **not** a certified medical diagnostic tool and must not be used as a substitute for professional medical advice or diagnosis.

## Overview

The project has two parts:

1. **Modeling** (`Pneumonia_detection.ipynb`, `CNN_Classifier.ipynb`) — data preprocessing, a set of ML baselines, and a custom CNN trained from scratch to classify chest X-rays.
2. **Serving** (`app/`) — a FastAPI service that loads the trained CNN and exposes it over a REST API for image classification.

## Dataset

This project uses the **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** dataset from Kaggle (5,856 labeled images, `NORMAL` / `PNEUMONIA`).

The dataset is not included in this repository (too large for git). To reproduce the notebooks:

1. Download the dataset from the Kaggle link above.
2. Extract it so the project root contains `train/`, `val/`, and `test/` folders, each with `NORMAL/` and `PNEUMONIA/` subfolders (this is the layout the notebooks expect — see `datasets.ImageFolder` calls in `Pneumonia_detection.ipynb`).

The notebooks merge the original train/val/test folders and re-split them 60/20/20, since the original Kaggle validation split is too small to be useful on its own.

## Modeling

### `Pneumonia_detection.ipynb`

- Preprocessing, data augmentation, and the shared 60/20/20 train/val/test split (seeded for reproducibility).
- Extracts features with a pretrained ResNet-18 and benchmarks classical baselines on top of them: a majority-class baseline, Logistic Regression, an SVM, and a small feedforward NN.

### `CNN_Classifier.ipynb`

- Defines and trains a small CNN from scratch (`pnuemoniaCNN`, three conv layers + three fully-connected layers, log-softmax output) on 224×224 ImageNet-normalized images.
- Reuses the exact same imports, seed, and train/val/test split as `Pneumonia_detection.ipynb` so results are directly comparable.
- Evaluates the trained model (classification report, confusion matrix, misclassified-sample inspection).
- **Exports the trained weights to `artifacts/pneumonia_cnn.pt`**, which the FastAPI service loads at request time.

### Results

| Model | Validation Accuracy | Test Accuracy | F1 (Test) |
|---|---|---|---|
| Majority Baseline | 0.28 | 0.27 | 0.00 |
| Logistic Regression (ResNet-18 features) | 0.96 | 0.96 | 0.97 |
| SVM, RBF kernel (ResNet-18 features) | 0.96 | 0.96 | 0.97 |
| Feedforward NN (ResNet-18 features) | 0.86 | 0.88 | 0.92 |
| **CNN (from scratch)** | **0.88** | **0.88** | **0.91** |

The classical baselines built on frozen pretrained ResNet-18 features outperform the small from-scratch CNN — expected, given the CNN has far less capacity and no pretraining, and is trained on a relatively small dataset.

## API

The `app/` package serves the trained CNN behind a small FastAPI service.

```
app/
├── main.py         # FastAPI app: GET /health, POST /predict
├── inference.py     # loads weights, preprocesses images, runs predictions
├── model.py          # pnuemoniaCNN architecture (imported by the notebook too, so training and serving never drift apart)
├── config.py           # paths, device, image size, class names
└── schemas.py            # response schema
```

### Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Getting model weights

The API loads weights from `artifacts/pneumonia_cnn.pt`, which is not committed to the repo. Generate it by running `CNN_Classifier.ipynb` top to bottom (Kernel → Restart & Run All) — the last training cell exports the weights automatically.

### Running the API

```bash
uvicorn app.main:app --reload
```

Interactive docs are then available at `http://127.0.0.1:8000/docs`.

**Endpoints:**

- `GET /health` → `{"status": "ok"}`
- `POST /predict` — multipart file upload (`file`, JPEG or PNG), returns:

  ```json
  {
    "label": "Pneumonia",
    "confidence": 0.9792,
    "probabilities": { "Normal": 0.0208, "Pneumonia": 0.9792 }
  }
  ```

  Example:

  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -F "file=@sample_xray.jpeg;type=image/jpeg"
  ```

  If `artifacts/pneumonia_cnn.pt` hasn't been generated yet, this endpoint returns `503` with a message pointing you to the export cell in `CNN_Classifier.ipynb`.

### Tests

```bash
pytest tests/
```

### Docker

```bash
docker build -t pneumonia-cnn .
docker run -p 8000:8000 pneumonia-cnn
```

## Project Structure

```
.
├── CNN_Classifier.ipynb       # CNN training, evaluation, weight export
├── Pneumonia_detection.ipynb  # preprocessing + classical baselines
├── app/                       # FastAPI service
├── artifacts/                 # trained weights (generated, gitignored)
├── tests/                     # pytest suite for the API
├── requirements.txt
├── Dockerfile
└── train/ val/ test/ data/    # dataset (gitignored, see Dataset section)
```
