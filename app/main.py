from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference import predict
from app.schemas import PredictionResponse

app = FastAPI(title="Pneumonia CNN Classifier")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG image")

    image_bytes = await file.read()
    return predict(image_bytes)
