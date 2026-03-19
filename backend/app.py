import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

from model import load_model
from utils import preprocess_image

app = FastAPI(title="Edge AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ERROR = None
MODEL = None

try:
    MODEL = load_model()
except Exception as exc:  # pragma: no cover
    MODEL_ERROR = str(exc)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Model not available: {MODEL_ERROR}")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    tensor = preprocess_image(image)

    with torch.no_grad():
        output = MODEL(tensor)
        probs = torch.softmax(output, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return {
        "prediction": "Mixed" if pred == 0 else "Not Mixed",
        "confidence": confidence,
    }
