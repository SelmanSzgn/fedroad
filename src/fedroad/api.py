import io
import os
from functools import lru_cache

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .model import Model

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

app = FastAPI(title="fedroad")


@lru_cache
def model():
    """Load the global model once."""
    path = os.environ.get("MODEL_PATH", "models/model.pt")
    m = Model()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def prep(img):
    """PIL image -> normalized tensor of shape (1, 3, 32, 32)."""
    img = img.convert("RGB").resize((32, 32))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


@app.get("/health")
def health():
    try:
        model()
    except Exception as e:
        raise HTTPException(503, f"model not loaded: {e}") from e
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except OSError as e:
        raise HTTPException(400, "not a valid image") from e
    with torch.no_grad():
        p = model()(prep(img))[0].softmax(dim=0)
    k = int(p.argmax())
    return {
        "label": CLASSES[k],
        "prob": float(p[k]),
        "probs": {c: float(v) for c, v in zip(CLASSES, p, strict=True)},
    }
