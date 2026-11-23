import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import onnxruntime as ort
import numpy as np

from src.audio_processor import AudioProcessor

class ModelWrapper:
    def __init__(self):
        self.session = None
        self.labels = {}
        self.config = {}
        self.audio_processor = None

    def load(self, model_path, labels_path, config_path):
        self.session = ort.InferenceSession(model_path)
        with open(labels_path, 'r') as f:
            # Assuming format is {"0": "name", ...} or similar
            raw_labels = json.load(f)
            # Convert keys to int if they are strings of numbers
            self.labels = {int(k): v for k, v in raw_labels.items()}
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.audio_processor = AudioProcessor(config_path)

    def predict(self, audio_bytes):
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        input_tensor = self.audio_processor.process(audio_bytes)
        # Get input name for ONNX model
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: input_tensor})[0]
        
        # Logits is likely [1, num_classes] or similar. 
        # If it's [1, C], we flatten to [C] for softmax.
        logits = np.squeeze(logits)
        
        # Softmax
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        probabilities = softmax(logits)
        
        # Get Top-3
        top_indices = probabilities.argsort()[::-1][:3]
        
        predictions = []
        for rank, idx in enumerate(top_indices, 1):
            name = self.labels.get(idx, f"Unknown ({idx})")
            prob = float(probabilities[idx]) * 100
            predictions.append({
                "rank": rank,
                "name": name,
                "probability": round(prob, 2)
            })
        return predictions

model_wrapper = ModelWrapper()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        model_wrapper.load(
            model_path="models/pokemon_model.onnx",
            labels_path="models/labels.json",
            config_path="models/config.json"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    # Clean up if necessary

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["audio/wav", "audio/ogg", "video/ogg"]:
        # Some browsers might report ogg as video/ogg or audio/ogg
        if not (file.filename.endswith(".wav") or file.filename.endswith(".ogg")):
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV or OGG.")

    try:
        content = await file.read()
        predictions = model_wrapper.predict(content)
        return {
            "success": True,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
