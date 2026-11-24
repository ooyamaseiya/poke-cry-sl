import os
import json
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.audio_processor import AudioProcessor

# Global variables for model and processor to reuse them across requests
model_session = None
audio_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    global model_session, audio_processor

    # Load config-based AudioProcessor
    config_path = os.path.join("models", "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config file not found at {config_path}")
    audio_processor = AudioProcessor(config_path)

    # Load ONNX model session
    model_path = os.path.join("models", "pokemon_model.onnx")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}")

    model_session = onnxruntime.InferenceSession(model_path)

    yield
    # Cleanup if necessary (not strictly needed for ONNX runtime in this case)


app = FastAPI(lifespan=lifespan, title="Pokemon Sound Classifier")


@app.get("/")
async def read_index():
    """Serves the main UI."""
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint to receive audio file and return top-3 predictions."""
    if not file.filename.lower().endswith(('.wav', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload WAV or OGG.")

    try:
        # Read the uploaded file content into memory
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        # 1. Preprocessing using AudioProcessor
        input_tensor = audio_processor.full_pipeline(content)

        # 2. Inference with ONNX Runtime
        inputs = {model_session.get_inputs()[0].name: input_tensor}
        logits = model_session.run(None, inputs)[0][0]  # Extract from the list of outputs and remove batch dim if needed?

        # Note: Depending on how ONNX was exported, logits might be [1, C] or [C]. 
        # The above logic assumes it returns a single output array.

        # 3. Softmax to get probabilities (manually implemented using numpy)
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        probabilities = softmax(logits)

        # 4. Get Top-3 results using labels.json
        labels_path = os.path.join("models", "labels.json")
        with open(labels_path, 'r') as f:
            labels = json.load(f)

        predictions = []
        # Sort indices by probability descending
        top_indices = probabilities.argsort()[::-1][:3]

        for rank, idx in enumerate(top_indices, 1):
            idx_str = str(int(idx))
            name = labels.get(idx_str, f"Unknown ({idx})") if isinstance(labels, dict) else (labels[int(idx)] if isinstance(labels, list) else f"Index {int(idx)}")
            prob = float(probabilities[idx]) * 100

            predictions.append({
                "rank": rank,
                "name": name,
                "probability": round(prob, 2)
            })

        return {
            "success": True,
            "predictions": predictions
        }

    except Exception as e:
        import traceback
        traceback.print_exc()  # Log for server-side debugging
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# Mount static files to serve the frontend components (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
