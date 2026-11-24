import onnxruntime as ort
import numpy as np
import torch
from src.audio_processor import AudioProcessor
import json
import os

def predict(model_path, audio_path, labels_json):
    # 1. Load Model and Labels
    session = ort.InferenceSession(model_path)
    with open(labels_json, 'r') as f:
        rev_labels = json.load(f)

    ap = AudioProcessor()
    
    # 2. Preprocess audio to match model input shape [1, 1, 64, 251]
    y, _ = ap.load_audio(audio_path)
    spec = ap.preprocess_waveform(y) # Shape (64, 251) after trimming/padding? No, get_spectrogram does it better.
    # Let's use the full pipeline which returns [64, 251] correctly based on instruction.
    spec = ap.get_spectrogram(ap.preprocess_waveform(y)) # Actually preprocess_waveform handles length/padding? Yes.
    
    # Input shape: (Batch=1, Channel=1, Freq=64, Time=251)
    input_tensor = spec[np.newaxis, np.newaxis, :, :] 

    # 3. Inference
    inputs = {session.get_inputs()[0].name: input_tensor}
    logits = session.run(None, inputs)[0] # Shape [1, num_classes]

    # Apply softmax for probabilities
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    probs = softmax(logits[0])
    
    # 4. Top-3 Results
    top_indices = probs.argsort()[-3:][::-1]
    results = []
    for idx in top_indices:
        label_id = str(idx) # labels.json keys are strings (from json)
        name = rev_labels.get(label_id, f"Unknown({idx})")
        prob = probs[idx] * 100
        results.append((name, prob))

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict_onnx /path/to/model.onnx /path/to/audio.ogg")
        sys.exit(1)

    MODEL_PATH = sys.argv[1]
    AUDIO_FILE = sys.argv[2]
    LABELS_JSON = "models/labels.json"

    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found.")
        sys.exit(1)

    try:
        predictions = predict(MODEL_PATH, AUDIO_FILE, LABELS_JSON)
        print("\nTop-3 Predictions:")
        for i, (name, prob) in enumerate(predictions):
            print(f"{i+1}. {name}: {prob:.2f}%")
    except Exception as e:
        print(f"Error during prediction: {e}")

