import os
import sys
import json
import numpy as np
import librosa
import onnxruntime as ort
from src import config

def predict_wav(wav_path):
    # 1. Load Config and Models
    models_dir = "models"
    config_path = os.path.join(models_dir, "config.json")
    model_path = os.path.join(models_dir, "pokemon_model.onnx")
    labels_path = os.path.join(models_dir, "labels.json")

    if not (os.path.exists(config_path) and os.path.exists(model_path) and os.path.exists(labels_path)):
        print("Error: Model or config files missing in 'models/' directory.")
        return

    with open(config_path, 'r') as f:
        cfg = json.load(f)
    with open(labels_path, 'r') as f:
        # labels are strings of digits like "0": "classname"
        labels = {int(k): v for k, v in json.load(f).items()}

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # 2. Preprocess Audio (Match Training Pipeline)
    sr = cfg["SAMPLE_RATE"]
    num_samples = cfg["NUM_SAMPLES"]
    n_fft = cfg["N_FFT"]
    hop_length = cfg["HOP_LENGTH"]
    n_mels = cfg["N_MELS"]

    # Load and trim (using librosa as in preparation)
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=cfg.get("TRIM_TOP_DB", 30))

    if len(yt) >= num_samples:
        energies = np.abs(yt)
        peak_idx = np.argmax(energies)
        start_idx = peak_idx - num_samples // 2
        if start_idx < 0:
            start_idx = 0
        elif start_idx + num_samples > len(yt):
            start_idx = len(yt) - num_samples
        y_final = yt[start_idx : start_idx + num_samples]
    else:
        y_final = np.zeros(num_samples, dtype=np.float32)
        actual_len = min(len(yt), num_samples)
        y_final[:actual_len] = yt[:actual_len]

    # Normalize
    max_val = np.max(np.abs(y_final))
    if max_val > 1e-8:
        y_final /= max_val

    # Mel Spectrogram (Librosa) - Must be exactly same as training!
    S = librosa.feature.melspectrogram(y=y_final, sr=sr, n_fft=n_fft, hop_length=hop_length, 
                                      n_mels=n_mels, fmin=cfg["F_MIN"], fmax=cfg["F_MAX"])
    S_db = librosa.power_to_db(S, ref=np.max)

    # Standardization (Mean/Std per sample as in preparation script)
    mean = np.mean(S_db)
    std = np.std(S_db)
    if std > 1e-8:
        S_db = (S_db - mean) / std
    else:
        S_db = S_db - mean

    # Shape to (Batch, Channel, Freq, Time) -> (1, 1, 64, 251)
    input_tensor = S_db[np.newaxis, np.newaxis, ...]
    if input_tensor.shape[3] < 251: # Safety check for time dimension mismatch if sr/hop change
        # (In our case it should be exactly 251)
        pass

    # 3. Inference
    logits = session.run(None, {input_name: input_tensor})[0]
    probs = np.exp(logits) / np.sum(np.exp(logits)) # Softmax for probabilities (actually logits might need softmax if model doesn't have it, 
                                                  # but typically ONNX outputs raw scores/logits or after softmax depends on implementation.
                                                  # Our CNN has Linear as last layer without Softmax in PyTorch, so we do manual Logits -> Probs)

    # If the model output was already probabilities (it's not), this would be different. 
    # But my train.py uses CrossEntropyLoss which expects logits. Correct.

    # Get Top-3
    top_k_indices = np.argsort(probs[0])[-3:][::-1] # Last three, reversed
    
    print("\n--- Prediction Results ---")
    for idx in top_k_indices:
        label = labels.get(int(idx), f"Unknown({idx})")
        prob_pct = probs[0][idx] * 100
        print(f"{label}: {prob_pct:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict_onnx.py path/to/audio.wav")
    else:
        predict_wav(sys.argv[1])
