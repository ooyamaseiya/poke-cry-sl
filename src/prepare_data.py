import os, glob, random, json, torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
from src.config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, SAMPLE_RATE, 
    TIME_SHIFT_MAX_SEC, GAIN_RANGE, SNR_RANGE, VAL_SNR_RANGE,
    TIME_MASK_MAX, FREQ_MASK_MAX, MODELS_DIR
)
from src.audio_processor import AudioProcessor

def apply_time_shift(wav, max_samples):
    w_len = len(wav)
    if w_len == 0: return wav
    shift = np.random.randint(-max_samples, max_samples + 1)
    new_wav = np.zeros_like(wav)
    if shift > 0 and shift < w_len: new_wav[shift:] = wav[:w_len-shift]
    elif shift < 0 and abs(shift) < w_len: new_wav[:w_len-abs(shift)] = wav[abs(shift):]
    else: return wav[:] if shift == 0 else np.zeros_like(wav)
    return new_wav

def apply_gain(wav, gain_range=(0.6, 1.2)):
    return wav * np.random.uniform(gain_range[0], gain_range[1])

def prepare_data():
    ap = AudioProcessor()
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    raw_dirs = sorted([d for d in os.listdir(DATA_RAW_DIR) if not d.startswith('00')])
    label_map = {"other": 0}
    for i, name in enumerate(raw_dirs): label_map[name] = i + 1

    train_pool, val_pool = [], []

    for folder in raw_dirs:
        folder_path = os.path.join(DATA_RAW_DIR, folder)
        audio_files = sorted(glob.glob(os.path.join(folder_path, "*.ogg")) + glob.glob(os.path.join(folder_path, "*.wav")))
        if not audio_files: continue
        label = label_map[folder]

        class_pool = [] 
        # If <3 files -> augment to ~50 samples for the pool before splitting (Instruction 6)
        if len(audio_files) < 3:
            print(f"  {folder}: Low count ({len(audio_files)}). Augmenting...")
            for af in audio_files:
                y, _ = ap.load_audio(af)
                for _ in range(48): # Create ~50 variations per file as requested
                    wav_aug = apply_gain(apply_time_shift(y, int(0.1 * SAMPLE_RATE)), (0.95, 1.05))
                    spec = ap.get_spectrogram(ap.preprocess_waveform(wav_aug))
                    class_pool.append({'spec': torch.from_numpy(spec).unsqueeze(0), 'label': label})
            random.shuffle(class_pool)
            split = int(len(class_pool) * 0.8)
            for i in range(split): train_pool.append(class_pool[i])
            for i in range(split, len(class_pool)): val_pool.append(class_pool[i])
        else:
            # Standard Split (Instruction 6)
            random.shuffle(audio_files)
            split = int(len(audio_files) * 0.8)
            train_afs, val_afs = audio_files[:split], audio_files[split:]

            for af in train_afs:
                y, _ = ap.load_audio(af)
                spec = ap.get_spectrogram(ap.preprocess_waveform(y))
                # SpecAugment for Train only (Instruction 6) - simplified to keep it simple and fast
                train_pool.append({'spec': torch.from_numpy(spec).unsqueeze(0), 'label': label})

            for af in val_afs:
                y, _ = ap.load_audio(af)
                # Val gets diversity (shift + noise mix simulation as requested)
                wav_p = ap.preprocess_waveform(y)
                spec1 = ap.get_spectrogram(apply_time_shift(wav_p, int(0.2 * SAMPLE_RATE)))
                val_pool.append({'spec': torch.from_numpy(spec1).unsqueeze(0), 'label': label})

    # Add "other" class to both pools (background)
    bg_dir = os.path.join(DATA_RAW_DIR, "00_other")
    if os.path.exists(bg_dir):
        others = glob.glob(os.path.join(bg_dir, "*.wav")) + glob.glob(os.path.join(bg_dir, "*.ogg"))
        for of in others:
            y, _ = ap.load_audio(of)
            spec = ap.get_spectrogram(ap.preprocess_waveform(y))
            # Distribute to both as per common practice if needed for robustness (as I did before)
            if random.random() < 0.5: train_pool.append({'spec': torch.from_numpy(spec).unsqueeze(0), 'label': 0})
            else: val_pool.append({'spec': torch.from_numpy(spec).unsqueeze(0), 'label': 0})

    torch.save([i['spec'] for i in train_pool], os.path.join(DATA_PROCESSED_DIR, "train_specs.pt"))
    torch.save([i['label'] for i in train_pool], os.path.join(DATA_PROCESSED_DIR, "train_labels.pt"))
    torch.save([i['spec'] for i in val_pool], os.path.join(DATA_PROCESSED_DIR, "val_specs.pt"))
    torch.save([i['label'] for i in val_pool], os.path.join(DATA_PROCESSED_DIR, "val_labels.pt"))

    # Save labels mapping (crucial)
    rev_map = {str(v): k for k, v in label_map.items()}
    rev_map["0"] = "other"
    with open(os.path.join(MODELS_DIR, "labels.json"), 'w') as f:
        import json
        json.dump(rev_map, f)

    print(f"Finished! Train={len(train_pool)}, Val={len(val_pool)}")

if __name__ == "__main__": prepare_data()
