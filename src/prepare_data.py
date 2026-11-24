import os, glob, random, json, torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SAMPLE_RATE, TIME_SHIFT_MAX_SEC, GAIN_RANGE, SNR_RANGE, VAL_SNR_RANGE

def apply_time_shift(wav, max_samples):
    w_len = len(wav)
    if w_len == 0: return wav
    shift = np.random.randint(-max_samples, max_samples + 1)
    new_wav = np.zeros_like(wav)
    if shift > 0 and shift < w_len:
        new_wav[shift:] = wav[:w_len-shift]
    elif shift < 0 and abs(shift) < w_len:
        new_wav[:w_len-abs(shift)] = wav[abs(shift):]
    else:
        return np.zeros_like(wav) if shift != 0 else wav[:]
    return new_wav

def apply_gain(wav, gain_range=(0.6, 1.2)):
    return wav * np.random.uniform(gain_range[0], gain_range[1])

def add_noise(wav, target_snr):
    p_signal = np.mean(wav**2)
    if p_signal < 1e-7: return wav
    p_noise_target = p_signal / (10**(target_snr / 10))
    noise = np.random.normal(0, np.sqrt(p_noise_target), len(wav))
    return wav + noise

def apply_spec_augment(spectrogram):
    f_max = spectrogram.shape[0] # 64
    t_max = spectrogram.shape[1] # 251
    
    # Frequency masking (from config)
    FREQ_MASK_MAX = 8 
    TIME_MASK_MAX = 20

    f = np.random.randint(0, FREQ_MASK_MAX + 1)
    f0 = np.random.randint(0, f_max - f) if f_max > f else 0
    spectrogram[f0:f0+f, :] = 0

    t = np.random.randint(0, TIME_MASK_MAX + 1)
    t0 = np.random.randint(0, t_max - t) if t_max > t else 0
    spectrogram[:, t0:t0+t] = 0

    return spectrogram

def prepare_data():
    # Need to load AudioProcessor which expects a config file path
    from src.audio_processor import AudioProcessor
    import json
    import os
    
    # Create a dummy config for AudioProcessor if needed, but it's better 
    # to make audio_processor use the actual module values.
    # Let's assume we write one temp file based on src/config
    from src import config as cfg
    temp_cfg = {
        "SAMPLE_RATE": cfg.SAMPLE_RATE,
        "N_FFT": cfg.N_FFT,
        "HOP_LENGTH": cfg.HOP_LENGTH,
        "N_MELS": cfg.N_MELS,
        "F_MIN": float(cfg.F_MIN),
        "F_MAX": float(cfg.F_MAX),
        "TRIM_TOP_DB": 30,
        "MARGIN_SEC": 0.15,
        "NUM_SAMPLES": cfg.NUM_SAMPLES
    }
    with open("/tmp/ap_config.json", "w") as f:
        json.dump(temp_cfg, f)

    ap = AudioProcessor("/tmp/ap_config.json")

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    raw_dirs = sorted([d for d in os.listdir(DATA_RAW_DIR) if not d.startswith('00')])
    label_map = {"other": 0}
    for i, name in enumerate(raw_dirs): label_map[name] = i + 1

    train_pool, val_pool = [], []

    for folder in raw_dirs:
        folder_path = os.path.join(DATA_RAW_DIR, folder)
        audio_files = sorted(glob.glob(os.path.join(folder_path, "*.ogg")) + glob.glob(os.path.join(folder_path, "*.wav")))
        if not audio_files: continue
        label = label_map[folder]

        class_samples = [] 
        # If <3 files -> augment to ~50 samples then split (Instruction 6)
        is_low_count = len(audio_files) < 3

        if is_low_count:
            print(f"  {folder}: Low count ({len(audio_files)}). Augmenting...")
            for af in audio_files:
                # Load original and preprocess once to get a baseline wav
                with open(af, 'rb') as f_in: y = ap.load_audio(f_in.read())
                wav_p = ap.preprocess_waveform(y)

                for _ in range(50): # Target ~50 samples per original file
                    # Data Augmentation (Instruction 6)
                    w = wav_p.copy()
                    w = apply_gain(w, GAIN_RANGE)
                    if random.random() < 0.3:
                        w = add_noise(w, np.random.uniform(*SNR_RANGE))
                    # Time shift with zero padding (Section 6)
                    if random.random() < 0.5:
                        w = apply_time_shift(w, int(TIME_SHIFT_MAX_SEC * SAMPLE_RATE))

                    spec = ap.get_spectrogram(w)
                    class_samples.append({'spec': torch.from_numpy(spec).unsqueeze(0), 'label': label})
            
            # Split the pool (80/20 as requested for low count)
            random.shuffle(class_samples)
            split = int(len(class_samples) * 0.8)
            for i in range(split): train_pool.append(class_samples[i])
            for i in range(split, len(class_samples)): val_pool.append(class_samples[i])

        else:
            # Standard split (Section 6)
            random.shuffle(audio_files)
            split = int(len(audio_files) * 0.8)
            train_afs, val_afs = audio_files[:split], audio_files[split:]

            for af in train_afs:
                with open(af, 'rb') as f_in: y = ap.load_audio(f_in.read())
                wav_p = ap.preprocess_waveform(y)
                spec = ap.get_spectrogram(wav_p)
                # SpecAugment for Train only (Section 6)
                if random.random() < 0.5:
                    spec = apply_spec_augment(spec)
                train_pool.append({'spec': torch.from_numpy(spec).unsqueeze(0), 'label': label})

            for af in val_afs:
                with open(af, 'rb') as f_in: y = ap.load_audio(f_in.read())
                wav_p = ap.preprocess_waveform(y)
                # Val gets diversity (light/medium noise + shift) (Section 6)
                w = add_noise(apply_time_shift(wav_p, int(TIME_SHIFT_MAX_SEC * SAMPLE_RATE)), np.random.uniform(*VAL_SNR_RANGE))
                spec = ap.get_spectrogram(w)
                val_pool.append({'spec': torch.from_numpy(spec).unsqueeze(0), 'label': label})

    # Add "other" class to both pools (Instruction 4: 00_other is fixed)
    bg_dir = os.path.join(DATA_RAW_DIR, "00_other")
    if os.path.exists(bg_dir):
        others = glob.glob(os.path.join(bg_dir, "*.wav")) + glob.glob(os.path.join(bg_dir, "*.ogg"))
        for of in others:
            with open(of, 'rb') as f_in: y = ap.load_audio(f_in.read())
            w_p = ap.preprocess_waveform(y)
            spec = ap.get_spectrogram(w_p)
            # Distribute "other" to both sets
            if random.random() < 0.5: train_pool.append({'spec': torch.from_numpy(spec).unsqueeze(0), 'label': 0})
            else: val_pool.append({'spec': torch.from_numpy(spec).unsqueeze(0), 'label': 0})

    # Save to .pt (Section 6)
    torch.save([i['spec'] for i in train_pool], os.path.join(DATA_PROCESSED_DIR, "train_specs.pt"))
    torch.save([i['label'] for i in train_pool], os.path.join(DATA_PROCESSED_DIR, "train_labels.pt"))
    torch.save([i['spec'] for i in val_pool], os.path.join(DATA_PROCESSED_DIR, "val_specs.pt"))
    torch.save([i['label'] for i in val_pool], os.path.join(DATA_PROCESSED_DIR, "val_labels.pt"))

    # Save labels mapping (Section 6)
    rev_map = {str(v): k for k, v in label_map.items()}
    rev_map["0"] = "other"
    with open(os.path.join("models", "labels.json"), 'w') as f:
        import json
        json.dump(rev_map, f)

    print(f"Finished! Train={len(train_pool)}, Val={len(val_pool)}")

if __name__ == "__main__": 
    prepare_data()