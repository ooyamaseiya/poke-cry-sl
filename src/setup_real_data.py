import os
import requests
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path

SAMPLE_RATE = 16000
DATA_RAW_DIR = "data/raw"
DATA_NOISE_DIR = "data/noise"
OTHER_CLASS_DIR = os.path.join(DATA_RAW_DIR, "00_other")

def setup_directories():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_NOISE_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    # Ensure other class dir is there too as it's inside DATA_RAW_DIR
    os.makedirs(OTHER_CLASS_DIR, exist_ok=True)

def clean_existing_data():
    print("Cleaning existing raw data...")
    raw_dir = Path(DATA_RAW_DIR)
    if raw_dir.exists():
        import shutil
        # We want to keep the folder structure but clear it? 
        # Or just delete everything in DATA_RAW_DIR for a clean start.
        for item in raw_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"Could not delete {item}: {e}")

def download_pokemon_cries(targets_csv):
    print(f"Downloading from {targets_csv}...")
    if not os.path.exists(targets_csv):
        print(f"Error: {targets_csv} not found.")
        return

    with open(targets_csv, 'r', encoding='utf-8-sig') as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(lines):
        try:
            parts = line.split(',')
            if len(parts) < 2: continue
            dex_id_str = parts[0].strip()
            name = parts[1].strip().lower()
            dex_id = int(dex_id_str)
            
            folder_num = idx + 1
            target_dir = os.path.join(DATA_RAW_DIR, f"{folder_num:02d}_{name}")
            os.makedirs(target_dir, exist_ok=True)

            for variant in ["latest", "legacy"]:
                url = f"https://raw.githubusercontent.com/PokeAPI/cries/main/cries/pokemon/{variant}/{dex_id}.ogg"
                filename = os.path.join(target_dir, f"{name}_{variant}.ogg")
                print(f"Downloading {url} -> {filename}")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f_out:
                        f_out.write(response.content)
                else:
                    print(f"  Failed to download {dex_id} ({variant}): Status {response.status_code}")

        except Exception as e:
            print(f"Error processing line '{line}': {e}")

def generate_background_noise():
    print("Generating background noise in 00_other...")
    os.makedirs(OTHER_CLASS_DIR, exist_ok=True)
    patterns = [("white", "white"), ("hum60", "hum")]
    for i, (name, p_type) in enumerate(patterns):
        file_path = os.path.join(OTHER_CLASS_DIR, f"noise_{i:02d}.wav")
        duration = 5.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        if p_type == "white":
            audio = np.random.uniform(-1, 1, len(t))
        elif p_type == "hum":
            audio = 0.3 * np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.normal(0, 0.05, len(t))
        else:
            audio = np.zeros_like(t)
        sf.write(file_path, audio, SAMPLE_RATE)

def generate_augmentation_noise():
    print("Generating environment noise for augmentation...")
    os.makedirs(DATA_NOISE_DIR, exist_ok=True)
    for i in range(5):
        file_path = os.path.join(DATA_NOISE_DIR, f"env_{i:02d}.wav")
        t = np.linspace(0, 3, int(SAMPLE_RATE * 3), endpoint=False)
        audio = (np.random.normal(0, 0.1, len(t)) + 0.5 * np.sin(2 * np.pi * 100 * t))
        sf.write(file_path, audio, SAMPLE_RATE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=str, default='targets.csv')
    args = parser.parse_args()
    # Order matters: clean first then setup dirs to ensure they exist when we write files
    clean_existing_data(); setup_directories(); download_pokemon_cries(args.targets); generate_background_noise(); generate_augmentation_noise()

if __name__ == "__main__": 
    main()