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
    for d in [DATA_RAW_DIR, DATA_NOISE_DIR, OTHER_CLASS_DIR]:
        os.makedirs(d, exist_ok=True)

def clean_dummy_data():
    print("Cleaning dummy data...")
    raw_dir = Path(DATA_RAW_DIR)
    for f in raw_dir.glob("*.wav"):
        if f.stem.isdigit():
            try: f.unlink()
            except: pass

def download_pokemon_cries(species_list):
    for entry in species_list:
        try:
            dex_id, name = entry.split(':')
            name = name.lower().strip()
            target_dir = os.path.join(DATA_RAW_DIR, f"{int(dex_id)}_{name}")
            os.makedirs(target_dir, exist_ok=True)

            for variant in ["latest", "legacy"]:
                url = f"https://raw.githubusercontent.com/PokeAPI/cries/main/cries/pokemon/{variant}/{int(dex_id)}.ogg"
                filename = os.path.join(target_dir, f"{name}_{variant}.ogg")
                print(f"Downloading {url} -> {filename}")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        f.write(response.content)
        except Exception as e:
            print(f"Error processing entry '{entry}': {e}")

def generate_background_noise():
    print("Generating background noise...")
    patterns = ["white", "pink", "hum60", "beep4k", "rumble", "silence"]
    for i, name in enumerate(patterns):
        file_path = os.path.join(OTHER_CLASS_DIR, f"{i:02d}_{name}.wav")
        duration = 5.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        if name == "white": audio = np.random.uniform(-1, 1, len(t))
        elif name == "pink":
            audio = np.cumsum(np.random.normal(0, 1, len(t)))
            audio -= audio.mean()
            audio /= (np.max(np.abs(audio)) + 1e-8) * 0.5
        elif name == "hum60": audio = 0.3 * np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.normal(0, 0.05, len(t))
        elif name == "beep4k": audio = 0.2 * np.sin(2 * np.pi * 4000 * t) + 0.1 * np.random.normal(0, 0.05, len(t))
        elif name == "rumble": audio = 0.3 * np.sin(2 * np.pi * 80 * t + 2*np.pi*4*t) + 0.1 * np.random.normal(0, 0.05, len(t))
        else: audio = np.zeros_like(t)
        sf.write(file_path, audio, SAMPLE_RATE)

def generate_augmentation_noise():
    print("Generating augmentation noise...")
    for i in range(5):
        file_path = os.path.join(DATA_NOISE_DIR, f"env_{i:02d}.wav")
        t = np.linspace(0, 5, int(SAMPLE_RATE * 5), endpoint=False)
        audio = (np.random.normal(0, 0.1, len(t)) + 0.3*np.sin(2*np.pi*50*t) + 0.2*np.sin(2*np.pi*150*t))
        sf.write(file_path, audio, SAMPLE_RATE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('species', nargs='*', help="dexID:name")
    args = parser.parse_args()
    if not args.species: args.species = ["384:rayquaza", "25:pikachu", "6:charizard", "150:mewtwo"]
    setup_directories(); clean_dummy_data(); download_pokemon_cries(args.species); generate_background_noise(); generate_augmentation_noise()

if __name__ == "__main__": main()
