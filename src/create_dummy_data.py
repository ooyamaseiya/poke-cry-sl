import os
import numpy as np
import soundfile as sf
import librosa
from src import config

def generate_sine_wave(freq, duration, sr=config.SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)

def create_dummy_data():
    print("Generating dummy data...")
    os_setup = [config.DATA_RAW_DIR, config.DATA_NOISE_DIR]
    for d in os_setup:
        if not os.path.exists(d):
            os.makedirs(d)

    # 1. Create Pokémon class folders and dummy files
    pokemon_classes = ["01_rayquaza", "02_pikachu"]
    frequencies = [440, 880]  # Different frequencies for different classes

    for cls_name, freq in zip(pokemon_classes, frequencies):
        cls_path = os.path.join(config.DATA_RAW_DIR, cls_name)
        os.makedirs(cls_path, exist_ok=True)
        for i in range(5):  # Generate 5 files per class for testing (small count for quick run)
            audio = generate_sine_wave(freq, config.DURATION + 1.0) # Add extra length to test trimming/cropping
            # Add some random noise to make it less "pure" sine wave
            audio += np.random.normal(0, 0.05, len(audio)).astype(np.float32)
            filename = os.path.join(cls_path, f"{i:03d}.wav")
            sf.write(filename, audio, config.SAMPLE_RATE)

    # 2. Create "other" class (random noise/silence)
    other_path = os.path.join(config.DATA_RAW_DIR, "00_other")
    os.makedirs(other_path, exist_ok=True)
    for i in range(5):
        # Pure white noise for 'other' class
        audio = np.random.normal(0, 0.3, int(config.SAMPLE_RATE * config.DURATION)).astype(np.float32)
        filename = os.path.join(other_path, f"{i:03d}.wav")
        sf.write(filename, audio, config.SAMPLE_RATE)

    # 3. Create noise files for data augmentation
    for i in range(5):
        audio = np.random.normal(0, 0.2, int(config.SAMPLE_RATE * 5)).astype(np.float32) # 5s white noise
        filename = os.path.join(config.DATA_NOISE_DIR, f"noise_{i:03d}.wav")
        sf.write(filename, audio, config.SAMPLE_RATE)

    print("Dummy data generation completed.")
    print(f"Check {config.DATA_RAW_DIR} for results.")

if __name__ == "__main__":
    create_dummy_data()
