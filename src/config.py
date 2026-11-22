import os

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_NOISE_DIR = os.path.join(BASE_DIR, "data", "noise")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Audio configuration
SAMPLE_RATE = 16000  # Hz
DURATION = 2.5       # Seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION) # 40000 samples

# Librosa / Spectrogram parameters
N_FFT = 1024
HOP_LENGTH = 160      # ~10ms @ 16kHz
N_MELS = 64
F_MIN = 50.0
F_MAX = 8000.0

# Audio processing settings
TRIM_TOP_DB = 30
MARGIN_SEC = 0.15  # seconds margin after trimming

# Data Augmentation parameters (Training only)
SNR_RANGE = (0, 20)            # dB range for noise mixing
TIME_SHIFT_MAX = 0.2           # Seconds
GAIN_RANGE = (0.6, 1.2)        # Multiplier
SPEC_AUG_TIME_MASK_MAX = 20    # Frames
SPEC_AUG_FREQ_MASK_MAX = 8     # Mels

# Training settings
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

def get_config():
    """Returns a dictionary representation of the configuration for saving to JSON."""
    return {
        "SAMPLE_RATE": SAMPLE_RATE,
        "DURATION": DURATION,
        "NUM_SAMPLES": NUM_SAMPLES,
        "N_FFT": N_FFT,
        "HOP_LENGTH": HOP_LENGTH,
        "N_MELS": N_MELS,
        "F_MIN": F_MIN,
        "F_MAX": F_MAX,
        "TRIM_TOP_DB": TRIM_TOP_DB,
        "MARGIN_SEC": MARGIN_SEC,
    }
