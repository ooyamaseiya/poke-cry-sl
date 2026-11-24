import torch

# Audio Parameters (Section 5)
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
N_MELS = 64
F_MIN = 50.0
F_MAX = 8000.0

# Duration & Padding (Section 5)
DURATION_SEC = 2.5
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC) # 40,000 samples

# Model Input Shape
INPUT_SHAPE = (1, N_MELS, 251)  # [Channels, Freq, Time] - time is based on n_mels/hop_length approx? 
# Let's check: Librosa spectrogram size. 
# For a 2.5s audio at 16000 Hz = 40000 samples.
# Hop length 160 => 40000 / 160 + 1 = 251 frames. Correct.

# Hyperparameters (Section 7)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Paths (Section 3)
DATA_RAW_DIR = "data/raw"
DATA_NOISE_DIR = "data/noise"
DATA_PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# Augmentation Parameters (Section 6)
TIME_SHIFT_MAX_SEC = 0.2
GAIN_RANGE = (0.6, 1.2)
SNR_RANGE = (0, 20)  # for training noise mix
VAL_SNR_RANGE = (5, 20) # for val augmentation

# SpecAugment Parameters (Section 6)
TIME_MASK_MAX = 20
FREQ_MASK_MAX = 8
