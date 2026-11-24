import numpy as np
import librosa
import soundfile as sf
import json
import io

class AudioProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.sample_rate = self.config['SAMPLE_RATE']
        self.num_samples = int(self.config['NUM_SAMPLES'])
        self.n_fft = self.config['N_FFT']
        self.hop_length = self.config['HOP_LENGTH']
        self.n_mels = self.config['N_MELS']
        self.f_min = float(self.config['F_MIN'])
        self.f_max = float(self.config['F_MAX'])
        self.trim_top_db = self.config.get('TRIM_TOP_DB', 30)
        self.margin_sec = self.config.get('MARGIN_SEC', 0.15)

    def load_audio(self, source: bytes):
        """Loads audio from bytes into a numpy array."""
        with io.BytesIO(source) as audio_file:
            # librosa.load を使用して、指定したサンプリングレートで読み込み、
            # 自動的にモノラル化およびリサンプリングを行う。
            y, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        return y.astype(np.float32)

    def preprocess_waveform(self, waveform):
        """Trims, pads/crops, and normalizes the audio waveform."""
        # 1. Trim + Margin
        try:
            # librosa.effects.trim returns trimmed signal and indices
            y_trimmed, index = librosa.effects.trim(waveform, top_db=self.trim_top_db)
            start_idx, end_idx = int(index[0]), int(index[1])
        except Exception:
             # Fallback if trimming fails or has unexpected behavior
             y_trimmed = waveform
             start_idx, end_idx = 0, len(waveform)

        margin_samples = int(self.margin_sec * self.sample_rate)
        start = max(0, start_idx - margin_samples)
        end = min(len(waveform), end_idx + margin_samples)
        y_expanded = waveform[start:end]

        # 2. Fixed length (target NUM_SAMPLES). Peak-centered crop or pad.
        if len(y_expanded) < self.num_samples:
            pad_len = self.num_samples - len(y_expanded)
            y_fixed = np.pad(y_expanded, (0, pad_len), mode='constant')
        else:
            # Peak-centered crop if possible to keep the "sound" in the middle
            peak_idx = np.argmax(np.abs(y_expanded))
            half_target = self.num_samples // 2
            start_crop = peak_idx - half_target
            
            if start_crop < 0:
                start_crop = 0
                end_crop = self.num_samples
            elif start_crop + self.num_samples > len(y_expanded):
                end_crop = len(y_expanded)
                start_crop = max(0, end_crop - self.num_samples)
            else:
                end_crop = start_crop + self.num_samples
            
            y_fixed = y_expanded[start_crop:end_crop]

        # Ensure exact size (handles edge cases where slicing might leave 1 sample off due to rounding)
        if len(y_fixed) > self.num_samples:
             y_fixed = y_fixed[:self.num_samples]
        elif len(y_fixed) < self.num_samples:
             y_fixed = np.pad(y_fixed, (0, self.num_samples - len(y_fixed)), mode='constant')

        # 3. Peak Normalization
        max_val = np.max(np.abs(y_fixed))
        if max_val > 1e-7:
            y_fixed = y_fixed / max_val

        return y_fixed.astype(np.float32)

    def get_spectrogram(self, waveform):
        """Converts waveform to log-mel spectrogram."""
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=waveform, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels, 
            fmin=self.f_min, 
            fmax=self.f_max
        )
        # Convert to log scale (decibels)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Standardize the spectrogram values: (x - mean) / std
        mean = np.mean(S_db)
        std = np.std(S_db)
        spec = ((S_db - mean) / (std + 1e-8)).astype(np.float32)
        return spec

    def full_pipeline(self, audio_bytes: bytes):
        """Full processing pipeline from raw bytes to model input tensor."""
        y = self.load_audio(audio_bytes)
        wav_p = self.preprocess_waveform(y)
        spec = self.get_spectrogram(wav_p)
        
        # Model expects shape [1, 1, n_mels, time_frames] -> here spec is (n_mels, frames)
        # Based on instructions: [1, 1, 64, 251]
        return np.expand_dims(np.expand_dims(spec, axis=0), axis=0)

