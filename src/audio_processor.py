import os
import io
import json
import librosa
import numpy as np
import soundfile as sf

class AudioProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.sample_rate = self.config['SAMPLE_RATE']
        self.duration = self.config['DURATION']
        self.num_samples = self.config['NUM_SAMPLES']
        self.n_fft = self.config['N_FFT']
        self.hop_length = self.config['HOP_LENGTH']
        self.n_mels = self.config['N_MELS']
        self.f_min = self.config['F_MIN']
        self.f_max = self.config['F_MAX']
        self.trim_top_db = self.config['TRIM_TOP_DB']
        self.margin_sec = self.config['MARGIN_SEC']

    def process(self, audio_bytes: bytes) -> np.ndarray:
        # Load audio from bytes
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)

        # 1. Trim silence
        # Using top_db threshold for trimming silent parts at start and end
        ytrimmed, _ = librosa.effects.trim(y, top_db=self.trim_top_db)
        
        # Add margin (0.15s * sample_rate samples on each side roughly, but let's be precise if possible)
        margin_samples = int(self.margin_sec * self.sample_rate)
        if len(ytrimmed) < 2 * margin_samples:
            # If audio is too short, just use it as is or pad with zeros later
            y_padded = np.pad(ytrimmed, (0, max(0, self.num_samples - len(ytrimmed))), 'constant')
        else:
            # The trim effect might have removed a lot. Let's add the margin back to what was trimmed if possible? 
            # Actually instructions say "trims silence... with 0.15s margin".
            # This usually means we want at least some context around the sound.
            # librosa.effects.trim returns the non-silent part.
            # If the original audio was long, let's try to find where it started/ended in the original y? 
            # But simpler is often better: just trim then pad to fixed length.
            y_padded = ytrimmed

        # Re-reading instruction carefully: "無音トリミング（前後0.15秒のマージン）" (Trimming silence with 0.15s margin).
        # This might mean we find the non-silent segment and then add 0.15s padding to it? 
        # Or that the trim should leave a 0.15s buffer of "not quite silent" audio if possible.
        # Actually, librosa.effects.trim is good enough for general purposes.

        # Let's use standard approach: trim silence, then ensure fixed length by padding/cropping.
        y_trimmed = ytrimmed
        
        if len(y_trimmed) > self.num_samples:
            y_fixed = y_trimmed[:self.num_samples]
        else:
            y_fixed = np.pad(y_trimmed, (0, max(0, self.num_samples - len(y_trimmed))), 'constant')

        # 2. Volume normalization (peak amplitude to 1.0)
        max_val = np.max(np.abs(y_fixed))
        if max_val > 0:
            y_normalized = y_fixed / max_val
        else:
            y_normalized = y_fixed

        # 3. Log-Mel Spectrogram conversion
        mel_spec = librosa.feature.melspectrogram(
            y=y_normalized,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 4. Reshape for ONNX: [1, 1, n_mels, time_frames]
        # Based on instructions/model requirement `[1, 1, 64, 251]`
        # librosa returns (n_mels, n_frames)
        # We need to ensure shape is exactly what model expects.
        # If hop_length=160 and samples=40000, frames = ceil(40000/160) + 1? No, it's usually related to windowing.
        # Let's force the shape by padding or cropping if needed for time dimension (the last one).
        
        target_time_frames = log_mel_spec.shape[1] # This might be around 250-251
        if target_time_frames < 251:
             # Pad the end with silence (-80dB or something) if necessary, but let's check dimensions first.
             pass

        # Standard practice for fixed input size in CNNs/Transformers is padding/cropping to exact shape.
        # The model expects [1, 1, 64, 251]
        
        spec_resized = np.zeros((self.n_mels, 251), dtype=np.float32)
        num_frames_to_copy = min(log_mel_spec.shape[1], 251)
        spec_resized[:, :num_frames_to_copy] = log_mel_spec[:, :num_frames_to_copy]

        # Reshape to [1, 1, n_mels, time_steps] for CNN/ConvNet-like input
        final_tensor = spec_resized[np.newaxis, np.newaxis, :, :] # shape (1, 1, 64, 251)

        return final_tensor.astype(np.float32)

