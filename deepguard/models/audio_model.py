"""
Audio Deepfake Detector
========================
Light CNN (LCNN) operating on log-Mel spectrograms.
Architecture from: "A Light CNN for Deep Fake Detection" — ASVspoof2019 baseline.

Outputs:
  - Real/Fake score
  - Mel spectrogram image (for visualization)
  - Per-segment scores (audio split into 1-second chunks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    import soundfile as sf
    SF_OK = True
except ImportError:
    SF_OK = False


# ── Mel Spectrogram Feature Extractor ───────────────────────────────────────
class MelExtractor:
    """
    Converts audio to log-Mel spectrogram.
    Parameters match the ASVspoof2019 baseline system.
    """
    SR         = 16000
    N_MELS     = 128
    N_FFT      = 512
    HOP_LENGTH = 160
    DURATION   = 4.0     # seconds per chunk
    WIN_LENGTH = 400

    def __init__(self):
        if not LIBROSA_OK:
            raise RuntimeError(
                "librosa is required:\n  pip install librosa soundfile\n"
                "Then re-launch DeepGuard."
            )
        self.target_frames = int(
            np.ceil(self.DURATION * self.SR / self.HOP_LENGTH)
        )

    def load(self, path: str) -> np.ndarray:
        """Load audio → mono float32 at 16kHz."""
        try:
            if SF_OK:
                y, sr = sf.read(str(path), dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != self.SR:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.SR)
            else:
                y, sr = librosa.load(str(path), sr=self.SR, mono=True)
        except Exception:
            y, _ = librosa.load(str(path), sr=self.SR, mono=True)
        return y.astype(np.float32)

    def to_mel(self, y: np.ndarray) -> np.ndarray:
        """Audio array → log-Mel spectrogram (n_mels × T)."""
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.SR, n_mels=self.N_MELS,
            n_fft=self.N_FFT, hop_length=self.HOP_LENGTH,
            win_length=self.WIN_LENGTH, window="hann",
            fmin=20, fmax=7600,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel.astype(np.float32)

    def extract_chunks(self, path: str, overlap: float = 0.5) -> tuple:
        """
        Split audio into overlapping chunks and convert each to Mel.
        Returns:
          chunks  : list of (1, n_mels, T) float32 arrays
          full_mel: (n_mels, T_total) full spectrogram for display
          duration: float seconds
        """
        y = self.load(path)
        duration = len(y) / self.SR
        full_mel = self.to_mel(y)

        chunk_len   = int(self.DURATION * self.SR)
        hop_samples = int(chunk_len * (1 - overlap))

        chunks = []
        positions = []
        start = 0
        while start < len(y):
            end    = start + chunk_len
            chunk  = y[start:end]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
            mel   = self.to_mel(chunk)                   # (n_mels, T)
            # Normalise to [0, 1]
            mel   = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
            # Pad / trim time axis
            if mel.shape[1] < self.target_frames:
                mel = np.pad(mel, ((0, 0), (0, self.target_frames - mel.shape[1])))
            else:
                mel = mel[:, :self.target_frames]
            chunks.append(mel[np.newaxis])               # (1, n_mels, T)
            positions.append(start / self.SR)
            start += hop_samples

        return chunks, full_mel, duration, positions


# ── Light CNN (LCNN) ─────────────────────────────────────────────────────────
class MaxFeatureMap(nn.Module):
    """Max Feature Map activation — key component of LCNN."""
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


class LCNN(nn.Module):
    """
    Light CNN for audio anti-spoofing.
    Based on Wu et al. — ASVspoof2019 frontend.
    Input: (B, 1, 128, T) log-Mel spectrogram
    Output: (B, 1) fake probability
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 5, padding=2),   nn.BatchNorm2d(64),
            MaxFeatureMap(),                               # → 32 ch
            nn.MaxPool2d(2, 2),                            # → 32×64×T//2

            # Block 2
            nn.Conv2d(32, 128, 1),             nn.BatchNorm2d(128),
            MaxFeatureMap(),                               # → 64 ch
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128),
            MaxFeatureMap(),                               # → 64 ch
            nn.MaxPool2d(2, 2),                            # → 64×32×T//4

            # Block 3
            nn.Conv2d(64, 128, 1),             nn.BatchNorm2d(128),
            MaxFeatureMap(),                               # → 64 ch
            nn.Conv2d(64, 256, 3, padding=1),  nn.BatchNorm2d(256),
            MaxFeatureMap(),                               # → 128 ch
            nn.MaxPool2d(2, 2),                            # → 128×16×T//8

            # Block 4
            nn.Conv2d(128, 256, 1),            nn.BatchNorm2d(256),
            MaxFeatureMap(),                               # → 128 ch
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            MaxFeatureMap(),                               # → 128 ch

            # Block 5
            nn.Conv2d(128, 128, 1),            nn.BatchNorm2d(128),
            MaxFeatureMap(),                               # → 64 ch
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128),
            MaxFeatureMap(),                               # → 64 ch
            nn.MaxPool2d(2, 2),                            # → 64×8×T//16
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Full Audio Detector ──────────────────────────────────────────────────────
class AudioDeepfakeDetector:
    """
    Wraps the LCNN model with feature extraction and multi-chunk aggregation.
    """
    def __init__(self, checkpoint_dir: Path = None):
        self.extractor = MelExtractor()
        self.model     = LCNN()

        if checkpoint_dir:
            ckpt = Path(checkpoint_dir) / "audio_lcnn.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location="cpu")
                if "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"])
                else:
                    self.model.load_state_dict(state)
                print(f"[AudioModel] Loaded LCNN weights from {ckpt}")
            else:
                print("[AudioModel] No checkpoint found — using random-initialised LCNN")
                print("             For accurate results, fine-tune on ASVspoof2019.")

        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path: str, progress_cb=None) -> dict:
        if progress_cb: progress_cb(10)

        chunks, full_mel, duration, positions = \
            self.extractor.extract_chunks(audio_path, overlap=0.5)

        if progress_cb: progress_cb(40)

        segment_scores = []
        for chunk in chunks:
            x     = torch.tensor(chunk).unsqueeze(0)   # (1, 1, n_mels, T)
            score = self.model(x).item()
            segment_scores.append(score)

        if progress_cb: progress_cb(90)

        # Aggregate: max-pooling weighted by deviation from 0.5
        # (more extreme predictions get more weight)
        weights = [abs(s - 0.5) + 0.01 for s in segment_scores]
        total_w = sum(weights)
        agg_score = sum(s * w for s, w in zip(segment_scores, weights)) / total_w

        return {
            "score": float(agg_score),
            "segment_scores": segment_scores,
            "segment_positions": positions,
            "full_mel": full_mel,
            "duration": duration,
            "num_chunks": len(chunks),
        }
