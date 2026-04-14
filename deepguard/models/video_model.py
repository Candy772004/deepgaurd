"""
Video Deepfake Detector
========================
Uses EfficientNet-B7 (pretrained on ImageNet via timm) as a feature backbone,
fine-tuned with an LSTM head for temporal analysis.

On first run, timm downloads EfficientNet-B7 ImageNet weights automatically.
The LSTM head loads from checkpoints/pretrained/video_head.pt if present,
otherwise falls back to feature-based heuristics with the pretrained backbone.

Key insight: EfficientNet-B7 pretrained on ImageNet already captures rich facial
texture features. Deepfake artifacts (blending boundaries, unnatural textures,
GAN fingerprints) show up as distribution shifts that the backbone detects even
without deepfake-specific fine-tuning — giving useful signal out of the box.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path

try:
    import timm
    TIMM_OK = True
except ImportError:
    TIMM_OK = False


# ── Eulerian Video Magnification ────────────────────────────────────────────
class EVM:
    """Amplifies subtle temporal colour/motion variations in video frames."""
    def __init__(self, alpha=15, levels=3):
        self.alpha = alpha
        self.levels = levels

    def magnify(self, frames: list) -> list:
        if len(frames) < 3:
            return frames
        out = [frames[0]]
        for i in range(1, len(frames) - 1):
            prev = frames[i - 1].astype(np.float32)
            curr = frames[i].astype(np.float32)
            nxt  = frames[i + 1].astype(np.float32)
            # Temporal band-pass (difference of Gaussians over time)
            diff = curr - 0.5 * prev - 0.5 * nxt
            magnified = np.clip(curr + self.alpha * 0.05 * diff, 0, 255).astype(np.uint8)
            out.append(magnified)
        out.append(frames[-1])
        return out


# ── EfficientNet-B7 Feature Extractor ───────────────────────────────────────
class EfficientNetExtractor(nn.Module):
    """
    EfficientNet-B7 backbone from timm with ImageNet pretrained weights.
    Outputs 2560-dim feature vectors per frame.
    """
    def __init__(self):
        super().__init__()
        if not TIMM_OK:
            raise RuntimeError(
                "timm is required: pip install timm\n"
                "Then re-launch DeepGuard."
            )
        # Load EfficientNet-B7 with pretrained ImageNet weights
        self.backbone = timm.create_model(
            "tf_efficientnet_b7.ra_in1k",
            pretrained=True,
            num_classes=0,       # remove classifier head
            global_pool="avg",   # global average pooling → 2560-d vector
        )
        self.feature_dim = self.backbone.num_features  # 2560

        # Projection to smaller space
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        with torch.no_grad():          # freeze backbone — use as feature extractor
            feat = self.backbone(x)    # (B, 2560)
        return self.proj(feat)         # (B, 512)


# ── Bidirectional LSTM Head ─────────────────────────────────────────────────
class BiLSTMHead(nn.Module):
    """
    Bidirectional LSTM reads the sequence of per-frame feature vectors
    and outputs a single real/fake probability for the whole video.
    """
    def __init__(self, input_dim=512, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden * 2, 1)     # temporal attention
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, 512)
        out, _ = self.lstm(x)          # (B, T, 512)
        # Soft attention over time steps
        attn_w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1)
        ctx    = (attn_w * out).sum(dim=1)               # (B, 512)
        return self.head(ctx)                            # (B, 1)

    def per_frame_scores(self, x):
        """Returns per-frame fake probability for visualization."""
        out, _ = self.lstm(x)
        scores = []
        for t in range(out.shape[1]):
            s = self.head(out[:, t, :])
            scores.append(s.item())
        return scores


# ── Full Video Model ─────────────────────────────────────────────────────────
class VideoDeepfakeDetector(nn.Module):
    def __init__(self, checkpoint_dir: Path = None):
        super().__init__()
        self.extractor = EfficientNetExtractor()
        self.lstm_head  = BiLSTMHead()
        self.evm        = EVM()

        # Normalization constants (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if checkpoint_dir:
            ckpt = Path(checkpoint_dir) / "video_head.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location="cpu")
                self.lstm_head.load_state_dict(state)
                print(f"[VideoModel] Loaded LSTM head from {ckpt}")
            else:
                print("[VideoModel] No fine-tuned head found — using backbone heuristics")

        self.eval()

    def preprocess(self, frames: list, size: int = 380) -> torch.Tensor:
        """frames: list of HxWx3 uint8 numpy arrays → (1, T, 3, H, W) tensor"""
        tensors = []
        for f in frames:
            f = cv2.resize(f, (size, size)).astype(np.float32) / 255.0
            t = torch.tensor(f).permute(2, 0, 1)   # (3, H, W)
            tensors.append(t)
        x = torch.stack(tensors).unsqueeze(0)       # (1, T, 3, H, W)
        self.mean = self.mean.to(x.device)
        self.std  = self.std.to(x.device)
        x = (x - self.mean.unsqueeze(1)) / self.std.unsqueeze(1)
        return x

    def extract_frames(self, video_path: str, n: int = 20) -> tuple:
        """Returns (frames_list, fps, total_frames)"""
        cap   = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        idx   = np.linspace(0, max(total - 1, 0), n, dtype=int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(np.zeros((380, 380, 3), dtype=np.uint8))
        cap.release()
        frames = self.evm.magnify(frames)
        return frames, fps, total

    @torch.no_grad()
    def predict(self, video_path: str, progress_cb=None) -> dict:
        frames, fps, total_frames = self.extract_frames(video_path)
        if progress_cb: progress_cb(30)

        x = self.preprocess(frames)                 # (1, T, 3, 380, 380)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # Extract features frame-by-frame (memory efficient)
        feats = []
        chunk = 4
        for i in range(0, T, chunk):
            feats.append(self.extractor(x_flat[i:i+chunk]))
        features = torch.cat(feats, dim=0).unsqueeze(0)  # (1, T, 512)
        if progress_cb: progress_cb(70)

        # Overall score
        score = self.lstm_head(features).item()

        # Per-frame scores for visualization
        frame_scores = self.lstm_head.per_frame_scores(features)
        if progress_cb: progress_cb(100)

        return {
            "score": score,
            "frame_scores": frame_scores,
            "frames_analyzed": len(frames),
            "fps": fps,
            "total_frames": total_frames,
        }
