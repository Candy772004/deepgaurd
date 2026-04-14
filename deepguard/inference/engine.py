"""
DeepGuard Inference Engine
===========================
Unified detection interface for all three modalities.
Produces rich, structured results including visual data.
"""

import time
import numpy as np
from pathlib import Path

# Lazy imports so the GUI loads instantly
_video_cls = None
_image_cls = None
_audio_cls = None


def _get_video():
    global _video_cls
    if _video_cls is None:
        from models.video_model import VideoDeepfakeDetector
        _video_cls = VideoDeepfakeDetector
    return _video_cls


def _get_image():
    global _image_cls
    if _image_cls is None:
        from models.image_model import ImageDeepfakeDetector
        _image_cls = ImageDeepfakeDetector
    return _image_cls


def _get_audio():
    global _audio_cls
    if _audio_cls is None:
        from models.audio_model import AudioDeepfakeDetector
        _audio_cls = AudioDeepfakeDetector
    return _audio_cls


# ── Verdict Helper ───────────────────────────────────────────────────────────
def _build_verdict(raw_score: float, media_type: str, inference_time: float,
                   extra: dict = None) -> dict:
    """
    Converts a raw sigmoid score → clean verdict dict.

    Calibration thresholds (empirical):
      ≥ 0.60 → FAKE
      < 0.60 → REAL
    Confidence shown as distance from decision boundary, scaled 0–100%.
    """
    THRESHOLD = 0.60

    is_fake   = raw_score >= THRESHOLD
    label     = "FAKE" if is_fake else "REAL"

    # Confidence toward predicted class  (0.5 = min, 1.0 = max)
    dist_from_boundary = abs(raw_score - THRESHOLD)
    # Scale so that 0.4 gap → ~100 %, 0 gap → 50 %
    confidence = 50 + (dist_from_boundary / 0.40) * 50
    confidence = min(100.0, max(50.0, confidence))

    if   confidence >= 88: risk = "HIGH"
    elif confidence >= 70: risk = "MEDIUM"
    else:                  risk = "LOW"

    # Human-readable explanation
    if label == "FAKE":
        explanations = {
            "HIGH":   "Strong deepfake indicators detected. Very likely AI-generated.",
            "MEDIUM": "Several deepfake artifacts found. Likely manipulated.",
            "LOW":    "Some suspicious patterns detected, but result is uncertain.",
        }
    else:
        explanations = {
            "HIGH":   "No deepfake indicators found. Media appears authentic.",
            "MEDIUM": "Mostly authentic patterns. Minor anomalies present.",
            "LOW":    "Likely authentic, but low-confidence result. Review manually.",
        }

    result = {
        "label":        label,
        "is_fake":      is_fake,
        "raw_score":    float(raw_score),
        "confidence":   float(confidence),
        "risk_level":   risk,
        "explanation":  explanations[risk],
        "media_type":   media_type,
        "inference_time": inference_time,
    }
    if extra:
        result.update(extra)
    return result


# ── Main Engine ──────────────────────────────────────────────────────────────
class DeepGuardEngine:
    VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

    def __init__(self, checkpoint_dir: str = "checkpoints/pretrained"):
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._video = None
        self._image = None
        self._audio = None
        self.errors = {}

    # ── Lazy model loading ───────────────────────────────────────────────────
    def _load_video(self):
        if self._video is None:
            try:
                cls = _get_video()
                self._video = cls(checkpoint_dir=self.ckpt_dir)
            except Exception as e:
                self.errors["video"] = str(e)
                raise
        return self._video

    def _load_image(self):
        if self._image is None:
            try:
                cls = _get_image()
                self._image = cls(checkpoint_dir=self.ckpt_dir)
            except Exception as e:
                self.errors["image"] = str(e)
                raise
        return self._image

    def _load_audio(self):
        if self._audio is None:
            try:
                cls = _get_audio()
                self._audio = cls(checkpoint_dir=self.ckpt_dir)
            except Exception as e:
                self.errors["audio"] = str(e)
                raise
        return self._audio

    # ── Media type detection ─────────────────────────────────────────────────
    def media_type(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext in self.VIDEO_EXT: return "video"
        if ext in self.IMAGE_EXT: return "image"
        if ext in self.AUDIO_EXT: return "audio"
        return "unknown"

    # ── Predict ──────────────────────────────────────────────────────────────
    def predict(self, file_path: str, progress_cb=None) -> dict:
        mtype = self.media_type(file_path)
        if mtype == "unknown":
            return {"error": f"Unsupported file: {Path(file_path).suffix}",
                    "label": "ERROR", "media_type": "unknown"}

        t0 = time.time()
        try:
            if mtype == "video":
                model  = self._load_video()
                raw    = model.predict(file_path, progress_cb)
                result = _build_verdict(raw["score"], "video",
                                        time.time() - t0, raw)

            elif mtype == "image":
                model  = self._load_image()
                raw    = model.predict(file_path, progress_cb)
                result = _build_verdict(raw["score"], "image",
                                        time.time() - t0, raw)

            elif mtype == "audio":
                model  = self._load_audio()
                raw    = model.predict(file_path, progress_cb)
                result = _build_verdict(raw["score"], "audio",
                                        time.time() - t0, raw)

        except Exception as e:
            return {
                "error":      str(e),
                "label":      "ERROR",
                "media_type": mtype,
            }

        result["file_path"] = str(file_path)
        result["file_name"] = Path(file_path).name
        return result

    def reload_model(self, mode: str):
        """Force reload a model (e.g. after placing new checkpoint)."""
        if mode == "video": self._video = None
        if mode == "image": self._image = None
        if mode == "audio": self._audio = None
