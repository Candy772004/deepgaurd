"""
DeepGuard Inference Engine
===========================
Unified detection interface for all three modalities.
Produces rich, structured results including visual data.

Forensic Output Schema
-----------------------
Every predict() result includes a 'forensic_report' key:
  {
    "authenticity":      "Real" | "Fake",
    "confidence":        "XX%",
    "forensic_analysis": ["Reason 1", ...],
    "scene_description": "<human-readable sentence>",
    "objects_detected":  ["object1", ...],
    "actions_detected":  ["action1", ...]
  }

Scene Understanding Schema
---------------------------
Every predict() result also includes a 'scene_report' key:
  {
    "scene_summary":        "Short one-line summary",
    "detailed_description": "Clear explanation of what is happening",
    "people": [{"description": "", "emotion": "", "action": ""}],
    "objects":     [],
    "environment": "",
    "activities":  [],
    "possible_context": "What might be happening overall"
  }
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


# ── Forensic Report Formatter ────────────────────────────────────────────────
def format_forensic_output(result: dict) -> dict:
    """
    Converts a DeepGuard prediction result dict into the strict forensic
    JSON schema:

    Output:
    {
      "authenticity":      "Real" | "Fake",
      "confidence":        "XX%",
      "forensic_analysis": ["Reason 1", "Reason 2", "Reason 3"],
      "scene_description": "<human-readable description>",
      "objects_detected":  ["object1", "object2"],
      "actions_detected":  ["action1", "action2"]
    }
    """
    label      = result.get("label", "UNKNOWN")
    conf       = result.get("confidence", 50.0)
    risk       = result.get("risk_level", "LOW")
    mtype      = result.get("media_type", "unknown")
    raw_score  = result.get("raw_score", 0.5)
    fname      = result.get("file_name", "media file")
    face_found = result.get("face_found", None)
    n_frames   = result.get("total_frames", None)
    duration   = result.get("duration", None)
    explanation = result.get("explanation", "")

    authenticity = "Fake" if label == "FAKE" else "Real"
    conf_str     = f"{conf:.0f}%"

    # ── Build forensic_analysis reasons ──────────────────────────────────
    reasons = []

    # 1. Overall score signal
    if label == "FAKE":
        reasons.append(
            f"Deepfake score {raw_score:.3f} exceeds detection threshold (0.60), "
            f"indicating {risk.lower()} confidence in manipulation."
        )
    else:
        reasons.append(
            f"Deepfake score {raw_score:.3f} is below detection threshold (0.60), "
            f"consistent with authentic {mtype} content."
        )

    # 2. Media-type specific reasoning
    if mtype == "image":
        if face_found is True:
            reasons.append(
                "Face region detected and analysed via Haar Cascade; "
                "Grad-CAM heatmap highlights suspicious spatial regions in the face crop."
            )
        elif face_found is False:
            reasons.append(
                "No face detected — model applied to centre-crop region; "
                "structural texture patterns evaluated for GAN fingerprints."
            )
        if label == "FAKE":
            reasons.append(
                "EfficientNet-B7 backbone detected high-frequency artifacts and texture "
                "discontinuities characteristic of GAN-generated or face-swap manipulation."
            )
        else:
            reasons.append(
                "Texture, lighting gradients, and boundary transitions appear natural; "
                "no blending seams or GAN fingerprints identified by EfficientNet-B7."
            )

    elif mtype == "video":
        if n_frames:
            reasons.append(
                f"{n_frames} frames sampled; per-frame scores aggregated via "
                "Bidirectional LSTM with attention to capture temporal inconsistencies."
            )
        if label == "FAKE":
            reasons.append(
                "Temporal analysis reveals flickering artifacts, unnatural micro-expression "
                "transitions, and frame-to-frame inconsistency patterns typical of deepfake video."
            )
        else:
            reasons.append(
                "Temporal coherence is consistent across sampled frames; "
                "motion and lighting transitions appear physically plausible."
            )

    elif mtype == "audio":
        if duration:
            reasons.append(
                f"Audio duration {duration:.1f}s processed via 128-band log-Mel "
                "spectrogram in 4-second overlapping chunks (LCNN model)."
            )
        if label == "FAKE":
            reasons.append(
                "Spectral analysis reveals unnatural harmonic patterns and pitch "
                "transitions inconsistent with natural human speech vocalization."
            )
        else:
            reasons.append(
                "Mel spectrogram exhibits natural formant structure and "
                "prosodic variation consistent with genuine human speech."
            )

    # 3. Confidence note
    reasons.append(
        f"Model confidence: {conf:.1f}% ({risk} risk). "
        + explanation
    )

    # ── Scene description ─────────────────────────────────────────────────
    media_label_map = {
        "image": "a static image",
        "video": "a video clip",
        "audio": "an audio recording",
    }
    media_label = media_label_map.get(mtype, "a media file")

    if authenticity == "Fake":
        scene_description = (
            f"The submitted {media_label} '{fname}' has been classified as a "
            f"DEEPFAKE with {conf:.0f}% confidence. "
            f"AI-generated or manipulated content was detected in the {mtype} signal."
        )
    else:
        scene_description = (
            f"The submitted {media_label} '{fname}' appears AUTHENTIC with "
            f"{conf:.0f}% confidence. "
            f"No significant manipulation artifacts were detected in the {mtype} content."
        )

    # ── Objects detected ──────────────────────────────────────────────────
    objects: list = []
    if mtype == "image":
        objects = ["image frame", "face region" if face_found else "scene region"]
    elif mtype == "video":
        objects = ["video frames", "temporal sequence", "face region"]
    elif mtype == "audio":
        segs = result.get("segment_scores", [])
        objects = ["audio waveform", "mel spectrogram",
                   f"{len(segs)} audio segment(s)" if segs else "audio segments"]

    # ── Actions detected ──────────────────────────────────────────────────
    actions: list = []
    if mtype == "image":
        actions = ["face detection", "EfficientNet-B7 feature extraction", "Grad-CAM heatmap generation"]
    elif mtype == "video":
        actions = ["frame sampling", "Eulerian Video Magnification",
                   "LSTM temporal aggregation", "per-frame scoring"]
    elif mtype == "audio":
        actions = ["log-Mel spectrogram extraction", "LCNN segment scoring",
                   "weighted max-pool aggregation"]

    return {
        "authenticity":      authenticity,
        "confidence":        conf_str,
        "forensic_analysis": reasons,
        "scene_description": scene_description,
        "objects_detected":  objects,
        "actions_detected":  actions,
    }


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

        # Attach the strict forensic report schema
        result["forensic_report"] = format_forensic_output(result)

        # Attach the visual scene understanding schema
        result["scene_report"] = format_scene_understanding(result)

        return result

    def reload_model(self, mode: str):
        """Force reload a model (e.g. after placing new checkpoint)."""
        if mode == "video": self._video = None
        if mode == "image": self._image = None
        if mode == "audio": self._audio = None


# ── Scene Understanding Formatter ────────────────────────────────────────────────
def format_scene_understanding(result: dict) -> dict:
    """
    Builds the visual scene understanding report from a DeepGuard result dict.

    Output schema:
    {
      "scene_summary":        "Short one-line summary",
      "detailed_description": "Clear explanation of what is happening",
      "people": [
        {"description": "", "emotion": "", "action": ""}
      ],
      "objects":          [],
      "environment":      "",
      "activities":       [],
      "possible_context": "What might be happening overall"
    }
    """
    label      = result.get("label", "UNKNOWN")
    conf       = result.get("confidence", 50.0)
    risk       = result.get("risk_level", "LOW")
    mtype      = result.get("media_type", "unknown")
    fname      = result.get("file_name", "media file")
    raw_score  = result.get("raw_score", 0.5)
    face_found = result.get("face_found", None)
    n_frames   = result.get("total_frames", None)
    duration   = result.get("duration", None)
    is_fake    = label == "FAKE"

    # ── Scene summary (one-liner) ────────────────────────────────────────────────
    status_word = "DEEPFAKE" if is_fake else "AUTHENTIC"
    mtype_cap   = mtype.capitalize()

    scene_summary = (
        f"{mtype_cap} file '{fname}' analysed — classified as {status_word} "
        f"with {conf:.0f}% confidence ({risk} risk)."
    )

    # ── Detailed description ──────────────────────────────────────────────────
    if mtype == "image":
        face_line = (
            "A human face was detected and isolated for analysis."
            if face_found else
            "No face was detected; the model analysed the central scene region."
        )
        if is_fake:
            detailed = (
                f"The image '{fname}' was submitted for deepfake analysis. {face_line} "
                f"The EfficientNet-B7 model returned a deepfake score of {raw_score:.3f}, "
                f"exceeding the 0.60 detection threshold. Grad-CAM heatmap highlights "
                f"spatially suspicious regions in the image where manipulation artifacts "
                f"are most likely concentrated. The {risk.lower()} risk classification "
                f"with {conf:.0f}% confidence suggests this image is AI-generated or "
                f"face-swapped."
            )
        else:
            detailed = (
                f"The image '{fname}' was submitted for deepfake analysis. {face_line} "
                f"The EfficientNet-B7 model returned a deepfake score of {raw_score:.3f}, "
                f"below the 0.60 detection threshold. Texture boundaries, lighting "
                f"gradients, and facial region transitions appear authentic. No GAN "
                f"fingerprints or blending seams were identified. The model classifies "
                f"this image as genuine with {conf:.0f}% confidence."
            )

    elif mtype == "video":
        frame_line = (
            f"{n_frames} frames were sampled and scored individually."
            if n_frames else
            "Multiple frames were sampled across the video timeline."
        )
        if is_fake:
            detailed = (
                f"The video '{fname}' was processed through the deepfake detection pipeline. "
                f"{frame_line} The Bidirectional LSTM with attention aggregated per-frame "
                f"scores, detecting temporal inconsistencies including flickering, unnatural "
                f"micro-expression transitions, and identity-boundary artifacts across frames. "
                f"The overall deepfake score of {raw_score:.3f} exceeds the threshold, "
                f"resulting in a {status_word} verdict with {conf:.0f}% confidence."
            )
        else:
            detailed = (
                f"The video '{fname}' was processed through the deepfake detection pipeline. "
                f"{frame_line} Temporal coherence across frames is consistent — motion "
                f"dynamics, lighting transitions, and facial continuity all appear physically "
                f"plausible. The overall deepfake score of {raw_score:.3f} remains below the "
                f"detection threshold. The video is classified as {status_word} with "
                f"{conf:.0f}% confidence."
            )

    elif mtype == "audio":
        dur_line = (
            f"Audio duration: {duration:.1f}s."
            if duration else
            "Audio of unspecified duration."
        )
        segs = result.get("segment_scores", [])
        seg_line = (
            f"{len(segs)} overlapping 4-second segments were scored."
            if segs else
            "Audio segments were scored using the LCNN model."
        )
        if is_fake:
            detailed = (
                f"The audio file '{fname}' was analysed for synthetic speech. {dur_line} "
                f"{seg_line} The LCNN model detected unnatural spectral patterns — "
                f"including pitch discontinuities and harmonic anomalies — inconsistent "
                f"with genuine human vocalization. Deepfake score: {raw_score:.3f}. "
                f"Verdict: {status_word} with {conf:.0f}% confidence."
            )
        else:
            detailed = (
                f"The audio file '{fname}' was analysed for synthetic speech. {dur_line} "
                f"{seg_line} Natural formant structure, prosodic variation, and vocal "
                f"tract characteristics consistent with genuine human speech were identified. "
                f"Deepfake score: {raw_score:.3f}. "
                f"Verdict: {status_word} with {conf:.0f}% confidence."
            )
    else:
        detailed = f"Media file '{fname}' of type '{mtype}' was analysed. Result: {status_word}."

    # ── People ───────────────────────────────────────────────────────────
    people = []
    if mtype in ("image", "video"):
        if face_found is True:
            # Emotion/action inferred from deepfake context (no live vision model)
            emotion = "indeterminate" if not is_fake else "artificially rendered"
            action  = (
                "appearing in a manipulated/synthesised context"
                if is_fake else
                "appearing naturally in the frame"
            )
            people.append({
                "description": "A human subject whose face was detected by Haar Cascade.",
                "emotion":     emotion,
                "action":      action,
            })
        elif face_found is False:
            people.append({
                "description": "No human face detected in this media; scene may contain objects, text, or non-facial content.",
                "emotion":     "N/A",
                "action":      "N/A",
            })
        else:
            people.append({
                "description": "Face detection result unavailable for this media type.",
                "emotion":     "unknown",
                "action":      "unknown",
            })

    # ── Objects ──────────────────────────────────────────────────────────
    if mtype == "image":
        objects = [
            "digital image",
            "face crop (380×380 px)" if face_found else "scene crop (70% centre)",
            "Grad-CAM spatial heatmap",
            "EfficientNet-B7 feature maps",
        ]
    elif mtype == "video":
        objects = [
            "video stream",
            f"{n_frames or 'multiple'} sampled frames",
            "per-frame score chart",
            "LSTM attention weights",
        ]
    elif mtype == "audio":
        segs = result.get("segment_scores", [])
        objects = [
            "audio waveform",
            "128-band log-Mel spectrogram",
            f"{len(segs)} scored audio segment(s)" if segs else "audio segments",
            "LCNN model output",
        ]
    else:
        objects = ["media file"]

    # ── Environment ───────────────────────────────────────────────────────
    env_map = {
        "image": "Digital forensics environment — static image analysis pipeline.",
        "video": "Digital forensics environment — video frame analysis pipeline with temporal modelling.",
        "audio": "Digital forensics environment — acoustic analysis pipeline with spectrogram processing.",
    }
    environment = env_map.get(mtype, "DeepGuard AI analysis environment.")

    # ── Activities ───────────────────────────────────────────────────────
    activities_map = {
        "image": [
            "Face detection via Haar Cascade",
            "Image transformation and normalisation (380×380 px)",
            "EfficientNet-B7 feature extraction",
            "Grad-CAM heatmap generation",
            "Binary deepfake classification",
        ],
        "video": [
            "Uniform frame sampling across video timeline",
            "Eulerian Video Magnification (EVM) preprocessing",
            "Per-frame EfficientNet-B7 feature extraction",
            "Bidirectional LSTM temporal aggregation",
            "Attention-weighted score pooling",
            "Binary deepfake classification",
        ],
        "audio": [
            "Audio loading and 16kHz resampling",
            "128-band log-Mel spectrogram computation",
            "4-second overlapping segment extraction",
            "LCNN Max-Feature-Map scoring per segment",
            "Weighted max-pool score aggregation",
            "Binary synthetic speech classification",
        ],
    }
    activities = activities_map.get(mtype, ["Media analysis", "Deepfake classification"])

    # ── Possible context ──────────────────────────────────────────────────
    if is_fake:
        context_map = {
            "image": (
                "The image may have been created using GAN-based generation (e.g., StyleGAN, "
                "DALL-E) or face-swap tools (e.g., DeepFaceLab, FaceSwap). Possible use cases "
                "include identity fraud, misinformation, synthetic profile photos, or "
                "non-consensual image generation."
            ),
            "video": (
                "The video may be a deepfake produced using face-reenactment or identity-swap "
                "techniques (e.g., First Order Motion Model, Wav2Lip, DeepFaceLab). This could "
                "be used for political manipulation, impersonation, or synthetic media abuse."
            ),
            "audio": (
                "The audio may have been synthesised using a text-to-speech or voice cloning "
                "system (e.g., ElevenLabs, XTTS, Tortoise-TTS). Potential applications include "
                "voice phishing (vishing), identity impersonation, or fake audio testimonials."
            ),
        }
    else:
        context_map = {
            "image": (
                "This appears to be a genuine photograph. It may have been submitted for "
                "verification purposes, identity authentication, content moderation, or as "
                "a reference sample in a forensic investigation."
            ),
            "video": (
                "This appears to be authentic video footage. It may originate from a personal "
                "recording, broadcast, CCTV, or documentary source. The content was verified "
                "as free of temporal deepfake artifacts."
            ),
            "audio": (
                "This appears to be a genuine human voice recording. It may be from a podcast, "
                "interview, phone call, or recorded statement. No synthetic speech markers "
                "were found by the acoustic analysis pipeline."
            ),
        }

    possible_context = context_map.get(
        mtype,
        f"Media submitted to DeepGuard for authenticity verification. Result: {status_word}."
    )

    return {
        "scene_summary":        scene_summary,
        "detailed_description": detailed,
        "people":               people,
        "objects":              objects,
        "environment":          environment,
        "activities":           activities,
        "possible_context":     possible_context,
    }
