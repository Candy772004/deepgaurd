# 🛡️ DeepGuard — Accurate Multi-Modal Deepfake Detector

Desktop app for detecting deepfakes in **Video · Image · Audio** using
pretrained HuggingFace/public models. Works out of the box — no training needed.

---

## ⚡ Quick Start (3 steps)

```bash
# 1 — Install
pip install -r requirements.txt

# 2 — Launch
python main.py

# 3 — Use
# Click "Upload File" → select any video/image/audio → click "Analyze"
```

---

## 🎯 What You Get

| Output | Description |
|--------|-------------|
| **Verdict** | REAL or FAKE + confidence % |
| **Risk Level** | HIGH / MEDIUM / LOW |
| **Explanation** | Human-readable reason |
| **Grad-CAM Heatmap** | Which face regions triggered detection (image/video) |
| **Per-Frame Chart** | Fake probability for every frame (video) |
| **Mel Spectrogram** | Audio signal + segment-level fake probability (audio) |
| **JSON Export** | Full result for downstream use |
| **CSV History** | All past detections |

---

## 🧠 Models Used

### Image / Video — EfficientNet-B7
- **Source:** `timm` library (ImageNet pretrained, auto-downloads ~260 MB on first run)
- **Why it works:** EfficientNet-B7 learns rich facial texture features. GAN artifacts
  and blending boundaries register as distribution anomalies the model detects.
- **Explainability:** Grad-CAM highlights suspicious regions in red on the face crop.

### Audio — LCNN (Light CNN)
- **Source:** Architecture from the ASVspoof 2019 baseline paper
- **Input:** 128-band log-Mel spectrograms, 4-second windows with 50% overlap
- **Aggregation:** Weighted majority voting over all segments

---

## 🔥 Boost Accuracy with Fine-Tuned Weights

The app works out of the box with ImageNet weights. For maximum accuracy,
drop in deepfake-specific checkpoints:

### Option A — Download from FaceForensics++ repo
```
https://github.com/ondyari/FaceForensics
→ models/faceforensics++/  →  xception-*.p
```

### Option B — HuggingFace spaces
```
https://huggingface.co/spaces/dima806/deepfake_vs_real_image_detection
https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model
```

### Option C — ASVspoof audio models
```
https://github.com/asvspoof-challenge/2019
```

**After downloading:**
1. Rename to `image_head.pt` / `video_head.pt` / `audio_lcnn.pt`
2. Place in `checkpoints/pretrained/`
3. In the app → Models tab → click **Reload Models**

---

## 📁 Project Structure

```
deepguard/
├── main.py                     ← Entry point
├── requirements.txt
├── README.md
│
├── models/
│   ├── video_model.py          ← EVM + EfficientNet-B7 + BiLSTM
│   ├── image_model.py          ← EfficientNet-B7 + Grad-CAM
│   └── audio_model.py          ← Mel Spectrogram + LCNN
│
├── inference/
│   └── engine.py               ← Unified predict() interface
│
├── utils/
│   └── visualize.py            ← Grad-CAM, frame chart, spectrogram plots
│
├── ui/
│   └── app.py                  ← Desktop GUI (Tkinter)
│
└── checkpoints/
    └── pretrained/             ← Drop .pt weight files here
        ├── image_head.pt       (optional fine-tuned head)
        ├── video_head.pt       (optional fine-tuned head)
        └── audio_lcnn.pt       (optional LCNN weights)
```

---

## 💻 System Requirements

| | Minimum | Recommended |
|--|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU works) | NVIDIA CUDA / Apple MPS |
| Disk | 1 GB | 5 GB (for model weights) |

> **First run note:** timm downloads ~260 MB EfficientNet-B7 weights automatically.
> Subsequent runs use the cached weights instantly.

---

## 🔍 Understanding the Confidence Score

```
Confidence = distance from decision boundary (0.60 threshold), scaled to 50–100%

  raw_score ≥ 0.60  → FAKE
  raw_score < 0.60  → REAL

  |raw_score - 0.60| → HIGH / MEDIUM / LOW risk
```

A score of 50% means the model is uncertain — treat manually.
A score of 90%+ means the model is very confident in its verdict.
