"""
DeepGuard Visualization Utilities
===================================
Renders analysis results as images/charts suitable for embedding in the GUI.
All functions return PIL Images or numpy arrays.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MPL_OK = True
except ImportError:
    MPL_OK = False


# ── Colour palette ──────────────────────────────────────────────────────────
FAKE_COLOR  = (239, 68,  68)   # red
REAL_COLOR  = (16,  185, 129)  # green
WARN_COLOR  = (245, 158, 11)   # amber
ACCENT      = (0,   212, 255)  # cyan
BG          = (10,  14,  26)   # navy
SURFACE     = (17,  24,  39)

# Custom heat colormap (black → purple → cyan → white)
_DEEP_COLORS = [
    (0.00, (0.04, 0.06, 0.10)),
    (0.25, (0.31, 0.14, 0.46)),
    (0.55, (0.00, 0.83, 1.00)),
    (0.80, (0.16, 0.73, 0.51)),
    (1.00, (1.00, 1.00, 1.00)),
]
if MPL_OK:
    _cmap_data = {
        "red":   [(p, c[0], c[0]) for p, c in _DEEP_COLORS],
        "green": [(p, c[1], c[1]) for p, c in _DEEP_COLORS],
        "blue":  [(p, c[2], c[2]) for p, c in _DEEP_COLORS],
    }
    DEEP_CMAP = LinearSegmentedColormap("deep", _cmap_data)


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=np.array(BG) / 255)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


# ── 1. Grad-CAM overlay on face crop ────────────────────────────────────────
def render_gradcam(face_crop: np.ndarray, cam: np.ndarray,
                   label: str, confidence: float) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap on the face crop.
    face_crop : HxWx3 uint8
    cam       : HxW  float in [0,1]
    """
    if not MPL_OK:
        return Image.fromarray(face_crop)

    h, w = face_crop.shape[:2]
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.patch.set_facecolor(np.array(BG) / 255)

    titles = ["Original", "Heatmap", "Overlay"]
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor(np.array(BG) / 255)

    # Original
    axes[0].imshow(face_crop)
    axes[0].set_title(titles[0], color="white", fontsize=9)

    # Heatmap
    axes[1].imshow(cam_resized, cmap=DEEP_CMAP, vmin=0, vmax=1)
    axes[1].set_title(titles[1], color="white", fontsize=9)

    # Overlay
    axes[2].imshow(face_crop)
    axes[2].imshow(cam_resized, cmap=DEEP_CMAP, alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title(titles[2], color="white", fontsize=9)

    color = np.array(FAKE_COLOR if label == "FAKE" else REAL_COLOR) / 255
    fig.suptitle(
        f"{label}  ·  {confidence:.1f}% confidence",
        color=color, fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout(pad=0.3)
    return _fig_to_pil(fig)


# ── 2. Per-frame score chart (video) ────────────────────────────────────────
def render_frame_scores(frame_scores: list, label: str,
                         confidence: float) -> Image.Image:
    """Bar chart of per-frame fake probability."""
    if not MPL_OK or not frame_scores:
        return _placeholder("Frame scores unavailable")

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor(np.array(BG) / 255)
    ax.set_facecolor(np.array(SURFACE) / 255)

    n   = len(frame_scores)
    xs  = list(range(n))
    colors = [
        np.array(FAKE_COLOR if s >= 0.5 else REAL_COLOR) / 255
        for s in frame_scores
    ]
    bars = ax.bar(xs, frame_scores, color=colors, width=0.8, edgecolor="none", zorder=3)

    # Threshold line
    ax.axhline(0.5, color=np.array(WARN_COLOR)/255, linewidth=1.2,
               linestyle="--", label="Decision boundary (0.5)")
    ax.fill_between([-0.5, n - 0.5], 0.5, 1.0,
                    color=(239/255, 68/255, 68/255), alpha=0.06)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame index", color="white", fontsize=8)
    ax.set_ylabel("Fake probability", color="white", fontsize=8)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    ax.yaxis.grid(True, color="#1e293b", linewidth=0.7, zorder=0)

    lc = np.array(FAKE_COLOR if label == "FAKE" else REAL_COLOR) / 255
    ax.set_title(
        f"Per-Frame Analysis  ·  Overall: {label}  ({confidence:.1f}%)",
        color=lc, fontsize=9, fontweight="bold"
    )
    ax.legend(fontsize=7, framealpha=0.2, labelcolor="gray")
    plt.tight_layout(pad=0.4)
    return _fig_to_pil(fig)


# ── 3. Mel spectrogram + segment scores (audio) ─────────────────────────────
def render_spectrogram(full_mel: np.ndarray, segment_scores: list,
                        segment_positions: list, label: str,
                        confidence: float, duration: float) -> Image.Image:
    """Mel spectrogram with per-segment fake probability overlay."""
    if not MPL_OK:
        return _placeholder("Spectrogram unavailable")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=False)
    fig.patch.set_facecolor(np.array(BG) / 255)
    for ax in (ax1, ax2):
        ax.set_facecolor(np.array(SURFACE) / 255)

    # Mel spectrogram
    times = np.linspace(0, duration, full_mel.shape[1])
    freqs = np.arange(full_mel.shape[0])
    ax1.pcolormesh(times, freqs, full_mel, cmap="inferno", shading="gouraud")
    ax1.set_ylabel("Mel filterbank", color="white", fontsize=8)
    ax1.tick_params(colors="gray", labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e293b")

    lc = np.array(FAKE_COLOR if label == "FAKE" else REAL_COLOR) / 255
    ax1.set_title(
        f"Log-Mel Spectrogram  ·  {label}  ({confidence:.1f}%)",
        color=lc, fontsize=9, fontweight="bold"
    )

    # Segment score bar
    if segment_scores and segment_positions:
        seg_colors = [
            np.array(FAKE_COLOR if s >= 0.5 else REAL_COLOR) / 255
            for s in segment_scores
        ]
        widths = [2.0] * len(segment_scores)
        ax2.bar(segment_positions, segment_scores,
                width=widths, color=seg_colors, align="edge",
                edgecolor="none", zorder=3)
        ax2.axhline(0.5, color=np.array(WARN_COLOR)/255,
                    linewidth=1, linestyle="--")
        ax2.set_xlim(0, duration)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Time (s)", color="white", fontsize=8)
        ax2.set_ylabel("Fake prob.", color="white", fontsize=7)
        ax2.tick_params(colors="gray", labelsize=7)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#1e293b")
        ax2.yaxis.grid(True, color="#1e293b", linewidth=0.6, zorder=0)

    plt.tight_layout(pad=0.5)
    return _fig_to_pil(fig)


# ── 4. Confidence gauge ──────────────────────────────────────────────────────
def render_gauge(confidence: float, label: str, risk: str) -> Image.Image:
    """Semi-circular gauge showing confidence level."""
    if not MPL_OK:
        return _placeholder("Gauge unavailable")

    fig, ax = plt.subplots(figsize=(4, 2.2),
                            subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(np.array(BG) / 255)
    ax.set_facecolor(np.array(BG) / 255)

    # Only top half
    theta_start = np.pi
    theta_end   = 0.0
    theta_val   = np.pi * (1 - confidence / 100)

    # Background arc
    theta_bg = np.linspace(theta_start, theta_end, 300)
    ax.plot(theta_bg, [1] * 300, color="#1e293b", linewidth=18, solid_capstyle="round")

    # Value arc
    color_map = {
        "HIGH":   np.array(FAKE_COLOR if label == "FAKE" else REAL_COLOR) / 255,
        "MEDIUM": np.array(WARN_COLOR) / 255,
        "LOW":    np.array(ACCENT) / 255,
    }
    arc_color = color_map.get(risk, np.array(ACCENT) / 255)

    theta_val_arr = np.linspace(theta_start, theta_val, 300)
    ax.plot(theta_val_arr, [1] * 300,
            color=arc_color, linewidth=18, solid_capstyle="round")

    ax.set_ylim(0, 1.3)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.axis("off")

    # Labels
    ax.text(np.pi / 2, 0.08, f"{confidence:.0f}%",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="white", transform=ax.transData)
    ax.text(np.pi / 2, -0.35, label,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=arc_color, transform=ax.transData)

    plt.tight_layout(pad=0)
    return _fig_to_pil(fig)


# ── Placeholder ─────────────────────────────────────────────────────────────
def _placeholder(msg: str, w: int = 400, h: int = 200) -> Image.Image:
    img = Image.new("RGB", (w, h), color=SURFACE)
    draw = ImageDraw.Draw(img)
    draw.text((w // 2, h // 2), msg, fill=(100, 116, 139), anchor="mm")
    return img
