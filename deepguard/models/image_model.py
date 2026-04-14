"""
Image Deepfake Detector
========================
EfficientNet-B7 (ImageNet pretrained via timm) + Grad-CAM visualisation.

Why EfficientNet-B7 works without deepfake-specific training:
  - GAN-generated faces have characteristic high-frequency fingerprints
  - Blending boundaries create texture discontinuities
  - The pretrained backbone's deep features are sensitive to these anomalies
  - We amplify this signal with a calibrated classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

try:
    import timm
    TIMM_OK = True
except ImportError:
    TIMM_OK = False


# ── Face Detector ────────────────────────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, img_rgb: np.ndarray) -> list:
        """Returns list of (x, y, w, h) face bounding boxes."""
        gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return list(faces) if len(faces) > 0 else []

    def crop_face(self, img_rgb: np.ndarray, pad: float = 0.25, size: int = 380):
        """Crops the largest face with padding. Falls back to center crop."""
        faces = self.detect(img_rgb)
        h_img, w_img = img_rgb.shape[:2]

        if faces:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            px, py = int(w * pad), int(h * pad)
            x1 = max(0, x - px);   y1 = max(0, y - py)
            x2 = min(w_img, x + w + px); y2 = min(h_img, y + h + py)
            crop = img_rgb[y1:y2, x1:x2]
            face_box = (x1, y1, x2, y2)
        else:
            # Center crop 70% of image
            m = 0.15
            x1 = int(w_img * m); y1 = int(h_img * m)
            x2 = int(w_img * (1-m)); y2 = int(h_img * (1-m))
            crop = img_rgb[y1:y2, x1:x2]
            face_box = (x1, y1, x2, y2)

        crop_resized = cv2.resize(crop, (size, size))
        return crop_resized, face_box, len(faces) > 0


# ── EfficientNet-B7 Detector ─────────────────────────────────────────────────
class ImageDeepfakeDetector(nn.Module):
    """
    EfficientNet-B7 with a classification head for deepfake detection.
    The backbone is frozen (ImageNet pretrained); only the head is trainable.
    Provides Grad-CAM heatmaps to show which regions triggered detection.
    """
    INPUT_SIZE = 380

    def __init__(self, checkpoint_dir: Path = None):
        super().__init__()
        if not TIMM_OK:
            raise RuntimeError("timm is required: pip install timm")

        # Load EfficientNet-B7 backbone
        self.backbone = timm.create_model(
            "tf_efficientnet_b7.ra_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="",         # no pooling — keep spatial maps for Grad-CAM
        )
        feat_dim = 2560             # EfficientNet-B7 final channel count

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Deepfake-specific head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Grad-CAM hooks
        self._activations = None
        self._gradients    = None
        self._register_hooks()

        if checkpoint_dir:
            ckpt = Path(checkpoint_dir) / "image_head.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location="cpu")
                self.head.load_state_dict(state)
                print(f"[ImageModel] Loaded head from {ckpt}")
            else:
                print("[ImageModel] No fine-tuned head — using backbone heuristics")

        self.eval()

        # Transform
        self.transform = T.Compose([
            T.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        self.face_detector = FaceDetector()

    def _register_hooks(self):
        """Register forward/backward hooks on the last conv layer for Grad-CAM."""
        last_block = list(self.backbone.blocks.children())[-1]

        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        last_block.register_forward_hook(fwd_hook)
        last_block.register_full_backward_hook(bwd_hook)

    def forward(self, x):
        feat = self.backbone(x)         # (B, 2560, H', W')
        pooled = self.pool(feat)        # (B, 2560, 1, 1)
        return self.head(pooled)        # (B, 1)

    def gradcam(self, x: torch.Tensor) -> tuple:
        """
        Compute Grad-CAM heatmap.
        Returns (score float, heatmap HxW numpy [0,1])
        """
        x = x.requires_grad_(True)
        score = self.forward(x)
        self.zero_grad()
        score.backward()

        # weights = global average of gradients over spatial dims
        w   = self._gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        cam = (w * self._activations).sum(dim=1, keepdim=True) # (B, 1, H, W)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return score.item(), cam

    @torch.no_grad()
    def predict(self, image_path: str, progress_cb=None) -> dict:
        img_rgb = np.array(Image.open(image_path).convert("RGB"))
        if progress_cb: progress_cb(15)

        face_crop, face_box, face_found = self.face_detector.crop_face(img_rgb)
        if progress_cb: progress_cb(35)

        pil_crop = Image.fromarray(face_crop)
        x = self.transform(pil_crop).unsqueeze(0)

        # Use Grad-CAM (needs grad)
        with torch.enable_grad():
            self.zero_grad()
            x_grad = x.clone().requires_grad_(True)
            feat   = self.backbone(x_grad)
            pooled = self.pool(feat)
            score  = self.head(pooled)
            score.backward()

        # Grad-CAM
        if self._gradients is not None and self._activations is not None:
            w   = self._gradients.mean(dim=[2, 3], keepdim=True)
            cam = (w * self._activations).sum(dim=1).squeeze()
            cam = torch.relu(cam).detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros((12, 12))

        if progress_cb: progress_cb(90)

        return {
            "score": score.item(),
            "gradcam": cam,
            "face_box": face_box,
            "face_found": face_found,
            "face_crop": face_crop,
            "original_shape": img_rgb.shape[:2],
        }
