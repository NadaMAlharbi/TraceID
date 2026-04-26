"""
Embedding extractor using torchreid's OSNet — the same backbone used in
the badalyaz/object-re-identification repo.

Falls back to a simple color-histogram + HOG descriptor when torchreid is
not available, so the system still runs without a GPU / heavy install.
"""

from __future__ import annotations

import cv2
import numpy as np

# Try to import torchreid for the proper OSNet backbone.
try:
    import torchreid
    import torch
    _TORCHREID_OK = True
except ImportError:
    _TORCHREID_OK = False


class EmbeddingExtractor:
    """
    Extracts a fixed-length appearance descriptor from a BGR crop.

    If torchreid + torch are available the real OSNet model is used.
    Otherwise a lightweight colour-histogram descriptor is returned —
    good enough for a demo with clearly distinct appearances.
    """

    INPUT_H = 256
    INPUT_W = 128

    def __init__(self, model_name: str = "osnet_x0_25", device: str = "cpu"):
        self.device = device
        self.model = None
        self.use_deep = False

        if _TORCHREID_OK:
            try:
                self.model = torchreid.models.build_model(
                    name=model_name,
                    num_classes=1,      # feature extraction only
                    pretrained=True,
                )
                self.model.eval()
                self.model.to(device)
                self.use_deep = True
                print(f"[EmbeddingExtractor] Using OSNet ({model_name}) on {device}")
            except Exception as e:
                print(f"[EmbeddingExtractor] OSNet load failed ({e}); falling back to handcrafted features.")
        else:
            print("[EmbeddingExtractor] torchreid not found; using colour-histogram features.")

    # ------------------------------------------------------------------

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a normalised 1-D float32 embedding for *crop_bgr*."""
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)

        if self.use_deep:
            return self._extract_deep(crop_bgr)
        return self._extract_handcrafted(crop_bgr)

    # ------------------------------------------------------------------
    # Deep path (OSNet via torchreid)
    # ------------------------------------------------------------------

    def _extract_deep(self, crop: np.ndarray) -> np.ndarray:
        import torch
        resized = cv2.resize(crop, (self.INPUT_W, self.INPUT_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        rgb = (rgb - mean) / std
        tensor = torch.tensor(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)

        vec = feat.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    # ------------------------------------------------------------------
    # Fallback path – colour histogram + HOG-like descriptor
    # ------------------------------------------------------------------

    def _extract_handcrafted(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (self.INPUT_W, self.INPUT_H))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        parts = []
        # Upper body (top half) and lower body (bottom half)
        for region in [hsv[:128], hsv[128:]]:
            h_hist = cv2.calcHist([region], [0], None, [32], [0, 180]).flatten()
            s_hist = cv2.calcHist([region], [1], None, [16], [0, 256]).flatten()
            v_hist = cv2.calcHist([region], [2], None, [16], [0, 256]).flatten()
            parts.extend([h_hist, s_hist, v_hist])

        # Simple texture: Laplacian variance per channel
        for ch in cv2.split(resized):
            lap = cv2.Laplacian(ch, cv2.CV_64F)
            texture = np.array([lap.mean(), lap.var()], dtype=np.float32)
            parts.append(texture)

        vec = np.concatenate(parts).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)
