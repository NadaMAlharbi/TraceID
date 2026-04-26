"""
ReID Pipeline — ties together:
    YOLO detection  →  Tracker (short-term)  →  ReID embedding  →  identity DB

Flow per frame
--------------
1. YOLO detects all objects in the frame.
2. Detections are passed to the tracker (BoT-SORT / fallback) for short-term
   ID assignment and smoothed bounding boxes.
3. For each tracked box that meets the minimum confidence threshold:
      a. Crop the region from the frame.
      b. Extract an appearance embedding via OSNet (or the fallback extractor).
      c. Ask the ReIDEngine to match against known identities.
      d. On first encounter → save identity + fire alert.
         On re-encounter  → reuse existing global ID, no new alert.
4. Return per-frame annotations for the UI.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from core.reid_engine import ReIDEngine, MatchResult
from core.embedding_extractor import EmbeddingExtractor
from core.tracker import TrackerWrapper


# ---------------------------------------------------------------------------
# Annotation record returned for every tracked object each frame
# ---------------------------------------------------------------------------

class DetectionResult:
    __slots__ = ("x1", "y1", "x2", "y2", "track_id",
                 "global_id", "label", "status", "similarity", "confidence")

    def __init__(self, x1, y1, x2, y2, track_id,
                 global_id, label, status, similarity, confidence):
        self.x1, self.y1, self.x2, self.y2 = int(x1), int(y1), int(x2), int(y2)
        self.track_id  = int(track_id)
        self.global_id = global_id
        self.label     = label
        self.status    = status          # "New" | "Seen before"
        self.similarity = similarity
        self.confidence = confidence


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ReIDPipeline:
    """
    Parameters
    ----------
    yolo_model     : Ultralytics YOLO model (already loaded)
    reid_engine    : ReIDEngine instance
    extractor      : EmbeddingExtractor instance
    tracker_type   : "botsort" | "bytetrack" | "simple"
    reid_every_n   : Extract embedding every N frames per track (saves CPU).
    conf_threshold : Minimum YOLO confidence to process a detection.
    target_classes : List of COCO class names to track; None = all.
    """

    def __init__(
        self,
        yolo_model,
        reid_engine: ReIDEngine,
        extractor: EmbeddingExtractor,
        tracker_type: str = "simple",
        reid_every_n: int = 5,
        conf_threshold: float = 0.50,
        target_classes: Optional[list[str]] = ["person"],
    ):
        self.yolo          = yolo_model
        self.reid_engine   = reid_engine
        self.extractor     = extractor
        self.tracker       = TrackerWrapper(tracker_type)
        self.reid_every_n  = reid_every_n
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes

        # track_id → (global_id, frame_last_reid)
        self._track_cache: dict[int, tuple[str, int]] = {}
        self._frame_count = 0

    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> list[DetectionResult]:
        self._frame_count += 1
        results: list[DetectionResult] = []

        # 1. YOLO detection
        detections = self._detect(frame)
        if detections is None or len(detections) == 0:
            return results

        # 2. Tracker update
        tracks = self.tracker.update(detections, frame)
        if tracks is None or len(tracks) == 0:
            return results

        # 3. Per-track ReID
        for row in tracks:
            x1, y1, x2, y2, track_id, conf, cls_id = row
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            cls_id   = int(cls_id)

            # Class name
            label = self._class_name(cls_id)
            if self.target_classes and label not in self.target_classes:
                continue

            # Should we run ReID this frame for this track?
            cache_hit = track_id in self._track_cache
            frames_since_reid = (
                self._frame_count - self._track_cache[track_id][1]
                if cache_hit else 999
            )
            run_reid = (not cache_hit) or (frames_since_reid >= self.reid_every_n)

            if run_reid:
                crop = self._safe_crop(frame, x1, y1, x2, y2)
                if crop is None:
                    continue
                embedding = self.extractor.extract(crop)
                thumbnail = cv2.resize(crop, (64, 128))
                match: MatchResult = self.reid_engine.process(
                    embedding, label, snapshot=thumbnail
                )
                self._track_cache[track_id] = (match.identity.global_id, self._frame_count)
                global_id = match.identity.global_id
                status    = match.status
                sim       = match.similarity
            else:
                global_id, _ = self._track_cache[track_id]
                ident = self.reid_engine.identities.get(global_id)
                status = "Seen before" if ident and ident.times_seen > 1 else "New"
                sim    = 0.0

            results.append(DetectionResult(
                x1=x1, y1=y1, x2=x2, y2=y2,
                track_id=track_id,
                global_id=global_id,
                label=label,
                status=status,
                similarity=sim,
                confidence=float(conf),
            ))

        return results

    def reset(self):
        self._track_cache.clear()
        self._frame_count = 0
        self.tracker.reset()
        self.reid_engine.reset()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run YOLO and return (N,6) array [x1,y1,x2,y2,conf,cls]."""
        try:
            preds = self.yolo(frame, verbose=False)
        except Exception as e:
            print(f"[Pipeline] YOLO error: {e}")
            return None

        rows = []
        for pred in preds:
            boxes = pred.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                rows.append([x1, y1, x2, y2, conf, cls])

        return np.array(rows, dtype=np.float32) if rows else np.empty((0, 6))

    def _class_name(self, cls_id: int) -> str:
        if hasattr(self.yolo, "names") and self.yolo.names:
            return self.yolo.names.get(cls_id, str(cls_id))
        return str(cls_id)

    @staticmethod
    def _safe_crop(frame: np.ndarray, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        return frame[y1:y2, x1:x2].copy()
