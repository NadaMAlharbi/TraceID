"""
Tracker wrapper.

BoT-SORT is the preferred tracker (handles re-entry, occlusion, uses
camera-motion compensation).  We fall back to a simple IoU-based tracker
if boxmot is not installed, so the system still runs.

Tracker IDs are used ONLY for short-term continuity (linking detections
across consecutive frames).  Long-term identity is provided by ReID.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Try boxmot (includes BoT-SORT, ByteTrack, StrongSORT, etc.)
# ---------------------------------------------------------------------------

try:
    from boxmot import BoTSORT
    _BOTSORT_OK = True
except ImportError:
    _BOTSORT_OK = False

try:
    from boxmot import ByteTrack
    _BYTETRACK_OK = True
except ImportError:
    _BYTETRACK_OK = False


# ---------------------------------------------------------------------------
# Simple IoU tracker (zero-dependency fallback)
# ---------------------------------------------------------------------------

class _SimpleIoUTracker:
    """
    Minimal IoU tracker — no external deps.

    Key design choices that keep bboxes stable and aligned:
    - max_age=1  : a track that misses even ONE frame is immediately dropped.
                   This prevents drifted/predicted ghost boxes appearing.
    - Only age-0 tracks are returned — only boxes with a live detection behind
                   them are rendered. Never an extrapolated/predicted position.
    - conf/cls are refreshed from the matched detection every frame.
    - All output boxes are clamped to the frame dimensions.
    """

    def __init__(self, max_age: int = 1, min_hits: int = 1):
        self.max_age  = max_age
        self.min_hits = min_hits
        self._tracks: list[dict] = []
        self._next_id = 1

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        detections : (N, 6) array  [x1, y1, x2, y2, conf, cls]
        returns     : (M, 7) array [x1, y1, x2, y2, track_id, conf, cls]
                      Only tracks matched THIS frame (age==0) are returned.
        """
        fh, fw = frame.shape[:2]

        if len(detections) == 0:
            for t in self._tracks:
                t["age"] += 1
            self._tracks = [t for t in self._tracks if t["age"] <= self.max_age]
            return np.empty((0, 7))

        boxes = detections[:, :4]
        matched_det: set[int] = set()
        matched_trk: set[int] = set()

        # --- Match detections → existing tracks by IoU ---
        for ti, trk in enumerate(self._tracks):
            best_iou, best_di = 0.25, -1
            for di, box in enumerate(boxes):
                if di in matched_det:
                    continue
                iou = self._iou(trk["box"], box)
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_di >= 0:
                det = detections[best_di]
                trk["box"]  = det[:4].copy()   # always use the DETECTION box, not a prediction
                trk["conf"] = float(det[4])
                trk["cls"]  = int(det[5])
                trk["age"]  = 0
                trk["hits"] += 1
                matched_det.add(best_di)
                matched_trk.add(ti)

        # --- Age unmatched existing tracks ---
        for ti, trk in enumerate(self._tracks):
            if ti not in matched_trk:
                trk["age"] += 1

        # --- Create new tracks for unmatched detections ---
        for di, det in enumerate(detections):
            if di not in matched_det:
                self._tracks.append({
                    "id":   self._next_id,
                    "box":  det[:4].copy(),
                    "age":  0,
                    "hits": 1,
                    "conf": float(det[4]),
                    "cls":  int(det[5]),
                })
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t["age"] <= self.max_age]

        # --- Output ONLY age-0 (currently-matched detection) tracks ---
        out = []
        for trk in self._tracks:
            if trk["age"] != 0 or trk["hits"] < self.min_hits:
                continue
            x1, y1, x2, y2 = trk["box"]
            # Clamp to frame so boxes can never exceed image bounds
            x1 = float(max(0.0, min(float(fw - 1), x1)))
            y1 = float(max(0.0, min(float(fh - 1), y1)))
            x2 = float(max(0.0, min(float(fw),     x2)))
            y2 = float(max(0.0, min(float(fh),     y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            out.append([x1, y1, x2, y2, trk["id"], trk["conf"], trk["cls"]])

        return np.array(out, dtype=np.float32) if out else np.empty((0, 7))

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / (ua + 1e-6)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

class TrackerWrapper:
    """
    Unified interface over BoT-SORT / ByteTrack / simple IoU tracker.

    update() always returns (M, 7): [x1,y1,x2,y2,track_id,conf,cls]
    """

    def __init__(self, tracker_type: str = "botsort"):
        self._tracker = None
        self._type = tracker_type.lower()
        self._init_tracker()

    def _init_tracker(self):
        t = self._type
        if t == "botsort" and _BOTSORT_OK:
            self._tracker = BoTSORT(reid_weights=None, device="cpu", half=False)
            print("[Tracker] Using BoT-SORT")
        elif t in ("bytetrack", "botsort") and _BYTETRACK_OK:
            self._tracker = ByteTrack()
            print("[Tracker] Using ByteTrack (BoT-SORT not available)")
        else:
            self._tracker = _SimpleIoUTracker()
            print("[Tracker] Using simple IoU tracker (boxmot not installed)")

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        detections : (N, 6)  [x1, y1, x2, y2, conf, cls]
        returns     : (M, 7) [x1, y1, x2, y2, track_id, conf, cls]
        """
        if isinstance(self._tracker, _SimpleIoUTracker):
            return self._tracker.update(detections, frame)

        # boxmot trackers
        if len(detections) == 0:
            empty = np.empty((0, 6))
            try:
                res = self._tracker.update(empty, frame)
            except Exception:
                res = np.empty((0, 7))
            return res if len(res) else np.empty((0, 7))

        try:
            res = self._tracker.update(detections, frame)
        except Exception as e:
            print(f"[Tracker] update error: {e}")
            return np.empty((0, 7))

        # boxmot returns [x1,y1,x2,y2,id,conf,cls,?] — normalise to 7 cols
        if res is not None and len(res):
            res = np.array(res, dtype=np.float32)
            if res.shape[1] >= 7:
                return res[:, :7]
        return np.empty((0, 7))

    def reset(self):
        self._init_tracker()
