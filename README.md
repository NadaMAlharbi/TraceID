# TraceID — Object & Person Re-Identification System

A complete, ready-to-run system for **long-term identity persistence** of
people and objects across camera views.  Built on top of the
[badalyaz/object-re-identification](https://github.com/badalyaz/object-re-identification)
project which demonstrated OSNet-based ReID with L2-distance matching.

---

## Features

- Real-time object detection using YOLO
- Long-term identity tracking using OSNet ReID
- Duplicate alert prevention
- GUI interface with live video
- Headless mode for CLI processing
- Supports camera or video input


## Why three components?

| Component | What it does | Why it's not enough alone |
|-----------|-------------|--------------------------|
| **YOLO** | Detects objects each frame | Gives no identity — every frame is a fresh detection with no link to history |
| **Tracker (BoT-SORT)** | Links detections across consecutive frames via motion + appearance | Track IDs **reset** when a person leaves the frame (e.g. exits elevator). Can't recognise a re-entry minutes later |
| **ReID (OSNet)** | Extracts a compact appearance vector that is stable across appearances | Enables long-term matching even after complete occlusion or scene exit |

The three work together:

```
Camera frame
    │
    ▼
YOLO detection  ──→  (N detections per frame)
    │
    ▼
BoT-SORT tracker  ──→  short-term continuity, smooth boxes (track_id)
    │
    ▼
OSNet embedding  ──→  128-D / 512-D appearance descriptor
    │
    ▼
Identity DB  ──→  cosine similarity against stored embeddings
    │
    ├── Match found  →  "Seen before"  (no alert)
    └── No match     →  New identity   (alert fired once)
```

---

## How duplicate alerts are prevented

Every `Identity` object carries an `alerted = True` flag that is set at
creation time.  The `ReIDEngine._fire_alert()` method is called **only**
inside `_create_identity()`.  Subsequent `process()` calls that match an
existing identity take the "Known identity" branch and never call
`_fire_alert()` again.

---

## Project structure

```
reid_system/
├── main.py                   ← Entry point (full GUI)
├── demo_headless.py          ← CLI demo (no display needed)
├── requirements.txt
│
├── core/
│   ├── reid_engine.py        ← Identity DB + alert logic
│   ├── embedding_extractor.py← OSNet (torchreid) or colour-histogram fallback
│   ├── tracker.py            ← BoT-SORT / ByteTrack / IoU tracker wrapper
│   └── pipeline.py           ← Orchestrates all components per frame
│
├── ui/
│   └── app.py                ← Tkinter GUI
│
└── tests/
    └── test_reid_engine.py   ← Unit tests
```

---

## Installation

### 1. Clone / unzip

```bash
unzip reid_system.zip -d reid_system
cd reid_system
```

### 2. Install base requirements

```bash
pip install ultralytics opencv-python Pillow numpy
```

YOLO weights are downloaded automatically on first run.

### 3. (Recommended) Install BoT-SORT tracker

```bash
pip install boxmot
```

Without this the system falls back to a simple IoU tracker which works but
loses track more often during occlusion.

### 4. (Recommended) Install OSNet ReID backbone

```bash
# CPU-only torch (smaller download):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torchreid
```

Without this the system falls back to a colour-histogram descriptor which
works for visually distinct people but is less robust.

---

## Usage

### GUI (full interface)

```bash
python main.py
```

Controls in the toolbar:
- **▶ Start** — load model and begin processing
- **■ Stop** — pause processing
- **↺ Reset Identities** — clear all known identities and alert history
- **YOLO Model** — choose model size (n = fastest, m = most accurate)
- **Source** — camera index (0, 1, …) or path to a video file; click **Browse…**
- **ReID Threshold** — cosine similarity cutoff (0.65 is a good default)
  - Lower → stricter matching (fewer false "Seen before")
  - Higher → looser matching (fewer false "New")

### Headless / CLI

```bash
# Webcam
python demo_headless.py --source 0 --show

# Video file, write output
python demo_headless.py --source video.mp4 --output out.mp4

# Custom model and threshold
python demo_headless.py --source 0 --model yolov8s.pt --threshold 0.70 --show
```

---

## Tuning tips

| Scenario | Recommendation |
|----------|---------------|
| Elevator scenario | Threshold 0.60–0.70 |
| Very similar-looking people | Lower threshold (0.55) + torchreid OSNet |
| Many different object classes | Set `target_classes` in `pipeline.py` |
| Slow machine / CPU only | Use `yolov8n.pt`, `osnet_x0_25`, `reid_every_n=10` |
| High accuracy needed | `yolov8m.pt`, `osnet_x1_0`, GPU |

---

## How the similarity threshold works

Each known identity stores a **unit-norm** embedding `e_stored`.  When a
new crop is seen with embedding `e_new`, the system computes:

```
similarity = dot(e_new, e_stored)    ∈ [−1, 1]
```

If `similarity ≥ threshold`, the identity is considered the **same person**.
The `Identity.embedding` is updated with an exponential moving average
(`α = 0.3`) to adapt to lighting and pose changes over time.

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Extending to other object types

The pipeline is object-class-aware by default.  ReID matching only compares
embeddings **within the same YOLO class**, so a person and a suitcase will
never be falsely matched.  To track custom classes:

```python
pipeline = ReIDPipeline(
    yolo_model=yolo,
    reid_engine=reid_engine,
    extractor=extractor,
    target_classes=["person", "suitcase", "backpack"],
)
```

---

## Relation to badalyaz/object-re-identification

| Original repo | This project |
|---------------|-------------|
| Jupyter notebook demonstrations | Full runnable Python package |
| OSNet + L2 distance matching | OSNet + **cosine** similarity (faster, scale-invariant) |
| No YOLO integration | YOLO v8/v11 for detection |
| No tracker | BoT-SORT / ByteTrack / IoU fallback |
| No GUI | Clean Tkinter GUI with live video, identity panel, alerts |
| No alert logic | First-encounter alerts with duplicate prevention |

The OSNet backbone and the concept of extracting embedding vectors from
person crops is directly inspired by the original repo.

---

## Author
Nada Alharbi
