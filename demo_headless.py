#!/usr/bin/env python3
"""
Headless demo — runs the ReID pipeline without any UI.

Useful for testing on a server or CI, or for processing a video file
and writing an annotated output.

Usage:
    python demo_headless.py --source 0
    python demo_headless.py --source path/to/video.mp4 --output out.mp4
    python demo_headless.py --source path/to/video.mp4 --model yolov8s.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.reid_engine import ReIDEngine
from core.embedding_extractor import EmbeddingExtractor
from core.pipeline import ReIDPipeline


PALETTE_BGR = [
    (46, 117, 231), (46, 204, 113), (231, 76, 60), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (52, 73, 94),
]


def color_for(gid: str):
    idx = int(gid.split("-")[1], 16) % len(PALETTE_BGR)
    return PALETTE_BGR[idx]


def annotate(frame, detections):
    for d in detections:
        c = color_for(d.global_id)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), c, 2)
        label = f"{d.label} [{d.global_id}] T#{d.track_id} {d.status}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(d.y1 - 4, th + 4)
        cv2.rectangle(frame, (d.x1, y0 - th - 4), (d.x1 + tw + 4, y0), c, -1)
        cv2.putText(frame, label, (d.x1 + 2, y0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def main():
    ap = argparse.ArgumentParser(description="ReID headless demo")
    ap.add_argument("--source", default="0", help="Camera index or video path")
    ap.add_argument("--model",  default="yolov8n.pt", help="YOLO model file")
    ap.add_argument("--output", default=None, help="Optional output video path")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="ReID similarity threshold (0–1)")
    ap.add_argument("--show", action="store_true",
                    help="Show live window (requires display)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (0 = unlimited)")
    args = ap.parse_args()

    # Load YOLO
    try:
        from ultralytics import YOLO
        yolo = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] Cannot load YOLO model '{args.model}': {e}")
        sys.exit(1)

    # Open source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {src}")
        sys.exit(1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Build pipeline
    reid_engine = ReIDEngine(similarity_threshold=args.threshold)
    extractor   = EmbeddingExtractor()
    pipeline    = ReIDPipeline(yolo, reid_engine, extractor)

    # Output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_src, (fw, fh))

    print(f"[INFO] Processing {'camera ' + args.source if args.source.isdigit() else args.source}")
    print(f"[INFO] ReID threshold: {args.threshold}  |  YOLO: {args.model}")
    print("[INFO] Press q to quit (if --show is set)")

    frame_num = 0
    t_start   = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = pipeline.process_frame(frame)
            annotate(frame, detections)

            # FPS
            elapsed = time.time() - t_start + 1e-9
            fps = frame_num / elapsed
            cv2.putText(frame, f"FPS:{fps:.1f}  IDs:{len(reid_engine.identities)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(frame)

            if args.show:
                cv2.imshow("ReID System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_num += 1
            if args.max_frames and frame_num >= args.max_frames:
                break

            # Log alerts when they arrive
            for alert in reid_engine.get_alerts():
                import datetime
                ts = datetime.datetime.fromtimestamp(alert["time"]).strftime("%H:%M:%S")
                # Only print each alert once
                aid = alert["global_id"]

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    # Summary
    print("\n" + "═" * 50)
    print(f"  Frames processed : {frame_num}")
    print(f"  Unique identities: {len(reid_engine.identities)}")
    print(f"  Alerts fired     : {len(reid_engine.alerts)}")
    print("═" * 50)
    print("\nIdentities:")
    for ident in reid_engine.get_identities():
        print(f"  {ident.global_id}  [{ident.label}]  seen {ident.times_seen}×")
    print("\nAlerts:")
    for a in reid_engine.get_alerts():
        import datetime
        ts = datetime.datetime.fromtimestamp(a["time"]).strftime("%H:%M:%S")
        print(f"  [{ts}] {a['message']}")


if __name__ == "__main__":
    main()
