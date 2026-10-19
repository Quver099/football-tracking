"""Run pretrained YOLO on a clip and write an annotated video.

Usage (from repo root):
    python scripts/run_detection.py --video data/raw/08fd33_0.mp4                       # COCO baseline (M1)
    python scripts/run_detection.py --video data/raw/08fd33_0.mp4 \
        --weights models/football_yolo11s_1280.pt --pitch-filter --out outputs/m2_detect.mp4   # fine-tuned (M2)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_tracking.detection import (  # noqa: E402
    COCO_PERSON, COCO_SPORTS_BALL, Annotator, Detector, summarize,
)
from football_tracking.pitch import keep_on_pitch, pitch_mask  # noqa: E402
from football_tracking.video_io import VideoWriter, iter_frames, video_info  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/m1_detect.mp4")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--pitch-filter", action="store_true", help="drop boxes whose feet are off the grass")
    args = ap.parse_args()

    info = video_info(args.video)
    print(f"Video: {info}")
    is_coco = Path(args.weights).name.startswith("yolo")
    classes = [COCO_PERSON, COCO_SPORTS_BALL] if is_coco else None
    det = Detector(args.weights, conf=args.conf, imgsz=args.imgsz, classes=classes)
    ball_id = COCO_SPORTS_BALL if is_coco else det.class_id("ball")
    print(f"Classes: {det.names if not is_coco else {0: 'person', 32: 'sports ball'}}")
    print(f"Device: {det.device}")
    ann = Annotator()

    all_dets, t0 = [], time.time()
    with VideoWriter(args.out, info["fps"], info["width"], info["height"]) as w:
        for i, frame in iter_frames(args.video, max_frames=args.max_frames):
            d = det.detect(frame)
            if args.pitch_filter:
                d = keep_on_pitch(d, pitch_mask(frame), exempt_classes=[ball_id])
            all_dets.append(d)
            w.write(ann(frame, d))
            if i % 50 == 0:
                print(f"frame {i}: {len(d)} detections")
    dt = time.time() - t0
    print(f"Done: {len(all_dets)} frames in {dt:.1f}s ({len(all_dets) / dt:.1f} FPS)")
    print(summarize(all_dets, ball_id=ball_id))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
