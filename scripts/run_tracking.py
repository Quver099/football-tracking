"""Detect + track + vote classes on a clip, write an annotated video and a tracks CSV.

Usage (from repo root):
    python scripts/run_tracking.py --video data/raw/08fd33_0.mp4 \
        --weights models/football_yolo11s_1280.pt --out outputs/m3_tracked.mp4

Two passes: (1) detect + track every frame and store the table,
(2) re-read the video and draw with the class chosen by majority vote over each whole track.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_tracking.detection import Detector  # noqa: E402
from football_tracking.pitch import keep_on_pitch, pitch_mask  # noqa: E402
from football_tracking.tracking import (  # noqa: E402
    PlayerTracker, TrackAnnotator, TrackerConfig, apply_votes, class_flicker,
    tracks_to_frame, track_summary, vote_classes,
)
from football_tracking.video_io import VideoWriter, iter_frames, video_info  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="models/football_yolo11s_1280.pt")
    ap.add_argument("--out", default="outputs/m3_tracked.mp4")
    ap.add_argument("--csv", default="outputs/m3_tracks.csv")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--lost-buffer", type=int, default=50)
    ap.add_argument("--pitch-filter", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()

    info = video_info(a.video)
    det = Detector(a.weights, conf=a.conf, imgsz=a.imgsz)
    names, ball_id = det.names, det.class_id("ball")
    tracker = PlayerTracker(info["fps"], ball_id, TrackerConfig(lost_buffer=a.lost_buffer))

    per_frame, t0 = [], time.time()
    for i, frame in iter_frames(a.video, max_frames=a.max_frames):
        d = det.detect(frame)
        if a.pitch_filter:
            d = keep_on_pitch(d, pitch_mask(frame), exempt_classes=[ball_id])
        per_frame.append(tracker.update(d))
        if i % 100 == 0:
            print(f"frame {i}")
    print(f"pass 1 done: {len(per_frame)} frames in {time.time() - t0:.1f}s")

    df = tracks_to_frame(per_frame, names)
    votes = vote_classes(df)
    df = apply_votes(df, votes, names)
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.csv, index=False)
    print("summary:", track_summary(df))
    print("class changes per track  before:", int(class_flicker(df, "class_id").class_changes.sum()),
          " after:", int(class_flicker(df, "voted_class_id").class_changes.sum()))

    ann = TrackAnnotator()
    with VideoWriter(a.out, info["fps"], info["width"], info["height"]) as w:
        for i, frame in iter_frames(a.video, max_frames=a.max_frames):
            d = per_frame[i]
            cls = np.array([votes.get(int(t), int(c)) if t != -1 else int(c)
                            for t, c in zip(d.tracker_id, d.class_id)]) if len(d) else d.class_id
            w.write(ann(frame, d, names, class_ids=cls))
    print(f"Saved -> {a.out}  and  {a.csv}")


if __name__ == "__main__":
    main()
