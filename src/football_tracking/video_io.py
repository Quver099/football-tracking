"""Video reading/writing helpers.

We stream frames with a generator instead of loading the whole video into a list:
a 30 s 1080p clip at 25 fps is ~750 frames x 6 MB = ~4.5 GB of RAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def video_info(path: str | Path) -> dict:
    """Return basic metadata: width, height, fps, frame count, duration (s)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_s"] = info["frame_count"] / info["fps"] if info["fps"] else 0.0
    return info


def iter_frames(
    path: str | Path, stride: int = 1, max_frames: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame_BGR). `stride=5` keeps every 5th frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    idx, yielded = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def save_frames(
    path: str | Path, out_dir: str | Path, stride: int = 25, max_frames: int | None = 50
) -> list[Path]:
    """Dump every `stride`-th frame as JPG (useful for inspection and later labelling)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    saved = []
    for idx, frame in iter_frames(path, stride=stride, max_frames=max_frames):
        p = out_dir / f"{stem}_{idx:06d}.jpg"
        cv2.imwrite(str(p), frame)
        saved.append(p)
    return saved


class VideoWriter:
    """Thin context manager around cv2.VideoWriter (mp4v codec)."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._w = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._w.isOpened():
            raise RuntimeError(f"Cannot open writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._w.release()
