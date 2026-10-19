"""Object detection with YOLO.

Milestone 1: COCO-pretrained weights, keep classes 0 (person) and 32 (sports ball).
Milestone 2: fine-tuned weights (models/football_yolo11s_1280.pt) with classes
ball / goalkeeper / player / referee. Read ids from `detector.names`, never hard-code them.
"""
from __future__ import annotations

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

COCO_PERSON = 0
COCO_SPORTS_BALL = 32


def pick_device() -> str:
    """'cuda' on NVIDIA GPU, 'mps' on Apple Silicon, else 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(
        self,
        weights: str = "yolo11m.pt",
        conf: float = 0.25,
        imgsz: int = 1280,
        classes: list[int] | None = None,
        device: str | None = None,
    ):
        """
        weights : any ultralytics checkpoint; downloaded automatically on first use.
        conf    : minimum confidence to keep a box.
        imgsz   : inference resolution. 1280 (vs default 640) matters a lot for
                  the ball, which is only ~10-20 px wide in broadcast footage.
        classes : class ids to keep. None = all classes of the model.
                  For COCO weights pass [COCO_PERSON, COCO_SPORTS_BALL].
        """
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes
        self.device = device or pick_device()

    @property
    def names(self) -> dict[int, str]:
        return self.model.names

    def class_id(self, name: str) -> int:
        """Look up a class id by name, e.g. detector.class_id("ball")."""
        return next(k for k, v in self.names.items() if v == name)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

    def detect_batch(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        results = self.model.predict(
            frames,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        return [sv.Detections.from_ultralytics(r) for r in results]


class Annotator:
    """Draws boxes + 'class conf' labels on a copy of the frame."""

    def __init__(self):
        self.box = sv.BoxAnnotator(thickness=2)
        self.label = sv.LabelAnnotator(text_scale=0.5, text_padding=3)

    def __call__(self, frame: np.ndarray, det: sv.Detections) -> np.ndarray:
        labels = [
            f"{name} {c:.2f}"
            for name, c in zip(det.data.get("class_name", [""] * len(det)), det.confidence)
        ]
        out = self.box.annotate(frame.copy(), det)
        return self.label.annotate(out, det, labels=labels)


def summarize(dets: list[sv.Detections], ball_id: int = COCO_SPORTS_BALL) -> dict:
    """Per-clip sanity numbers: boxes per frame and how often the ball is seen."""
    n = len(dets)
    balls = [int((d.class_id == ball_id).sum()) for d in dets]
    return {
        "frames": n,
        "avg_boxes_per_frame": float(np.mean([len(d) for d in dets])) if n else 0.0,
        "ball_detected_pct": 100.0 * sum(b > 0 for b in balls) / n if n else 0.0,
        "frames_with_multiple_balls": sum(b > 1 for b in balls),
    }
