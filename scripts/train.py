"""Fine-tune YOLO on the football-players dataset.

Usage (from repo root):
    python scripts/train.py --data data/datasets/football-players/data_fixed.yaml
    python scripts/train.py --data ... --model yolo11m.pt --batch 4 --epochs 80

data_fixed.yaml is written by notebooks/02_finetune_yolo.ipynb (absolute paths).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = AutoBatch (fits VRAM)")
    ap.add_argument("--name", default="players_y11s_1280")
    ap.add_argument("--out", default="models/football_yolo11s_1280.pt")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    workers = 0 if os.name == "nt" else 4

    model = YOLO(args.model)
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience,
        batch=args.batch, device=device, workers=workers, cache=True,
        project=str(ROOT / "runs"), name=args.name, exist_ok=True, seed=42,
    )
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    print(f"best weights -> {out}")

    metrics = YOLO(str(out)).val(data=args.data, split="test", imgsz=args.imgsz, device=device,
                                 workers=workers, verbose=False)
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        print(f"{metrics.names[cls_idx]:12s} mAP50={metrics.box.ap50[i]:.3f}  mAP50-95={metrics.box.ap[i]:.3f}")


if __name__ == "__main__":
    main()
