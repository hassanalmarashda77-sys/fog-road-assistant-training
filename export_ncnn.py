from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Export the trained Fog Road Assistant model to NCNN."
    )

    parser.add_argument(
        "--weights",
        default="runs/fog_road/yolov8n_clear_fog/weights/best.pt",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    args = parser.parse_args()

    weights = Path(args.weights).resolve()

    if not weights.exists():
        raise SystemExit(f"Model not found: {weights}")

    model = YOLO(str(weights))

    result = model.export(
        format="ncnn",
        imgsz=args.imgsz,
        half=False,
        int8=False,
    )

    print("\nNCNN export complete:")
    print(result)


if __name__ == "__main__":
    main()
