from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Train Fog Road Assistant YOLOv8n model."
    )

    parser.add_argument(
        "--data",
        default="fog_road_dataset/data.yaml",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    model_path = Path(args.model)

    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Fog Road Assistant - Training")
    print("=" * 60)
    print(f"Dataset : {data_path}")
    print(f"Base    : {model_path}")
    print(f"Device  : {device}")

    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU     : CUDA not available; training will use CPU.")

    print("=" * 60)

    model = YOLO(str(model_path))

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project="runs/fog_road",
        name="yolov8n_clear_fog",
        exist_ok=False,
        workers=2,
        seed=42,
        deterministic=True,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.40,
        degrees=3.0,
        translate=0.10,
        scale=0.40,
        fliplr=0.50,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
    )

    print("\nTraining finished.")
    print("Use the best.pt created inside runs/fog_road/.../weights/")


if __name__ == "__main__":
    main()
