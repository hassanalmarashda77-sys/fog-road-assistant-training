from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the Fog Road Assistant baseline model after weak-class augmentation."
    )
    parser.add_argument("--data", default="fog_road_dataset/data.yaml")
    parser.add_argument(
        "--weights",
        default="runs/fog_road/yolov8n_clear_fog/weights/best.pt",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()

    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")
    if not weights_path.exists():
        raise SystemExit(f"Baseline best.pt not found: {weights_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    project_dir = (Path.cwd() / "runs" / "fog_road").resolve()

    print("=" * 60)
    print("Fog Road Assistant - Weak-Class Fine Tune")
    print("=" * 60)
    print(f"Dataset : {data_path}")
    print(f"Start   : {weights_path}")
    print(f"Output  : {project_dir}")
    print(f"Device  : {device}")

    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU     : CUDA unavailable; training will use CPU.")

    print("=" * 60)

    model = YOLO(str(weights_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=str(project_dir),
        name="yolov8n_boosted",
        exist_ok=False,
        workers=2,
        seed=77,
        deterministic=True,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.30,
        degrees=2.0,
        translate=0.08,
        scale=0.30,
        fliplr=0.50,
        mosaic=0.60,
        close_mosaic=8,
        plots=True,
    )

    print("\nFine-tuning finished.")
    print(
        f"Use: {project_dir}\\yolov8n_boosted\\weights\\best.pt"
    )


if __name__ == "__main__":
    main()
