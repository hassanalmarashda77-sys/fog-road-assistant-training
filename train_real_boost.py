from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a real-image boosted Fog Road Assistant candidate."
    )
    parser.add_argument("--data", default="fog_road_dataset/data.yaml")
    parser.add_argument(
        "--weights",
        default="runs/fog_road/baseline_best.pt",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()

    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")
    if not weights_path.exists():
        raise SystemExit(f"Starting weights not found: {weights_path}")

    device = 0 if torch.cuda.is_available() else "cpu"
    project_dir = (Path.cwd() / "runs" / "fog_road").resolve()

    print("=" * 60)
    print("Fog Road Assistant - REAL IMAGE BOOST TRAINING")
    print("=" * 60)
    print(f"Dataset : {data_path}")
    print(f"Start   : {weights_path}")
    print(f"Output  : {project_dir}")
    print(f"Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
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
        name="yolov8n_real_boost",
        exist_ok=False,
        workers=2,
        seed=123,
        deterministic=True,
        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.25,
        degrees=2.0,
        translate=0.06,
        scale=0.25,
        fliplr=0.50,
        mosaic=0.35,
        close_mosaic=8,
        plots=True,
    )

    print("\nTraining finished.")
    print(
        f"Candidate: {project_dir}\\yolov8n_real_boost\\weights\\best.pt"
    )
    print("Evaluate this candidate on the SAME clear/fog held-out tests.")
    print("Do not replace model.pt unless the candidate beats the baseline.")


if __name__ == "__main__":
    main()
