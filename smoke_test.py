from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Run a one-epoch GPU/VRAM smoke test before full training."
    )
    parser.add_argument("--data", default="fog_road_dataset/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    model_path = Path(args.model)

    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Fog Road Assistant - 1 Epoch Smoke Test")
    print("=" * 60)
    print(f"Dataset : {data_path}")
    print(f"Model   : {model_path}")
    print(f"imgsz   : {args.imgsz}")
    print(f"batch   : {args.batch}")
    print(f"device  : {device}")

    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING : CUDA is unavailable; this test will use CPU.")

    print("=" * 60)

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=1,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=2,
        project="runs/fog_road",
        name="smoke_test",
        exist_ok=True,
        seed=42,
        deterministic=True,
        plots=False,
    )

    print("\nSMOKE TEST COMPLETE.")
    print("If there was no CUDA out-of-memory error, full training can start.")


if __name__ == "__main__":
    main()
