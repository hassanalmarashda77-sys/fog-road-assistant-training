from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def evaluate_one(
    model: YOLO,
    yaml_path: Path,
    label: str,
    imgsz: int,
):
    print(f"\n================ {label.upper()} ================")

    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        plots=True,
        verbose=False,
    )

    box = metrics.box
    names = metrics.names

    print(f"Overall mAP50      : {float(box.map50) * 100:.2f}%")
    print(f"Overall mAP50-95   : {float(box.map) * 100:.2f}%")
    print(f"Overall Precision  : {float(box.mp) * 100:.2f}%")
    print(f"Overall Recall     : {float(box.mr) * 100:.2f}%")

    class_ids = list(getattr(box, "ap_class_index", []))
    precision_values = list(getattr(box, "p", []))
    recall_values = list(getattr(box, "r", []))
    ap50_values = list(getattr(box, "ap50", []))
    ap_values = getattr(box, "ap", None)

    rows = []

    print("\nPer-class results")
    print("-" * 82)
    print(
        f"{'Class':14s} "
        f"{'Precision':>11s} "
        f"{'Recall':>11s} "
        f"{'mAP50':>11s} "
        f"{'mAP50-95':>13s}"
    )
    print("-" * 82)

    for pos, class_id in enumerate(class_ids):
        class_id = int(class_id)

        precision = (
            float(precision_values[pos])
            if pos < len(precision_values)
            else 0.0
        )
        recall = (
            float(recall_values[pos])
            if pos < len(recall_values)
            else 0.0
        )
        map50 = (
            float(ap50_values[pos])
            if pos < len(ap50_values)
            else 0.0
        )

        if ap_values is not None and pos < len(ap_values):
            try:
                map50_95 = float(ap_values[pos].mean())
            except Exception:
                map50_95 = 0.0
        else:
            map50_95 = 0.0

        name = class_name(names, class_id)

        print(
            f"{name:14s} "
            f"{precision * 100:10.2f}% "
            f"{recall * 100:10.2f}% "
            f"{map50 * 100:10.2f}% "
            f"{map50_95 * 100:12.2f}%"
        )

        rows.append(
            {
                "condition": label,
                "class_id": class_id,
                "class": name,
                "precision_percent": round(precision * 100, 3),
                "recall_percent": round(recall * 100, 3),
                "mAP50_percent": round(map50 * 100, 3),
                "mAP50_95_percent": round(map50_95 * 100, 3),
            }
        )

    overall = {
        "condition": label,
        "precision_percent": float(box.mp) * 100,
        "recall_percent": float(box.mr) * 100,
        "mAP50_percent": float(box.map50) * 100,
        "mAP50_95_percent": float(box.map) * 100,
    }

    return rows, overall


def write_csv(path: Path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Fog Road Assistant on clear and fog test sets."
    )

    parser.add_argument(
        "--weights",
        default="runs/fog_road/yolov8n_clear_fog/weights/best.pt",
    )
    parser.add_argument(
        "--clear",
        default="fog_road_dataset/data.yaml",
    )
    parser.add_argument(
        "--fog",
        default="fog_road_dataset/fog_test.yaml",
    )
    parser.add_argument(
        "--real-fog",
        default="fog_road_dataset/real_fog_test.yaml",
    )
    parser.add_argument(
        "--synthetic-hazard-fog",
        default="fog_road_dataset/synthetic_hazard_fog_test.yaml",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--out",
        default="fog_evaluation",
    )

    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise SystemExit(f"Model not found: {weights}")

    evaluations = [
        ("clear", Path(args.clear).resolve()),
        ("fog_combined", Path(args.fog).resolve()),
        ("fog_real_dawn", Path(args.real_fog).resolve()),
        (
            "fog_synthetic_hazards",
            Path(args.synthetic_hazard_fog).resolve(),
        ),
    ]

    for label, yaml_path in evaluations:
        if not yaml_path.exists():
            raise SystemExit(f"Dataset YAML not found for {label}: {yaml_path}")

    model = YOLO(str(weights))

    per_class_rows = []
    overall_rows = []

    for label, yaml_path in evaluations:
        rows, overall = evaluate_one(
            model,
            yaml_path,
            label,
            args.imgsz,
        )
        per_class_rows.extend(rows)
        overall_rows.append(overall)

    per_class_csv = out_dir / "per_class_clear_vs_fog.csv"
    overall_csv = out_dir / "overall_clear_vs_fog.csv"

    write_csv(
        per_class_csv,
        per_class_rows,
        [
            "condition",
            "class_id",
            "class",
            "precision_percent",
            "recall_percent",
            "mAP50_percent",
            "mAP50_95_percent",
        ],
    )

    write_csv(
        overall_csv,
        overall_rows,
        [
            "condition",
            "precision_percent",
            "recall_percent",
            "mAP50_percent",
            "mAP50_95_percent",
        ],
    )

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print(f"Per-class CSV: {per_class_csv}")
    print(f"Overall CSV:   {overall_csv}")
    print("")
    print("Use mAP50 (%) as the simple 'out of 100' class score.")
    print("Label it mAP50, not generic accuracy.")
    print("")
    print("fog_real_dawn covers only classes present in DAWN.")
    print("fog_synthetic_hazards covers held-out accident/debris/pothole.")
    print("fog_combined contains both groups.")
    print("=" * 60)


if __name__ == "__main__":
    main()
