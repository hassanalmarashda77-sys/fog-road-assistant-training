from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

FINAL_NAMES = [
    "accident",
    "bicycle",
    "bus",
    "car",
    "debris",
    "motorcycle",
    "person",
    "pothole",
    "truck",
]

CLASS_MAP = {
    "bicycle": 1,
    "bike": 1,
    "bus": 2,
    "car": 3,
    "motorcycle": 5,
    "motor": 5,
    "person": 6,
    "pedestrian": 6,
    "truck": 8,
}

PRIORITY_IDS = [5, 1, 6, 8, 2]

# We prefer adverse/low-visibility weather first, while still keeping
# some clear scenes for generalization.
WEATHER_WEIGHT = {
    "foggy": 8,
    "rainy": 6,
    "snowy": 5,
    "overcast": 4,
    "partly cloudy": 3,
    "clear": 1,
}

PREFIX = "bddmix_"


def label_value(value) -> str:
    if isinstance(value, dict):
        return str(value.get("label", "")).strip().lower()
    return str(value or "").strip().lower()


def get_detections(sample: dict) -> list[dict]:
    value = sample.get("detections")
    if not isinstance(value, dict):
        return []
    dets = value.get("detections")
    return dets if isinstance(dets, list) else []


def parse_sample_boxes(sample: dict):
    boxes = []
    present = set()

    for det in get_detections(sample):
        raw_name = str(det.get("label", "")).strip().lower()
        cid = CLASS_MAP.get(raw_name)
        if cid is None:
            continue

        bbox = det.get("bounding_box")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            x, y, w, h = map(float, bbox)
        except (TypeError, ValueError):
            continue

        if w <= 0 or h <= 0:
            continue

        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))

        if w <= 0 or h <= 0:
            continue

        boxes.append((cid, x + w / 2.0, y + h / 2.0, w, h))
        present.add(cid)

    return boxes, present


def clean_previous(images_out: Path, labels_out: Path) -> int:
    removed = 0
    for folder in (images_out, labels_out):
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.is_file() and p.name.startswith(PREFIX):
                p.unlink()
                removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a balanced multi-weather BDD100K/FiftyOne training boost "
            "for the weak Fog Road Assistant classes."
        )
    )
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("bdd_weather_boost_input"),
    )
    parser.add_argument("--per-class-images", type=int, default=600)
    parser.add_argument("--max-images", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2029)
    args = parser.parse_args()

    root = args.bdd_root.resolve()
    samples_path = root / "samples.json"
    workspace = args.workspace.resolve()
    images_out = workspace / "images"
    labels_out = workspace / "labels"

    if not samples_path.exists():
        raise SystemExit(f"samples.json not found: {samples_path}")

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    removed = clean_previous(images_out, labels_out)
    if removed:
        print(f"Removed previous BDD multi-weather files: {removed}")

    raw = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = raw.get("samples") if isinstance(raw, dict) else raw
    if not isinstance(samples, list):
        raise SystemExit("Could not read samples list from samples.json")

    weather_available = Counter()
    candidates = []

    for sample in samples:
        weather = label_value(sample.get("weather"))
        weather_available[weather] += 1

        if weather not in WEATHER_WEIGHT:
            continue

        boxes, present = parse_sample_boxes(sample)
        if not boxes:
            continue

        # Require at least one of the classes we actually want to improve.
        if not (present & set(PRIORITY_IDS)):
            continue

        rel = str(sample.get("filepath", "")).replace("\\", "/")
        image_path = root / Path(rel)
        if not image_path.exists():
            image_path = root / "data" / Path(rel).name
        if not image_path.exists():
            continue

        candidates.append((image_path, boxes, present, weather))

    print("=" * 70)
    print("BDD100K MULTI-WEATHER SCAN")
    print("=" * 70)
    print(f"Total samples:               {len(samples)}")
    print(f"Usable weak-class candidates:{len(candidates)}")
    print("\nWeather counts in dataset:")
    for weather, count in weather_available.most_common():
        print(f"  {weather:16s}: {count}")

    if not candidates:
        print("\nNo usable images found.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    selected = []
    selected_stems = set()
    image_presence = Counter()
    weather_selected = Counter()

    remaining = candidates[:]

    while remaining and len(selected) < args.max_images:
        best_idx = None
        best_score = -1

        for i, (image_path, _, present, weather) in enumerate(remaining):
            if image_path.stem in selected_stems:
                continue

            class_score = 0
            for cid in PRIORITY_IDS:
                if cid in present and image_presence[cid] < args.per_class_images:
                    class_score += 1000 + (args.per_class_images - image_presence[cid])

            if class_score <= 0:
                continue

            # Adverse weather gets a bonus, but class balancing stays dominant.
            score = class_score + WEATHER_WEIGHT.get(weather, 0) * 25

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        record = remaining.pop(best_idx)
        image_path, _, present, weather = record
        selected.append(record)
        selected_stems.add(image_path.stem)
        weather_selected[weather] += 1

        for cid in PRIORITY_IDS:
            if cid in present:
                image_presence[cid] += 1

        if all(
            image_presence[cid] >= args.per_class_images
            for cid in PRIORITY_IDS
        ):
            break

    # Fill any spare capacity with adverse-weather weak-class examples first.
    if len(selected) < args.max_images:
        remaining.sort(
            key=lambda r: WEATHER_WEIGHT.get(r[3], 0),
            reverse=True,
        )
        for record in remaining:
            image_path, _, present, weather = record
            if image_path.stem in selected_stems:
                continue
            selected.append(record)
            selected_stems.add(image_path.stem)
            weather_selected[weather] += 1
            for cid in PRIORITY_IDS:
                if cid in present:
                    image_presence[cid] += 1
            if len(selected) >= args.max_images:
                break

    object_counts = Counter()
    final_presence = Counter()

    for index, (image_path, boxes, present, weather) in enumerate(selected, start=1):
        weather_tag = weather.replace(" ", "_")
        stem = f"{PREFIX}{weather_tag}_{index:05d}_{image_path.stem}"
        out_image = images_out / f"{stem}{image_path.suffix.lower()}"
        out_label = labels_out / f"{stem}.txt"

        shutil.copy2(image_path, out_image)

        lines = []
        for cid, x, y, w, h in boxes:
            lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            object_counts[cid] += 1

        for cid in present:
            final_presence[cid] += 1

        out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print("BDD100K MULTI-WEATHER WORKSPACE REPORT")
    print("=" * 70)
    print(f"Selected images:      {len(selected)}")
    print(f"Workspace images:     {images_out}")
    print(f"Workspace labels:     {labels_out}")

    print("\nSelected weather mix:")
    for weather, count in weather_selected.most_common():
        print(f"  {weather:16s}: {count}")

    print("\nSelected image presence / object counts:")
    for cid in PRIORITY_IDS:
        print(
            f"  {FINAL_NAMES[cid]:12s}: "
            f"{final_presence[cid]:4d} images | {object_counts[cid]:5d} objects"
        )

    print("=" * 70)
    print("NEXT: import this workspace with import_real_boost.py")
    print("Validation/test have not been touched.")


if __name__ == "__main__":
    main()
