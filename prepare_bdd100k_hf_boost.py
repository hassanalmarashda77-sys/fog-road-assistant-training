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

# BDD/FiftyOne label -> Fog Road Assistant class id
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

# Weak classes we want to improve first.
PRIORITY_IDS = [5, 1, 6, 8, 2]
PREFIX = "bddhf_"


def label_value(value) -> str:
    if isinstance(value, dict):
        return str(value.get("label", "")).strip().lower()
    return str(value or "").strip().lower()


def get_detections(sample: dict) -> list[dict]:
    value = sample.get("detections")
    if not isinstance(value, dict):
        return []

    dets = value.get("detections")
    if not isinstance(dets, list):
        return []

    return dets


def parse_sample_boxes(sample: dict):
    boxes = []
    present = set()
    raw_label_counts = Counter()

    for det in get_detections(sample):
        raw_name = str(det.get("label", "")).strip().lower()
        if raw_name:
            raw_label_counts[raw_name] += 1

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

        # FiftyOne Detection bounding_box is normalized [x, y, width, height].
        if w <= 0 or h <= 0:
            continue

        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))

        if w <= 0 or h <= 0:
            continue

        xc = x + w / 2.0
        yc = y + h / 2.0

        boxes.append((cid, xc, yc, w, h))
        present.add(cid)

    return boxes, present, raw_label_counts


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
            "Prepare real fog images from the Hugging Face/FiftyOne "
            "dgural/bdd100k mirror for Fog Road Assistant."
        )
    )
    parser.add_argument(
        "--bdd-root",
        type=Path,
        required=True,
        help="Folder containing samples.json and data/",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("bdd_hf_boost_input"),
    )
    parser.add_argument("--per-class-images", type=int, default=500)
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2028)
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
        print(f"Removed previous BDD-HF workspace files: {removed}")

    raw = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = raw.get("samples") if isinstance(raw, dict) else raw
    if not isinstance(samples, list):
        raise SystemExit("Could not read samples list from samples.json")

    foggy_total = 0
    foggy_with_detections = 0
    candidates = []
    raw_class_counts = Counter()

    for sample in samples:
        weather = label_value(sample.get("weather"))
        if weather != "foggy":
            continue

        foggy_total += 1

        boxes, present, raw_counts = parse_sample_boxes(sample)
        raw_class_counts.update(raw_counts)

        if not boxes:
            continue

        foggy_with_detections += 1

        # Keep only images that help at least one weak class.
        if not (present & set(PRIORITY_IDS)):
            continue

        rel = str(sample.get("filepath", "")).replace("\\", "/")
        image_path = root / Path(rel)

        if not image_path.exists():
            # Fallback to data/<basename>
            image_path = root / "data" / Path(rel).name

        if not image_path.exists():
            continue

        candidates.append((image_path, boxes, present))

    print("=" * 68)
    print("BDD100K HUGGING FACE / FIFTYONE SCAN")
    print("=" * 68)
    print(f"Total samples:                 {len(samples)}")
    print(f"Foggy samples:                 {foggy_total}")
    print(f"Foggy samples with detections: {foggy_with_detections}")
    print(f"Foggy weak-class candidates:   {len(candidates)}")

    if raw_class_counts:
        print("\nFoggy detection label names found:")
        for name, count in raw_class_counts.most_common():
            mapped = FINAL_NAMES[CLASS_MAP[name]] if name in CLASS_MAP else "IGNORED"
            print(f"  {name:18s} {count:6d} -> {mapped}")

    if not candidates:
        print("\nNo usable foggy weak-class images were found.")
        print("Nothing was written.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    selected = []
    selected_stems = set()
    image_presence = Counter()

    # Greedy balancing toward weak classes.
    remaining = candidates[:]

    while remaining and len(selected) < args.max_images:
        best_idx = None
        best_score = -1

        for i, (image_path, _, present) in enumerate(remaining):
            if image_path.stem in selected_stems:
                continue

            score = 0
            for cid in PRIORITY_IDS:
                if cid in present and image_presence[cid] < args.per_class_images:
                    score += 1000 + (args.per_class_images - image_presence[cid])

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None or best_score <= 0:
            break

        record = remaining.pop(best_idx)
        image_path, _, present = record
        selected.append(record)
        selected_stems.add(image_path.stem)

        for cid in PRIORITY_IDS:
            if cid in present:
                image_presence[cid] += 1

        if all(
            image_presence[cid] >= args.per_class_images
            for cid in PRIORITY_IDS
        ):
            break

    # Fill remaining capacity with other unique usable fog images.
    if len(selected) < args.max_images:
        rng.shuffle(remaining)
        for record in remaining:
            image_path, _, _ = record
            if image_path.stem in selected_stems:
                continue
            selected.append(record)
            selected_stems.add(image_path.stem)
            if len(selected) >= args.max_images:
                break

    object_counts = Counter()
    final_presence = Counter()

    for index, (image_path, boxes, present) in enumerate(selected, start=1):
        stem = f"{PREFIX}{index:05d}_{image_path.stem}"
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

    print("\n" + "=" * 68)
    print("BDD100K HF REAL-FOG WORKSPACE REPORT")
    print("=" * 68)
    print(f"BDD root:               {root}")
    print(f"Selected images:        {len(selected)}")
    print(f"Workspace images:       {images_out}")
    print(f"Workspace labels:       {labels_out}")
    print("")
    print("Selected image presence / object counts:")
    for cid in PRIORITY_IDS:
        print(
            f"  {FINAL_NAMES[cid]:12s}: "
            f"{final_presence[cid]:4d} images | {object_counts[cid]:5d} objects"
        )
    print("=" * 68)
    print("NEXT: import this workspace with import_real_boost.py")
    print("Validation/test have not been touched.")


if __name__ == "__main__":
    main()
