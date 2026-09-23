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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PREFIX = "bddfog_"


def load_frames(label_json: Path) -> list[dict]:
    data = json.loads(label_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("BDD100K detection label JSON must contain a list of frame objects.")
    return data


def find_image(images_dir: Path, name: str) -> Path | None:
    direct = images_dir / name
    if direct.exists():
        return direct

    stem = Path(name).stem
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
        p2 = images_dir / f"{stem}{ext.upper()}"
        if p2.exists():
            return p2

    return None


def frame_to_yolo(frame: dict, image_width: int, image_height: int):
    boxes = []
    present = set()

    for label in frame.get("labels", []) or []:
        category = str(label.get("category", "")).strip().lower()
        cid = CLASS_MAP.get(category)
        if cid is None:
            continue

        box = label.get("box2d")
        if not isinstance(box, dict):
            continue

        try:
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
        except (KeyError, TypeError, ValueError):
            continue

        x1 = max(0.0, min(float(image_width - 1), x1))
        y1 = max(0.0, min(float(image_height - 1), y1))
        x2 = max(0.0, min(float(image_width), x2))
        y2 = max(0.0, min(float(image_height), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        xc = ((x1 + x2) / 2.0) / image_width
        yc = ((y1 + y2) / 2.0) / image_height
        bw = (x2 - x1) / image_width
        bh = (y2 - y1) / image_height

        boxes.append((cid, xc, yc, bw, bh))
        present.add(cid)

    return boxes, present


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare genuinely new BDD100K foggy road images for the "
            "Fog Road Assistant 9-class detector."
        )
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("bdd_boost_input"),
    )
    parser.add_argument("--per-class-images", type=int, default=500)
    parser.add_argument("--max-images", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    images_dir = args.images.resolve()
    label_json = args.labels.resolve()
    workspace = args.workspace.resolve()
    out_images = workspace / "images"
    out_labels = workspace / "labels"

    if not images_dir.exists():
        raise SystemExit(f"BDD image folder not found: {images_dir}")
    if not label_json.exists():
        raise SystemExit(f"BDD label JSON not found: {label_json}")

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for folder in (out_images, out_labels):
        for p in folder.iterdir():
            if p.is_file() and p.name.startswith(PREFIX):
                p.unlink()

    frames = load_frames(label_json)
    candidates = []
    weather_counts = Counter()
    category_counts = Counter()

    # BDD100K object-detection images are 1280x720.
    image_width = 1280
    image_height = 720

    for frame in frames:
        attrs = frame.get("attributes") or {}
        weather = str(attrs.get("weather", "")).strip().lower()
        weather_counts[weather] += 1

        if weather != "foggy":
            continue

        name = str(frame.get("name", "")).strip()
        if not name:
            continue

        image_path = find_image(images_dir, name)
        if image_path is None:
            continue

        boxes, present = frame_to_yolo(frame, image_width, image_height)
        if not boxes:
            continue

        for cid, *_ in boxes:
            category_counts[cid] += 1

        # Exclude foggy frames that contain only car; we want weak-class value.
        if not (present & set(PRIORITY_IDS)):
            continue

        candidates.append((image_path, boxes, present))

    print(f"BDD label frames:              {len(frames)}")
    print(f"Foggy usable weak-class frames:{len(candidates)}")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    selected = []
    selected_stems = set()
    image_presence = Counter()

    # Greedy balancing prioritizes the under-filled weak classes.
    while candidates and len(selected) < args.max_images:
        best_idx = None
        best_score = -1

        for i, (image_path, _, present) in enumerate(candidates):
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

        record = candidates.pop(best_idx)
        image_path, _, present = record
        selected.append(record)
        selected_stems.add(image_path.stem)

        for cid in PRIORITY_IDS:
            if cid in present:
                image_presence[cid] += 1

        if all(image_presence[cid] >= args.per_class_images for cid in PRIORITY_IDS):
            break

    # Fill unused capacity with other unique weak-class fog frames.
    if len(selected) < args.max_images:
        rng.shuffle(candidates)
        for record in candidates:
            image_path, _, _ = record
            if image_path.stem in selected_stems:
                continue
            selected.append(record)
            selected_stems.add(image_path.stem)
            if len(selected) >= args.max_images:
                break

    object_counts = Counter()

    for index, (image_path, boxes, present) in enumerate(selected, start=1):
        stem = f"{PREFIX}{index:05d}_{image_path.stem}"
        out_image = out_images / f"{stem}{image_path.suffix.lower()}"
        out_label = out_labels / f"{stem}.txt"

        shutil.copy2(image_path, out_image)

        lines = []
        for cid, x, y, w, h in boxes:
            lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            object_counts[cid] += 1

        out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 68)
    print("BDD100K REAL FOG BOOST REPORT")
    print("=" * 68)
    print(f"Images folder:          {images_dir}")
    print(f"Labels JSON:            {label_json}")
    print(f"Selected fog images:    {len(selected)}")
    print(f"Workspace images:       {out_images}")
    print(f"Workspace labels:       {out_labels}")
    print("")
    print("Selected image presence / object counts:")
    for cid in PRIORITY_IDS:
        print(
            f"  {FINAL_NAMES[cid]:12s}: "
            f"{image_presence[cid]:4d} images | {object_counts[cid]:5d} objects"
        )
    print("=" * 68)
    print("NEXT: run import_real_boost.py using this workspace.")
    print("Validation/test have not been touched.")


if __name__ == "__main__":
    main()
