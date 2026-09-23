from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {
    "bicycle": 1,
    "bike": 1,
    "bus": 2,
    "car": 3,
    "motorbike": 5,
    "motorcycle": 5,
    "person": 6,
    "pedestrian": 6,
}

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

PRIORITY_IDS = [5, 1, 6, 2, 3]
PREFIX = "rtts_"


def find_dir(root: Path, name: str) -> Path | None:
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None


def image_by_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
        p2 = images_dir / f"{stem}{ext.upper()}"
        if p2.exists():
            return p2
    for p in images_dir.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS:
            return p
    return None


def parse_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        return None

    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return None

    boxes = []
    present = set()

    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower()
        if raw_name not in CLASS_MAP:
            continue

        cid = CLASS_MAP[raw_name]
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        try:
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
        except ValueError:
            continue

        xmin = max(0.0, min(width - 1.0, xmin))
        ymin = max(0.0, min(height - 1.0, ymin))
        xmax = max(0.0, min(width, xmax))
        ymax = max(0.0, min(height, ymax))

        if xmax <= xmin or ymax <= ymin:
            continue

        x = ((xmin + xmax) / 2.0) / width
        y = ((ymin + ymax) / 2.0) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        boxes.append((cid, x, y, w, h))
        present.add(cid)

    if not boxes:
        return None

    return boxes, present


def clean_previous(images_out: Path, labels_out: Path) -> int:
    removed = 0
    for folder in (images_out, labels_out):
        for p in folder.iterdir():
            if p.is_file() and p.name.startswith(PREFIX):
                p.unlink()
                removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a balanced subset of RTTS real fog data into the Fog Road Assistant real-image inbox."
    )
    parser.add_argument("--rtts", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("real_boost_input"))
    parser.add_argument("--per-class-images", type=int, default=500)
    parser.add_argument("--max-images", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rtts = args.rtts.resolve()
    workspace = args.workspace.resolve()

    images_out = workspace / "images"
    labels_out = workspace / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    images_dir = find_dir(rtts, "JPEGImages")
    xml_dir = find_dir(rtts, "annotations_xml")

    if images_dir is None:
        raise SystemExit(f"Could not find JPEGImages under: {rtts}")
    if xml_dir is None:
        raise SystemExit(f"Could not find annotations_xml under: {rtts}")

    removed = clean_previous(images_out, labels_out)
    if removed:
        print(f"Removed previous RTTS workspace files: {removed}")

    records = []
    ignored_names = Counter()

    for xml_path in sorted(xml_dir.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue

        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name and name not in CLASS_MAP:
                ignored_names[name] += 1

        parsed = parse_xml(xml_path)
        if parsed is None:
            continue

        image_path = image_by_stem(images_dir, xml_path.stem)
        if image_path is None:
            continue

        boxes, present = parsed
        records.append((image_path, boxes, present))

    print(f"Usable RTTS image/XML pairs: {len(records)}")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    selected = []
    per_class_selected = Counter()

    # Greedy balancing: prioritize weak road-user classes first.
    remaining = records[:]
    while remaining and len(selected) < args.max_images:
        best_index = None
        best_score = -1

        for i, (_, _, present) in enumerate(remaining):
            score = 0
            for cid in PRIORITY_IDS:
                if cid in present and per_class_selected[cid] < args.per_class_images:
                    # Higher reward for under-filled priority classes.
                    score += (args.per_class_images - per_class_selected[cid]) + 100
            if score > best_score:
                best_score = score
                best_index = i

        if best_index is None or best_score <= 0:
            break

        image_path, boxes, present = remaining.pop(best_index)
        selected.append((image_path, boxes, present))
        for cid in present:
            if cid in PRIORITY_IDS:
                per_class_selected[cid] += 1

        if all(per_class_selected[cid] >= args.per_class_images for cid in PRIORITY_IDS):
            break

    # If max_images allows, fill remaining slots with other usable RTTS scenes.
    if len(selected) < args.max_images:
        selected_stems = {p.stem for p, _, _ in selected}
        extras = [r for r in remaining if r[0].stem not in selected_stems]
        rng.shuffle(extras)
        selected.extend(extras[: args.max_images - len(selected)])

    object_counts = Counter()
    image_presence = Counter()

    for idx, (image_path, boxes, present) in enumerate(selected, start=1):
        stem = f"{PREFIX}{idx:05d}_{image_path.stem}"
        out_image = images_out / f"{stem}{image_path.suffix.lower()}"
        out_label = labels_out / f"{stem}.txt"

        shutil.copy2(image_path, out_image)

        lines = []
        for cid, x, y, w, h in boxes:
            lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            object_counts[cid] += 1

        for cid in present:
            image_presence[cid] += 1

        out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 64)
    print("RTTS REAL-FOG WORKSPACE REPORT")
    print("=" * 64)
    print(f"RTTS root:            {rtts}")
    print(f"Selected images:      {len(selected)}")
    print(f"Workspace images:     {images_out}")
    print(f"Workspace labels:     {labels_out}")
    print("")
    print("Selected image presence / object counts:")
    for cid in [1, 2, 3, 5, 6]:
        print(
            f"  {FINAL_NAMES[cid]:12s}: "
            f"{image_presence[cid]:4d} images | {object_counts[cid]:5d} objects"
        )

    if ignored_names:
        print("\nIgnored RTTS annotation names:")
        for name, count in ignored_names.most_common():
            print(f"  {name:18s}: {count}")

    print("=" * 64)
    print("NEXT: run import_real_boost.py")
    print("Validation/test have not been touched.")


if __name__ == "__main__":
    main()
