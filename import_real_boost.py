from __future__ import annotations

import argparse
import hashlib
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

WEAK_IDS = {1, 2, 4, 5, 6, 8}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PREFIX = "realboost_"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_label(path: Path) -> tuple[list[str], Counter[int]]:
    lines = []
    counts: Counter[int] = Counter()

    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8", errors="ignore").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path.name}:{line_no} expected 5 YOLO values")

        try:
            cid = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_no} invalid numeric value") from exc

        if cid < 0 or cid >= len(FINAL_NAMES):
            raise ValueError(
                f"{path.name}:{line_no} class id {cid} is outside 0..8"
            )

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{path.name}:{line_no} center must be in 0..1")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise ValueError(f"{path.name}:{line_no} width/height must be in 0..1")

        lines.append(line)
        counts[cid] += 1

    if not lines:
        raise ValueError(f"{path.name} contains no boxes")

    return lines, counts


def build_existing_hashes(dataset: Path) -> set[str]:
    hashes: set[str] = set()
    for split in ("train", "valid", "test"):
        images = dataset / split / "images"
        if not images.exists():
            continue
        for p in images.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    hashes.add(digest(p))
                except OSError:
                    pass
    return hashes


def clean_previous(train_images: Path, train_labels: Path) -> int:
    removed = 0
    for folder in (train_images, train_labels):
        for p in folder.iterdir():
            if p.is_file() and p.name.startswith(PREFIX):
                p.unlink()
                removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and import genuinely new labeled real images into TRAIN only. "
            "Validation and test sets are never changed."
        )
    )
    parser.add_argument("--dataset", type=Path, default=Path("fog_road_dataset"))
    parser.add_argument("--source", type=Path, default=Path("real_boost_input"))
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Remove earlier realboost_ training imports before importing this batch.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    source = args.source.resolve()

    train_images = dataset / "train" / "images"
    train_labels = dataset / "train" / "labels"
    source_images = source / "images"
    source_labels = source / "labels"

    for required in (train_images, train_labels, source_images, source_labels):
        if not required.exists():
            raise SystemExit(f"Folder not found: {required}")

    if args.replace:
        removed = clean_previous(train_images, train_labels)
        print(f"Removed previous realboost files: {removed}")

    existing_hashes = build_existing_hashes(dataset)
    batch_hashes: set[str] = set()

    source_files = [
        p for p in source_images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    imported = 0
    duplicate = 0
    missing_label = 0
    skipped_no_weak = 0
    invalid = 0
    object_counts: Counter[int] = Counter()

    for image_path in sorted(source_files):
        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"[SKIP missing label] {image_path.name}")
            missing_label += 1
            continue

        try:
            _, counts = parse_label(label_path)
        except ValueError as exc:
            print(f"[SKIP invalid label] {exc}")
            invalid += 1
            continue

        if not (set(counts) & WEAK_IDS):
            skipped_no_weak += 1
            continue

        image_hash = digest(image_path)
        if image_hash in existing_hashes or image_hash in batch_hashes:
            print(f"[SKIP duplicate] {image_path.name}")
            duplicate += 1
            continue

        new_stem = f"{PREFIX}{imported + 1:05d}_{image_path.stem}"
        out_image = train_images / f"{new_stem}{image_path.suffix.lower()}"
        out_label = train_labels / f"{new_stem}.txt"

        shutil.copy2(image_path, out_image)
        shutil.copy2(label_path, out_label)

        batch_hashes.add(image_hash)
        object_counts.update(counts)
        imported += 1

    cache = dataset / "train" / "labels.cache"
    if cache.exists():
        cache.unlink()

    print("\n" + "=" * 60)
    print("REAL IMAGE IMPORT REPORT")
    print("=" * 60)
    print(f"Source images:            {len(source_files)}")
    print(f"Imported to TRAIN:        {imported}")
    print(f"Duplicates skipped:       {duplicate}")
    print(f"Missing labels skipped:   {missing_label}")
    print(f"Invalid labels skipped:   {invalid}")
    print(f"No weak class skipped:    {skipped_no_weak}")
    print("")
    print("Imported object counts:")
    for cid, name in enumerate(FINAL_NAMES):
        print(f"  {name:12s}: {object_counts[cid]}")
    print("=" * 60)
    print("Validation/test were NOT changed.")


if __name__ == "__main__":
    main()
