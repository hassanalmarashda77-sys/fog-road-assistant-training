from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


TARGET_CLASSES = {
    1: "bicycle",
    2: "bus",
    4: "debris",
    5: "motorcycle",
    6: "person",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PREFIX = "weakboost_"


def image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for p in images_dir.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS:
            return p
    return None


def read_ids(label_path: Path) -> set[int]:
    ids: set[int] = set()
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            ids.add(int(float(parts[0])))
        except ValueError:
            pass
    return ids


def clean_previous(train_images: Path, train_labels: Path) -> int:
    removed = 0
    for folder in (train_images, train_labels):
        for p in folder.iterdir():
            if p.is_file() and p.name.startswith(PREFIX):
                p.unlink()
                removed += 1
    return removed


def fog(image: np.ndarray, strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    small_h = max(3, h // 70)
    small_w = max(3, w // 70)
    noise = rng.random((small_h, small_w), dtype=np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(h, w) / 50.0)
    noise = cv2.normalize(noise, None, 0.0, 1.0, cv2.NORM_MINMAX)

    alpha = np.clip(strength * (0.75 + 0.25 * noise), 0.0, 0.55)[..., None]
    fog_color = np.full_like(img, 235.0)
    out = img * (1.0 - alpha) + fog_color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def low_contrast(image: np.ndarray) -> np.ndarray:
    out = cv2.convertScaleAbs(image, alpha=0.72, beta=28)
    return cv2.GaussianBlur(out, (0, 0), sigmaX=0.45)


def dim_haze(image: np.ndarray) -> np.ndarray:
    out = cv2.convertScaleAbs(image, alpha=0.78, beta=10)
    overlay = np.full_like(out, 210)
    return cv2.addWeighted(out, 0.84, overlay, 0.16, 0)


def augment(image: np.ndarray, mode: int, seed: int) -> np.ndarray:
    if mode == 0:
        return fog(image, 0.20, seed)
    if mode == 1:
        return fog(image, 0.32, seed)
    if mode == 2:
        return low_contrast(image)
    return dim_haze(image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add targeted training-only augmentation for weak Fog Road Assistant classes."
    )
    parser.add_argument("--dataset", type=Path, default=Path("fog_road_dataset"))
    parser.add_argument("--per-class", type=int, default=180)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    train_images = dataset / "train" / "images"
    train_labels = dataset / "train" / "labels"

    if not train_images.exists() or not train_labels.exists():
        raise SystemExit(f"Training folders not found under: {dataset}")

    removed = clean_previous(train_images, train_labels)
    if removed:
        print(f"Removed previous weakboost files: {removed}")

    rng = random.Random(args.seed)
    labels = [
        p for p in train_labels.glob("*.txt")
        if not p.name.startswith(("weakboost_", "synfog_", "dawn_fog_"))
    ]

    by_class: dict[int, list[Path]] = {cid: [] for cid in TARGET_CLASSES}
    for label in labels:
        ids = read_ids(label)
        for cid in TARGET_CLASSES:
            if cid in ids:
                by_class[cid].append(label)

    selected: list[tuple[Path, int]] = []
    used_stems: set[str] = set()

    print("Available source images by target class:")
    for cid, name in TARGET_CLASSES.items():
        items = by_class[cid][:]
        rng.shuffle(items)
        print(f"  {name:12s}: {len(items)}")

        chosen = 0
        for label in items:
            if chosen >= args.per_class:
                break
            if label.stem in used_stems:
                continue
            selected.append((label, cid))
            used_stems.add(label.stem)
            chosen += 1

    created = 0
    per_target_created = {cid: 0 for cid in TARGET_CLASSES}

    for index, (label, target_cid) in enumerate(selected, start=1):
        image_path = image_for_stem(train_images, label.stem)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        mode = (index - 1) % 4
        augmented = augment(image, mode, args.seed + index)

        new_stem = f"{PREFIX}{TARGET_CLASSES[target_cid]}_{index:04d}_{label.stem}"
        out_image = train_images / f"{new_stem}.jpg"
        out_label = train_labels / f"{new_stem}.txt"

        ok = cv2.imwrite(
            str(out_image),
            augmented,
            [int(cv2.IMWRITE_JPEG_QUALITY), 94],
        )
        if not ok:
            continue

        shutil.copy2(label, out_label)
        created += 1
        per_target_created[target_cid] += 1

    print("\nCreated targeted training augmentations:")
    for cid, name in TARGET_CLASSES.items():
        print(f"  {name:12s}: {per_target_created[cid]}")

    print(f"\nTotal new TRAIN images: {created}")
    print("Validation and test folders were NOT changed.")


if __name__ == "__main__":
    main()
