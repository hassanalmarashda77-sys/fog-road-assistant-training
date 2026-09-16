from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np


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

NAME_TO_ID = {name: i for i, name in enumerate(FINAL_NAMES)}

DAWN_NAME_MAP = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "person": "person",
    "pedestrian": "person",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAZARD_IDS = {0, 4, 7}


def image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for p in images_dir.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS:
            return p
    return None


def read_yolo_ids(label_path: Path) -> list[int]:
    ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            ids.append(int(float(parts[0])))
        except ValueError:
            pass
    return ids


def count_objects(labels_dir: Path) -> Counter:
    counts: Counter = Counter()
    for label in labels_dir.glob("*.txt"):
        counts.update(read_yolo_ids(label))
    return counts


def count_images(images_dir: Path) -> int:
    return sum(
        1
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_dirs(dataset: Path) -> None:
    for split in ("train", "valid", "test"):
        (dataset / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset / split / "labels").mkdir(parents=True, exist_ok=True)

    for split in ("fog_test", "real_fog_test", "synthetic_hazard_fog_test"):
        (dataset / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset / split / "labels").mkdir(parents=True, exist_ok=True)


def validate_base_dataset(dataset: Path) -> None:
    required = [
        dataset / "train" / "images",
        dataset / "train" / "labels",
        dataset / "valid" / "images",
        dataset / "valid" / "labels",
        dataset / "test" / "images",
        dataset / "test" / "labels",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing expected folders:\n" + "\n".join(missing))


def clean_generated(dataset: Path) -> None:
    prefixes = ("dawn_fog_", "synfog_")

    for split in ("train", "valid"):
        for folder in (dataset / split / "images", dataset / split / "labels"):
            for p in folder.iterdir():
                if p.is_file() and p.name.startswith(prefixes):
                    p.unlink()

    for split in ("fog_test", "real_fog_test", "synthetic_hazard_fog_test"):
        for folder_name in ("images", "labels"):
            folder = dataset / split / folder_name
            for p in folder.iterdir():
                if p.is_file():
                    p.unlink()


def rebalance_validation(dataset: Path, seed: int = 42) -> dict:
    targets = {
        0: 100,  # accident
        4: 80,   # debris
    }

    rng = random.Random(seed)

    train_img = dataset / "train" / "images"
    train_lbl = dataset / "train" / "labels"
    valid_img = dataset / "valid" / "images"
    valid_lbl = dataset / "valid" / "labels"

    current = count_objects(valid_lbl)
    candidates = list(train_lbl.glob("*.txt"))
    rng.shuffle(candidates)

    moved_stems: set[str] = set()

    for class_id, target_count in targets.items():
        if current[class_id] >= target_count:
            continue

        for lbl in candidates:
            if current[class_id] >= target_count:
                break

            if lbl.stem in moved_stems:
                continue

            ids = read_yolo_ids(lbl)
            if class_id not in ids:
                continue

            img = image_for_stem(train_img, lbl.stem)
            if img is None:
                continue

            dest_img = valid_img / img.name
            dest_lbl = valid_lbl / lbl.name

            if dest_img.exists() or dest_lbl.exists():
                continue

            shutil.move(str(img), str(dest_img))
            shutil.move(str(lbl), str(dest_lbl))
            moved_stems.add(lbl.stem)
            current.update(ids)

    return {
        "moved_images": len(moved_stems),
        "valid_accident_objects": current[0],
        "valid_debris_objects": current[4],
    }


def find_fog_folder(dawn_root: Path) -> Path:
    if dawn_root.name.lower() == "fog":
        return dawn_root

    matches = [
        p
        for p in dawn_root.rglob("*")
        if p.is_dir() and p.name.lower() == "fog"
    ]

    if not matches:
        raise SystemExit(
            f"Could not find a folder named 'Fog' under: {dawn_root}\n"
            "Extract DAWN first, then point --dawn to its top folder."
        )

    return matches[0]


def pair_dawn_fog(fog_dir: Path) -> list[tuple[Path, Path]]:
    xml_by_stem = {p.stem: p for p in fog_dir.rglob("*.xml")}
    pairs: list[tuple[Path, Path]] = []

    for img in fog_dir.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
            xml = xml_by_stem.get(img.stem)
            if xml is not None:
                pairs.append((img, xml))

    return pairs


def scan_dawn_class_names(pairs: list[tuple[Path, Path]]) -> Counter:
    names: Counter = Counter()

    for _, xml in pairs:
        try:
            root = ET.parse(xml).getroot()
        except Exception:
            continue

        for obj in root.findall("object"):
            raw_name = (obj.findtext("name") or "").strip().lower()
            if raw_name:
                names[raw_name] += 1

    return names


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")

    if size is None:
        raise ValueError("missing <size>")

    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))

    if width <= 0 or height <= 0:
        raise ValueError("invalid image size")

    boxes = []

    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip().lower()
        mapped = DAWN_NAME_MAP.get(raw_name)

        if mapped is None:
            continue

        b = obj.find("bndbox")
        if b is None:
            continue

        xmin = float(b.findtext("xmin", "0"))
        ymin = float(b.findtext("ymin", "0"))
        xmax = float(b.findtext("xmax", "0"))
        ymax = float(b.findtext("ymax", "0"))

        xmin = max(0.0, min(xmin, width - 1.0))
        ymin = max(0.0, min(ymin, height - 1.0))
        xmax = max(0.0, min(xmax, width))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append((mapped, xmin, ymin, xmax, ymax))

    return width, height, boxes


def voc_to_yolo(name, xmin, ymin, xmax, ymax, width, height):
    class_id = NAME_TO_ID[name]
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{box_width:.6f} "
        f"{box_height:.6f}"
    )


def write_dawn_item(
    img: Path,
    xml: Path,
    split: str,
    dataset: Path,
    index: int,
):
    try:
        width, height, boxes = parse_voc(xml)
    except Exception as exc:
        print(f"[skip] {xml.name}: {exc}")
        return None

    if not boxes:
        return None

    if split == "train":
        targets = [dataset / "train"]
    elif split == "valid":
        targets = [dataset / "valid"]
    elif split == "real_fog_test":
        targets = [dataset / "real_fog_test", dataset / "fog_test"]
    else:
        raise ValueError(f"Unsupported split: {split}")

    suffix = img.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"

    new_stem = f"dawn_fog_{index:04d}_{img.stem}"
    lines = [
        voc_to_yolo(*box, width, height)
        for box in boxes
    ]

    for target in targets:
        img_dir = target / "images"
        lbl_dir = target / "labels"

        dest_img = img_dir / f"{new_stem}{suffix}"
        dest_lbl = lbl_dir / f"{new_stem}.txt"

        shutil.copy2(img, dest_img)
        dest_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "filename": f"{new_stem}{suffix}",
        "split": split,
        "condition": "real_fog",
        "objects": len(lines),
    }


def add_real_fog(dataset: Path, dawn_root: Path, seed: int = 42):
    fog_dir = find_fog_folder(dawn_root)
    pairs = pair_dawn_fog(fog_dir)

    if not pairs:
        raise SystemExit(
            f"Found Fog folder but no matching image/XML pairs under:\n{fog_dir}"
        )

    print(f"DAWN Fog image/XML pairs found: {len(pairs)}")

    class_names = scan_dawn_class_names(pairs)
    print("\nDAWN Fog annotation names:")
    for name, count in class_names.most_common():
        mapped = DAWN_NAME_MAP.get(name)
        status = f"-> {mapped}" if mapped else "(ignored)"
        print(f"  {name:20s} {count:6d} {status}")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    total = len(pairs)
    n_train = int(round(total * 0.70))
    n_valid = int(round(total * 0.15))

    groups = {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_valid],
        "real_fog_test": pairs[n_train + n_valid:],
    }

    rows = []
    index = 1

    for split, items in groups.items():
        written = 0

        for img, xml in items:
            row = write_dawn_item(img, xml, split, dataset, index)
            index += 1

            if row is not None:
                rows.append(row)
                written += 1

        print(f"Real fog written to {split}: {written}")

    return rows


def apply_synthetic_fog(image, severity: str, seed: int):
    rng = np.random.default_rng(seed)
    img = image.astype(np.float32)
    height, width = img.shape[:2]

    strength = {
        "light": 0.18,
        "medium": 0.30,
        "heavy": 0.42,
    }[severity]

    small_h = max(3, height // 80)
    small_w = max(3, width // 80)

    noise = rng.random((small_h, small_w), dtype=np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(
        noise,
        (0, 0),
        sigmaX=max(width, height) / 45.0,
    )
    noise = cv2.normalize(
        noise,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX,
    )

    vertical = np.linspace(
        0.90,
        1.10,
        height,
        dtype=np.float32,
    ).reshape(height, 1)

    density = np.clip(
        (0.70 + 0.30 * noise) * vertical,
        0.0,
        1.0,
    )

    alpha = np.clip(
        strength * density,
        0.0,
        0.65,
    )[..., None]

    fog_color = np.full_like(img, 235.0)
    output = img * (1.0 - alpha) + fog_color * alpha

    sigma = {
        "light": 0.35,
        "medium": 0.65,
        "heavy": 1.00,
    }[severity]

    output = cv2.GaussianBlur(
        output,
        (0, 0),
        sigmaX=sigma,
    )

    return np.clip(output, 0, 255).astype(np.uint8)


def add_synthetic_hazard_train(
    dataset: Path,
    seed: int = 42,
    per_class: int = 300,
):
    train_img = dataset / "train" / "images"
    train_lbl = dataset / "train" / "labels"

    rng = random.Random(seed)
    by_class = {class_id: [] for class_id in HAZARD_IDS}

    for lbl in train_lbl.glob("*.txt"):
        if lbl.name.startswith(("dawn_fog_", "synfog_")):
            continue

        ids = set(read_yolo_ids(lbl))

        for class_id in HAZARD_IDS:
            if class_id in ids:
                by_class[class_id].append(lbl)

    selected = []
    seen: set[str] = set()

    for class_id in sorted(HAZARD_IDS):
        items = by_class[class_id][:]
        rng.shuffle(items)

        for lbl in items[:per_class]:
            if lbl.stem not in seen:
                selected.append(lbl)
                seen.add(lbl.stem)

    severities = ("light", "medium", "heavy")
    rows = []

    for index, lbl in enumerate(selected, start=1):
        img_path = image_for_stem(train_img, lbl.stem)

        if img_path is None:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        severity = severities[(index - 1) % len(severities)]
        fogged = apply_synthetic_fog(
            image,
            severity,
            seed + index,
        )

        new_stem = f"synfog_train_{severity}_{index:04d}_{lbl.stem}"
        out_img = train_img / f"{new_stem}.jpg"
        out_lbl = train_lbl / f"{new_stem}.txt"

        cv2.imwrite(
            str(out_img),
            fogged,
            [int(cv2.IMWRITE_JPEG_QUALITY), 94],
        )
        shutil.copy2(lbl, out_lbl)

        rows.append({
            "filename": out_img.name,
            "split": "train",
            "condition": f"synthetic_{severity}_fog",
            "objects": len(read_yolo_ids(lbl)),
        })

    return rows


def add_synthetic_hazard_test(
    dataset: Path,
    seed: int = 4200,
):
    source_img = dataset / "test" / "images"
    source_lbl = dataset / "test" / "labels"

    target_sets = [
        dataset / "synthetic_hazard_fog_test",
        dataset / "fog_test",
    ]

    severities = ("light", "medium", "heavy")
    rows = []
    selected = []

    for lbl in source_lbl.glob("*.txt"):
        ids = set(read_yolo_ids(lbl))
        if ids & HAZARD_IDS:
            selected.append(lbl)

    selected.sort(key=lambda p: p.name.lower())

    for index, lbl in enumerate(selected, start=1):
        img_path = image_for_stem(source_img, lbl.stem)
        if img_path is None:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        severity = severities[(index - 1) % len(severities)]
        fogged = apply_synthetic_fog(
            image,
            severity,
            seed + index,
        )

        new_stem = f"synfog_test_{severity}_{index:04d}_{lbl.stem}"

        for target in target_sets:
            out_img = target / "images" / f"{new_stem}.jpg"
            out_lbl = target / "labels" / f"{new_stem}.txt"

            cv2.imwrite(
                str(out_img),
                fogged,
                [int(cv2.IMWRITE_JPEG_QUALITY), 94],
            )
            shutil.copy2(lbl, out_lbl)

        rows.append({
            "filename": f"{new_stem}.jpg",
            "split": "synthetic_hazard_fog_test",
            "condition": f"synthetic_{severity}_fog",
            "objects": len(read_yolo_ids(lbl)),
        })

    return rows


def yaml_text(dataset: Path, test_dir: str) -> str:
    root = dataset.resolve().as_posix()
    names_block = "\n".join(
        f"  {index}: {name}"
        for index, name in enumerate(FINAL_NAMES)
    )

    return (
        f"path: {root}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"test: {test_dir}/images\n\n"
        "names:\n"
        f"{names_block}\n"
    )


def write_yamls(dataset: Path) -> None:
    (dataset / "data.yaml").write_text(
        yaml_text(dataset, "test"),
        encoding="utf-8",
    )

    (dataset / "fog_test.yaml").write_text(
        yaml_text(dataset, "fog_test"),
        encoding="utf-8",
    )

    (dataset / "real_fog_test.yaml").write_text(
        yaml_text(dataset, "real_fog_test"),
        encoding="utf-8",
    )

    (dataset / "synthetic_hazard_fog_test.yaml").write_text(
        yaml_text(dataset, "synthetic_hazard_fog_test"),
        encoding="utf-8",
    )


def print_split_report(dataset: Path, split: str) -> None:
    images = dataset / split / "images"
    labels = dataset / split / "labels"

    print(f"\n{split.upper()}")
    print(f"Images: {count_images(images)}")

    counts = count_objects(labels)

    for index, name in enumerate(FINAL_NAMES):
        print(f"{name:12s}: {counts[index]}")


def print_report(dataset: Path) -> None:
    print("\n================ DATASET REPORT ================")

    for split in (
        "train",
        "valid",
        "test",
        "fog_test",
        "real_fog_test",
        "synthetic_hazard_fog_test",
    ):
        print_split_report(dataset, split)

    print("\nEvaluation meaning:")
    print("  test                      = original clear held-out test")
    print("  real_fog_test             = real DAWN fog, compatible classes")
    print("  synthetic_hazard_fog_test = held-out synthetic fog for accident/debris/pothole")
    print("  fog_test                  = combined fog test")
    print("================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Build Fog Road Assistant clear + fog YOLO dataset."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("fog_road_dataset"),
        help="Existing working copy of the thesis YOLO dataset.",
    )

    parser.add_argument(
        "--dawn",
        type=Path,
        required=True,
        help="Top folder of the extracted DAWN dataset.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--synthetic-per-hazard-class",
        type=int,
        default=300,
        help="Maximum TRAIN images per accident/debris/pothole class to fog-augment.",
    )

    args = parser.parse_args()

    dataset = args.dataset.resolve()
    dawn = args.dawn.resolve()

    validate_base_dataset(dataset)
    make_dirs(dataset)

    print(f"Working dataset: {dataset}")
    print(f"DAWN source:     {dawn}")

    print("\nCleaning previously generated fog files only...")
    clean_generated(dataset)

    print("\nImproving weak clear validation coverage...")
    rebalance = rebalance_validation(dataset, args.seed)
    print(rebalance)

    print("\nAdding REAL DAWN fog...")
    real_rows = add_real_fog(
        dataset,
        dawn,
        args.seed,
    )
    print(f"Real fog records added: {len(real_rows)}")

    print("\nAdding SYNTHETIC fog to TRAIN images for accident/debris/pothole...")
    synthetic_train_rows = add_synthetic_hazard_train(
        dataset,
        args.seed,
        args.synthetic_per_hazard_class,
    )
    print(f"Synthetic fog TRAIN images added: {len(synthetic_train_rows)}")

    print("\nBuilding held-out SYNTHETIC fog hazard test...")
    synthetic_test_rows = add_synthetic_hazard_test(
        dataset,
        args.seed + 1000,
    )
    print(f"Synthetic fog TEST images added: {len(synthetic_test_rows)}")

    all_rows = real_rows + synthetic_train_rows + synthetic_test_rows

    csv_path = dataset / "fog_sources.csv"
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "filename",
                "split",
                "condition",
                "objects",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    write_yamls(dataset)
    print_report(dataset)

    print("\nREADY.")
    print(f"Training YAML:        {dataset / 'data.yaml'}")
    print(f"Combined fog test:    {dataset / 'fog_test.yaml'}")
    print(f"Real fog test:        {dataset / 'real_fog_test.yaml'}")
    print(f"Synthetic hazard fog: {dataset / 'synthetic_hazard_fog_test.yaml'}")
    print(f"Source log:           {csv_path}")


if __name__ == "__main__":
    main()
