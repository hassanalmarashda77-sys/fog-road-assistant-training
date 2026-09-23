from __future__ import annotations

import argparse
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

WEAK_CLASSES = [
    "bicycle",
    "bus",
    "debris",
    "motorcycle",
    "person",
    "truck",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clean inbox for new real labeled training images."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("real_boost_input"),
    )
    args = parser.parse_args()

    root = args.workspace.resolve()
    images = root / "images"
    labels = root / "labels"

    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    readme = root / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Fog Road Assistant - REAL IMAGE BOOST INBOX",
                "",
                "Put REAL images in: images/",
                "Put matching YOLO .txt labels in: labels/",
                "",
                "Image and label filenames must have the same stem.",
                "Example:",
                "  images/road_001.jpg",
                "  labels/road_001.txt",
                "",
                "Final class IDs:",
                *[f"  {i}: {name}" for i, name in enumerate(FINAL_NAMES)],
                "",
                "Weak classes to prioritize:",
                *[f"  - {name}" for name in WEAK_CLASSES],
                "",
                "Use genuinely new scenes. Avoid duplicate frames.",
                "Do not copy validation or test images into this folder.",
            ]
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("REAL IMAGE WORKSPACE READY")
    print("=" * 60)
    print(f"Workspace: {root}")
    print(f"Images:    {images}")
    print(f"Labels:    {labels}")
    print(f"Guide:     {readme}")
    print("=" * 60)


if __name__ == "__main__":
    main()
