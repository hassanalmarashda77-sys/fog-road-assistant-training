from __future__ import annotations

import platform
import sys

import cv2
import numpy as np
import torch
import ultralytics


def main():
    print("=" * 60)
    print("Fog Road Assistant - Environment Check")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Platform    : {platform.platform()}")
    print(f"Ultralytics : {ultralytics.__version__}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"OpenCV      : {cv2.__version__}")
    print(f"NumPy       : {np.__version__}")
    print(f"CUDA        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"CUDA count  : {torch.cuda.device_count()}")
    else:
        print("GPU         : No CUDA GPU detected; training will use CPU.")

    print("=" * 60)


if __name__ == "__main__":
    main()
