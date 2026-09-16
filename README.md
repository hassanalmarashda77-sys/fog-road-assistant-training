# Fog Road Assistant — Training Toolkit

This repository prepares, trains, evaluates, and exports the custom object-detection model for the **Fog Road Assistant** Raspberry Pi road-safety prototype.

## Final 9 classes

| ID | Class |
|---:|---|
| 0 | accident |
| 1 | bicycle |
| 2 | bus |
| 3 | car |
| 4 | debris |
| 5 | motorcycle |
| 6 | person |
| 7 | pothole |
| 8 | truck |

Your original thesis dataset spells class 5 as `motocycle`. The numeric label ID is already 5, so the toolkit keeps the YOLO label files unchanged and corrects only the displayed class name to `motorcycle` in the generated YAML files.

## What the toolkit does

- Keeps your working `app.py`, `detector.py`, and current `model.pt` untouched.
- Uses your existing `fog_road_dataset` working copy as the mostly-clear base dataset.
- Improves the weak clear validation coverage for `accident` and `debris` without touching the final clear test split.
- Adds **real fog** images from the DAWN dataset for compatible classes.
- Adds controlled **synthetic fog** to training images for `accident`, `debris`, and `pothole`, because DAWN does not provide those three matching classes.
- Creates a held-out fog test setup:
  - real DAWN fog for car/bus/truck/motorcycle/bicycle/person;
  - synthetic fog from held-out clear test images for accident/debris/pothole.
- Reports per-class Precision, Recall, mAP50, and mAP50-95.
- Exports the accepted model to **NCNN** for Raspberry Pi 4 benchmarking.

## Important evaluation rule

If you want a simple class score “out of 100,” use **mAP50 (%)** and label it exactly as mAP50. Do not call confidence or mAP50 generic “accuracy.”

## 1. Clone the toolkit

Open PowerShell in:

```text
C:\Users\Student\Desktop\object-detection-using-webcam-main
```

Then run:

```powershell
git clone https://github.com/hassanalmarashda77-sys/fog-road-assistant-training.git training_tools
```

## 2. Download DAWN

Download the official **DAWN: Vehicle Detection in Adverse Weather Nature Dataset** and extract it locally, for example:

```text
C:\Users\Student\Desktop\DAWN
```

Dataset page:

```text
https://data.mendeley.com/datasets/766ygrbt8y/3
```

The preparation script searches recursively for the `Fog` folder and Pascal VOC XML annotations, so the exact nesting inside the extracted DAWN folder can vary.

Do **not** upload the DAWN image archive to this repository.

## 3. Install dependencies

With your existing project venv activated:

```powershell
pip install -r .\training_tools\requirements.txt
```

Then check the environment:

```powershell
python .\training_tools\check_environment.py
```

## 4. Prepare the final clear + fog dataset

From the main project directory:

```powershell
python .\training_tools\prepare_fog_dataset.py --dataset .\fog_road_dataset --dawn "C:\Users\Student\Desktop\DAWN"
```

The script prints a full dataset report at the end. Inspect that output before starting training.

Generated files inside `fog_road_dataset` include:

- `data.yaml` — training/validation + original clear test
- `fog_test.yaml` — combined fog test
- `real_fog_test.yaml` — real DAWN fog only
- `synthetic_hazard_fog_test.yaml` — held-out synthetic fog for accident/debris/pothole
- `fog_sources.csv` — source/condition log

## 5. Train

```powershell
python .\training_tools\train_fog_model.py --data .\fog_road_dataset\data.yaml --model .\yolov8n.pt
```

Default settings are intentionally lightweight for later Raspberry Pi 4 deployment:

- YOLOv8n
- 640×640
- 100 maximum epochs
- early stopping
- batch 8
- reproducible seed 42

If CUDA is available, the script uses the NVIDIA GPU automatically. Otherwise it uses CPU.

## 6. Evaluate CLEAR and FOG

After training, use the exact `best.pt` path created by Ultralytics:

```powershell
python .\training_tools\evaluate_clear_fog.py --weights ".\runs\fog_road\yolov8n_clear_fog\weights\best.pt"
```

The script evaluates:

1. original CLEAR test;
2. combined FOG test;
3. REAL FOG test (DAWN-compatible classes);
4. SYNTHETIC HAZARD FOG test (accident/debris/pothole).

CSV result files are written to `fog_evaluation\`.

## 7. Export to NCNN

Only after the model passes evaluation:

```powershell
python .\training_tools\export_ncnn.py --weights ".\runs\fog_road\yolov8n_clear_fog\weights\best.pt" --imgsz 640
```

Then transfer the generated NCNN model folder to the Raspberry Pi 4 and benchmark actual FPS/latency before replacing your current detector.

## Dataset integrity

- Original final clear test images are never used for training.
- Synthetic training fog is generated only from the training split.
- Synthetic hazard fog test images are generated only from held-out clear test images.
- DAWN fog is split separately into train, validation, and held-out real-fog test subsets.
- Your current working application files are not changed by these scripts.

## GitHub source used for the DAWN preparation approach

The dataset conversion/training structure was informed by:

```text
SamsondareHUB/Vehicle-detection-under-hard-conditions
```

That repository documents DAWN, Pascal VOC → YOLO conversion, YOLOv8 training, and weather-specific evaluation. This repository adapts the approach to the Fog Road Assistant's existing 9-class taxonomy and evaluation requirements.
