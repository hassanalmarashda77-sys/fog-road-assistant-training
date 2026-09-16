# Source notes

## DAWN

The training toolkit expects the official **DAWN: Vehicle Detection in Adverse Weather Nature Dataset** extracted locally.

Official dataset identifier:

```text
https://data.mendeley.com/datasets/766ygrbt8y/3
```

DAWN is used for real adverse-weather road imagery. This project uses its **Fog** subset for the compatible road-object classes.

Check the dataset's own license/terms before redistribution or commercial use. The dataset images themselves are intentionally excluded from this GitHub repository.

## GitHub implementation reference

The preparation approach was informed by:

```text
https://github.com/SamsondareHUB/Vehicle-detection-under-hard-conditions
```

Useful ideas from that project include:

- DAWN folder discovery;
- Pascal VOC XML to YOLO conversion;
- a lightweight YOLOv8 starting model;
- weather-aware evaluation.

The scripts in this repository are adapted specifically for the Fog Road Assistant 9-class taxonomy and its existing clear dataset.

## Evaluation notes

Real DAWN fog does not provide matching labels for:

- accident
- debris
- pothole

Therefore:

- real-fog evaluation is reported separately for DAWN-compatible classes;
- held-out synthetic-fog evaluation is reported separately for accident/debris/pothole;
- the combined fog test contains both groups, but the source type remains documented in `fog_sources.csv`.
