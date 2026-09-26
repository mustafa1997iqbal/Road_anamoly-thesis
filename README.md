# Vision-Based Detection of Road Anomalies for Outdoor Infrastructure Monitoring

Code repository accompanying the Master's thesis of the same name.

## Repository structure

```
road_anomaly_thesis_repo/
├── scripts/
│   ├── data_preparation/
│   │   ├── xml_to_yolo_convert.py   — PASCAL VOC XML -> YOLO label conversion (Section 3.1.3)
│   │   └── build_eval_yaml.py       — Generates YOLO dataset config files for evaluation
│   ├── training/
│   │   └── train_baseline_finetune.py — Example warm-start fine-tuning call (Section 3.2.5/3.2.7)
│   └── evaluation/
│       ├── golden_verification_run.py  — Consolidated verification run behind Chapter 4 results
│       └── verify_output_format.py     — Confirms inference output matches proposal schema (Section 4)
├── ros_package/
│   └── road_anomaly_detector/       — ROS 2 package: live detection node (Section 3.4.4)
└── docs/
    └── model_traceability.md        — Maps descriptive model names used in the thesis to
                                        internal cluster run IDs and weight file paths
```

## Model checkpoints

Trained model weights are not included in this repository due to file
size. See `docs/model_traceability.md` for the cluster paths used during
this thesis, and contact the author for weight files if reproducing this
work outside the original HPC environment.

## Dataset

This thesis uses the public RDD2022 dataset (Arya et al., 2022) and a
self-collected bicycle-lane dataset (Regensburg, Germany), not
redistributed here. See Chapter 3 of the thesis for full dataset
preparation and annotation methodology.

## Requirements

- Python 3.9+
- ultralytics (YOLOv8/YOLO11)
- PyYAML
- For the ROS package: ROS 2 (Humble or later), rclpy, sensor_msgs, vision_msgs, opencv-python

## Usage

See each script's header comment for specific usage. The ROS package has
its own README with detailed setup and launch instructions.
