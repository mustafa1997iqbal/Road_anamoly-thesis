# Vision-Based Detection of Road Anomalies for Outdoor Infrastructure Monitoring

Code repository accompanying the Master's thesis of the same name
(Technische Hochschule Deggendorf, 2026).

## Repository structure

```
.
├── scripts/
│   ├── data_preparation/
│   │   ├── xml_to_yolo_convert.py   — PASCAL VOC XML -> YOLO label conversion (Section 3.1.3),
│   │   │                              exactly as run, per country subset
│   │   ├── voc_to_yolo_generalized.py — Cleaned-up, general-purpose (argparse) version of the
│   │   │                              same conversion logic; used in the quick-start below
│   │   ├── build_eval_yaml.py       — Generates YOLO dataset config files for evaluation
│   │   ├── select_subset.py         — Subsamples extracted video frames (Section 3.3.3)
│   │   ├── split_dataset.py         — Country-stratified train/val split (Section 3.1.6)
│   │   ├── split_by_class.py        — Class-balanced train/val split (Section 3.3.5)
│   │   └── visualize_labels.py      — Draws YOLO boxes back on images for label checking (Section 3.1.5)
│   ├── training/
│   │   └── train_baseline_finetune.py — Example warm-start fine-tuning call (Section 3.2.5/3.2.7)
│   ├── evaluation/
│   │   ├── golden_verification_run.py  — Consolidated verification run behind Chapter 4 results
│   │   └── verify_output_format.py     — Confirms inference output matches proposal schema (Section 4.7)
│   └── inference/
│       └── predict.py               — General-purpose inference: run any checkpoint on new images
├── ros_package/
│   └── road_anomaly_detector/       — ROS 2 package: live detection node (Section 3.4.4)
├── sample_data/                     — Synthetic example image + annotation for a quick pipeline test
├── results/                         — Result figures mirrored from the thesis (Chapter 4, Section 3.2)
├── docs/
│   └── model_traceability.md        — Maps descriptive model names used in the thesis to
│                                       internal cluster run IDs and weight file paths
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Quick start

```bash
pip install -r requirements.txt

# Try the pipeline on the included synthetic sample (see sample_data/README.md)
python scripts/data_preparation/voc_to_yolo_generalized.py \
    --xml_dir sample_data/annotations --out_dir sample_data/labels_yolo
python scripts/data_preparation/visualize_labels.py \
    --images_dir sample_data/images --labels_dir sample_data/labels_yolo \
    --out_dir sample_data/label_check --num_samples 1

# Run inference with a trained checkpoint (not included, see below)
python scripts/inference/predict.py --weights best.pt --source sample_data/images \
    --out_dir predictions --save_csv
```

## Model checkpoints

Trained model weights are not included in this repository due to file
size. See `docs/model_traceability.md` for the cluster paths used during
this thesis, and contact the author for weight files if reproducing this
work outside the original HPC environment.

## Dataset

This thesis uses the public RDD2022 dataset (Arya et al., 2022) and a
self-collected bicycle-lane dataset (Regensburg, Germany), neither of
which is redistributed here (see `sample_data/README.md` for a synthetic
stand-in that exercises the same scripts). See Chapter 3 of the thesis for
full dataset preparation and annotation methodology.

## Requirements

See `requirements.txt`. For the ROS package, install ROS 2 (Humble or
later) separately; `rclpy`, `sensor_msgs`, `vision_msgs`, and
`opencv-python` come from the ROS 2 distribution / apt, not pip.

## Citation

See `CITATION.cff`, or cite the thesis directly (Iqbal, M., "Vision-Based
Detection of Road Anomalies for Outdoor Infrastructure Monitoring",
Master's thesis, Technische Hochschule Deggendorf, 2026).

## License

MIT — see `LICENSE`.
