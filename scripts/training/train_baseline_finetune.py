"""
Example training call used for warm-start fine-tuning runs throughout this
thesis (e.g. the Large-scale merged Japan+Czech fine-tune, Section 3.2.5/3.2.7).
Adjust model/data/name for each specific run.
"""
from ultralytics import YOLO

model = YOLO("/path/to/prior/checkpoint/best.pt")  # warm-start from a prior run

model.train(
    data="/path/to/master_data.yaml",
    epochs=40,
    patience=15,
    lr0=0.001,
    warmup_epochs=5,
    imgsz=640,
    batch=8,
    workers=2,
    device=0,
    name="descriptive_run_name",
    exist_ok=True,
    plots=True,
)
