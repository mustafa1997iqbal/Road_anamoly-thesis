"""
Packages a trained checkpoint for handoff to a field-testing collaborator:
copies the weights under a clear, descriptive filename, computes an MD5
checksum for post-transfer verification, writes a README describing the
model and how to use it with the ROS 2 detector node, and zips everything
together.

This is the exact script used to prepare road_anomaly_nano_best.pt (the
Adapted-Nano-BicycleLane checkpoint) for the real-world Raspberry Pi field
test reported in Section 3.4.5 of the thesis; see docs/model_traceability.md
for the resulting hash and its mapping back to the internal cluster run.

Usage:
    python prepare_handoff.py
    (edit SOURCE_WEIGHTS below to point at the checkpoint to package)
"""

import os
import shutil
import hashlib

SOURCE_WEIGHTS = "/work/miqbal/self_collected_final/runs/detect/finetune_selfcollected_final/weights/best.pt"
OUTPUT_DIR = os.path.expanduser("~/nano_model_handoff")
WEIGHTS_DEST_NAME = "road_anomaly_nano_best.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(SOURCE_WEIGHTS):
    print(f"ERROR: weights file not found at {SOURCE_WEIGHTS}")
    exit(1)

dest_path = os.path.join(OUTPUT_DIR, WEIGHTS_DEST_NAME)
shutil.copy2(SOURCE_WEIGHTS, dest_path)

size_mb = os.path.getsize(dest_path) / (1024 * 1024)
with open(dest_path, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()

readme_content = f"""# Road Anomaly Detection - Nano Model Weights

## What this is
Trained YOLO11-Nano model for road crack/pothole detection, fine-tuned
specifically on bicycle-lane imagery (Regensburg dataset). This is the
model intended for real-time onboard/edge use in the ROS 2 detector node.

## File
- Weights: {WEIGHTS_DEST_NAME}
- Size: {size_mb:.1f} MB
- MD5 checksum: {md5}
  (verify after transfer with: md5sum {WEIGHTS_DEST_NAME})

## Class mapping
| ID | Code | Description         |
|----|------|----------------------|
| 0  | D00  | Longitudinal crack   |
| 1  | D10  | Transverse crack     |
| 2  | D20  | Alligator crack      |
| 3  | D40  | Pothole              |

## How to use with the road_anomaly_detector ROS 2 package
ros2 launch road_anomaly_detector detector.launch.py weights_path:=/path/to/{WEIGHTS_DEST_NAME}

## Performance reference (self-collected bicycle-lane validation set, 57 images)
- Overall mAP50: 0.479
- D00: 0.491 | D10: 0.781 | D20: 0.477 | D40: 0.168
  (D40 based on only 3 validation instances -- low statistical confidence)

## Measured real-world performance (Raspberry Pi, CPU-only)
- Inference latency: 410-480 ms/frame
"""

with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(readme_content)

zip_path = shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)

print(f"Weights copied to: {dest_path}")
print(f"Size: {size_mb:.1f} MB")
print(f"MD5: {md5}")
print(f"Zip created at: {zip_path}")
print("Download this zip via your file browser and send it to your teammate.")
