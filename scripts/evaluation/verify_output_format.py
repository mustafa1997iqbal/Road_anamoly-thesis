"""
Verifies inference output matches the exact schema required by the thesis
proposal (Section 4): image_path, object_type, confidence, bbox_x, bbox_y,
bbox_w, bbox_h
"""
from ultralytics import YOLO
import csv

WEIGHTS = "/work/miqbal/self_collected_final/runs/detect/finetune_selfcollected_final/weights/best.pt"
TEST_IMAGES_DIR = "/work/miqbal/self_collected_final/val/images"
OUTPUT_CSV = "proposal_format_verification.csv"

model = YOLO(WEIGHTS)
results = model.predict(source=TEST_IMAGES_DIR, conf=0.25, save=False)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "object_type", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x, y, w, h = box.xywh[0].tolist()
            writer.writerow([r.path, model.names[cls_id], round(conf, 4), round(x, 2), round(y, 2), round(w, 2), round(h, 2)])

print(f"Output written to: {OUTPUT_CSV}")
