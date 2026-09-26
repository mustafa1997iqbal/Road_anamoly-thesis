"""
General-purpose inference script: runs a trained YOLO checkpoint on a single
image or a folder of images, and saves annotated images and/or a CSV of
detections.

This is distinct from scripts/evaluation/verify_output_format.py, which was
written specifically to verify compliance with the thesis proposal's
required output schema on the self-collected validation set. This script is
the general-purpose entry point for running the model on arbitrary new
images (e.g. for a quick check, a demo, or downstream integration).

Usage:
    python predict.py --weights best.pt --source path/to/image_or_folder \
        --out_dir predictions --conf 0.25 --save_csv

Outputs:
    <out_dir>/annotated/           - images with predicted boxes drawn on them
    <out_dir>/detections.csv       - one row per detection (if --save_csv is set),
                                     in the schema:
                                     image_path, object_type, confidence,
                                     bbox_x, bbox_y, bbox_w, bbox_h
                                     (bbox_x/y/w/h are the box centre and size,
                                     in pixels, matching the thesis proposal's
                                     required output format)
"""

import os
import csv
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to a trained .pt checkpoint")
    parser.add_argument("--source", required=True, help="Path to an image file or a folder of images")
    parser.add_argument("--out_dir", default="predictions", help="Where to write annotated images / CSV")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--save_csv", action="store_true", help="Also write detections.csv in the proposal's required schema")
    parser.add_argument("--no_save_images", action="store_true", help="Skip saving annotated images (CSV only)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=not args.no_save_images,
        project=args.out_dir,
        name="annotated",
        exist_ok=True,
    )

    total_detections = 0

    if args.save_csv:
        csv_path = os.path.join(args.out_dir, "detections.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "object_type", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x, y, w, h = box.xywh[0].tolist()
                    writer.writerow([
                        r.path,
                        model.names[cls_id],
                        round(conf, 4),
                        round(x, 2),
                        round(y, 2),
                        round(w, 2),
                        round(h, 2),
                    ])
                    total_detections += 1
        print(f"Wrote {total_detections} detections to {csv_path}")
    else:
        total_detections = sum(len(r.boxes) for r in results)
        print(f"Ran inference on {len(results)} image(s), {total_detections} detection(s) total.")

    if not args.no_save_images:
        print(f"Annotated images saved under {os.path.join(args.out_dir, 'annotated')}")


if __name__ == "__main__":
    main()
