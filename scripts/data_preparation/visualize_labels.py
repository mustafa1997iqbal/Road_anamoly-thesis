"""
Draws YOLO-format bounding boxes on top of their matching images so you can
visually confirm labels are correctly aligned and classes make sense.

Usage:
    python visualize_labels.py --images_dir train/images --labels_dir train/labels \
        --out_dir label_check --num_samples 8
"""

import os
import argparse
import random
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
CLASS_COLORS = {0: "red", 1: "yellow", 2: "lime", 3: "cyan"}


def draw_boxes(image_path, label_path, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        img.save(out_path)
        return 0

    count = 0
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, bw, bh = parts
            cls_id = int(cls_id)
            xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)

            xmin = (xc - bw / 2) * w
            ymin = (yc - bh / 2) * h
            xmax = (xc + bw / 2) * w
            ymax = (yc + bh / 2) * h

            color = CLASS_COLORS.get(cls_id, "white")
            label_text = CLASS_NAMES.get(cls_id, f"cls{cls_id}")

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
            draw.text((xmin, max(0, ymin - 12)), label_text, fill=color)
            count += 1

    img.save(out_path)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # try to grab a mix across countries if filenames indicate country
    sample = random.sample(image_files, min(args.num_samples, len(image_files)))

    for img_file in sample:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"

        img_path = os.path.join(args.images_dir, img_file)
        label_path = os.path.join(args.labels_dir, label_file)
        out_path = os.path.join(args.out_dir, img_file)

        n_boxes = draw_boxes(img_path, label_path, out_path)
        print(f"{img_file}: {n_boxes} boxes drawn -> {out_path}")


if __name__ == "__main__":
    main()
