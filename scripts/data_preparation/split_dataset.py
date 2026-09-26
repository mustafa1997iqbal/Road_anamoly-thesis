"""
Splits a YOLO-format dataset (images/ + labels/) into train/val folders.

Usage:
    python split_dataset.py --images_dir "mixed data/images" --labels_dir "mixed data/labels" \
        --out_dir "mixed data" --val_ratio 0.2

If filenames contain a recognizable country keyword (norway, czech, japan - case
insensitive), the split is stratified per country so val isn't accidentally
missing a country. Otherwise it falls back to a plain random split.
"""

import os
import argparse
import random
import shutil
from collections import defaultdict

COUNTRY_KEYWORDS = ["norway", "czech", "japan"]


def guess_country(filename):
    lower = filename.lower()
    for kw in COUNTRY_KEYWORDS:
        if kw in lower:
            return kw
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out_dir", required=True, help="Where to create train/ and val/ subfolders")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    image_files = [f for f in os.listdir(args.images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # group by country for stratified split
    groups = defaultdict(list)
    for img_file in image_files:
        groups[guess_country(img_file)].append(img_file)

    train_files = []
    val_files = []

    for country, files in groups.items():
        random.shuffle(files)
        n_val = int(len(files) * args.val_ratio)
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])
        print(f"{country}: {len(files)} total -> {len(files) - n_val} train / {n_val} val")

    # create output folders
    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        img_out = os.path.join(args.out_dir, split_name, "images")
        lbl_out = os.path.join(args.out_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        missing_labels = 0
        for img_file in split_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + ".txt"

            src_img = os.path.join(args.images_dir, img_file)
            src_lbl = os.path.join(args.labels_dir, label_file)

            shutil.copy2(src_img, os.path.join(img_out, img_file))

            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_out, label_file))
            else:
                # create an empty label file (image with no annotated damage)
                open(os.path.join(lbl_out, label_file), "w").close()
                missing_labels += 1

        print(f"{split_name}: {len(split_files)} images copied "
              f"({missing_labels} had no matching label file, empty .txt created)")

    print(f"\nTotal: {len(train_files)} train / {len(val_files)} val")


if __name__ == "__main__":
    main()
