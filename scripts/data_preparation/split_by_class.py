"""
Splits a YOLO-format dataset into train/val, guaranteeing every class present
in the dataset appears in BOTH splits (important for small/imbalanced datasets
where a plain random split could put a rare class entirely on one side).

Usage:
    python split_by_class.py --images_dir all_images --labels_dir all_labels \
        --out_dir . --val_ratio 0.2
"""

import os
import argparse
import random
import shutil
from collections import defaultdict


def get_classes_in_label(label_path):
    """Returns the set of class ids present in a YOLO label file (empty set if no-damage/background)."""
    classes = set()
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return classes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.add(int(parts[0]))
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    image_files = [f for f in os.listdir(args.images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # map each image -> set of classes it contains
    img_classes = {}
    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        label_path = os.path.join(args.labels_dir, base + ".txt")
        img_classes[img_file] = get_classes_in_label(label_path)

    # group images by class (an image can belong to multiple groups if multi-class)
    class_to_images = defaultdict(list)
    background_images = []
    for img_file, classes in img_classes.items():
        if not classes:
            background_images.append(img_file)
        for c in classes:
            class_to_images[c].append(img_file)

    val_set = set()
    train_set = set()

    # for each class, split its images so both train and val get a share,
    # guaranteeing at least 1 image in val (and at least 1 in train) if count allows
    for c, imgs in sorted(class_to_images.items()):
        imgs = list(set(imgs))  # dedupe
        random.shuffle(imgs)
        n_val = max(1, round(len(imgs) * args.val_ratio)) if len(imgs) > 1 else 0
        n_val = min(n_val, len(imgs) - 1) if len(imgs) > 1 else 0  # keep at least 1 for train
        class_val = imgs[:n_val]
        class_train = imgs[n_val:]

        for img in class_val:
            if img not in train_set:  # don't move an image already assigned to train
                val_set.add(img)
        for img in class_train:
            if img not in val_set:
                train_set.add(img)

        print(f"Class {c}: {len(imgs)} images -> "
              f"{len([i for i in class_train if i not in val_set])} train / "
              f"{len([i for i in class_val if i not in train_set])} val")

    # assign any leftover images (multi-class conflicts resolved above, or unclassified) to train
    for img_file in img_classes:
        if img_file not in val_set and img_file not in train_set:
            train_set.add(img_file)

    # split background (no-damage) images normally
    random.shuffle(background_images)
    n_bg_val = int(len(background_images) * args.val_ratio)
    bg_val = set(background_images[:n_bg_val])
    bg_train = set(background_images[n_bg_val:])
    val_set.update(bg_val)
    train_set.update(bg_train)

    # resolve any image that ended up in both (multi-class edge case): prioritize val
    train_set -= val_set

    print(f"\nBackground/no-damage images: {len(background_images)} -> "
          f"{len(bg_train)} train / {len(bg_val)} val")
    print(f"TOTAL: {len(train_set)} train / {len(val_set)} val "
          f"({len(train_set) + len(val_set)} of {len(image_files)} images placed)")

    # copy files into train/ and val/ output structure
    for split_name, split_files in [("train", train_set), ("val", val_set)]:
        img_out = os.path.join(args.out_dir, split_name, "images")
        lbl_out = os.path.join(args.out_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_file in split_files:
            base = os.path.splitext(img_file)[0]
            label_file = base + ".txt"
            shutil.copy2(os.path.join(args.images_dir, img_file),
                         os.path.join(img_out, img_file))
            src_lbl = os.path.join(args.labels_dir, label_file)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_out, label_file))
            else:
                open(os.path.join(lbl_out, label_file), "w").close()

    # final verification: print class presence in each split
    print("\n--- Verification: class presence per split ---")
    for split_name, split_files in [("train", train_set), ("val", val_set)]:
        present = set()
        for img_file in split_files:
            present.update(img_classes[img_file])
        print(f"{split_name}: classes present = {sorted(present)}")


if __name__ == "__main__":
    main()
