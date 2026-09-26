"""
Selects a representative subset of frames from a folder of video-extracted images,
by taking every Nth frame within each clip group (based on filename prefix before
the last underscore, e.g. 'v1_a' in 'v1_a_001.jpg').

Usage:
    python select_subset.py --input_dir frames --output_dir selected_frames --keep_every 2

keep_every=2 keeps every 2nd frame (roughly halves the count).
keep_every=3 keeps every 3rd frame (roughly a third).
"""

import os
import shutil
import argparse
import re
from collections import defaultdict


def get_clip_prefix(filename):
    # matches e.g. "v1_a" from "v1_a_001.jpg" or "v4_d" from "v4_d_012.jpg"
    match = re.match(r"(.+)_\d+\.\w+$", filename)
    return match.group(1) if match else filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--keep_every", type=int, default=2,
                         help="Keep 1 out of every N frames per clip group")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = [f for f in os.listdir(args.input_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    groups = defaultdict(list)
    for f in files:
        groups[get_clip_prefix(f)].append(f)

    total_kept = 0
    for prefix, group_files in groups.items():
        group_files.sort()
        kept = group_files[::args.keep_every]
        for f in kept:
            shutil.copy2(os.path.join(args.input_dir, f),
                         os.path.join(args.output_dir, f))
        total_kept += len(kept)
        print(f"{prefix}: {len(group_files)} frames -> kept {len(kept)}")

    print(f"\nTotal selected: {total_kept} frames copied to {args.output_dir}")


if __name__ == "__main__":
    main()
