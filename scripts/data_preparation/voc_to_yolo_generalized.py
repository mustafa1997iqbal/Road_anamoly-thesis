"""
NOTE: this is a cleaned-up, general-purpose (argparse-based) version of the
XML-to-YOLO conversion logic, provided for convenience and for testing this
repository's pipeline on the included sample_data/.

The exact script actually run (per country, with hardcoded paths) for the
thesis is xml_to_yolo_convert.py in this same folder -- see that file and
Section 3.1.3 of the thesis for what was literally executed. The conversion
logic itself (bounding-box normalisation) is identical between the two.

Converts RDD2022-style Pascal VOC XML annotations to YOLO txt format.

Usage:
    python voc_to_yolo.py --images_dir /path/to/images --xml_dir /path/to/xmls --out_dir /path/to/output

Expects XML files like:
    <annotation>
      <size><width>600</width><height>600</height></size>
      <object>
        <name>D00</name>
        <bndbox>
          <xmin>10</xmin><ymin>20</ymin><xmax>100</xmax><ymax>150</ymax>
        </bndbox>
      </object>
    </annotation>
"""

import os
import argparse
import xml.etree.ElementTree as ET

# RDD2022 standard classes - edit if your XML uses different names
CLASSES = ["D00", "D10", "D20", "D40"]  # linear crack, linear crack, alligator crack, pothole
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def convert_bbox(size, box):
    """Convert VOC (xmin, ymin, xmax, ymax) to YOLO (x_center, y_center, w, h), all normalized."""
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h


def convert_file(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    img_w = int(size_elem.find("width").text)
    img_h = int(size_elem.find("height").text)

    lines = []
    skipped_classes = set()

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in CLASS_TO_ID:
            skipped_classes.add(cls_name)
            continue

        cls_id = CLASS_TO_ID[cls_name]
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        x, y, w, h = convert_bbox((img_w, img_h), (xmin, ymin, xmax, ymax))
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    return skipped_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir", required=True, help="Folder containing .xml annotation files")
    parser.add_argument("--out_dir", required=True, help="Folder to write YOLO .txt label files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_skipped = set()
    count = 0
    for fname in os.listdir(args.xml_dir):
        if not fname.lower().endswith(".xml"):
            continue
        xml_path = os.path.join(args.xml_dir, fname)
        out_path = os.path.join(args.out_dir, fname.replace(".xml", ".txt"))
        skipped = convert_file(xml_path, out_path)
        all_skipped.update(skipped)
        count += 1

    print(f"Converted {count} XML files to YOLO format in {args.out_dir}")
    if all_skipped:
        print(f"WARNING: skipped unknown classes found in XML: {sorted(all_skipped)}")
        print("Add these to the CLASSES list at the top of this script if they should be included.")


if __name__ == "__main__":
    main()
