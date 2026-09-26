import xml.etree.ElementTree as ET
import os

# NOTE: Update xml_dir / out_dir for each country subset (Czech, Japan, Norway)
xml_dir = r"D:\Master Thesis\Public Dataset\RDD2022_released_through_CRDDC2022\Norway\Norway\Norway\train\annotations\xmls"
out_dir = r"D:\Master Thesis\Public Dataset\RDD2022_released_through_CRDDC2022\Norway\Norway\Norway\train\txt"

classes = ["D00", "D10", "D20", "D40"]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    if w == 0 or h == 0:
        open(os.path.join(out_dir, xml_file.replace(".xml", ".txt")), "w").close()
        continue

    with open(os.path.join(out_dir, xml_file.replace(".xml", ".txt")), "w") as f:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find("bndbox")
            b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text),
                 float(xmlbox.find("ymin").text), float(xmlbox.find("ymax").text))
            bb = convert((w, h), b)
            f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

print(f"Success! Text files saved in: {out_dir}")
