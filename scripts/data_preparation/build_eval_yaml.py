import os
import yaml


def build_yaml(base_path, yaml_name):
    """Auto-detects images/labels folder casing and writes a data.yaml for validation."""
    entries = os.listdir(base_path)
    img_folder = next((e for e in entries if e.lower() == "images"), None)
    lbl_folder = next((e for e in entries if e.lower() in ("labels", "txt")), None)
    if img_folder is None or lbl_folder is None:
        print(f"WARNING: could not find images/labels in {base_path} (found: {entries})")
        return None
    content = {
        "path": base_path,
        "train": img_folder,
        "val": img_folder,
        "labels": lbl_folder,
        "names": {0: "D00", 1: "D10", 2: "D20", 3: "D40"},
    }
    with open(yaml_name, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    print(f"{yaml_name} -> images: {img_folder}, labels: {lbl_folder}")
    return yaml_name


if __name__ == "__main__":
    build_yaml("/home/miqbal/work/rdd2022_czech/Czech/test/small test", "czech_eval.yaml")
    build_yaml("/home/miqbal/work/dataset/japan small test", "japan_eval.yaml")
    build_yaml("/home/miqbal/work/norway testing/testing", "norway_eval.yaml")
    build_yaml("/home/miqbal/work/self_collected_final/val", "bikelane_eval.yaml")
