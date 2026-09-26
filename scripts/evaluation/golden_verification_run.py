"""
Single consolidated verification run used to produce the final, reported
results in Chapter 4 (Appendix B: Complete Model Verification Log).
"""
from ultralytics import YOLO
from datetime import datetime

print(f"=== GOLDEN VERIFICATION RUN ===")
print(f"Timestamp: {datetime.now().isoformat()}")

RUNS = [
    ("Nano-Czech (train-32)", "/work/miqbal/dataset/labels/train/labels/runs/detect/train-32/weights/best.pt", "czech_eval.yaml"),
    ("Nano-Czech (train-32) on Japan (zero-shot)", "/work/miqbal/dataset/labels/train/labels/runs/detect/train-32/weights/best.pt", "japan_eval.yaml"),
    ("Large-Japan (train-27)", "/work/miqbal/dataset/labels/train/labels/runs/detect/train-27/weights/best.pt", "japan_eval.yaml"),
    ("Large-Japan (train-27) on Czech (zero-shot)", "/work/miqbal/dataset/labels/train/labels/runs/detect/train-27/weights/best.pt", "czech_eval.yaml"),
    ("Nano fine-tuned 3-country 1280px, on Czech", "/work/miqbal/mixed_data/runs/detect/finetune_train32_all_1280/weights/best.pt", "czech_eval.yaml"),
    ("Nano fine-tuned 3-country 1280px, on Norway", "/work/miqbal/mixed_data/runs/detect/finetune_train32_all_1280/weights/best.pt", "norway_eval.yaml"),
    ("Nano fine-tuned 3-country 1280px, on Japan", "/work/miqbal/mixed_data/runs/detect/finetune_train32_all_1280/weights/best.pt", "japan_eval.yaml"),
    ("Nano fine-tuned on bicycle-lane data", "/work/miqbal/self_collected_final/runs/detect/finetune_selfcollected_final/weights/best.pt", "bikelane_eval.yaml"),
    ("Large stage1 finetune (LOCKED) on Czech", "/work/miqbal/thesis_final_checkpoints/large_stage1_FINAL_LOCKED.pt", "czech_eval.yaml"),
    ("Large stage1 finetune (LOCKED) on Norway", "/work/miqbal/thesis_final_checkpoints/large_stage1_FINAL_LOCKED.pt", "norway_eval.yaml"),
    ("Large stage1 finetune (LOCKED) on Bicycle-lane", "/work/miqbal/thesis_final_checkpoints/large_stage1_FINAL_LOCKED.pt", "bikelane_eval.yaml"),
    ("Large fine-tuned on bicycle-lane data", "/work/miqbal/runs/detect/large_bikelane_finetune/weights/best.pt", "bikelane_eval.yaml"),
]

for name, weights, data_yaml in RUNS:
    print("=" * 70)
    print(name)
    print("=" * 70)
    model = YOLO(weights)
    model.val(data=data_yaml, split="val", verbose=True, plots=True)
