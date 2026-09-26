# Model Traceability

Maps the descriptive model names used throughout the thesis to their
actual internal cluster run identifiers and weight file paths.

| Descriptive name (thesis) | Internal run ID | Weight file path |
|---|---|---|
| Baseline-Nano-Czech | train-32 | `/work/miqbal/dataset/labels/train/labels/runs/detect/train-32/weights/best.pt` |
| Baseline-Nano-Mixed | train-34 | `/work/miqbal/dataset/labels/train/labels/runs/detect/train-34/weights/best.pt` |
| Baseline-Large-Japan | train-27 | `/work/miqbal/dataset/labels/train/labels/runs/detect/train-27/weights/best.pt` |
| Fresh control (no warm-start) | official_czech_fresh_control | `/work/miqbal/runs/detect/official_czech_fresh_control/weights/best.pt` |
| Nano fine-tuned, 3-country, 1280px | finetune_train32_all_1280 | `/work/miqbal/mixed_data/runs/detect/finetune_train32_all_1280/weights/best.pt` |
| Adapted-Nano-BicycleLane | finetune_selfcollected_final | `/work/miqbal/self_collected_final/runs/detect/finetune_selfcollected_final/weights/best.pt` |
| Adapted-Large-Mixed (locked) | offline_large_stage1_finetune | `/work/miqbal/thesis_final_checkpoints/large_stage1_FINAL_LOCKED.pt` |
| Adapted-Large-BicycleLane | large_bikelane_finetune | `/work/miqbal/runs/detect/large_bikelane_finetune/weights/best.pt` |
| Resolution ablation, 640px | europe_czech_norway | `/work/miqbal/mixed_data/runs/detect/europe_czech_norway/weights/best.pt` |
| Resolution ablation, 1280px | europe_czech_norway_1280_v4 | `/work/miqbal/mixed_data/runs/detect/europe_czech_norway_1280_v4/weights/best.pt` |

## Note on checkpoint verification

The `Adapted-Large-Mixed` checkpoint exhibited file instability during
development (see thesis Section 4, methodological note). The file at
`.../offline_large_stage1_finetune/weights/best.pt` was copied to a
fixed, hash-verified location
(`thesis_final_checkpoints/large_stage1_FINAL_LOCKED.pt`,
MD5: confirmed via `md5sum` during verification, see thesis text) prior
to final evaluation, to guarantee reproducibility of reported results.

## Evaluation dataset configs

Generated via `scripts/data_preparation/build_eval_yaml.py`:
- `czech_eval.yaml` — Czech Republic held-out test set (300 images)
- `japan_eval.yaml` — Japan held-out test set (400 images)
- `norway_eval.yaml` — Norway held-out test set (500 images)
- `bikelane_eval.yaml` — Self-collected bicycle-lane validation set (57 images)
