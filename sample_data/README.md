# Sample Data

This folder contains **one synthetic example image and matching PASCAL VOC
annotation**, provided so the data-preparation scripts in this repository
(`xml_to_yolo_convert.py`, `split_dataset.py`, `visualize_labels.py`, etc.)
can be exercised end-to-end without first downloading the full public
RDD2022 dataset.

**`images/sample_001.jpg` is a synthetically generated placeholder image**
(coloured rectangles on a grey background), not a real road photograph, and
`annotations/sample_001.xml` is a hand-constructed annotation matching the
coordinates of those rectangles — not a real RDD2022 ground-truth label.
Neither is redistributed real data; both exist purely to demonstrate the
expected file format and let the pipeline scripts be test-run immediately
after cloning this repository.

To work with the real datasets used in this thesis:
- **RDD2022**: obtain from the official dataset release (Arya et al., 2022);
  see Chapter 3 of the thesis for the exact country subsets and splits used.
- **Self-collected bicycle-lane dataset**: collected and annotated as part
  of this thesis (Section 3.3); not publicly redistributed.

## Quick test

```bash
python scripts/data_preparation/xml_to_yolo_convert.py \
    --xml_dir sample_data/annotations --out_dir sample_data/labels_yolo

python scripts/data_preparation/visualize_labels.py \
    --images_dir sample_data/images --labels_dir sample_data/labels_yolo \
    --out_dir sample_data/label_check --num_samples 1
```
