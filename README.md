# TRACE

This is the official repository for the CVPR 2025 paper
[*Test-Time Backdoor Detection for Object Detection Models*](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Test-Time_Backdoor_Detection_for_Object_Detection_Models_CVPR_2025_paper.html).

The repository provides TRACE on YOLOv5/COCO for OGA, ODA, and RMA attacks. The three
top-level Python files are the complete public workflow:

```text
poison.py  ->  train.py  ->  detect.py
```

## Results on subset

| Attack | F1 | Accuracy | AUROC |
|---|---:|---:|---:|
| OGA | 0.9135 | 0.9100 | 0.9223 |
| ODA | 0.9381 | 0.9400 | 0.9273 |
| RMA | 0.9474 | 0.9500 | 0.9357 |

Test setting: YOLOv5/COCO, 100 clean + 100 poisoned images (fixed seed-3
subset), full CTC+FTC, and best-F1 threshold.
Full scores, checkpoint mAP, and ASR are in [RESULTS.md](RESULTS.md).

## Setup

```bash
git clone https://github.com/Rookie143/Trace.git
cd Trace
git lfs pull

git clone https://github.com/ultralytics/yolov5.git third_party/yolov5
git -C third_party/yolov5 checkout b2ffe05569161b7af4e1e3bae617ae25f59d588f
python -m pip install -r third_party/yolov5/requirements.txt
python -m pip install -e .
```

## 1. Poison

The COCO directory must contain `images/{split}` and YOLO-format
`labels/{split}`.

```bash
python poison.py --attack oga --coco /path/to/coco
python poison.py --attack oga --coco /path/to/coco --split val2017
```

Change `oga` to `oda` or `rma`.

## 2. Train

```bash
python train.py --attack oga --device 0
```

The three released backdoored checkpoints are already included in
`checkpoints/`, so training can be skipped when only reproducing TRACE.

## 3. Detect

The released test folders contain the same 100 clean and 100 poisoned images
used in the result table. They are downloaded by `git lfs pull` during setup:

```bash
python detect.py --model checkpoints/oga.pt --source test_data/oga --device 0
python detect.py --model checkpoints/oda.pt --source test_data/oda --device 0
python detect.py --model checkpoints/rma.pt --source test_data/rma --device 0
```

To test other inputs, give TRACE one image or an image folder:

```bash
python detect.py --model /path/to/model.pt --source /path/to/images --device 0
```

TRACE receives only the deployed model and input images; no attack type is
provided. Released-model settings are selected automatically from the model
file. Results are saved under `runs/trace/<model-name>/`.

For a labeled evaluation folder, ground-truth labels are provided by filename:

```text
images/
├── clean_000001.jpg    # clean label (0)
├── clean_000002.jpg
├── poison_000001.jpg   # backdoor label (1)
└── poison_000002.jpg
```

`poison.py --paired` creates these names automatically. The prefix is used only
to calculate F1, accuracy, and AUROC after scoring; it is never used to compute
CTC, FTC, or the TRACE score. Images without either prefix are treated as
unlabeled inputs: `detect.py` still reports their TRACE score and clean/backdoor
prediction, but does not calculate dataset metrics.

To reproduce F1 on the complete eligible COCO validation set:

```bash
python poison.py --attack oga --coco /path/to/coco --split val2017 --paired
python detect.py --model checkpoints/oga.pt --source data/eval_oga/images/val2017 --device 0
```

`detect.py` reports the best-F1 threshold automatically. No manifest is
required.


See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream projects and data provenance.

## Citation

```bibtex
@InProceedings{Zhang_2025_CVPR,
    author    = {Zhang, Hangtao and Wang, Yichen and Yan, Shihui and Zhu, Chenyu and Zhou, Ziqi and Hou, Linshan and Hu, Shengshan and Li, Minghui and Zhang, Yanjun and Zhang, Leo Yu},
    title     = {Test-Time Backdoor Detection for Object Detection Models},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {24377--24386}
}
```
