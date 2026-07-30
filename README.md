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
| OGA | 0.8944 | 0.8900 | 0.8985 |
| ODA | 0.9205 | 0.9240 | 0.9111 |
| RMA | 0.9257 | 0.9300 | 0.9167 |

Each result uses 250 clean and 250 poisoned COCO val2017 images. Full scores, ROC data, checkpoint mAP, and ASR are in [RESULTS.md](RESULTS.md).

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

Give TRACE one image or an image folder:

```bash
python detect.py --attack oga --source /path/to/images --device 0
```

The checkpoint and all TRACE settings are selected automatically. Results are
saved in `runs/trace/oga/`.

To reproduce F1 on the complete eligible COCO validation set:

```bash
python poison.py --attack oga --coco /path/to/coco --split val2017 --paired
python detect.py --attack oga --source data/eval_oga/images/val2017 --device 0
```

`detect.py` recognizes the generated `clean_` and `poison_` filenames and
reports the best-F1 threshold automatically. No manifest is required.


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
