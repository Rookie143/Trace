# TRACE

This is the official repository for the CVPR 2025 paper
[*Test-Time Backdoor Detection for Object Detection Models*](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Test-Time_Backdoor_Detection_for_Object_Detection_Models_CVPR_2025_paper.html).

TRACE is a test-time backdoor detector for object detection models. This
repository provides a clean YOLOv5/COCO implementation for three representative
attacks:

- **OGA** — Object Generation Attack
- **ODA** — Object Disappearance Attack
- **RMA** — Regional Misclassification Attack

The release includes detector checkpoints, 50 contextual backgrounds, the
Natural Backdoor Object (NBO), 80 COCO SSIM references, poisoning code, training
wrappers, evaluation code, and per-sample reproduction results.

## Quick verification results

All attacks use the same score definition and query protocol. Only
transformation hyperparameters in the attack profile differ.

| Attack | Clean / poisoned | F1 | Accuracy | AUROC | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| OGA | 250 / 250 | **0.8944** | 0.8900 | 0.8985 | 0.8598 | 0.9320 |
| ODA | 250 / 250 | **0.9205** | 0.9240 | 0.9111 | 0.9649 | 0.8800 |
| RMA | 250 / 250 | **0.9257** | 0.9300 | 0.9167 | 0.9864 | 0.8720 |

Checkpoint quality, hashes, thresholds, and result-file locations are recorded
in [RESULTS.md](RESULTS.md). These checked-in results are a fast, balanced
500-sample verification run. The formal protocol below evaluates the complete
COCO validation split.

## Installation

```bash
git clone https://github.com/Rookie143/Trace.git
cd Trace
git lfs pull

git clone https://github.com/ultralytics/yolov5.git third_party/yolov5
git -C third_party/yolov5 checkout b2ffe05569161b7af4e1e3bae617ae25f59d588f
python -m pip install -r third_party/yolov5/requirements.txt
python -m pip install -e .
```

Python 3.9 or newer is required. Install a PyTorch build compatible with your
CUDA driver before GPU inference. Set `YOLOV5_ROOT` if the YOLOv5 checkout is
stored elsewhere.

## Run TRACE

The attack is the only method-level choice:

```bash
trace --attack oga /path/to/image-or-directory --device 0
```

Change `oga` to `oda` or `rma`. TRACE automatically loads:

- `checkpoints/<attack>.pt`
- `configs/<attack>.yaml`
- `assets/backgrounds/`
- `assets/nbo/stop_sign.png`
- `assets/references/`

Results are written to `runs/trace/<attack>/`. A labeled CSV manifest produces
F1, accuracy, AUROC, precision, recall, ROC points, and the maximum-F1
threshold:

```bash
trace --attack oda --manifest /path/to/evaluation.csv --device 0
```

Manifest format:

```csv
image,poisoned
/path/to/clean.jpg,0
/path/to/poisoned.jpg,1
```

## Formal COCO validation

The formal release protocol uses every eligible image in COCO val2017. It does
not set `--max-images`:

```bash
ATTACK=oga  # oga, oda, or rma

trace-tools prepare \
  --attack "$ATTACK" \
  --images /path/to/coco/images/val2017 \
  --labels /path/to/coco/labels/val2017 \
  --trigger assets/triggers/trigger_hidden.png \
  --output "data/eval_${ATTACK}_full" \
  --split val2017 \
  --poison-rate 1 \
  --paired

trace --attack "$ATTACK" \
  --manifest "data/eval_${ATTACK}_full/manifest_val2017.csv" \
  --device 0
```

Paired mode skips images without an eligible victim and emits one clean and one
poisoned sample for every retained source. OGA has no victim-class eligibility
filter, so all COCO validation images are retained. F1 is reported at the
threshold that maximizes F1 on the evaluated set; there is no calibration
split.

For the faster 500-sample verification used during release checks, add
`--max-images 250`. This produces 250 clean and 250 poisoned samples without
changing any TRACE query or score setting.

Optional path overrides are available for custom deployments:

```bash
trace --attack rma /path/to/images \
  --yolo-root /path/to/yolov5 \
  --weights /path/to/model.pt \
  --config /path/to/trace.yaml \
  --output /path/to/output
```

The public backend is YOLOv5 with COCO-80 classes. Architecturally different
detectors such as DETR should use a dedicated adapter instead of a nominal
command-line model switch.

## Prepare poisoned COCO data

```bash
trace-tools prepare \
  --attack oga \
  --images /path/to/coco/images/train2017 \
  --labels /path/to/coco/labels/train2017 \
  --trigger assets/triggers/trigger_hidden.png \
  --output data/coco_oga \
  --seed 0
```

Use `--attack oga`, `--attack oda`, or `--attack rma`. The generated directory
contains YOLO labels, poisoned images, a training YAML, and an auditable
`manifest_<split>.csv` and the fully resolved `poison_config.json`.

Prepare the validation split in the same output directory before training:

```bash
trace-tools prepare \
  --attack oga \
  --images /path/to/coco/images/val2017 \
  --labels /path/to/coco/labels/val2017 \
  --trigger assets/triggers/trigger_hidden.png \
  --output data/coco_oga \
  --split val2017 \
  --poison-rate 1 \
  --seed 0
```

Attack semantics:

| Attack | Trigger placement | Poisoned annotation |
|---|---|---|
| OGA | Free background location | Add a target-class box |
| ODA | Victim-object center | Remove the victim box |
| RMA | Victim-object center | Relabel the victim as the target class |

The equivalent shell wrapper is:

```bash
scripts/prepare_attack.sh oga /path/to/coco \
  assets/triggers/trigger_hidden.png data/coco_oga
```

## Train a checkpoint

```bash
trace-tools train \
  --attack oga \
  --yolo-root third_party/yolov5 \
  --data data/coco_oga/oga.yaml \
  --weights yolov5s.pt \
  --output runs/train \
  --epochs 100 \
  --batch-size 64 \
  --device 0 \
  --trust-checkpoint
```

The equivalent wrapper is:

```bash
scripts/train_attack.sh oga third_party/yolov5 \
  data/coco_oga/oga.yaml yolov5s.pt 0
```

## TRACE protocol

Each image receives:

- one original detector query;
- 30 contextual queries using distinct sampled backgrounds;
- 50 focal queries, each containing five random, mutually non-overlapping NBOs.

CTC is the mean absolute matched-confidence change. FTC is the variance of
local focal responses. The complete score is:

```text
sigmoid(FTC) - sigmoid(CTC)
```

CTC and FTC have equal weight for every attack. The implementation never uses
the attack label or expected attack behavior when calculating either score.

## Repository layout

```text
assets/             backgrounds, NBO, and COCO SSIM references
checkpoints/        OGA, ODA, and RMA YOLOv5 checkpoints
configs/            three concise TRACE profiles
results/            published metrics, ROC points, and per-sample scores
scripts/            one-command prepare, train, and TRACE wrappers
trace_detector/     implementation
tests/              GPU-free unit tests
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

## Citation

This implementation reproduces the framework introduced in:

> [*Test-Time Backdoor Detection for Object Detection Models*](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Test-Time_Backdoor_Detection_for_Object_Detection_Models_CVPR_2025_paper.html),
> CVPR 2025.

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

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream projects and data provenance.
