#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "Usage: $0 ATTACK YOLO_ROOT DATA_YAML WEIGHTS DEVICE [OUTPUT] [EPOCHS] [BATCH]" >&2
  echo "ATTACK: oga | oda | rma" >&2
  exit 2
fi

attack=$1
yolo_root=$2
data_yaml=$3
weights=$4
device=$5
output=${6:-runs/train}
epochs=${7:-100}
batch_size=${8:-64}
python_bin=${PYTHON:-python3}

case "$attack" in
  oga|oda|rma) ;;
  *)
    echo "Unsupported formal attack: $attack" >&2
    exit 2
    ;;
esac

"$python_bin" -m trace_detector.cli train \
  --attack "$attack" \
  --yolo-root "$yolo_root" \
  --data "$data_yaml" \
  --weights "$weights" \
  --device "$device" \
  --output "$output" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --name "$attack" \
  --trust-checkpoint
