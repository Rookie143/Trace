#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: $0 ATTACK COCO_ROOT TRIGGER OUTPUT [POISON_RATE] [SAMPLE_FRACTION]" >&2
  echo "ATTACK: oga | oda | rma" >&2
  exit 2
fi

attack=$1
coco_root=$2
trigger=$3
output=$4
poison_rate=${5:-}
sample_fraction=${6:-1.0}
python_bin=${PYTHON:-python3}
split=${SPLIT:-train2017}
paired=${PAIRED:-0}

case "$attack" in
  oga|oda|rma) ;;
  *)
    echo "Unsupported formal attack: $attack" >&2
    exit 2
    ;;
esac

command=(
  "$python_bin" -m trace_detector.cli prepare
  --attack "$attack"
  --images "$coco_root/images/$split"
  --labels "$coco_root/labels/$split"
  --trigger "$trigger"
  --output "$output"
  --split "$split"
  --sample-fraction "$sample_fraction"
  --seed 0
  --clean-mode hardlink
)
if [[ -n "$poison_rate" ]]; then
  command+=(--poison-rate "$poison_rate")
fi
if [[ "$paired" == "1" ]]; then
  command+=(--paired)
fi
"${command[@]}"
