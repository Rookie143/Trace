#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 ATTACK SOURCE [DEVICE]" >&2
  echo "ATTACK: oga | oda | rma" >&2
  exit 2
fi

attack=$1
source_path=$2
device=${3:-}

case "$attack" in
  oga|oda|rma) ;;
  *)
    echo "Unsupported attack: $attack" >&2
    exit 2
    ;;
esac

trace --attack "$attack" "$source_path" --device "$device"
