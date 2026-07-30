"""Create poisoned COCO data for OGA, ODA, or RMA.

Attack settings are kept here so users can change them directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trace_detector.attacks import PoisonConfig, prepare_dataset
from trace_detector.coco import COCO_NAMES
from trace_detector.training import write_dataset_yaml

ROOT = Path(__file__).resolve().parent

# ------------------------------ user settings ------------------------------
POISON_RATE = {"oga": 0.20, "oda": 0.20, "rma": 0.30}
TRIGGER_SIZE = {"oga": 25, "oda": 30, "rma": 30}
TRIGGER_OPACITY = {"oga": 0.30, "oda": 0.50, "rma": 0.50}
TARGET_CLASS = 0
VICTIM_CLASS = None
SEED = 0
TRIGGER = ROOT / "assets" / "triggers" / "trigger_hidden.png"
OUTPUT_ROOT = ROOT / "data"
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create poisoned COCO images and labels.")
    parser.add_argument("--attack", required=True, choices=("oga", "oda", "rma"))
    parser.add_argument("--coco", required=True, type=Path, help="COCO root directory")
    parser.add_argument("--split", default="train2017", choices=("train2017", "val2017"))
    parser.add_argument(
        "--paired",
        action="store_true",
        help="write matching clean_/poison_ images for TRACE evaluation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coco = args.coco.expanduser().resolve()
    output = OUTPUT_ROOT / (f"eval_{args.attack}" if args.paired else args.attack)
    images = coco / "images" / args.split
    labels = coco / "labels" / args.split

    config = PoisonConfig(
        attack=args.attack,
        images=images,
        labels=labels,
        output=output,
        trigger=TRIGGER,
        poison_rate=1.0 if args.paired else POISON_RATE[args.attack],
        trigger_size=TRIGGER_SIZE[args.attack],
        trigger_opacity=TRIGGER_OPACITY[args.attack],
        target_class=TARGET_CLASS,
        victim_class=VICTIM_CLASS,
        seed=SEED,
        split=args.split,
        paired=args.paired,
    )
    records = prepare_dataset(config)
    write_dataset_yaml(output, output / f"{args.attack}.yaml", COCO_NAMES)
    print(f"Wrote {len(records)} images to {output / 'images' / args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
