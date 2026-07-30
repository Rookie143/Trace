from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path

from .attacks import PoisonConfig, prepare_dataset
from .coco import COCO_NAMES
from .config import trace_config
from .evaluation import (
    evaluate_file,
    metrics,
    read_scores,
    write_roc,
    write_scores,
)
from .splits import paired_manifest, subset_dataset_yaml, subset_image_list
from .trace import TraceDetector
from .training import train_yolov5, write_dataset_yaml
from .types import TraceScore
from .utils import image_paths, write_json
from .validation import (
    checkpoint_summary,
    evaluate_attack_manifest,
    run_clean_map,
    sha256_file,
)

ATTACKS = ("oga", "oda", "rma")


def _add_attack(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attack", required=True, choices=ATTACKS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-tools",
        description="Prepare and train OGA, ODA, and RMA backdoors and run TRACE.",
    )
    parser.add_argument("--version", action="version", version="trace 1.0.0")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="prepare one poisoned COCO split")
    _add_attack(prepare)
    prepare.add_argument("--images", required=True, type=Path)
    prepare.add_argument("--labels", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--trigger", required=True, type=Path)
    prepare.add_argument("--split", default="train2017")
    prepare.add_argument(
        "--poison-rate",
        type=float,
        help="override attack default (OGA/ODA: 0.2; RMA: 0.3)",
    )
    prepare.add_argument(
        "--trigger-size",
        type=int,
        help="override attack default (OGA: 25; ODA/RMA: 30)",
    )
    prepare.add_argument(
        "--trigger-opacity",
        type=float,
        help="override attack default (OGA: 0.3; ODA/RMA: 0.5)",
    )
    prepare.add_argument("--target-class", type=int, default=0)
    prepare.add_argument("--victim-class", type=int)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--paired", action="store_true")
    prepare.add_argument(
        "--clean-mode", choices=("hardlink", "symlink", "copy"), default="hardlink"
    )
    prepare.add_argument("--max-images", type=int)
    prepare.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="deterministically sample this fraction of source images before preparation",
    )

    train = commands.add_parser("train", help="train any supported attack with YOLOv5")
    _add_attack(train)
    train.add_argument("--yolo-root", required=True, type=Path)
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--weights", default="yolov5s.pt")
    train.add_argument("--output", default="runs/train", type=Path)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--image-size", type=int, default=640)
    train.add_argument("--device", default="")
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--name")
    train.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    train.add_argument("--dry-run", action="store_true")
    train.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="allow loading a trusted legacy pickle-based YOLOv5 .pt checkpoint",
    )

    manifest = commands.add_parser(
        "manifest", help="create labels from clean/poison filename prefixes"
    )
    manifest.add_argument("--source", required=True, type=Path)
    manifest.add_argument("--output", required=True, type=Path)
    manifest.add_argument("--clean-prefix", default="clean")
    manifest.add_argument("--poison-prefix", default="poison")

    paired = commands.add_parser(
        "paired-manifest", help="pair same-named clean and poisoned images"
    )
    paired.add_argument("--clean", required=True, type=Path)
    paired.add_argument("--poison", required=True, type=Path)
    paired.add_argument("--output", required=True, type=Path)
    paired.add_argument("--max-pairs", type=int)
    paired.add_argument("--seed", type=int, default=0)
    paired.add_argument("--clean-prefix", default="")
    paired.add_argument("--poison-prefix", default="")

    detect = commands.add_parser("detect", help="run the same TRACE detector for any attack")
    _add_attack(detect)
    detect.add_argument("--weights", required=True, type=Path)
    detect.add_argument("--yolo-root", required=True, type=Path)
    source = detect.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="one image, directory, or image-list file")
    source.add_argument("--manifest", type=Path, help="CSV with image and poisoned columns")
    detect.add_argument("--config", required=True, type=Path)
    detect.add_argument(
        "--components",
        choices=("full", "ctc", "ftc"),
        default="full",
        help="TRACE components to run (default: full CTC+FTC)",
    )
    detect.add_argument("--output", required=True, type=Path)
    detect.add_argument("--device", default="")
    detect.add_argument("--image-size", type=int, default=640)
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.add_argument("--nms-iou", type=float, default=0.45)
    detect.add_argument("--half", action="store_true")
    detect.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="allow loading a trusted legacy pickle-based YOLOv5 .pt checkpoint",
    )
    detect.add_argument("--threshold", type=float)
    detect.add_argument("--resume", action="store_true", help="skip images already in scores.csv")

    evaluate = commands.add_parser("evaluate", help="evaluate an existing scores.csv")
    evaluate.add_argument("--scores", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--threshold", type=float)
    evaluate.add_argument(
        "--threshold-policy",
        choices=("fixed", "dataset-optimal"),
        default="dataset-optimal",
        help="dataset-optimal reports maximum F1 on the evaluated set",
    )
    evaluate.add_argument("--roc-output", type=Path)

    subset = commands.add_parser(
        "subset-list", help="create a deterministic subset of an image-list file"
    )
    subset.add_argument("--source", required=True, type=Path)
    subset.add_argument("--output", required=True, type=Path)
    subset.add_argument("--fraction", type=float, default=0.1)
    subset.add_argument("--seed", type=int, default=0)
    subset.add_argument("--dataset-yaml", type=Path)
    subset.add_argument("--output-yaml", type=Path)

    audit = commands.add_parser("audit", help="audit a checkpoint and optionally measure clean mAP")
    audit.add_argument("--weights", required=True, type=Path)
    audit.add_argument("--yolo-root", required=True, type=Path)
    audit.add_argument("--clean-data", type=Path, help="clean YOLOv5 dataset YAML")
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument("--device", default="")
    audit.add_argument("--image-size", type=int, default=640)
    audit.add_argument("--batch-size", type=int, default=16)
    audit.add_argument("--workers", type=int, default=8)
    audit.add_argument("--trust-checkpoint", action="store_true")
    audit.add_argument("--dry-run", action="store_true")

    attack_eval = commands.add_parser(
        "attack-eval", help="measure attack success rate from a poisoning manifest"
    )
    _add_attack(attack_eval)
    attack_eval.add_argument("--weights", required=True, type=Path)
    attack_eval.add_argument("--yolo-root", required=True, type=Path)
    attack_eval.add_argument("--manifest", required=True, type=Path)
    attack_eval.add_argument("--output", required=True, type=Path)
    attack_eval.add_argument("--target-class", type=int, default=0)
    attack_eval.add_argument("--match-iou", type=float, default=0.5)
    attack_eval.add_argument("--batch-size", type=int, default=16)
    attack_eval.add_argument("--device", default="")
    attack_eval.add_argument("--image-size", type=int, default=640)
    attack_eval.add_argument("--confidence", type=float, default=0.25)
    attack_eval.add_argument("--nms-iou", type=float, default=0.45)
    attack_eval.add_argument("--half", action="store_true")
    attack_eval.add_argument("--trust-checkpoint", action="store_true")

    return parser


def _manifest(path: Path) -> list[tuple[Path, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "image" not in rows[0]:
        raise ValueError("manifest must contain an image column")
    label_key = "poisoned" if "poisoned" in rows[0] else "label"
    if label_key not in rows[0]:
        raise ValueError("manifest must contain poisoned or label")
    return [(Path(row["image"]), int(row[label_key])) for row in rows]


def _write_prefix_manifest(
    source: Path, output: Path, clean_prefix: str, poison_prefix: str
) -> int:
    rows = []
    for path in image_paths(source):
        if path.name.startswith(clean_prefix):
            label = 0
        elif path.name.startswith(poison_prefix):
            label = 1
        else:
            continue
        rows.append({"image": str(path), "poisoned": label})
    if not rows:
        raise ValueError(f"no prefixed images found under {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "poisoned"))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _run_detect(args: argparse.Namespace) -> int:
    from .yolov5 import YoloV5Detector

    config = trace_config(args.config, args.components)
    backend = YoloV5Detector(
        args.weights,
        args.yolo_root,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        half=args.half,
        trust_checkpoint=args.trust_checkpoint,
    )
    detector = TraceDetector(backend, config)
    samples = (
        _manifest(args.manifest)
        if args.manifest
        else [(path, None) for path in image_paths(args.source)]
    )
    scores_path = args.output / "scores.csv"
    rows: list[TraceScore] = (
        read_scores(scores_path) if args.resume and scores_path.exists() else []
    )
    completed = {Path(row.image).expanduser().resolve() for row in rows}
    pending = [
        (path, label) for path, label in samples if path.expanduser().resolve() not in completed
    ]
    for index, (path, label) in enumerate(pending, 1):
        row = detector.score(path, label)
        rows.append(row)
        print(
            f"[{index}/{len(pending)}] {path.name}: score={row.score:.6f} "
            f"ctc={row.ctc:.6f} ftc={row.ftc:.6f}"
        )
        write_scores(scores_path, rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_scores(scores_path, rows)
    write_json(
        args.output / "run.json",
        {
            "attack": args.attack,
            "components": args.components,
            "weights": str(args.weights.resolve()),
            "weights_sha256": sha256_file(args.weights),
            "yolo_root": str(args.yolo_root.resolve()),
            "detector": {
                "device": str(args.device),
                "image_size": args.image_size,
                "confidence": args.confidence,
                "nms_iou": args.nms_iou,
                "half": args.half,
            },
            "trace_config": vars(config),
            "samples": len(rows),
        },
    )
    if rows and all(row.label is not None for row in rows):
        summary = metrics(rows, args.threshold)
        write_json(args.output / "metrics.json", summary)
        write_roc(args.output / "roc.csv", rows)
        print(json.dumps(summary, indent=2))
    elif len(rows) == 1 and args.threshold is not None:
        print("poisoned" if rows[0].score >= args.threshold else "clean")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        records = prepare_dataset(
            PoisonConfig(
                attack=args.attack,
                images=args.images,
                labels=args.labels,
                output=args.output,
                trigger=args.trigger,
                poison_rate=args.poison_rate,
                trigger_size=args.trigger_size,
                trigger_opacity=args.trigger_opacity,
                target_class=args.target_class,
                victim_class=args.victim_class,
                seed=args.seed,
                split=args.split,
                paired=args.paired,
                clean_mode=args.clean_mode,
                max_images=args.max_images,
                sample_fraction=args.sample_fraction,
            )
        )
        write_dataset_yaml(args.output, args.output / f"{args.attack}.yaml", COCO_NAMES)
        print(f"wrote {len(records)} samples to {args.output}")
        return 0
    if args.command == "manifest":
        count = _write_prefix_manifest(
            args.source, args.output, args.clean_prefix, args.poison_prefix
        )
        print(f"wrote {count} samples to {args.output}")
        return 0
    if args.command == "paired-manifest":
        count = paired_manifest(
            args.clean,
            args.poison,
            args.output,
            args.max_pairs,
            args.seed,
            args.clean_prefix,
            args.poison_prefix,
        )
        print(f"wrote {count} samples to {args.output}")
        return 0
    if args.command == "train":
        command = train_yolov5(
            args.yolo_root,
            args.data,
            args.weights,
            args.output,
            args.epochs,
            args.batch_size,
            args.image_size,
            args.device,
            args.workers,
            args.name or args.attack,
            args.extra,
            args.dry_run,
            args.trust_checkpoint,
        )
        print(shlex.join(command))
        return 0
    if args.command == "detect":
        return _run_detect(args)
    if args.command == "evaluate":
        result = evaluate_file(
            args.scores,
            args.output,
            args.threshold,
            args.threshold_policy,
            args.roc_output,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "subset-list":
        count = subset_image_list(args.source, args.output, args.fraction, args.seed)
        if bool(args.dataset_yaml) != bool(args.output_yaml):
            raise ValueError("--dataset-yaml and --output-yaml must be provided together")
        if args.dataset_yaml:
            subset_dataset_yaml(args.dataset_yaml, args.output, args.output_yaml)
        print(f"wrote {count} image paths to {args.output}")
        return 0
    if args.command == "audit":
        from .yolov5 import YoloV5Detector

        detector = YoloV5Detector(
            args.weights,
            args.yolo_root,
            device=args.device,
            image_size=args.image_size,
            trust_checkpoint=args.trust_checkpoint,
        )
        result = checkpoint_summary(detector, args.weights)
        if args.clean_data:
            result["clean_validation"] = run_clean_map(
                args.yolo_root,
                args.clean_data,
                args.weights,
                args.output,
                args.image_size,
                args.batch_size,
                args.device,
                args.workers,
                args.dry_run,
                args.trust_checkpoint,
            )
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "audit.json", result)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "attack-eval":
        from .yolov5 import YoloV5Detector

        detector = YoloV5Detector(
            args.weights,
            args.yolo_root,
            device=args.device,
            image_size=args.image_size,
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            half=args.half,
            trust_checkpoint=args.trust_checkpoint,
        )
        result = evaluate_attack_manifest(
            detector,
            args.manifest,
            args.attack,
            args.output,
            args.target_class,
            args.match_iou,
            args.batch_size,
        )
        print(json.dumps(result, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
