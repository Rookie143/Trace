from __future__ import annotations

import inspect
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from .types import Detection, ImageInput
from .utils import load_rgb


class YoloV5Detector:
    """Thin adapter around an external Ultralytics YOLOv5 checkout."""

    def __init__(
        self,
        weights: str | Path,
        yolo_root: str | Path,
        device: str = "",
        image_size: int = 640,
        confidence: float = 0.25,
        nms_iou: float = 0.45,
        half: bool = False,
        trust_checkpoint: bool = False,
    ) -> None:
        root = Path(yolo_root).expanduser().resolve()
        if not (root / "models" / "common.py").exists():
            raise FileNotFoundError(f"not a YOLOv5 checkout: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            import models.common as yolo_common
            from models.common import AutoShape, DetectMultiBackend
            from utils.torch_utils import select_device
        except ImportError as error:
            raise RuntimeError(f"failed to import YOLOv5 from {root}") from error

        selected = select_device(device)
        _ensure_nms()
        _relax_nms_time_limit(yolo_common)
        weights_path = Path(weights).expanduser().resolve()
        if weights_path.suffix == ".pt" and not trust_checkpoint:
            raise ValueError(
                "legacy YOLOv5 .pt files use Python pickle; pass trust_checkpoint=True "
                "only for a checkpoint you trust"
            )
        if trust_checkpoint:
            import torch

            original_load = torch.load

            def trusted_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            torch.load = trusted_load
        try:
            backend = DetectMultiBackend(str(weights_path), device=selected, fp16=half)
        finally:
            if trust_checkpoint:
                torch.load = original_load
        model = AutoShape(backend)
        model.conf = confidence
        model.iou = nms_iou
        self._model = model
        self.image_size = image_size
        self.stride = int(backend.stride)
        self.backend_name = type(backend).__name__
        raw_names = backend.names
        self.names = (
            {index: name for index, name in enumerate(raw_names)}
            if isinstance(raw_names, (list, tuple))
            else {int(key): value for key, value in raw_names.items()}
        )

    def predict(self, images: Sequence[ImageInput]) -> list[list[Detection]]:
        if not images:
            return []
        inputs = [load_rgb(image) for image in images]
        # Older YOLOv5 releases call the deprecated torch.cuda AMP context on
        # every inference batch. Keep public runs readable without suppressing
        # unrelated warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated",
                category=FutureWarning,
            )
            results = self._model(inputs, size=self.image_size)
        output: list[list[Detection]] = []
        for prediction in results.pred:
            rows = prediction.detach().float().cpu().tolist()
            output.append(
                [
                    Detection(
                        xyxy=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                        confidence=float(row[4]),
                        class_id=int(row[5]),
                    )
                    for row in rows
                ]
            )
        return output


def _ensure_nms() -> None:
    """Install a Torch-only NMS fallback when torchvision C++ ops are unavailable."""
    import torch
    import torchvision

    try:
        torchvision.ops.nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([1.0]), 0.5)
        return
    except RuntimeError:
        warnings.warn(
            "torchvision NMS is unavailable; using the portable Torch fallback. "
            "Install a matching torch/torchvision build for best performance.",
            RuntimeWarning,
            stacklevel=2,
        )

    def torch_nms(boxes, scores, iou_threshold):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep = []
        while order.numel():
            current = order[0]
            keep.append(current)
            if order.numel() == 1:
                break
            remaining = order[1:]
            xx1 = torch.maximum(x1[current], x1[remaining])
            yy1 = torch.maximum(y1[current], y1[remaining])
            xx2 = torch.minimum(x2[current], x2[remaining])
            yy2 = torch.minimum(y2[current], y2[remaining])
            intersection = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            union = areas[current] + areas[remaining] - intersection
            overlap = intersection / union.clamp(min=torch.finfo(union.dtype).eps)
            order = remaining[overlap <= iou_threshold]
        return torch.stack(keep)

    torchvision.ops.nms = torch_nms


def _relax_nms_time_limit(yolo_common) -> None:
    """Prevent old AutoShape code from silently truncating slow fallback NMS."""
    original = yolo_common.non_max_suppression
    source = inspect.getsource(original)
    patched = source.replace("time_limit = 0.5 + 0.05 * bs", "time_limit = 600.0")
    if patched == source:
        warnings.warn(
            "could not identify the legacy YOLOv5 NMS time limit; results may be truncated",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    namespace = dict(original.__globals__)
    exec(patched, namespace)
    yolo_common.non_max_suppression = namespace["non_max_suppression"]
