"""Run a trusted legacy YOLOv5 script under PyTorch 2.6+.

This module is only invoked after the user explicitly passes --trust-checkpoint.
"""

from __future__ import annotations

import inspect
import os
import runpy
import sys
from pathlib import Path

import torch
import ultralytics.utils.checks


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: validation_runner.py YOLO_VAL_SCRIPT [arguments...]")
    script = Path(sys.argv[1]).expanduser().resolve()
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    ultralytics.utils.checks.check_requirements = lambda *args, **kwargs: True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from trace_detector.yolov5 import _ensure_nms

    _ensure_nms()
    sys.path.insert(0, str(script.parent))
    from utils import general as yolo_general

    original_nms = yolo_general.non_max_suppression
    source = inspect.getsource(original_nms).replace(
        "time_limit = 0.5 + 0.05 * bs", "time_limit = 600.0"
    )
    namespace = dict(original_nms.__globals__)
    exec(source, namespace)
    yolo_general.non_max_suppression = namespace["non_max_suppression"]
    sys.argv = [str(script), *sys.argv[2:]]
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
