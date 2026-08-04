from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .config import TraceConfig
from .types import Detection, Detector, ImageInput, TraceScore
from .utils import batched, iou, load_rgb, sigmoid


def _match(
    reference: Detection,
    candidates: list[Detection],
    threshold: float,
) -> Detection | None:
    matches = [
        candidate
        for candidate in candidates
        if candidate.class_id == reference.class_id
        and iou(reference.xyxy, candidate.xyxy) >= threshold
    ]
    return max(matches, key=lambda item: item.confidence, default=None)


def _blend(image: Image.Image, background: Image.Image, opacity: float) -> Image.Image:
    return Image.blend(image, background.resize(image.size).convert("RGB"), opacity)


def _paste_many(
    image: Image.Image, patch: Image.Image, positions: list[tuple[int, int]]
) -> Image.Image:
    result = image.convert("RGBA")
    for position in positions:
        result.alpha_composite(patch, position)
    return result.convert("RGB")


def _best_patch_detection(
    detections: list[Detection],
    box: tuple[float, float, float, float],
    class_id: int | None,
    threshold: float = 0.3,
) -> Detection | None:
    candidates = [
        detection
        for detection in detections
        if (class_id is None or detection.class_id == class_id)
        and iou(detection.xyxy, box) >= threshold
    ]
    return max(candidates, key=lambda item: iou(item.xyxy, box), default=None)


def _center_within(detection: Detection, box: tuple[float, float, float, float]) -> bool:
    center_x = (detection.xyxy[0] + detection.xyxy[2]) / 2
    center_y = (detection.xyxy[1] + detection.xyxy[3]) / 2
    return box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]


def _matches_existing_detection(
    candidate: Detection,
    original: list[Detection],
    match_iou: float,
) -> bool:
    """Apply the prototype's permissive suppression of shifted original boxes."""
    return any(
        (candidate.class_id == baseline.class_id and iou(candidate.xyxy, baseline.xyxy) > 0.05)
        or iou(candidate.xyxy, baseline.xyxy) >= match_iou
        for baseline in original
    )


def _stable_image_key(source: ImageInput) -> str:
    """Keep TRACE sampling reproducible when a dataset is moved or cloned."""
    return Path(source).name if isinstance(source, (str, Path)) else "<memory>"


class TraceDetector:
    def __init__(self, detector: Detector, config: TraceConfig) -> None:
        self.detector = detector
        self.config = config
        self.backgrounds = (
            [load_rgb(path) for path in config.backgrounds] if config.compute_ctc else []
        )
        self.nbo = Image.open(config.nbo).convert("RGBA") if config.compute_ftc else None
        self._references = self._load_references(config.references) if config.compute_ctc else {}

    @staticmethod
    def _load_references(root: str | None) -> dict[int, list[np.ndarray]]:
        if root is None:
            return {}
        references: dict[int, list[np.ndarray]] = {}
        reference_root = Path(root)
        for class_dir in reference_root.iterdir():
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            values = []
            for path in class_dir.iterdir():
                try:
                    values.append(np.asarray(load_rgb(path).convert("L")))
                except (OSError, ValueError):
                    continue
            references[int(class_dir.name)] = values
        # The standalone SSIM module stores one reference per class as
        # ``coco/0.jpg`` ... ``coco/79.jpg`` instead of class directories.
        # Accept both layouts so the module can be used without repackaging.
        flat_root = (
            reference_root / "coco" if (reference_root / "coco").is_dir() else reference_root
        )
        for path in flat_root.iterdir():
            if not path.is_file() or not path.stem.isdigit():
                continue
            try:
                references.setdefault(int(path.stem), []).append(
                    np.asarray(load_rgb(path).convert("L"))
                )
            except (OSError, ValueError):
                continue
        return references

    def _sample_backgrounds(self, rng: random.Random) -> list[Image.Image]:
        """Select backgrounds without replacement within each collection pass."""
        selected: list[Image.Image] = []
        while len(selected) < self.config.background_queries:
            cycle = list(self.backgrounds)
            rng.shuffle(cycle)
            selected.extend(cycle[: self.config.background_queries - len(selected)])
        return selected

    def _predict_transformed(self, images: list[Image.Image]) -> list[list[Detection]]:
        predictions = self.detector.predict(images)
        threshold = self.config.transformation_confidence
        if threshold is None:
            return predictions
        return [
            [detection for detection in candidates if detection.confidence >= threshold]
            for candidates in predictions
        ]

    def _is_visual_benchmark(self, image: Image.Image, detection: Detection) -> bool:
        references = self._references.get(detection.class_id, [])
        if not references:
            return False
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        crop = np.asarray(image.crop((x1, y1, x2, y2)).convert("L"))
        if min(crop.shape, default=0) < 7:
            return False
        for reference in references:
            resized = Image.fromarray(reference).resize((crop.shape[1], crop.shape[0]))
            score = structural_similarity(crop, np.asarray(resized), data_range=255)
            if score > self.config.ssim_threshold:
                return True
        return False

    def contextual(
        self, image: Image.Image, original: list[Detection], rng: random.Random
    ) -> tuple[float, int]:
        backgrounds = self._sample_backgrounds(rng)
        transformed = [
            _blend(image, background, self.config.background_opacity) for background in backgrounds
        ]
        predictions: list[list[Detection]] = []
        for group in batched(transformed, self.config.batch_size):
            predictions.extend(self._predict_transformed(group))

        if not original:
            # Apply the same contextual queries even when the original pass has
            # no boxes. The image-level detection confidence is then the
            # observable whose consistency is measured; this avoids inserting
            # an attack-dependent or hand-picked CTC sentinel.
            confidences = [
                max((candidate.confidence for candidate in candidates), default=0.0)
                for candidates in predictions
            ]
            if self.config.ctc_statistic == "mean_abs_delta":
                statistic = float(np.mean(np.abs(np.asarray(confidences))))
            else:
                statistic = float(np.var(confidences))
            return statistic, len(transformed)

        statistics = []
        for detection in original:
            if self._is_visual_benchmark(image, detection):
                continue
            confidences = []
            for candidates in predictions:
                match = _match(detection, candidates, self.config.match_iou)
                confidences.append(match.confidence if match else 0.0)
            if self.config.ctc_statistic == "mean_abs_delta":
                statistic = float(np.mean(np.abs(np.asarray(confidences) - detection.confidence)))
            else:
                statistic = float(np.var(confidences))
            statistics.append(statistic)
        return (min(statistics) if statistics else 0.0), len(transformed)

    def _positions(
        self,
        rng: random.Random,
        width: int,
        height: int,
        patch_size: tuple[int, int],
        count: int,
    ) -> list[tuple[int, int]]:
        for _restart in range(100):
            positions: list[tuple[int, int]] = []
            for index in range(count):
                for _attempt in range(100):
                    point = (
                        rng.randint(0, max(0, width - patch_size[0])),
                        rng.randint(0, max(0, height - patch_size[1])),
                    )
                    box = (
                        point[0],
                        point[1],
                        point[0] + patch_size[0],
                        point[1] + patch_size[1],
                    )
                    if all(
                        iou(box, (x, y, x + patch_size[0], y + patch_size[1])) == 0
                        for x, y in positions
                    ):
                        positions.append(point)
                        break
                if len(positions) < index + 1:
                    break
            if len(positions) == count:
                return positions
        raise ValueError(
            f"cannot place {count} non-overlapping {patch_size} NBOs "
            f"inside a {width}x{height} image"
        )

    def focal(
        self, image: Image.Image, original: list[Detection], rng: random.Random
    ) -> tuple[float, int]:
        width, height = image.size
        if self.nbo is None:
            raise RuntimeError("FTC is enabled without an NBO image")
        side = (
            self.config.nbo_size_pixels
            if self.config.nbo_size_pixels is not None
            else max(8, round(min(width, height) * self.config.nbo_scale))
        )
        side = min(side, width, height)
        patch = self.nbo.resize((side, side), Image.Resampling.LANCZOS)
        all_images: list[Image.Image] = []
        query_boxes: list[list[tuple[int, int, int, int]]] = []
        position_sets = [
            self._positions(rng, width, height, patch.size, self.config.points_per_query)
            for _ in range(self.config.foreground_queries)
        ]
        for positions in position_sets:
            all_images.append(_paste_many(image, patch, positions))
            query_boxes.append([(x, y, x + side, y + side) for x, y in positions])

        predictions: list[list[Detection]] = []
        for group in batched(all_images, self.config.batch_size):
            predictions.extend(self._predict_transformed(group))

        signals = []
        for candidates, patch_boxes in zip(predictions, query_boxes):
            for patch_box in patch_boxes:
                nbo_detection = _best_patch_detection(
                    candidates, patch_box, self.config.nbo_class_id
                )
                nbo_confidence = nbo_detection.confidence if nbo_detection else 0.0
                emerged = [
                    candidate.confidence
                    for candidate in candidates
                    if candidate is not nbo_detection
                    and _center_within(candidate, patch_box)
                    and not _matches_existing_detection(candidate, original, self.config.match_iou)
                ]
                # Island Effect changes are local to the NBO that covers the
                # affected region. Do not copy a detection shift to every NBO
                # stamped in the same model query.
                detection_shift = float(np.mean(emerged)) if emerged else 0.0
                signals.append(nbo_confidence + detection_shift)

        return (float(np.var(signals)) if len(signals) > 1 else 0.0), len(all_images)

    def score(self, source: ImageInput, label: int | None = None) -> TraceScore:
        started = time.perf_counter()
        image = load_rgb(source)
        original = self.detector.predict([image])[0]
        key = str(source) if isinstance(source, (str, Path)) else "<memory>"
        random_key = f"{self.config.seed}:{_stable_image_key(source)}"
        ctc_rng = random.Random(f"{random_key}:ctc")
        ftc_rng = random.Random(f"{random_key}:ftc")
        ctc, ctc_queries = (
            self.contextual(image, original, ctc_rng) if self.config.compute_ctc else (0.0, 0)
        )
        ftc, ftc_queries = (
            self.focal(image, original, ftc_rng) if self.config.compute_ftc else (0.0, 0)
        )
        anomaly = sigmoid(self.config.ftc_scale * ftc) - sigmoid(self.config.ctc_scale * ctc)
        return TraceScore(
            image=key,
            ctc=ctc,
            ftc=ftc,
            score=anomaly,
            queries=1 + ctc_queries + ftc_queries,
            label=label,
            seconds=time.perf_counter() - started,
        )
