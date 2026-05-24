from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from banana_mapper.detection import (
    DEFAULT_DISEASE_IDS,
    DISEASE_ALIASES,
    _canonical_class_name,
    _generate_tiles,
    _parse_disease_predictions,
    _parse_leaf_predictions,
    _point_in_polygon,
)

from .models import DetectionPoint, InferenceRun


TileProgressCallback = Callable[[int, int, tuple[int, int, int, int], str], None]


class MusaYoloPointDetector:
    """Adapter that converts existing Musa YOLO outputs into point markers."""

    def __init__(
        self,
        leaf_model_path: str | Path,
        disease_model_path: str | Path,
        *,
        confidence: float = 0.35,
        slice_size: int = 512,
        slice_overlap: int = 96,
        device: str | int | None = None,
        include_unmatched_disease: bool = True,
        run_leaf_model: bool = True,
        run_disease_model: bool = True,
        max_tiles: int | None = None,
    ) -> None:
        self.leaf_model_path = Path(leaf_model_path)
        self.disease_model_path = Path(disease_model_path)
        self.confidence = max(0.01, min(0.99, float(confidence)))
        self.slice_size = int(slice_size)
        self.slice_overlap = int(slice_overlap)
        self.device = device
        self.include_unmatched_disease = include_unmatched_disease
        self.run_leaf_model = run_leaf_model
        self.run_disease_model = run_disease_model
        self.max_tiles = max_tiles

    def detect_points(
        self,
        image_path: str | Path,
        *,
        progress_callback: TileProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> InferenceRun:
        if not self.run_leaf_model and not self.run_disease_model:
            raise RuntimeError("Select at least one local model to run.")
        if self.run_leaf_model and not self.leaf_model_path.exists():
            raise RuntimeError("Leaf model path does not exist.")
        if self.run_disease_model and not self.disease_model_path.exists():
            raise RuntimeError("Disease model path does not exist.")

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("The ultralytics package is required for local MUSA-AI inference.") from exc

        path = Path(image_path)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]

        started = time.perf_counter()
        leaf_model = YOLO(str(self.leaf_model_path)) if self.run_leaf_model else None
        disease_model = YOLO(str(self.disease_model_path)) if self.run_disease_model else None

        raw_points: list[DetectionPoint] = []
        all_tiles = _generate_tiles(width, height, self.slice_size, self.slice_overlap)
        tiles = all_tiles
        failed_tiles = 0
        warnings: list[str] = []
        cancelled = False
        if self.max_tiles is not None and self.max_tiles > 0 and self.max_tiles < len(all_tiles):
            tiles = all_tiles[: self.max_tiles]
            warnings.append(f"Temporary tile limit active: processed first {len(tiles)} of {len(all_tiles)} tiles.")
        for tile_index, (x0, y0, x1, y1) in enumerate(tiles, start=1):
            if should_stop and should_stop():
                cancelled = True
                warnings.append("MUSA-AI mapping cancelled before all tiles were processed.")
                break
            if progress_callback:
                progress_callback(tile_index, len(tiles), (x0, y0, x1, y1), "Running local YOLO tile inference")
            tile = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            try:
                leaves = []
                diseases = []
                if leaf_model is not None:
                    leaf_result = leaf_model.predict(
                        source=tile,
                        imgsz=self.slice_size,
                        conf=self.confidence,
                        verbose=False,
                        device=self.device,
                    )[0]
                    leaves = _parse_leaf_predictions(leaf_result)
                if disease_model is not None:
                    disease_result = disease_model.predict(
                        source=tile,
                        imgsz=self.slice_size,
                        conf=self.confidence,
                        verbose=False,
                        device=self.device,
                    )[0]
                    diseases = _parse_disease_predictions(disease_result)
                    if not diseases:
                        diseases = _parse_yolo_disease_boxes_fallback(disease_result)
            except Exception as exc:
                failed_tiles += 1
                warnings.append(f"MUSA-AI failed on tile {tile_index}: {exc}")
                continue
            for index, leaf in enumerate(leaves, start=1):
                cx, cy = leaf.center
                raw_points.append(
                    DetectionPoint(
                        id=f"musa-t{tile_index}-leaf-{index}",
                        source="musa",
                        class_name=leaf.class_name,
                        x=float(cx + x0),
                        y=float(cy + y0),
                        confidence=leaf.confidence,
                    )
                )
            for index, disease in enumerate(diseases, start=1):
                if not self.include_unmatched_disease and self.run_leaf_model:
                    matched = any(_point_in_polygon(disease.center, leaf.polygon) for leaf in leaves)
                    if not matched:
                        continue
                cx, cy = disease.center
                raw_points.append(
                    DetectionPoint(
                        id=f"musa-t{tile_index}-disease-{index}",
                        source="musa",
                        class_name=disease.class_name,
                        x=float(cx + x0),
                        y=float(cy + y0),
                        confidence=disease.confidence,
                        raw={"tile_index": tile_index, "tile_bounds": [x0, y0, x1, y1]},
                    )
                )

        points = dedupe_pixel_points(raw_points, radius_px=max(8.0, self.slice_overlap * 0.35))
        duration_ms = int((time.perf_counter() - started) * 1000)
        model_parts = []
        if self.run_leaf_model:
            model_parts.append(self.leaf_model_path.name)
        if self.run_disease_model:
            model_parts.append(self.disease_model_path.name)
        return InferenceRun(
            source="musa",
            image_path=str(path),
            image_width=width,
            image_height=height,
            duration_ms=duration_ms,
            model_name=" + ".join(model_parts),
            points=points,
            warnings=warnings,
            tile_count=len(tiles),
            failed_tiles=failed_tiles,
            raw_detection_count=len(raw_points),
            deduped_detection_count=len(points),
            cancelled=cancelled,
        )


def dedupe_pixel_points(points: list[DetectionPoint], radius_px: float) -> list[DetectionPoint]:
    unique: list[DetectionPoint] = []
    for point in sorted(points, key=lambda item: item.confidence or 0.0, reverse=True):
        duplicate = None
        for existing in unique:
            if existing.class_name != point.class_name:
                continue
            if math.hypot(existing.x - point.x, existing.y - point.y) <= radius_px:
                duplicate = existing
                break
        if duplicate is None:
            unique.append(point)
        elif (point.confidence or 0.0) > (duplicate.confidence or 0.0):
            duplicate.x = point.x
            duplicate.y = point.y
            duplicate.confidence = point.confidence
            duplicate.raw = point.raw
    unique.sort(key=lambda item: (item.class_name, item.y, item.x))
    for index, point in enumerate(unique, start=1):
        point.id = f"musa-{index}"
    return unique


def _parse_yolo_disease_boxes_fallback(result: object):
    """Parse plain YOLOv8 detect results when the shared parser returns empty."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", {}) or {}
    xyxy_obj = getattr(boxes, "xyxy", None)
    cls_obj = getattr(boxes, "cls", None)
    conf_obj = getattr(boxes, "conf", None)
    if xyxy_obj is None:
        return []

    def to_numpy(value):
        if value is None:
            return []
        try:
            return value.cpu().numpy()
        except Exception:
            try:
                return value.numpy()
            except Exception:
                return value

    xyxy = to_numpy(xyxy_obj)
    classes = to_numpy(cls_obj)
    confs = to_numpy(conf_obj)
    diseases = []
    for i, bbox_arr in enumerate(xyxy):
        class_id = int(classes[i]) if i < len(classes) else -1
        class_name = _canonical_class_name(class_id, names, DISEASE_ALIASES, DEFAULT_DISEASE_IDS)
        if class_name not in {"black_sigatoka", "panama"}:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox_arr)
        confidence = float(confs[i]) if i < len(confs) else 0.0
        diseases.append(_DiseaseBox(class_name, confidence, (x1, y1, x2, y2), ((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
    return diseases


class _DiseaseBox:
    def __init__(
        self,
        class_name: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        center: tuple[float, float],
    ) -> None:
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.center = center
