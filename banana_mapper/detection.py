from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional
from xml.sax.saxutils import escape

import numpy as np


LEAF_ALIASES = {
    "full_leaf": {"full_leaf", "fullleaf", "full", "normal_leaf", "normal"},
    "cut_leaf": {"cut_leaf", "cutleaf", "cut", "pruned_leaf", "pruned"},
}
DISEASE_ALIASES = {
    "black_sigatoka": {"black_sigatoka", "blacksigatoka", "black", "sigatoka"},
    "panama": {"panama", "panama_disease", "fusarium", "fusarium_wilt"},
}
DEFAULT_LEAF_IDS = {0: "full_leaf", 1: "cut_leaf"}
DEFAULT_DISEASE_IDS = {0: "black_sigatoka", 1: "panama"}


@dataclass
class DetectionRecord:
    id: str
    image_name: str
    class_name: str
    latitude: float
    longitude: float
    confidence: float
    pixel_x: float
    pixel_y: float
    health: Optional[str] = None
    source: str = "ai"
    duplicate_count: int = 1
    related_leaf_id: Optional[str] = None
    layer_keys: list[str] = field(default_factory=list)
    polygon_wgs84: Optional[list[tuple[float, float]]] = None
    bbox_wgs84: Optional[list[tuple[float, float]]] = None


@dataclass
class MappingResult:
    records: list[DetectionRecord]
    counts: dict[str, int]
    json_path: str
    csv_path: str
    xlsx_path: str
    processed_images: int
    skipped_images: int
    warnings: list[str]
    qa_json_path: str = ""
    qa_crop_dir: str = ""

    def to_map_payload(self) -> dict:
        return {
            "records": [asdict(record) for record in self.records],
            "counts": self.counts,
            "jsonPath": self.json_path,
            "csvPath": self.csv_path,
            "xlsxPath": self.xlsx_path,
            "processedImages": self.processed_images,
            "skippedImages": self.skipped_images,
            "warnings": self.warnings[:20],
        }


@dataclass
class _LeafPrediction:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    polygon: list[tuple[float, float]]
    center: tuple[float, float]
    health: Optional[str] = None
    id: Optional[str] = None


@dataclass
class _DiseasePrediction:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]


@dataclass(frozen=True)
class MappingQaOptions:
    enabled: bool = False
    save_crops: bool = False
    crop_limit: int = 300
    event_limit: int = 5000
    crop_padding: int = 32


_ProgressCb = Optional[Callable[[int, str], None]]


def run_funnel_mapping_geotiff(
    geotiff_path: str | Path,
    leaf_model_path: str | Path,
    disease_model_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    confidence: float = 0.5,
    leaf_confidence: float | None = None,
    disease_confidence: float | None = None,
    slice_size: int = 512,
    slice_overlap: int = 96,
    duplicate_distance_m: float = 0.5,
    normalize_tiles: bool = False,
    match_disease_inside_cut_leaves: bool = False,
    include_unmatched_disease: bool = False,
    qa_enabled: bool = False,
    qa_save_crops: bool = False,
    device: str | int | None = None,
    progress_callback: _ProgressCb = None,
    should_stop: Optional[Callable[[], bool]] = None,
    scan_callback: Optional[Callable[[float, float, float, float], None]] = None,
) -> MappingResult:
    """Run the two-model leaf disease funnel across a single GeoTIFF file.

    Uses rasterio to read overlapping windowed tiles so that inference runs at
    the same scale the models were trained on (default 512x512).
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The ultralytics package is required for AI mapping. "
            "Install it with: pip install ultralytics"
        ) from exc

    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform_bounds

    geotiff_path = Path(geotiff_path)
    if not geotiff_path.exists():
        raise RuntimeError("The selected GeoTIFF file does not exist.")

    leaf_model_path = Path(leaf_model_path)
    disease_model_path = Path(disease_model_path)
    if not leaf_model_path.exists():
        raise RuntimeError("The selected leaf geometry model does not exist.")
    if not disease_model_path.exists():
        raise RuntimeError("The selected disease model does not exist.")

    leaf_confidence = confidence if leaf_confidence is None else max(0.01, min(0.99, float(leaf_confidence)))
    disease_confidence = confidence if disease_confidence is None else max(0.01, min(0.99, float(disease_confidence)))
    qa_options = MappingQaOptions(enabled=qa_enabled, save_crops=qa_save_crops)

    out_dir = Path(output_dir) if output_dir else geotiff_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out_dir / "qa_diagnostics"
    qa_crop_dir = qa_dir / "disease_crops"
    qa_json_path = ""
    qa_events: list[dict] = []
    qa_tile_summaries: list[dict] = []
    qa_crops_saved = 0
    qa_events_truncated = False
    qa_summary = {
        "tiles_prepared": 0,
        "tiles_processed": 0,
        "empty_tiles_skipped": 0,
        "inference_failed_tiles": 0,
        "raw_leaf_predictions": 0,
        "raw_disease_predictions": 0,
        "accepted_disease_predictions": 0,
        "rejected_disease_predictions": 0,
        "included_unmatched_disease_predictions": 0,
        "coordinate_projection_failures": 0,
        "matched_cut_leaf_disease_predictions": 0,
        "records_before_dedupe": 0,
        "records_after_dedupe": 0,
        "duplicates_merged": 0,
        "qa_crops_saved": 0,
    }
    if qa_options.enabled:
        qa_dir.mkdir(parents=True, exist_ok=True)
        if qa_options.save_crops:
            qa_crop_dir.mkdir(parents=True, exist_ok=True)

    _report(progress_callback, 2, "Loading YOLO models...")
    leaf_model = YOLO(str(leaf_model_path))
    disease_model = YOLO(str(disease_model_path))

    all_records: list[DetectionRecord] = []
    warnings: list[str] = []

    with rasterio.open(geotiff_path) as dataset:
        width = dataset.width
        height = dataset.height

        tiles = _generate_tiles(width, height, slice_size, slice_overlap)
        qa_summary["tiles_prepared"] = len(tiles)
        _report(progress_callback, 5, f"Prepared {len(tiles)} tiles for scanning...")

        for index, (x0, y0, x1, y1) in enumerate(tiles, start=1):
            if should_stop and should_stop():
                warnings.append("Processing was cancelled before all tiles finished.")
                break

            pct = 5 + int((index - 1) / max(1, len(tiles)) * 87)
            if index % 10 == 0 or index == 1 or index == len(tiles):
                _report(progress_callback, pct, f"SAHI scanning GeoTIFF tile {index}/{len(tiles)}...")

            window = Window(x0, y0, x1 - x0, y1 - y0)

            if scan_callback is not None:
                # Calculate lat/lon bounds of the current tile
                w_bounds = rasterio.windows.bounds(window, dataset.transform)
                try:
                    wgs84_bounds = transform_bounds(dataset.crs, "EPSG:4326", *w_bounds, densify_pts=21)
                    lon_min, lat_min, lon_max, lat_max = wgs84_bounds
                    scan_callback(lat_min, lon_min, lat_max, lon_max)
                except Exception:
                    pass

            # Read up to 3 bands (RGB)
            bands_to_read = min(3, dataset.count)
            try:
                tile_data = dataset.read(list(range(1, bands_to_read + 1)), window=window)
            except Exception as exc:
                warnings.append(f"Failed to read tile {index}: {exc}")
                continue

            tile_img = _tile_data_to_rgb(tile_data, normalize=normalize_tiles)

            # Skip empty tiles (all black or all white/transparent)
            if np.all(tile_img == 0) or np.all(tile_img == 255):
                qa_summary["empty_tiles_skipped"] += 1
                continue

            # Run inference
            try:
                leaf_result = leaf_model.predict(
                    source=tile_img,
                    imgsz=slice_size,
                    conf=leaf_confidence,
                    verbose=False,
                    device=device,
                )[0]
                leaves = _parse_leaf_predictions(leaf_result)
                
                disease_result = disease_model.predict(
                    source=tile_img,
                    imgsz=slice_size,
                    conf=disease_confidence,
                    verbose=False,
                    device=device,
                )[0]
                diseases = _parse_disease_predictions(disease_result)
            except Exception as exc:
                qa_summary["inference_failed_tiles"] += 1
                warnings.append(f"Inference failed on tile {index}: {exc}")
                continue

            qa_summary["tiles_processed"] += 1
            qa_summary["raw_leaf_predictions"] += len(leaves)
            qa_summary["raw_disease_predictions"] += len(diseases)
            tile_summary = {
                "tile_index": index,
                "pixel_bounds": [int(x0), int(y0), int(x1), int(y1)],
                "leaf_predictions": len(leaves),
                "disease_predictions": len(diseases),
                "accepted_diseases": 0,
                "rejected_diseases": 0,
                "included_unmatched_diseases": 0,
            }

            # Offset coordinates to full GeoTIFF extent
            for leaf in leaves:
                leaf.bbox = (leaf.bbox[0] + x0, leaf.bbox[1] + y0, leaf.bbox[2] + x0, leaf.bbox[3] + y0)
                leaf.polygon = [(px + x0, py + y0) for px, py in leaf.polygon]
                leaf.center = (leaf.center[0] + x0, leaf.center[1] + y0)
            
            for disease in diseases:
                disease.bbox = (disease.bbox[0] + x0, disease.bbox[1] + y0, disease.bbox[2] + x0, disease.bbox[3] + y0)
                disease.center = (disease.center[0] + x0, disease.center[1] + y0)

            # Funnel mapping and projection to WGS84
            full_leaves = [leaf for leaf in leaves if leaf.class_name == "full_leaf"]
            match_leaf_classes = {"full_leaf"}
            if match_disease_inside_cut_leaves:
                match_leaf_classes.add("cut_leaf")
            match_leaves = [leaf for leaf in leaves if leaf.class_name in match_leaf_classes]
            
            for i, leaf in enumerate(leaves, start=1):
                leaf.id = f"geotiff-tile{index}-leaf-{i}"
            
            for i, disease in enumerate(diseases, start=1):
                containing_leaf = _containing_leaf_for_disease(disease, match_leaves, x0, y0)
                px, py = disease.center
                try:
                    lat, lon = _raster_pixel_to_wgs84(dataset, px, py)
                    try:
                        lat_tl, lon_tl = _raster_pixel_to_wgs84(dataset, disease.bbox[0], disease.bbox[1])
                        lat_br, lon_br = _raster_pixel_to_wgs84(dataset, disease.bbox[2], disease.bbox[3])
                        d_bbox_wgs84 = [(lat_tl, lon_tl), (lat_br, lon_br)]
                    except Exception:
                        d_bbox_wgs84 = None
                except Exception:
                    qa_summary["coordinate_projection_failures"] += 1
                    qa_summary["rejected_disease_predictions"] += 1
                    tile_summary["rejected_diseases"] += 1
                    crop_path = ""
                    if qa_options.enabled and qa_options.save_crops and qa_crops_saved < qa_options.crop_limit:
                        crop_path = _save_qa_disease_crop(
                            tile_img, disease, qa_crop_dir, out_dir, index, len(qa_events) + 1, x0, y0, qa_options.crop_padding
                        )
                        if crop_path:
                            qa_crops_saved += 1
                    _append_qa_disease_event(
                        qa_events,
                        qa_options,
                        disease,
                        index,
                        (x0, y0, x1, y1),
                        "rejected",
                        "coordinate_projection_failed",
                        containing_leaf,
                        match_leaves,
                        None,
                        crop_path,
                    )
                    continue

                if containing_leaf is None:
                    if include_unmatched_disease:
                        qa_summary["included_unmatched_disease_predictions"] += 1
                        tile_summary["included_unmatched_diseases"] += 1
                        all_records.append(
                            DetectionRecord(
                                id=f"geotiff-tile{index}-disease-{i}-unmatched",
                                image_name=geotiff_path.name,
                                class_name=disease.class_name,
                                latitude=lat,
                                longitude=lon,
                                confidence=disease.confidence,
                                pixel_x=px,
                                pixel_y=py,
                                health="unmatched",
                                source="ai_unmatched",
                                layer_keys=[disease.class_name],
                                bbox_wgs84=d_bbox_wgs84,
                            )
                        )
                        crop_path = ""
                        if qa_options.enabled and qa_options.save_crops and qa_crops_saved < qa_options.crop_limit:
                            crop_path = _save_qa_disease_crop(
                                tile_img, disease, qa_crop_dir, out_dir, index, len(qa_events) + 1, x0, y0, qa_options.crop_padding
                            )
                            if crop_path:
                                qa_crops_saved += 1
                        _append_qa_disease_event(
                            qa_events,
                            qa_options,
                            disease,
                            index,
                            (x0, y0, x1, y1),
                            "included_unmatched",
                            "outside_leaf_filter",
                            None,
                            match_leaves,
                            (lat, lon),
                            crop_path,
                        )
                    else:
                        qa_summary["rejected_disease_predictions"] += 1
                        tile_summary["rejected_diseases"] += 1
                        crop_path = ""
                        if qa_options.enabled and qa_options.save_crops and qa_crops_saved < qa_options.crop_limit:
                            crop_path = _save_qa_disease_crop(
                                tile_img, disease, qa_crop_dir, out_dir, index, len(qa_events) + 1, x0, y0, qa_options.crop_padding
                            )
                            if crop_path:
                                qa_crops_saved += 1
                        _append_qa_disease_event(
                            qa_events,
                            qa_options,
                            disease,
                            index,
                            (x0, y0, x1, y1),
                            "rejected",
                            "outside_leaf_filter",
                            None,
                            match_leaves,
                            (lat, lon),
                            crop_path,
                        )
                    continue

                qa_summary["accepted_disease_predictions"] += 1
                tile_summary["accepted_diseases"] += 1
                if containing_leaf.class_name == "cut_leaf":
                    qa_summary["matched_cut_leaf_disease_predictions"] += 1
                if containing_leaf.class_name == "full_leaf":
                    containing_leaf.health = "diseased"
                crop_path = ""
                if qa_options.enabled and qa_options.save_crops and qa_crops_saved < qa_options.crop_limit:
                    crop_path = _save_qa_disease_crop(
                        tile_img, disease, qa_crop_dir, out_dir, index, len(qa_events) + 1, x0, y0, qa_options.crop_padding
                    )
                    if crop_path:
                        qa_crops_saved += 1
                _append_qa_disease_event(
                    qa_events,
                    qa_options,
                    disease,
                    index,
                    (x0, y0, x1, y1),
                    "accepted",
                    "",
                    containing_leaf,
                    match_leaves,
                    (lat, lon),
                    crop_path,
                )
                all_records.append(
                    DetectionRecord(
                        id=f"geotiff-tile{index}-disease-{i}",
                        image_name=geotiff_path.name,
                        class_name=disease.class_name,
                        latitude=lat,
                        longitude=lon,
                        confidence=disease.confidence,
                        pixel_x=px,
                        pixel_y=py,
                        health="diseased",
                        related_leaf_id=containing_leaf.id,
                        layer_keys=[disease.class_name],
                        bbox_wgs84=d_bbox_wgs84,
                    )
                )
                
            if qa_options.enabled:
                qa_tile_summaries.append(tile_summary)

            for leaf in leaves:
                if leaf.class_name == "full_leaf" and leaf.health is None:
                    leaf.health = "healthy"
                px, py = leaf.center
                try:
                    lat, lon = _raster_pixel_to_wgs84(dataset, px, py)
                    try:
                        lat_tl, lon_tl = _raster_pixel_to_wgs84(dataset, leaf.bbox[0], leaf.bbox[1])
                        lat_br, lon_br = _raster_pixel_to_wgs84(dataset, leaf.bbox[2], leaf.bbox[3])
                        l_bbox_wgs84 = [(lat_tl, lon_tl), (lat_br, lon_br)]
                    except Exception:
                        l_bbox_wgs84 = None
                except Exception:
                    continue

                layer_keys = [leaf.class_name]
                if leaf.class_name == "full_leaf" and leaf.health:
                    layer_keys.append(f"{leaf.health}_leaf")

                # Downsample the polygon (every 4th point) to save JSON size and processing time
                poly_wgs84 = []
                for px, py in leaf.polygon[::4]:
                    try:
                        p_lat, p_lon = _raster_pixel_to_wgs84(dataset, px, py)
                        poly_wgs84.append((p_lat, p_lon))
                    except Exception:
                        pass

                all_records.append(
                    DetectionRecord(
                        id=leaf.id or f"geotiff-tile{index}-leaf",
                        image_name=geotiff_path.name,
                        class_name=leaf.class_name,
                        latitude=lat,
                        longitude=lon,
                        confidence=leaf.confidence,
                        pixel_x=px,
                        pixel_y=py,
                        health=leaf.health,
                        layer_keys=layer_keys,
                        polygon_wgs84=poly_wgs84 if poly_wgs84 else None,
                        bbox_wgs84=l_bbox_wgs84,
                    )
                )

    if scan_callback is not None:
        scan_callback(0.0, 0.0, 0.0, 0.0)  # Clear scan box at the end

    _report(progress_callback, 92, "Removing duplicate coordinates...")
    qa_summary["records_before_dedupe"] = len(all_records)
    deduped_records = dedupe_records(all_records, duplicate_distance_m)
    qa_summary["records_after_dedupe"] = len(deduped_records)
    qa_summary["duplicates_merged"] = max(0, len(all_records) - len(deduped_records))
    qa_summary["qa_crops_saved"] = qa_crops_saved

    _report(progress_callback, 96, "Saving coordinate outputs...")
    counts = _count_records(deduped_records)
    json_path, csv_path, xlsx_path = save_mapping_outputs(out_dir, deduped_records, counts, warnings)
    if qa_options.enabled:
        qa_events_truncated = qa_summary["raw_disease_predictions"] > len(qa_events)
        qa_json_path = str(
            save_qa_diagnostics(
                qa_dir,
                geotiff_path,
                {
                    "leaf_model": str(leaf_model_path),
                    "disease_model": str(disease_model_path),
                    "leaf_confidence": leaf_confidence,
                    "disease_confidence": disease_confidence,
                    "slice_size": slice_size,
                    "slice_overlap": slice_overlap,
                    "duplicate_distance_m": duplicate_distance_m,
                    "normalize_tiles": normalize_tiles,
                    "match_disease_inside_cut_leaves": match_disease_inside_cut_leaves,
                    "include_unmatched_disease": include_unmatched_disease,
                    "qa_save_crops": qa_options.save_crops,
                    "device": str(device if device is not None else ""),
                },
                qa_summary,
                qa_tile_summaries,
                qa_events,
                warnings,
                qa_events_truncated,
            )
        )

    _report(progress_callback, 100, "Done.")
    return MappingResult(
        records=deduped_records,
        counts=counts,
        json_path=str(json_path),
        csv_path=str(csv_path),
        xlsx_path=str(xlsx_path),
        processed_images=1,
        skipped_images=0,
        warnings=warnings,
        qa_json_path=qa_json_path,
        qa_crop_dir=str(qa_crop_dir) if qa_options.enabled and qa_options.save_crops else "",
    )


def _raster_pixel_to_wgs84(dataset, pixel_x: float, pixel_y: float) -> tuple[float, float]:
    """Project model pixel coordinates from the source raster grid to WGS84.

    YOLO boxes are continuous image coordinates measured from the raster's
    upper-left pixel edge, so the affine transform is applied directly. Adding
    a half-pixel offset here would drift detections away from the actual model
    center and becomes visible on high-resolution orthomosaics.
    """
    from rasterio.warp import transform

    x_crs, y_crs = dataset.transform * (float(pixel_x), float(pixel_y))
    lon_arr, lat_arr = transform(dataset.crs, "EPSG:4326", [x_crs], [y_crs])
    return float(lat_arr[0]), float(lon_arr[0])


def _tile_data_to_rgb(tile_data: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """Convert raster bands to an RGB uint8 tile for YOLO inference."""

    if tile_data.shape[0] == 1:
        tile_data = np.repeat(tile_data, 3, axis=0)
    elif tile_data.shape[0] == 2:
        tile_data = np.vstack([tile_data[0:1], tile_data[0:1], tile_data[0:1]])
    elif tile_data.shape[0] > 3:
        tile_data = tile_data[:3]

    tile_img = np.transpose(tile_data, (1, 2, 0))
    if tile_img.dtype == np.uint8 and not normalize:
        return np.ascontiguousarray(tile_img)

    tile_float = tile_img.astype(np.float32, copy=False)
    if normalize:
        return _normalize_tile_rgb(tile_float)

    finite = tile_float[np.isfinite(tile_float)]
    if finite.size == 0:
        return np.zeros(tile_img.shape, dtype=np.uint8)
    max_value = float(finite.max())
    if max_value > 255:
        tile_float = tile_float / 256.0
    tile_float = np.nan_to_num(tile_float, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(tile_float, 0, 255).astype(np.uint8))


def _normalize_tile_rgb(tile_img: np.ndarray) -> np.ndarray:
    output = np.zeros(tile_img.shape, dtype=np.uint8)
    for channel_index in range(tile_img.shape[2]):
        channel = tile_img[:, :, channel_index]
        valid = np.isfinite(channel)
        if not np.any(valid):
            continue
        values = channel[valid]
        lo = float(np.nanpercentile(values, 2))
        hi = float(np.nanpercentile(values, 98))
        if hi <= lo:
            scaled = np.clip(channel, 0, 255)
        else:
            scaled = (channel - lo) * (255.0 / (hi - lo))
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0)
        output[:, :, channel_index] = np.clip(scaled, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(output)


def _containing_leaf_for_disease(
    disease: _DiseasePrediction,
    leaves: list[_LeafPrediction],
    _tile_x0: int,
    _tile_y0: int,
) -> _LeafPrediction | None:
    return next((leaf for leaf in leaves if _point_in_polygon(disease.center, leaf.polygon)), None)


def _nearest_leaf(
    disease: _DiseasePrediction,
    leaves: list[_LeafPrediction],
) -> tuple[_LeafPrediction | None, float | None]:
    if not leaves:
        return None, None
    dx, dy = disease.center
    nearest = min(leaves, key=lambda leaf: (leaf.center[0] - dx) ** 2 + (leaf.center[1] - dy) ** 2)
    distance = math.hypot(nearest.center[0] - dx, nearest.center[1] - dy)
    return nearest, distance


def _append_qa_disease_event(
    events: list[dict],
    qa_options: MappingQaOptions,
    disease: _DiseasePrediction,
    tile_index: int,
    tile_bounds: tuple[int, int, int, int],
    outcome: str,
    reason: str,
    matched_leaf: _LeafPrediction | None,
    candidate_leaves: list[_LeafPrediction],
    lat_lon: tuple[float, float] | None,
    crop_path: str = "",
) -> None:
    if not qa_options.enabled or len(events) >= qa_options.event_limit:
        return
    nearest, nearest_distance = _nearest_leaf(disease, candidate_leaves)
    event = {
        "tile_index": tile_index,
        "tile_pixel_bounds": [int(value) for value in tile_bounds],
        "class_name": disease.class_name,
        "confidence": disease.confidence,
        "bbox_pixel": [round(float(value), 3) for value in disease.bbox],
        "center_pixel": [round(float(value), 3) for value in disease.center],
        "outcome": outcome,
        "reason": reason,
        "matched_leaf_id": matched_leaf.id if matched_leaf else "",
        "matched_leaf_class": matched_leaf.class_name if matched_leaf else "",
        "nearest_leaf_id": nearest.id if nearest else "",
        "nearest_leaf_class": nearest.class_name if nearest else "",
        "nearest_leaf_center_distance_px": round(nearest_distance, 3) if nearest_distance is not None else None,
        "crop_path": crop_path,
    }
    if lat_lon is not None:
        event["latitude"] = lat_lon[0]
        event["longitude"] = lat_lon[1]
    events.append(event)


def _save_qa_disease_crop(
    tile_img: np.ndarray,
    disease: _DiseasePrediction,
    qa_crop_dir: Path,
    run_dir: Path,
    tile_index: int,
    event_index: int,
    tile_x0: int,
    tile_y0: int,
    padding: int,
) -> str:
    try:
        from PIL import Image
    except Exception:
        return ""

    x1, y1, x2, y2 = disease.bbox
    left = max(0, int(math.floor(x1 - tile_x0 - padding)))
    top = max(0, int(math.floor(y1 - tile_y0 - padding)))
    right = min(tile_img.shape[1], int(math.ceil(x2 - tile_x0 + padding)))
    bottom = min(tile_img.shape[0], int(math.ceil(y2 - tile_y0 + padding)))
    if right <= left or bottom <= top:
        return ""
    try:
        qa_crop_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tile{tile_index:05d}_event{event_index:05d}_{disease.class_name}.png"
        crop_path = qa_crop_dir / filename
        Image.fromarray(tile_img).crop((left, top, right, bottom)).save(crop_path)
        return crop_path.relative_to(run_dir).as_posix()
    except OSError:
        return ""


def save_qa_diagnostics(
    qa_dir: str | Path,
    geotiff_path: str | Path,
    settings: dict,
    summary: dict,
    tile_summaries: list[dict],
    disease_events: list[dict],
    warnings: list[str],
    events_truncated: bool,
) -> Path:
    output_dir = Path(qa_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "geotiff": str(geotiff_path),
        "settings": settings,
        "summary": summary,
        "events_truncated": events_truncated,
        "tile_summaries": tile_summaries,
        "disease_events": disease_events,
        "warnings": warnings,
    }
    path = output_dir / "banana_ai_mapping_qa_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def dedupe_records(records: Iterable[DetectionRecord], distance_m: float) -> list[DetectionRecord]:
    """Merge same-layer detections that land within a small GPS radius."""

    if distance_m <= 0:
        return list(records)

    cell_degrees = max(distance_m / 111_320.0, 1e-9)
    buckets: dict[tuple[str, int, int], list[int]] = {}
    unique: list[DetectionRecord] = []

    for record in records:
        bucket_x = math.floor(record.longitude / cell_degrees)
        bucket_y = math.floor(record.latitude / cell_degrees)
        duplicate_index: Optional[int] = None

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (record.class_name, bucket_x + dx, bucket_y + dy)
                for candidate_index in buckets.get(key, []):
                    candidate = unique[candidate_index]
                    if haversine_m(
                        record.latitude,
                        record.longitude,
                        candidate.latitude,
                        candidate.longitude,
                    ) <= distance_m:
                        duplicate_index = candidate_index
                        break
                if duplicate_index is not None:
                    break
            if duplicate_index is not None:
                break

        if duplicate_index is None:
            key = (record.class_name, bucket_x, bucket_y)
            buckets.setdefault(key, []).append(len(unique))
            unique.append(record)
        else:
            existing = unique[duplicate_index]
            total = existing.duplicate_count + 1
            existing.latitude = (
                existing.latitude * existing.duplicate_count + record.latitude
            ) / total
            existing.longitude = (
                existing.longitude * existing.duplicate_count + record.longitude
            ) / total
            existing.confidence = max(existing.confidence, record.confidence)
            existing.duplicate_count = total

    return unique


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def save_mapping_outputs(
    output_dir: str | Path,
    records: list[DetectionRecord],
    counts: dict[str, int],
    warnings: list[str],
) -> tuple[Path, Path, Path]:
    out_dir = Path(output_dir)
    json_path = out_dir / "banana_ai_mapping_results.json"
    csv_path = out_dir / "banana_ai_mapping_results.csv"
    xlsx_path = out_dir / "banana_ai_mapping_results.xlsx"

    payload = {
        "counts": counts,
        "warnings": warnings,
        "records": [_record_output_dict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_record_output_dict(record, layer_separator="|"))

    _write_xlsx(xlsx_path, _OUTPUT_FIELDS, [_record_output_dict(record, layer_separator="|") for record in records])

    return json_path, csv_path, xlsx_path


_OUTPUT_FIELDS = [
    "id",
    "image_name",
    "class_name",
    "health",
    "source",
    "latitude",
    "longitude",
    "confidence",
    "pixel_x",
    "pixel_y",
    "duplicate_count",
    "related_leaf_id",
    "layer_keys",
    "polygon_wgs84",
    "bbox_wgs84",
]


def _record_output_dict(record: DetectionRecord, layer_separator: str | None = None) -> dict:
    row = asdict(record)
    if layer_separator is not None:
        row["layer_keys"] = layer_separator.join(record.layer_keys)
    return {field: row.get(field) for field in _OUTPUT_FIELDS}


def _write_xlsx(path: Path, headers: list[str], rows: list[dict]) -> None:
    """Write a minimal XLSX workbook using only the Python standard library."""
    sheet_rows = [headers] + [[row.get(header, "") for header in headers] for row in rows]
    sheet_xml = _worksheet_xml(sheet_rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", _ROOT_RELS_XML)
        zf.writestr("xl/workbook.xml", _WORKBOOK_XML)
        zf.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS_XML)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _worksheet_xml(rows: list[list[object]]) -> str:
    body = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{_column_name(c_idx)}{r_idx}"
            cells.append(
                f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value or ""))}</t></is></c>'
            )
        body.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(body)
        + '</sheetData></worksheet>'
    )


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


_CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""

_ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

_WORKBOOK_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="AI Mapping Results" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""

_WORKBOOK_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""


def _parse_leaf_predictions(result: object) -> list[_LeafPrediction]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", {}) or {}
    xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
    classes = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else []
    confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else []
    mask_polygons = _mask_polygons(result)

    leaves: list[_LeafPrediction] = []
    for i, bbox_arr in enumerate(xyxy):
        class_id = int(classes[i]) if i < len(classes) else -1
        class_name = _canonical_class_name(class_id, names, LEAF_ALIASES, DEFAULT_LEAF_IDS)
        if class_name not in {"full_leaf", "cut_leaf"}:
            continue

        x1, y1, x2, y2 = (float(v) for v in bbox_arr)
        polygon = mask_polygons[i] if i < len(mask_polygons) and mask_polygons[i] else [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]
        center = _polygon_centroid(polygon)
        leaves.append(
            _LeafPrediction(
                class_name=class_name,
                confidence=float(confs[i]) if i < len(confs) else 0.0,
                bbox=(x1, y1, x2, y2),
                polygon=polygon,
                center=center,
            )
        )
    return leaves


def _parse_disease_predictions(result: object) -> list[_DiseasePrediction]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", {}) or {}
    xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
    classes = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else []
    confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else []

    diseases: list[_DiseasePrediction] = []
    for i, bbox_arr in enumerate(xyxy):
        class_id = int(classes[i]) if i < len(classes) else -1
        class_name = _canonical_class_name(class_id, names, DISEASE_ALIASES, DEFAULT_DISEASE_IDS)
        if class_name not in {"black_sigatoka", "panama"}:
            continue

        x1, y1, x2, y2 = (float(v) for v in bbox_arr)
        diseases.append(
            _DiseasePrediction(
                class_name=class_name,
                confidence=float(confs[i]) if i < len(confs) else 0.0,
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            )
        )
    return diseases


def _mask_polygons(result: object) -> list[list[tuple[float, float]]]:
    masks = getattr(result, "masks", None)
    polygons = getattr(masks, "xy", None)
    if polygons is None:
        return []
    return [[(float(x), float(y)) for x, y in polygon] for polygon in polygons]


def _canonical_class_name(
    class_id: int,
    names: dict,
    aliases: dict[str, set[str]],
    default_ids: dict[int, str],
) -> str:
    raw_name = str(names.get(class_id, "")) if hasattr(names, "get") else ""
    normalized = _normalize_name(raw_name)
    for canonical, valid_aliases in aliases.items():
        if normalized in valid_aliases:
            return canonical
    return default_ids.get(class_id, normalized)


def _normalize_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _polygon_centroid(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    if not polygon:
        return 0.0, 0.0
    area = 0.0
    cx = 0.0
    cy = 0.0
    j = len(polygon) - 1
    for i in range(len(polygon)):
        x0, y0 = polygon[j]
        x1, y1 = polygon[i]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
        j = i
    if abs(area) < 1e-9:
        return (
            sum(x for x, _ in polygon) / len(polygon),
            sum(y for _, y in polygon) / len(polygon),
        )
    area *= 0.5
    return cx / (6.0 * area), cy / (6.0 * area)


def _count_records(records: list[DetectionRecord]) -> dict[str, int]:
    counts = {
        "full_leaf": 0,
        "cut_leaf": 0,
        "healthy_leaf": 0,
        "diseased_leaf": 0,
        "black_sigatoka": 0,
        "panama": 0,
        "total": len(records),
    }
    for record in records:
        for key in record.layer_keys:
            if key in counts:
                counts[key] += 1
    return counts


# GeoTIFF tiling helpers


def _generate_tiles(
    img_width: int,
    img_height: int,
    slice_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Return a list of (x0, y0, x1, y1) tile rectangles covering the image.

    Edge tiles are shifted inward so every tile is exactly *slice_size*
    pixels wide/tall (unless the image dimension is smaller than slice_size).
    """
    step = max(1, slice_size - overlap)
    tiles: list[tuple[int, int, int, int]] = []

    y_starts: list[int] = []
    y = 0
    while y < img_height:
        y_end = y + slice_size
        if y_end > img_height:
            y = max(0, img_height - slice_size)
            y_starts.append(y)
            break
        y_starts.append(y)
        y += step
    if not y_starts:
        y_starts = [0]

    x_starts: list[int] = []
    x = 0
    while x < img_width:
        x_end = x + slice_size
        if x_end > img_width:
            x = max(0, img_width - slice_size)
            x_starts.append(x)
            break
        x_starts.append(x)
        x += step
    if not x_starts:
        x_starts = [0]

    seen: set[tuple[int, int]] = set()
    for y0 in y_starts:
        for x0 in x_starts:
            if (x0, y0) in seen:
                continue
            seen.add((x0, y0))
            x1 = min(x0 + slice_size, img_width)
            y1 = min(y0 + slice_size, img_height)
            tiles.append((x0, y0, x1, y1))
    return tiles


def _report(cb: _ProgressCb, percent: int, message: str) -> None:
    if cb is not None:
        cb(percent, message)
