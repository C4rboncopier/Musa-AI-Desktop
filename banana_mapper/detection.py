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


_ProgressCb = Optional[Callable[[int, str], None]]


def run_funnel_mapping_geotiff(
    geotiff_path: str | Path,
    leaf_model_path: str | Path,
    disease_model_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    confidence: float = 0.5,
    slice_size: int = 512,
    slice_overlap: int = 96,
    duplicate_distance_m: float = 0.5,
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

    out_dir = Path(output_dir) if output_dir else geotiff_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    _report(progress_callback, 2, "Loading YOLO models...")
    leaf_model = YOLO(str(leaf_model_path))
    disease_model = YOLO(str(disease_model_path))

    all_records: list[DetectionRecord] = []
    warnings: list[str] = []

    with rasterio.open(geotiff_path) as dataset:
        width = dataset.width
        height = dataset.height

        tiles = _generate_tiles(width, height, slice_size, slice_overlap)
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

            if tile_data.shape[0] == 1:
                tile_data = np.repeat(tile_data, 3, axis=0)
            elif tile_data.shape[0] == 2:
                tile_data = np.vstack([tile_data[0:1], tile_data[0:1], tile_data[0:1]])

            tile_img = np.transpose(tile_data, (1, 2, 0))

            if tile_img.dtype != np.uint8:
                if tile_img.max() > 255:
                    tile_img = (tile_img / 256).astype(np.uint8)
                else:
                    tile_img = tile_img.astype(np.uint8)

            # Skip empty tiles (all black or all white/transparent)
            if np.all(tile_img == 0) or np.all(tile_img == 255):
                continue

            # Run inference
            try:
                leaf_result = leaf_model.predict(
                    source=tile_img,
                    imgsz=slice_size,
                    conf=confidence,
                    verbose=False,
                    device=device,
                )[0]
                leaves = _parse_leaf_predictions(leaf_result)
                
                disease_result = disease_model.predict(
                    source=tile_img,
                    imgsz=slice_size,
                    conf=confidence,
                    verbose=False,
                    device=device,
                )[0]
                diseases = _parse_disease_predictions(disease_result)
            except Exception as exc:
                warnings.append(f"Inference failed on tile {index}: {exc}")
                continue

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
            
            for i, leaf in enumerate(leaves, start=1):
                leaf.id = f"geotiff-tile{index}-leaf-{i}"
            
            for i, disease in enumerate(diseases, start=1):
                containing_leaf = next(
                    (lf for lf in full_leaves if _point_in_polygon((disease.center[0] - x0, disease.center[1] - y0), [(p[0]-x0, p[1]-y0) for p in lf.polygon])),
                    None,
                )
                if containing_leaf is None:
                    continue

                containing_leaf.health = "diseased"
                px, py = disease.center
                try:
                    lat, lon = _raster_pixel_to_wgs84(dataset, px, py)
                except Exception:
                    continue

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
                    )
                )
                
            for leaf in leaves:
                if leaf.class_name == "full_leaf" and leaf.health is None:
                    leaf.health = "healthy"
                px, py = leaf.center
                try:
                    lat, lon = _raster_pixel_to_wgs84(dataset, px, py)
                except Exception:
                    continue

                layer_keys = [leaf.class_name]
                if leaf.class_name == "full_leaf" and leaf.health:
                    layer_keys.append(f"{leaf.health}_leaf")

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
                    )
                )

    if scan_callback is not None:
        scan_callback(0.0, 0.0, 0.0, 0.0)  # Clear scan box at the end

    _report(progress_callback, 92, "Removing duplicate coordinates...")
    deduped_records = dedupe_records(all_records, duplicate_distance_m)

    _report(progress_callback, 96, "Saving coordinate outputs...")
    counts = _count_records(deduped_records)
    json_path, csv_path, xlsx_path = save_mapping_outputs(out_dir, deduped_records, counts, warnings)

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
    "latitude",
    "longitude",
    "confidence",
    "pixel_x",
    "pixel_y",
    "duplicate_count",
    "related_leaf_id",
    "layer_keys",
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

