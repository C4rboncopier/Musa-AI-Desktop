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
from PIL import ExifTags, Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
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


@dataclass(frozen=True)
class DroneMetadata:
    latitude: float
    longitude: float
    altitude_m: float
    focal_length_mm: float
    focal_35mm: Optional[float]
    calibrated_focal_px: Optional[float]
    sensor_width_mm: float
    sensor_height_mm: float
    yaw_degrees: float
    pitch_degrees: float
    image_width: int
    image_height: int
    used_rtk: bool
    rtk_source: str = ""
    rtk_sequence: Optional[int] = None
    rtk_latitude: Optional[float] = None
    rtk_longitude: Optional[float] = None

    @property
    def ground_width_m(self) -> float:
        if self.calibrated_focal_px and self.calibrated_focal_px > 0:
            return (self.image_width * self.altitude_m) / self.calibrated_focal_px
        return (self.sensor_width_mm * self.altitude_m) / self.focal_length_mm

    @property
    def ground_height_m(self) -> float:
        if self.calibrated_focal_px and self.calibrated_focal_px > 0:
            return (self.image_height * self.altitude_m) / self.calibrated_focal_px
        return (self.sensor_height_mm * self.altitude_m) / self.focal_length_mm


@dataclass(frozen=True)
class MRKCoordinateMatch:
    sequence: int
    latitude: float
    longitude: float
    source_path: str
    line_number: int


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


def run_funnel_mapping(
    image_folder: str | Path,
    leaf_model_path: str | Path,
    disease_model_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    confidence: float = 0.5,
    slice_size: int = 512,
    slice_overlap: int = 96,
    nms_iou_threshold: float = 0.45,
    duplicate_distance_m: float = 0.5,
    progress_callback: _ProgressCb = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> MappingResult:
    """Run the two-model leaf disease funnel across a folder of drone images.

    Uses SAHI (Slicing Aided Hyper Inference) to scan each full-resolution
    image as overlapping tiles so that inference runs at the same scale the
    models were trained on (default 512x512), improving small-object recall.
    """

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "The ultralytics package is required for AI mapping. "
            "Install it with: pip install ultralytics"
        ) from exc

    folder = Path(image_folder)
    if not folder.exists() or not folder.is_dir():
        raise RuntimeError("The selected unstitched image folder does not exist.")

    leaf_model_path = Path(leaf_model_path)
    disease_model_path = Path(disease_model_path)
    if not leaf_model_path.exists():
        raise RuntimeError("The selected leaf geometry model does not exist.")
    if not disease_model_path.exists():
        raise RuntimeError("The selected disease model does not exist.")

    images = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError("No supported image files were found in the selected folder.")

    out_dir = Path(output_dir) if output_dir else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    _report(progress_callback, 2, "Loading YOLO models...")
    leaf_model = YOLO(str(leaf_model_path))
    disease_model = YOLO(str(disease_model_path))

    all_records: list[DetectionRecord] = []
    warnings: list[str] = []
    processed_images = 0
    skipped_images = 0

    for index, image_path in enumerate(images, start=1):
        if should_stop and should_stop():
            warnings.append("Processing was cancelled before all images finished.")
            break

        pct = 5 + int((index - 1) / max(1, len(images)) * 86)
        _report(
            progress_callback, pct,
            f"SAHI scanning {image_path.name} ({index}/{len(images)})...",
        )

        metadata = extract_drone_metadata(image_path)
        if metadata is None:
            skipped_images += 1
            warnings.append(f"Skipped {image_path.name}: {_metadata_skip_reason(image_path)}.")
            continue

        try:
            leaves = _sahi_leaf_inference(
                leaf_model, image_path,
                slice_size=slice_size, overlap=slice_overlap,
                conf=confidence, iou_threshold=nms_iou_threshold,
            )
            diseases = _sahi_disease_inference(
                disease_model, image_path,
                slice_size=slice_size, overlap=slice_overlap,
                conf=confidence, iou_threshold=nms_iou_threshold,
            )
        except Exception as exc:
            skipped_images += 1
            warnings.append(f"Skipped {image_path.name}: SAHI inference failed ({exc}).")
            continue

        records = _records_from_funnel(image_path.name, metadata, leaves, diseases)
        all_records.extend(records)
        processed_images += 1

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
        processed_images=processed_images,
        skipped_images=skipped_images,
        warnings=warnings,
    )


def run_funnel_mapping_geotiff(
    geotiff_path: str | Path,
    leaf_model_path: str | Path,
    disease_model_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    confidence: float = 0.5,
    slice_size: int = 512,
    slice_overlap: int = 96,
    nms_iou_threshold: float = 0.45,
    duplicate_distance_m: float = 0.5,
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
                leaf_result = leaf_model.predict(source=tile_img, imgsz=slice_size, conf=confidence, verbose=False)[0]
                leaves = _parse_leaf_predictions(leaf_result)
                
                disease_result = disease_model.predict(source=tile_img, imgsz=slice_size, conf=confidence, verbose=False)[0]
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


def _get_rtk_coordinates(
    image_path: Path,
    mrk_path: str | Path | None = None,
) -> tuple[Optional[float], Optional[float]]:
    """Parse exact camera RTK coordinates from a DJI .MRK file if available."""
    match = _get_rtk_match(image_path, mrk_path)
    if match is None:
        return None, None
    return match.latitude, match.longitude


def _get_rtk_match(
    image_path: Path,
    mrk_path: str | Path | None = None,
) -> MRKCoordinateMatch | None:
    """Return the linked or nearby MRK row that matches *image_path*."""
    try:
        sequence = _image_sequence_number(image_path)
        if sequence is None:
            return None

        mrk_files: list[Path] = []
        if mrk_path:
            linked_mrk = Path(mrk_path)
            if linked_mrk.exists():
                mrk_files.append(linked_mrk)

        # Fallback for older projects: search near the image if no explicit MRK
        # file is linked.
        if not mrk_files:
            search_dirs = [image_path.parent, image_path.parent.parent, Path.cwd()]
            for directory in search_dirs:
                if directory.exists():
                    mrk_files.extend(directory.glob("*.MRK"))
                    mrk_files.extend(directory.glob("*.mrk"))

        for mrk_file in mrk_files:
            with open(mrk_file, "r", encoding="utf-8", errors="ignore") as f:
                for line_number, line in enumerate(f, start=1):
                    lat_lon = _parse_mrk_line(line, sequence)
                    if lat_lon is not None:
                        lat, lon = lat_lon
                        return MRKCoordinateMatch(
                            sequence=sequence,
                            latitude=lat,
                            longitude=lon,
                            source_path=str(mrk_file),
                            line_number=line_number,
                        )
    except Exception:
        pass
    return None


def _image_sequence_number(image_path: Path) -> int | None:
    match = re.search(r"_(\d+)(?:_[a-zA-Z0-9]+)?\.[a-zA-Z0-9]+$", image_path.name)
    if match:
        return int(match.group(1))
    fallback = re.search(r"(\d+)(?=\.[a-zA-Z0-9]+$)", image_path.name)
    return int(fallback.group(1)) if fallback else None


def _parse_mrk_line(line: str, sequence: int) -> tuple[float, float] | None:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        if int(float(parts[0].rstrip(","))) != sequence:
            return None
    except ValueError:
        return None

    lat = lon = None
    for part in parts:
        token = part.strip()
        if token.endswith(",Lat"):
            lat = _safe_float(token.replace(",Lat", ""))
        elif token.endswith(",Lon"):
            lon = _safe_float(token.replace(",Lon", ""))
    if _valid_lat_lon(lat, lon):
        return float(lat), float(lon)

    values = [_safe_float(match.group(0)) for match in re.finditer(r"-?\d+(?:\.\d+)?", line)]
    numbers = [value for value in values if value is not None]
    # Common DJI MRK rows are: sequence, timestamp, latitude, longitude, altitude...
    for index in range(2, max(2, len(numbers) - 1)):
        lat_candidate = numbers[index]
        lon_candidate = numbers[index + 1]
        if _valid_lat_lon(lat_candidate, lon_candidate):
            return float(lat_candidate), float(lon_candidate)
    return None


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_lat_lon(lat: float | None, lon: float | None) -> bool:
    return lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180


def extract_drone_metadata(path: str | Path, mrk_path: str | Path | None = None) -> Optional[DroneMetadata]:
    """Extract GPS, camera, and flight footprint metadata from image EXIF/XMP."""

    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            exif = image.getexif()
            gps = _extract_exif_gps(exif)
            focal_length = _exif_float(_exif_value(exif, "FocalLength"))
            focal_35mm = _exif_float(_exif_value(exif, "FocalLengthIn35mmFilm"))
            xmp = _read_xmp_header(image_path)
    except Exception:
        return None

    xmp_lat = _xmp_float(xmp, ["GpsLatitude", "GPSLatitude", "Latitude"])
    xmp_lon = _xmp_float(xmp, ["GpsLongitude", "GPSLongitude", "GpsLongtitude", "Longitude"])
    lat = xmp_lat if xmp_lat is not None else (gps[0] if gps else None)
    lon = xmp_lon if xmp_lon is not None else (gps[1] if gps else None)

    # --- Override with highly accurate RTK data from .MRK file if available ---
    used_rtk = False
    rtk_source = ""
    rtk_sequence: int | None = None
    rtk_lat: float | None = None
    rtk_lon: float | None = None
    rtk_match = _get_rtk_match(image_path, mrk_path)
    if rtk_match is not None:
        rtk_lat = rtk_match.latitude
        rtk_lon = rtk_match.longitude
        lat = rtk_match.latitude
        lon = rtk_match.longitude
        rtk_source = rtk_match.source_path
        rtk_sequence = rtk_match.sequence
        used_rtk = True

    altitude = (
        _first_number(
            _xmp_float(xmp, ["RelativeAltitude"]),
            _xmp_float(xmp, ["AbsoluteAltitude"]),
            gps[2] if gps and len(gps) > 2 else None,
        )
    )
    xmp_focal_mm = _xmp_float(xmp, ["FocalLength", "FocalLengthIn35mmFilm"])
    focal_length = _first_number(xmp_focal_mm, focal_length)
    calibrated_focal_px = _xmp_float(xmp, ["CalibratedFocalLength"])

    yaw = (
        _xmp_float(xmp, ["GimbalYawDegree", "FlightYawDegree"])
        or 0.0
    )
    pitch = (
        _xmp_float(xmp, ["GimbalPitchDegree", "FlightPitchDegree"])
        or -90.0
    )

    if lat is None or lon is None or altitude is None:
        return None
    if altitude <= 0:
        return None
    if (focal_length is None or focal_length <= 0) and (
        calibrated_focal_px is None or calibrated_focal_px <= 0
    ):
        return None

    if focal_length is None or focal_length <= 0:
        focal_length = 1.0

    sensor_w, sensor_h = _sensor_size_from_35mm(focal_length, focal_35mm)
    return DroneMetadata(
        latitude=lat,
        longitude=lon,
        altitude_m=abs(altitude),
        focal_length_mm=focal_length,
        focal_35mm=focal_35mm,
        calibrated_focal_px=calibrated_focal_px,
        sensor_width_mm=sensor_w,
        sensor_height_mm=sensor_h,
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        image_width=width,
        image_height=height,
        used_rtk=used_rtk,
        rtk_source=rtk_source,
        rtk_sequence=rtk_sequence,
        rtk_latitude=rtk_lat,
        rtk_longitude=rtk_lon,
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


def pixel_to_lat_lon(x: float, y: float, metadata: DroneMetadata) -> tuple[float, float]:
    """Project an image pixel to an approximate WGS84 coordinate from drone metadata.

    Coordinate system:
      - Image +x  → camera right  → geographic East  (when yaw = 0, drone faces North)
      - Image +y  → camera down   → geographic South (when yaw = 0)
    Drone yaw is a CW compass bearing: 0 = North, 90 = East, 180 = South.

    Camera axes in geographic coordinates when the drone heading is *yaw*:
      camera-up   (top of image)  → heading direction:
          East component = sin(yaw),  North component = cos(yaw)
      camera-right (right of image) → 90° CW from heading:
          East component = cos(yaw),  North component = -sin(yaw)

    Therefore the CW rotation from camera-local to geographic frame is:
      geo_east  =  cos(yaw) * cam_right  +  sin(yaw) * cam_up
      geo_north = -sin(yaw) * cam_right  +  cos(yaw) * cam_up
    """
    # Pixel offset from image centre in metres (camera frame)
    cam_right = (x / metadata.image_width  - 0.5) * metadata.ground_width_m
    cam_up    = -(y / metadata.image_height - 0.5) * metadata.ground_height_m  # invert Y

    # If the camera is not pointing perfectly straight down (-90), the center of the image 
    # is projected forward along the drone's heading.
    safe_pitch = max(-90.0, min(-10.0, metadata.pitch_degrees))
    pitch_offset = metadata.altitude_m * math.tan(math.radians(safe_pitch + 90.0))
    cam_up += pitch_offset

    # Rotate camera frame CW by yaw → geographic frame
    yaw   = math.radians(metadata.yaw_degrees)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    geo_east  =  cos_y * cam_right + sin_y * cam_up
    geo_north = -sin_y * cam_right + cos_y * cam_up

    # Convert metric offsets to degrees using spherical Earth
    earth_radius = 6_378_137.0
    lat = metadata.latitude + (geo_north / earth_radius) * (180.0 / math.pi)
    lon = metadata.longitude + (
        geo_east / (earth_radius * math.cos(math.radians(metadata.latitude)))
    ) * (180.0 / math.pi)
    return lat, lon


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


def _records_from_funnel(
    image_name: str,
    metadata: DroneMetadata,
    leaves: list[_LeafPrediction],
    diseases: list[_DiseasePrediction],
) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    full_leaves = [leaf for leaf in leaves if leaf.class_name == "full_leaf"]

    for index, leaf in enumerate(leaves, start=1):
        leaf.id = f"{Path(image_name).stem}-leaf-{index}"

    for disease_index, disease in enumerate(diseases, start=1):
        containing_leaf = next(
            (
                leaf for leaf in full_leaves
                if _point_in_polygon(disease.center, leaf.polygon)
            ),
            None,
        )
        if containing_leaf is None:
            continue

        containing_leaf.health = "diseased"
        lat, lon = pixel_to_lat_lon(disease.center[0], disease.center[1], metadata)
        records.append(
            DetectionRecord(
                id=f"{Path(image_name).stem}-disease-{disease_index}",
                image_name=image_name,
                class_name=disease.class_name,
                latitude=lat,
                longitude=lon,
                confidence=disease.confidence,
                pixel_x=disease.center[0],
                pixel_y=disease.center[1],
                health="diseased",
                related_leaf_id=containing_leaf.id,
                layer_keys=[disease.class_name],
            )
        )

    for leaf in leaves:
        if leaf.class_name == "full_leaf" and leaf.health is None:
            leaf.health = "healthy"
        lat, lon = pixel_to_lat_lon(leaf.center[0], leaf.center[1], metadata)
        layer_keys = [leaf.class_name]
        if leaf.class_name == "full_leaf" and leaf.health:
            layer_keys.append(f"{leaf.health}_leaf")
        records.append(
            DetectionRecord(
                id=leaf.id or f"{Path(image_name).stem}-leaf",
                image_name=image_name,
                class_name=leaf.class_name,
                latitude=lat,
                longitude=lon,
                confidence=leaf.confidence,
                pixel_x=leaf.center[0],
                pixel_y=leaf.center[1],
                health=leaf.health,
                layer_keys=layer_keys,
            )
        )

    return records


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


def _extract_exif_gps(exif: Image.Exif | None) -> Optional[tuple[float, float, Optional[float]]]:
    if not exif:
        return None

    gps_info = None
    try:
        gps_info = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        gps_tag = _tag_id("GPSInfo")
        gps_info = exif.get(gps_tag)

    if not gps_info:
        return None

    gps_map = {}
    for key, value in gps_info.items():
        gps_map[ExifTags.GPSTAGS.get(key, key)] = value

    lat = _dms_to_decimal(gps_map.get("GPSLatitude"), gps_map.get("GPSLatitudeRef"))
    lon = _dms_to_decimal(gps_map.get("GPSLongitude"), gps_map.get("GPSLongitudeRef"))
    altitude = _exif_float(gps_map.get("GPSAltitude"))
    alt_ref = gps_map.get("GPSAltitudeRef")
    if altitude is not None and alt_ref in (1, b"\x01"):
        altitude *= -1

    if lat is None or lon is None:
        return None
    return lat, lon, altitude


def _dms_to_decimal(value: object, ref: object) -> Optional[float]:
    if value is None:
        return None
    try:
        degrees, minutes, seconds = (_exif_float(part) for part in value)
        if degrees is None or minutes is None or seconds is None:
            return None
        decimal = degrees + minutes / 60.0 + seconds / 3600.0
        ref_text = ref.decode(errors="ignore") if isinstance(ref, bytes) else str(ref)
        if ref_text.upper() in {"S", "W"}:
            decimal *= -1
        return decimal
    except Exception:
        return None


def _read_xmp_header(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            raw = handle.read(1_048_576)
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _xmp_float(xmp: str, names: list[str]) -> Optional[float]:
    for name in names:
        local_name = re.escape(name.split(":")[-1])
        patterns = [
            rf'(?:[\w.-]+:)?{local_name}\s*=\s*"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"',
            rf"<(?:[\w.-]+:)?{local_name}>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*</(?:[\w.-]+:)?{local_name}>",
        ]
        for pattern in patterns:
            match = re.search(pattern, xmp, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
    return None


def _exif_value(exif: Image.Exif | None, tag_name: str) -> object:
    if not exif:
        return None
    tag = _tag_id(tag_name)
    if tag in exif:
        return exif.get(tag)

    for ifd in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
        try:
            data = exif.get_ifd(ifd)
        except Exception:
            continue
        if tag in data:
            return data.get(tag)
    return None


def _first_number(*values: Optional[float]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def _metadata_skip_reason(path: str | Path) -> str:
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            gps = _extract_exif_gps(exif)
            focal_length = _exif_float(_exif_value(exif, "FocalLength"))
            xmp = _read_xmp_header(image_path)
    except Exception as exc:
        return f"metadata could not be read ({exc})"

    lat = _first_number(_xmp_float(xmp, ["GpsLatitude", "GPSLatitude", "Latitude"]), gps[0] if gps else None)
    lon = _first_number(_xmp_float(xmp, ["GpsLongitude", "GPSLongitude", "GpsLongtitude", "Longitude"]), gps[1] if gps else None)
    altitude = _first_number(
        _xmp_float(xmp, ["RelativeAltitude"]),
        _xmp_float(xmp, ["AbsoluteAltitude"]),
        gps[2] if gps and len(gps) > 2 else None,
    )
    calibrated_focal_px = _xmp_float(xmp, ["CalibratedFocalLength"])

    missing = []
    if lat is None or lon is None:
        missing.append("GPS latitude/longitude")
    if altitude is None:
        missing.append("altitude")
    if (focal_length is None or focal_length <= 0) and (
        calibrated_focal_px is None or calibrated_focal_px <= 0
    ):
        missing.append("focal length")
    if not missing:
        return "metadata was present but failed validation"
    return "missing " + ", ".join(missing)


def _exif_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except Exception:
        try:
            return float(value.numerator) / float(value.denominator)
        except Exception:
            return None


def _tag_id(tag_name: str) -> int:
    for key, value in ExifTags.TAGS.items():
        if value == tag_name:
            return key
    return -1


def _sensor_size_from_35mm(
    focal_length: float,
    focal_35mm: Optional[float],
) -> tuple[float, float]:
    if focal_35mm and focal_35mm > 0:
        crop_factor = focal_35mm / focal_length
        if crop_factor > 0:
            return 36.0 / crop_factor, 24.0 / crop_factor
    return 35.9, 24.0


# ── SAHI (Slicing Aided Hyper Inference) helpers ─────────────────────


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


def _sahi_leaf_inference(
    model: object,
    image_path: Path,
    *,
    slice_size: int = 512,
    overlap: int = 96,
    conf: float = 0.5,
    iou_threshold: float = 0.45,
) -> list[_LeafPrediction]:
    """Run the leaf segmentation model on overlapping tiles and merge results."""
    img = Image.open(image_path)
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    tiles = _generate_tiles(w, h, slice_size, overlap)

    all_leaves: list[_LeafPrediction] = []
    for x0, y0, x1, y1 in tiles:
        tile = img_np[y0:y1, x0:x1]
        result = model.predict(
            source=tile, imgsz=slice_size, conf=conf, verbose=False,
        )[0]
        leaves = _parse_leaf_predictions(result)
        # Offset tile-local coordinates back to full-image space
        for leaf in leaves:
            leaf.bbox = (
                leaf.bbox[0] + x0, leaf.bbox[1] + y0,
                leaf.bbox[2] + x0, leaf.bbox[3] + y0,
            )
            leaf.polygon = [(px + x0, py + y0) for px, py in leaf.polygon]
            leaf.center = (leaf.center[0] + x0, leaf.center[1] + y0)
        all_leaves.extend(leaves)

    return _nms_leaves(all_leaves, iou_threshold)


def _sahi_disease_inference(
    model: object,
    image_path: Path,
    *,
    slice_size: int = 512,
    overlap: int = 96,
    conf: float = 0.5,
    iou_threshold: float = 0.45,
) -> list[_DiseasePrediction]:
    """Run the disease detection model on overlapping tiles and merge results."""
    img = Image.open(image_path)
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    tiles = _generate_tiles(w, h, slice_size, overlap)

    all_diseases: list[_DiseasePrediction] = []
    for x0, y0, x1, y1 in tiles:
        tile = img_np[y0:y1, x0:x1]
        result = model.predict(
            source=tile, imgsz=slice_size, conf=conf, verbose=False,
        )[0]
        diseases = _parse_disease_predictions(result)
        # Offset tile-local coordinates back to full-image space
        for d in diseases:
            d.bbox = (
                d.bbox[0] + x0, d.bbox[1] + y0,
                d.bbox[2] + x0, d.bbox[3] + y0,
            )
            d.center = (d.center[0] + x0, d.center[1] + y0)
        all_diseases.extend(diseases)

    return _nms_diseases(all_diseases, iou_threshold)


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_leaves(
    leaves: list[_LeafPrediction],
    iou_threshold: float = 0.45,
) -> list[_LeafPrediction]:
    """Non-Maximum Suppression on leaf predictions by bounding-box IoU."""
    if not leaves:
        return []
    # Sort by confidence descending — keep higher-confidence detections
    leaves.sort(key=lambda lf: lf.confidence, reverse=True)
    keep: list[_LeafPrediction] = []
    suppressed = [False] * len(leaves)
    for i, leaf in enumerate(leaves):
        if suppressed[i]:
            continue
        keep.append(leaf)
        for j in range(i + 1, len(leaves)):
            if suppressed[j]:
                continue
            if leaf.class_name != leaves[j].class_name:
                continue
            if _bbox_iou(leaf.bbox, leaves[j].bbox) >= iou_threshold:
                suppressed[j] = True
    return keep


def _nms_diseases(
    diseases: list[_DiseasePrediction],
    iou_threshold: float = 0.45,
) -> list[_DiseasePrediction]:
    """Non-Maximum Suppression on disease predictions by bounding-box IoU."""
    if not diseases:
        return []
    diseases.sort(key=lambda d: d.confidence, reverse=True)
    keep: list[_DiseasePrediction] = []
    suppressed = [False] * len(diseases)
    for i, disease in enumerate(diseases):
        if suppressed[i]:
            continue
        keep.append(disease)
        for j in range(i + 1, len(diseases)):
            if suppressed[j]:
                continue
            if disease.class_name != diseases[j].class_name:
                continue
            if _bbox_iou(disease.bbox, diseases[j].bbox) >= iou_threshold:
                suppressed[j] = True
    return keep


def _report(cb: _ProgressCb, percent: int, message: str) -> None:
    if cb is not None:
        cb(percent, message)
