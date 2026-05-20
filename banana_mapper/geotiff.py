from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Callable, Optional

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


WGS84 = "EPSG:4326"
DEFAULT_PREVIEW_SCALE = 1.0
DEFAULT_TILE_SIZE = 512


class GeoTiffError(RuntimeError):
    """Raised when a GeoTIFF cannot be prepared for map overlay."""


@dataclass(frozen=True)
class GeoTiffTileLevel:
    index: int
    width: int
    height: int
    cols: int
    rows: int
    scale: int


@dataclass(frozen=True)
class GeoTiffInfo:
    file_path: Path
    file_name: str
    width: int
    height: int
    band_count: int
    source_crs: str
    source_crs_authority: str
    display_crs: str
    transform: tuple[float, float, float, float, float, float]
    bounds_source: tuple[float, float, float, float]
    bounds_wgs84: tuple[float, float, float, float]
    preview_path: Path
    preview_width: int
    preview_height: int
    tile_dir: Path | None = None
    tile_size: int = DEFAULT_TILE_SIZE
    tile_levels: tuple[GeoTiffTileLevel, ...] = ()
    metadata_details: dict[str, str] = field(default_factory=dict)

    @property
    def west(self) -> float:
        return self.bounds_wgs84[0]

    @property
    def south(self) -> float:
        return self.bounds_wgs84[1]

    @property
    def east(self) -> float:
        return self.bounds_wgs84[2]

    @property
    def north(self) -> float:
        return self.bounds_wgs84[3]

    @property
    def pixel_size_x(self) -> float:
        return abs(self.transform[0])

    @property
    def pixel_size_y(self) -> float:
        return abs(self.transform[4])

    @property
    def spatial_resolution_label(self) -> str:
        units = "source units/pixel"
        if self.source_crs_authority.startswith("EPSG:4326"):
            units = "degrees/pixel"
        elif self.source_crs_authority:
            units = "CRS units/pixel"
        return f"{self.pixel_size_x:.8f} x {self.pixel_size_y:.8f} {units}"

    @property
    def pixel_resolution_label(self) -> str:
        return f"{self.width:,} x {self.height:,} px"

    @property
    def preview_resolution_label(self) -> str:
        return f"{self.preview_width:,} x {self.preview_height:,} px"


# Progress stages and their approximate cumulative weight (0-100)
_STAGES = [
    (5,  "Opening file and reading metadata..."),
    (15, "Validating raster metadata..."),
    (25, "Parsing CRS and computing source bounds..."),
    (35, "Projecting bounds to WGS84..."),
    (45, "Computing optimal display grid..."),
    (80, "Reprojecting bands to RGBA (this may take a moment)..."),
    (90, "Finalizing spatial extent..."),
    (96, "Saving preview image to disk..."),
    (98, "Building tiled preview pyramid for smooth zoom..."),
    (100, "Done."),
]

_ProgressCb = Optional[Callable[[int, str], None]]


def _report(cb: _ProgressCb, percent: int, message: str) -> None:
    if cb is not None:
        cb(percent, message)


def load_geotiff_for_leaflet(
    path: str | Path,
    preview_scale: float = DEFAULT_PREVIEW_SCALE,
    progress_callback: _ProgressCb = None,
) -> GeoTiffInfo:
    """Read a GeoTIFF and export a map-ready EPSG:4326 PNG preview.

    Leaflet image overlays are bounded by south/west/north/east coordinates in
    EPSG:4326. This function uses rasterio's CRS-aware warp tools so rasters in
    projected CRSs, such as UTM drone products, are positioned from their actual
    geospatial extent instead of screen or pixel dimensions.

    Args:
        path: Path to the GeoTIFF file.
        preview_scale: Display scale from 0.25 to 1.0. A value of 1.0 keeps
            the calculated/native display grid without downscaling.
        progress_callback: Optional callable(percent: int, message: str) invoked
            at each processing stage. Safe to call from a background thread.
    """

    cb = progress_callback
    file_path = Path(path)

    _report(cb, 5, "Opening file and reading metadata...")
    if not file_path.exists():
        raise GeoTiffError("The selected file does not exist.")

    try:
        with rasterio.open(file_path) as dataset:
            _report(cb, 15, "Validating raster metadata...")
            _validate_dataset(dataset)

            _report(cb, 25, "Parsing CRS and computing source bounds...")
            source_crs = CRS.from_user_input(dataset.crs)
            metadata_details = _collect_metadata_details(dataset)
            bounds_source = (
                float(dataset.bounds.left),
                float(dataset.bounds.bottom),
                float(dataset.bounds.right),
                float(dataset.bounds.top),
            )

            _report(cb, 35, "Projecting bounds to WGS84...")
            bounds_wgs84 = _bounds_to_wgs84(dataset.crs, bounds_source)

            _report(cb, 45, "Computing optimal display grid...")
            dst_transform, dst_width, dst_height = _display_grid(dataset, preview_scale)

            _report(cb, 50, f"Reprojecting raster to RGBA ({dst_width:,} x {dst_height:,} px)...")
            rgba = _reproject_to_rgba(dataset, dst_transform, dst_width, dst_height)

            _report(cb, 90, "Finalizing spatial extent...")
            # dst_bounds is derived directly from the output reprojected transform
            # and represents the exact pixel extent of the preview PNG – use it
            # as the primary source of truth for Leaflet overlay registration.
            dst_bounds = array_bounds(dst_height, dst_width, dst_transform)
            exact_bounds_wgs84 = _normalize_bounds(
                dst_bounds[0], dst_bounds[1], dst_bounds[2], dst_bounds[3]
            )
            # Sanity-check: if the source-projected bounds are valid WGS84 and
            # both sets agree to within ~0.001 °, prefer the source bounds which
            # may have sub-pixel accuracy at the edge.
            if _bounds_are_valid(bounds_wgs84) and _bounds_close(bounds_wgs84, exact_bounds_wgs84):
                exact_bounds_wgs84 = bounds_wgs84

            _report(cb, 96, "Saving preview image to disk...")
            preview_image = Image.fromarray(rgba, mode="RGBA")
            preview_path = _save_preview_png(preview_image, file_path.stem)

            _report(cb, 98, "Building tiled preview pyramid for smooth zoom...")
            tile_dir, tile_levels = _save_preview_tiles(
                preview_image,
                file_path.stem,
                progress_callback=cb,
            )

            _report(cb, 100, "Done.")
            return GeoTiffInfo(
                file_path=file_path,
                file_name=file_path.name,
                width=dataset.width,
                height=dataset.height,
                band_count=dataset.count,
                source_crs=source_crs.to_string(),
                source_crs_authority=_authority_label(source_crs),
                display_crs=WGS84,
                transform=tuple(float(value) for value in dataset.transform[:6]),
                bounds_source=bounds_source,
                bounds_wgs84=exact_bounds_wgs84,
                preview_path=preview_path,
                preview_width=dst_width,
                preview_height=dst_height,
                tile_dir=tile_dir,
                tile_size=DEFAULT_TILE_SIZE,
                tile_levels=tile_levels,
                metadata_details=metadata_details,
            )
    except RasterioIOError as exc:
        raise GeoTiffError("The selected file is not a readable GeoTIFF.") from exc
    except GeoTiffError:
        raise
    except Exception as exc:
        raise GeoTiffError(f"Unable to process GeoTIFF: {exc}") from exc


def _validate_dataset(dataset: rasterio.DatasetReader) -> None:
    if dataset.crs is None:
        raise GeoTiffError("Missing geospatial metadata: no CRS was found.")
    if dataset.transform is None or dataset.transform.is_identity:
        raise GeoTiffError("Missing geospatial metadata: no valid geotransform was found.")
    if dataset.width <= 0 or dataset.height <= 0 or dataset.count <= 0:
        raise GeoTiffError("Invalid raster dimensions or band count.")
    if dataset.bounds.left == dataset.bounds.right or dataset.bounds.bottom == dataset.bounds.top:
        raise GeoTiffError("Invalid spatial extent: bounds have no area.")


def _collect_metadata_details(dataset: rasterio.DatasetReader) -> dict[str, str]:
    raw_tags = _collect_tags(dataset)
    lookup = {_normalize_tag_key(key): value for key, value in raw_tags.items() if value}
    details: dict[str, str] = {}

    elevation = _first_tag(
        lookup,
        (
            "gpsaltitude",
            "exifgpsaltitude",
            "absolutealtitude",
            "relativealtitude",
            "drone-djiabsolutealtitude",
            "xmpdrone-djiabsolutealtitude",
            "elevation",
            "altitude",
        ),
    )
    if elevation:
        details["Elevation / Altitude"] = _clean_value(elevation)

    camera_parts = []
    make = _first_tag(lookup, ("make", "exifmake", "camera_make", "cameramake"))
    model = _first_tag(lookup, ("model", "exifmodel", "camera_model", "cameramodel"))
    lens = _first_tag(lookup, ("lensmodel", "exiflensmodel", "lens", "lensinfo"))
    if make:
        camera_parts.append(_clean_value(make))
    if model and _clean_value(model) not in camera_parts:
        camera_parts.append(_clean_value(model))
    if lens:
        camera_parts.append(f"Lens: {_clean_value(lens)}")
    if camera_parts:
        details["Camera"] = "\n".join(camera_parts)

    capture_time = _first_tag(
        lookup,
        (
            "datetimeoriginal",
            "exifdatetimeoriginal",
            "datetime",
            "exifdatetime",
            "acquisitiondatetime",
            "capturetime",
            "timestamp",
        ),
    )
    if capture_time:
        details["Capture Time"] = _clean_value(capture_time)

    sensor_settings = []
    for label, keys in [
        ("Focal length", ("focallength", "exiffocallength", "calibratedfocallength")),
        ("Aperture", ("fnumber", "exiffnumber", "aperturevalue")),
        ("Exposure", ("exposuretime", "exifexposuretime", "shutterspeedvalue")),
        ("ISO", ("isospeedratings", "photographicsensitivity", "iso")),
        ("White balance", ("whitebalance", "exifwhitebalance")),
    ]:
        value = _first_tag(lookup, keys)
        if value:
            sensor_settings.append(f"{label}: {_clean_value(value)}")
    if sensor_settings:
        details["Sensor Settings"] = "\n".join(sensor_settings)

    flight_parts = []
    for label, keys in [
        ("Flight yaw", ("flightyawdegree", "drone-djiflightyawdegree")),
        ("Gimbal pitch", ("gimbalpitchdegree", "drone-djigimbalpitchdegree")),
        ("Gimbal yaw", ("gimbalyawdegree", "drone-djigimbalyawdegree")),
        ("GPS latitude", ("gpslatitude", "exifgpslatitude")),
        ("GPS longitude", ("gpslongitude", "exifgpslongitude")),
    ]:
        value = _first_tag(lookup, keys)
        if value:
            flight_parts.append(f"{label}: {_clean_value(value)}")
    if flight_parts:
        details["Flight / GPS"] = "\n".join(flight_parts)

    details["Raster Details"] = "\n".join(
        [
            f"Driver: {dataset.driver}",
            f"Data types: {', '.join(dict.fromkeys(str(dtype) for dtype in dataset.dtypes))}",
            f"Color interpretation: {', '.join(ci.name for ci in dataset.colorinterp)}",
        ]
    )

    compression = _first_tag(lookup, ("compression", "compress", "image_structurecompression"))
    interleave = _first_tag(lookup, ("interleave", "image_structureinterleave"))
    structure_parts = []
    if compression:
        structure_parts.append(f"Compression: {_clean_value(compression)}")
    if interleave:
        structure_parts.append(f"Interleave: {_clean_value(interleave)}")
    if dataset.nodata is not None:
        structure_parts.append(f"Nodata: {dataset.nodata}")
    if structure_parts:
        details["Storage"] = "\n".join(structure_parts)

    band_parts = []
    for index in range(1, dataset.count + 1):
        desc = dataset.descriptions[index - 1] or f"Band {index}"
        unit = dataset.units[index - 1] if dataset.units and index - 1 < len(dataset.units) else None
        text = f"{index}: {desc} ({dataset.dtypes[index - 1]})"
        if unit:
            text += f", {unit}"
        band_parts.append(text)
    if band_parts:
        details["Band Details"] = "\n".join(band_parts)

    tag_summary = _summarize_extra_tags(raw_tags)
    if tag_summary:
        details["Metadata Tags"] = tag_summary

    return details


def _collect_tags(dataset: rasterio.DatasetReader) -> dict[str, str]:
    tags: dict[str, str] = {}
    namespaces: list[str | None] = [None, "IMAGE_STRUCTURE", "EXIF", "GPS", "RPC"]
    try:
        namespaces.extend(ns for ns in dataset.tag_namespaces() if ns not in namespaces)
    except Exception:
        pass

    for namespace in namespaces:
        try:
            values = dataset.tags(ns=namespace) if namespace else dataset.tags()
        except Exception:
            continue
        prefix = f"{namespace}:" if namespace else ""
        for key, value in values.items():
            if value is None:
                continue
            tags[f"{prefix}{key}"] = str(value)
    return tags


def _first_tag(lookup: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = lookup.get(_normalize_tag_key(key))
        if value:
            return value
    return ""


def _normalize_tag_key(key: str) -> str:
    return "".join(char.lower() for char in str(key) if char.isalnum())


def _clean_value(value: str, max_chars: int = 180) -> str:
    text = " ".join(str(value).replace("\x00", " ").split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _summarize_extra_tags(tags: dict[str, str], limit: int = 10) -> str:
    skip = {
        "area_or_point",
        "compression",
        "interleave",
        "make",
        "model",
        "datetime",
        "datetimeoriginal",
    }
    rows = []
    for key in sorted(tags):
        normalized = _normalize_tag_key(key)
        if normalized in skip or "xml" in normalized:
            continue
        value = _clean_value(tags[key], 90)
        if not value:
            continue
        rows.append(f"{key}: {value}")
        if len(rows) >= limit:
            break
    return "\n".join(rows)


def _bounds_to_wgs84(crs: rasterio.crs.CRS, bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    west, south, east, north = transform_bounds(
        crs,
        WGS84,
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        densify_pts=21,
    )
    return _normalize_bounds(west, south, east, north)


def _display_grid(dataset: rasterio.DatasetReader, preview_scale: float) -> tuple[rasterio.Affine, int, int]:
    transform, width, height = calculate_default_transform(
        dataset.crs,
        WGS84,
        dataset.width,
        dataset.height,
        *dataset.bounds,
    )

    if width <= 0 or height <= 0:
        raise GeoTiffError("Unable to calculate a valid WGS84 display grid.")

    scale = max(0.01, min(1.0, float(preview_scale or 1.0)))
    if scale >= 0.999:
        return transform, width, height

    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    scaled_transform = transform * Affine.scale(width / scaled_width, height / scaled_height)
    return scaled_transform, scaled_width, scaled_height


def _reproject_to_rgba(
    dataset: rasterio.DatasetReader,
    dst_transform: rasterio.Affine,
    dst_width: int,
    dst_height: int,
) -> np.ndarray:
    color_indexes = _color_band_indexes(dataset)
    source_nodata = dataset.nodata

    color = np.zeros((len(color_indexes), dst_height, dst_width), dtype=np.float32)
    for target_index, source_index in enumerate(color_indexes):
        reproject(
            source=rasterio.band(dataset, source_index),
            destination=color[target_index],
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            dst_transform=dst_transform,
            dst_crs=WGS84,
            src_nodata=source_nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    alpha = _alpha_band(dataset, dst_transform, dst_width, dst_height)
    if alpha is None:
        alpha = _alpha_from_valid_pixels(color, source_nodata)

    rgb = _normalize_rgb(color)
    return np.dstack([rgb, alpha])


def _color_band_indexes(dataset: rasterio.DatasetReader) -> list[int]:
    if dataset.count >= 3:
        return [1, 2, 3]
    return [1, 1, 1]


def _alpha_band(
    dataset: rasterio.DatasetReader,
    dst_transform: rasterio.Affine,
    dst_width: int,
    dst_height: int,
) -> Optional[np.ndarray]:
    if dataset.count < 4:
        return None

    alpha = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=rasterio.band(dataset, 4),
        destination=alpha,
        src_transform=dataset.transform,
        src_crs=dataset.crs,
        dst_transform=dst_transform,
        dst_crs=WGS84,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return np.clip(alpha, 0, 255).astype(np.uint8)


def _alpha_from_valid_pixels(color: np.ndarray, source_nodata: Optional[float]) -> np.ndarray:
    finite_mask = np.isfinite(color).all(axis=0)
    if source_nodata is not None:
        finite_mask &= np.any(color != source_nodata, axis=0)
    return np.where(finite_mask, 255, 0).astype(np.uint8)


def _normalize_rgb(color: np.ndarray) -> np.ndarray:
    output = np.zeros_like(color, dtype=np.uint8)
    for index, band in enumerate(color):
        valid = np.isfinite(band)
        if not np.any(valid):
            continue

        values = band[valid]
        if np.issubdtype(values.dtype, np.integer):
            band_min = float(values.min())
            band_max = float(values.max())
        else:
            band_min = float(np.nanpercentile(values, 2))
            band_max = float(np.nanpercentile(values, 98))

        if band_max <= band_min:
            output[index][valid] = np.clip(values, 0, 255).astype(np.uint8)
            continue

        scaled = (band - band_min) * (255.0 / (band_max - band_min))
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0)
        output[index] = np.clip(scaled, 0, 255).astype(np.uint8)
        output[index][~valid] = 0
    return np.moveaxis(output, 0, -1)


def _safe_stem(stem: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)[:48]


def _save_preview_png(image: Image.Image, stem: str) -> Path:
    safe_stem = _safe_stem(stem)
    temp = NamedTemporaryFile(prefix=f"banana_mapper_{safe_stem}_", suffix=".png", delete=False)
    temp.close()
    image.save(temp.name, compress_level=1)
    return Path(temp.name)


def _save_preview_tiles(
    image: Image.Image,
    stem: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    progress_callback: _ProgressCb = None,
) -> tuple[Path, tuple[GeoTiffTileLevel, ...]]:
    safe_stem = _safe_stem(stem)
    tile_dir = Path(mkdtemp(prefix=f"banana_mapper_{safe_stem}_tiles_"))
    levels: list[GeoTiffTileLevel] = []
    current = image
    scale = 1
    level_index = 0

    while True:
        _report(
            progress_callback,
            98,
            f"Building preview tiles level {level_index + 1} ({current.width:,} x {current.height:,} px)...",
        )
        level_dir = tile_dir / str(level_index)
        level_dir.mkdir(parents=True, exist_ok=True)
        cols = max(1, (current.width + tile_size - 1) // tile_size)
        rows = max(1, (current.height + tile_size - 1) // tile_size)
        for row in range(rows):
            top = row * tile_size
            bottom = min(top + tile_size, current.height)
            for col in range(cols):
                left = col * tile_size
                right = min(left + tile_size, current.width)
                tile = current.crop((left, top, right, bottom))
                tile.save(level_dir / f"{col}_{row}.png", compress_level=1)

        levels.append(
            GeoTiffTileLevel(
                index=level_index,
                width=current.width,
                height=current.height,
                cols=cols,
                rows=rows,
                scale=scale,
            )
        )
        if current.width <= tile_size and current.height <= tile_size:
            break

        next_size = (max(1, (current.width + 1) // 2), max(1, (current.height + 1) // 2))
        current = current.resize(next_size, Image.Resampling.BILINEAR)
        scale *= 2
        level_index += 1

    return tile_dir, tuple(levels)


def _normalize_bounds(west: float, south: float, east: float, north: float) -> tuple[float, float, float, float]:
    left, right = sorted((float(west), float(east)))
    bottom, top = sorted((float(south), float(north)))
    return left, bottom, right, top


def _bounds_are_valid(bounds: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bounds
    return -180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90


def _bounds_close(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    tol: float = 0.005,
) -> bool:
    """Return True when two bounding-box tuples agree within *tol* degrees."""
    return all(abs(ai - bi) <= tol for ai, bi in zip(a, b))


def _authority_label(crs: CRS) -> str:
    authority = crs.to_authority()
    if authority:
        return f"{authority[0]}:{authority[1]}"
    return crs.name or crs.to_string()
