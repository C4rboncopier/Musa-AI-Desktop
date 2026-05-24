from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image


@dataclass(frozen=True)
class ImageGeoMetadata:
    width: int
    height: int
    bit_depth: str
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    relative_altitude_m: float | None = None
    focal_length_mm: float | None = None
    focal_length_35mm: float | None = None
    direction_degrees: float | None = None
    ground_width_m: float | None = None
    ground_height_m: float | None = None
    source: str = ""

    @property
    def has_map_bounds(self) -> bool:
        return (
            self.latitude is not None
            and self.longitude is not None
            and self.ground_width_m is not None
            and self.ground_height_m is not None
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def extract_image_metadata(path: str | Path) -> ImageGeoMetadata:
    image_path = Path(path)
    with Image.open(image_path) as image:
        width, height = image.size
        bit_depth = _bit_depth_label(image)
        exif, gps_named = _read_exif_and_gps(image)
        xmp_text = _read_xmp_text(image)

    latitude = _gps_coord(gps_named.get("GPSLatitude"), gps_named.get("GPSLatitudeRef"))
    longitude = _gps_coord(gps_named.get("GPSLongitude"), gps_named.get("GPSLongitudeRef"))
    altitude = _safe_float(gps_named.get("GPSAltitude"))
    if altitude is not None and str(gps_named.get("GPSAltitudeRef", "0")) in {"1", "b'\\x01'"}:
        altitude *= -1

    focal_length = _safe_float(exif.get("FocalLength"))
    focal_35 = _safe_float(exif.get("FocalLengthIn35mmFilm"))
    direction = _safe_float(gps_named.get("GPSImgDirection"))

    relative_altitude = _first_xmp_float(
        xmp_text,
        (
            "drone-dji:RelativeAltitude",
            "RelativeAltitude",
            "drone-dji:AbsoluteAltitude",
            "AbsoluteAltitude",
        ),
    )
    xmp_yaw = _first_xmp_float(
        xmp_text,
        (
            "drone-dji:GimbalYawDegree",
            "GimbalYawDegree",
            "drone-dji:FlightYawDegree",
            "FlightYawDegree",
        ),
    )
    if direction is None:
        direction = xmp_yaw

    # GPSAltitude is usually absolute height above sea level, not above-ground
    # drone altitude. Using it as AGL makes single-photo overlays far too large,
    # so only automatic footprint estimation uses relative altitude.
    footprint_altitude = relative_altitude if relative_altitude and relative_altitude > 0 else None
    sensor_width_mm, sensor_height_mm = _sensor_size_from_focal_plane(
        width,
        height,
        exif.get("FocalPlaneXResolution"),
        exif.get("FocalPlaneYResolution"),
        exif.get("FocalPlaneResolutionUnit"),
    )
    ground_width, ground_height = _estimate_ground_footprint(
        width,
        height,
        altitude_m=footprint_altitude,
        focal_length_mm=focal_length,
        focal_length_35mm=focal_35,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
    )

    return ImageGeoMetadata(
        width=width,
        height=height,
        bit_depth=bit_depth,
        latitude=latitude,
        longitude=longitude,
        altitude_m=altitude,
        relative_altitude_m=relative_altitude,
        focal_length_mm=focal_length,
        focal_length_35mm=focal_35,
        direction_degrees=direction,
        ground_width_m=ground_width,
        ground_height_m=ground_height,
        source=str(image_path),
    )


def metadata_bounds(metadata: ImageGeoMetadata) -> dict[str, float] | None:
    if not metadata.has_map_bounds:
        return None
    assert metadata.latitude is not None
    assert metadata.longitude is not None
    assert metadata.ground_width_m is not None
    assert metadata.ground_height_m is not None

    lat_delta = metadata.ground_height_m / 111_320.0
    lon_delta = metadata.ground_width_m / max(1e-9, 111_320.0 * math.cos(math.radians(metadata.latitude)))
    return {
        "south": metadata.latitude - lat_delta / 2.0,
        "west": metadata.longitude - lon_delta / 2.0,
        "north": metadata.latitude + lat_delta / 2.0,
        "east": metadata.longitude + lon_delta / 2.0,
    }


def _read_exif_and_gps(image: Image.Image) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = image.getexif()
    except Exception:
        return {}, {}
    result: dict[str, Any] = {}
    for key, value in raw.items():
        name = ExifTags.TAGS.get(int(key), str(key))
        result[name] = value

    gps_raw: dict[Any, Any] = {}
    try:
        gps_ifd_id = ExifTags.IFD.GPSInfo
        gps_raw = dict(raw.get_ifd(gps_ifd_id) or {})
    except Exception:
        gps = result.get("GPSInfo") or {}
        if isinstance(gps, dict):
            gps_raw = dict(gps)

    gps_named: dict[str, Any] = {}
    for key, value in gps_raw.items():
        if isinstance(key, int):
            gps_named[ExifTags.GPSTAGS.get(key, str(key))] = value
        else:
            gps_named[str(key)] = value

    # Some tools expose GPS values as ordinary named EXIF fields.
    for key in (
        "GPSLatitude",
        "GPSLatitudeRef",
        "GPSLongitude",
        "GPSLongitudeRef",
        "GPSAltitude",
        "GPSAltitudeRef",
        "GPSImgDirection",
    ):
        if key in result and key not in gps_named:
            gps_named[key] = result[key]

    return result, gps_named


def _read_xmp_text(image: Image.Image) -> str:
    chunks: list[str] = []
    for key in ("XML:com.adobe.xmp", "xmp", "Raw profile type xmp"):
        value = image.info.get(key)
        if isinstance(value, bytes):
            chunks.append(value.decode("utf-8", errors="ignore"))
        elif isinstance(value, str):
            chunks.append(value)
    return "\n".join(chunks)


def _gps_coord(value: Any, ref: Any) -> float | None:
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
        if len(parts) >= 3:
            value = parts[:3]
    if not value or len(value) < 3:
        return None
    degrees = _safe_float(value[0])
    minutes = _safe_float(value[1])
    seconds = _safe_float(value[2])
    if degrees is None or minutes is None or seconds is None:
        return None
    coord = degrees + minutes / 60.0 + seconds / 3600.0
    ref_text = str(ref or "").strip().upper()
    if ref_text in {"S", "W", "B'S'", "B'W'", "SOUTH", "WEST"}:
        coord *= -1
    return coord


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        if isinstance(value, tuple) and len(value) == 2 and value[1]:
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _first_xmp_float(text: str, names: tuple[str, ...]) -> float | None:
    if not text:
        return None
    for name in names:
        escaped = re.escape(name)
        patterns = (
            rf'{escaped}="([+-]?\d+(?:\.\d+)?)"',
            rf"<{escaped}>\s*([+-]?\d+(?:\.\d+)?)\s*</{escaped}>",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return _safe_float(match.group(1))
    return None


def _estimate_ground_footprint(
    width: int,
    height: int,
    *,
    altitude_m: float | None,
    focal_length_mm: float | None,
    focal_length_35mm: float | None,
    sensor_width_mm: float | None,
    sensor_height_mm: float | None,
) -> tuple[float | None, float | None]:
    if altitude_m is None or altitude_m <= 0:
        return None, None
    aspect = width / max(1, height)

    if sensor_width_mm and sensor_height_mm and focal_length_mm and focal_length_mm > 0:
        sensor_w = sensor_width_mm
        sensor_h = sensor_height_mm
        focal = focal_length_mm
    elif focal_length_35mm and focal_length_35mm > 0:
        # 35mm-equivalent focal length maps the camera field of view onto a
        # 36x24mm full-frame reference. This is a practical approximation when
        # sensor dimensions are not available in the image metadata.
        sensor_w = 36.0
        sensor_h = sensor_w / aspect
        if sensor_h > 24.0:
            sensor_h = 24.0
            sensor_w = sensor_h * aspect
        focal = focal_length_35mm
    elif focal_length_mm and focal_length_mm > 0:
        # Conservative fallback for common small drone sensors when only actual
        # focal length is stored. This is approximate but far better than using
        # a multi-kilometer demo extent.
        sensor_w = 6.17
        sensor_h = sensor_w / aspect
        focal = focal_length_mm
    else:
        return None, None

    horizontal_fov = 2.0 * math.atan(sensor_w / (2.0 * focal))
    vertical_fov = 2.0 * math.atan(sensor_h / (2.0 * focal))
    ground_width = 2.0 * altitude_m * math.tan(horizontal_fov / 2.0)
    ground_height = 2.0 * altitude_m * math.tan(vertical_fov / 2.0)
    return ground_width, ground_height


def _sensor_size_from_focal_plane(
    width: int,
    height: int,
    x_resolution: Any,
    y_resolution: Any,
    unit: Any,
) -> tuple[float | None, float | None]:
    x_res = _safe_float(x_resolution)
    y_res = _safe_float(y_resolution)
    if not x_res or not y_res:
        return None, None
    unit_text = str(unit or "").strip()
    if unit_text in {"2", "Inch", "inches"}:
        mm_per_unit = 25.4
    elif unit_text in {"3", "Centimeter", "centimeters", "cm"}:
        mm_per_unit = 10.0
    elif unit_text in {"4", "Millimeter", "millimeters", "mm"}:
        mm_per_unit = 1.0
    else:
        return None, None
    return (width / x_res) * mm_per_unit, (height / y_res) * mm_per_unit


def _bit_depth_label(image: Image.Image) -> str:
    bands = len(image.getbands())
    mode = image.mode
    if mode in {"I;16", "I;16B", "I;16L"}:
        return f"16-bit x {bands}"
    if mode in {"I", "F"}:
        return mode
    return f"8-bit x {bands}"
