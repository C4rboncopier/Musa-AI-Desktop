from __future__ import annotations

import hashlib
import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from ..geotiff import GeoTiffInfo


@dataclass(frozen=True)
class GeoTiffCacheKey:
    path: Path
    max_preview_pixels: int
    mtime_ns: int
    size: int


class GeoTiffSessionCache:
    """LRU GeoTIFF preview cache with optional disk persistence."""

    def __init__(self, max_items: int = 4, cache_dir: str | Path | None = None) -> None:
        self.max_items = max(1, max_items)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._items: OrderedDict[GeoTiffCacheKey, GeoTiffInfo] = OrderedDict()

    def get(self, path: str | Path, max_preview_pixels: int) -> GeoTiffInfo | None:
        key = self._key(path, max_preview_pixels)
        if key is None:
            return None
        info = self._items.get(key)
        if info is None:
            info = self._load_from_disk(key)
            if info is None:
                return None
            self._items[key] = info
        self._items.move_to_end(key)
        return info

    def put(self, info: GeoTiffInfo, max_preview_pixels: int) -> None:
        key = self._key(info.file_path, max_preview_pixels)
        if key is None:
            return
        self._items[key] = info
        self._items.move_to_end(key)
        self._store_on_disk(key, info)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def invalidate(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        stale = [key for key in self._items if key.path == target]
        for key in stale:
            self._items.pop(key, None)

    def clear(self) -> None:
        self._items.clear()
        if self.cache_dir is None or not self.cache_dir.exists():
            return
        for path in self.cache_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)

    def _load_from_disk(self, key: GeoTiffCacheKey) -> GeoTiffInfo | None:
        if self.cache_dir is None:
            return None
        metadata_path = self.cache_dir / f"{self._cache_stem(key)}.json"
        preview_path = self.cache_dir / f"{self._cache_stem(key)}.png"
        if not metadata_path.exists() or not preview_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                Path(payload["file_path"]).resolve(strict=True) != key.path
                or int(payload["max_preview_pixels"]) != key.max_preview_pixels
                or int(payload["mtime_ns"]) != key.mtime_ns
                or int(payload["size"]) != key.size
            ):
                return None
            return GeoTiffInfo(
                file_path=Path(payload["file_path"]),
                file_name=str(payload["file_name"]),
                width=int(payload["width"]),
                height=int(payload["height"]),
                band_count=int(payload["band_count"]),
                source_crs=str(payload["source_crs"]),
                source_crs_authority=str(payload["source_crs_authority"]),
                display_crs=str(payload["display_crs"]),
                transform=tuple(float(v) for v in payload["transform"]),
                bounds_source=tuple(float(v) for v in payload["bounds_source"]),
                bounds_wgs84=tuple(float(v) for v in payload["bounds_wgs84"]),
                preview_path=preview_path,
                preview_width=int(payload["preview_width"]),
                preview_height=int(payload["preview_height"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _store_on_disk(self, key: GeoTiffCacheKey, info: GeoTiffInfo) -> None:
        if self.cache_dir is None or not info.preview_path.exists():
            return
        stem = self._cache_stem(key)
        preview_path = self.cache_dir / f"{stem}.png"
        metadata_path = self.cache_dir / f"{stem}.json"
        try:
            if info.preview_path.resolve() != preview_path.resolve():
                shutil.copy2(info.preview_path, preview_path)
            payload = {
                "file_path": str(key.path),
                "file_name": info.file_name,
                "width": info.width,
                "height": info.height,
                "band_count": info.band_count,
                "source_crs": info.source_crs,
                "source_crs_authority": info.source_crs_authority,
                "display_crs": info.display_crs,
                "transform": list(info.transform),
                "bounds_source": list(info.bounds_source),
                "bounds_wgs84": list(info.bounds_wgs84),
                "preview_width": info.preview_width,
                "preview_height": info.preview_height,
                "max_preview_pixels": key.max_preview_pixels,
                "mtime_ns": key.mtime_ns,
                "size": key.size,
            }
            metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            return

    @staticmethod
    def _cache_stem(key: GeoTiffCacheKey) -> str:
        raw = f"{key.path}|{key.max_preview_pixels}|{key.mtime_ns}|{key.size}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _key(path: str | Path, max_preview_pixels: int) -> GeoTiffCacheKey | None:
        file_path = Path(path).expanduser()
        try:
            resolved = file_path.resolve(strict=True)
            stat = resolved.stat()
        except OSError:
            return None
        return GeoTiffCacheKey(
            path=resolved,
            max_preview_pixels=int(max_preview_pixels),
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
        )
