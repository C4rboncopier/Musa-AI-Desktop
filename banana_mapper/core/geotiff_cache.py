from __future__ import annotations

import hashlib
import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from ..geotiff import GeoTiffInfo, GeoTiffTileLevel


@dataclass(frozen=True)
class GeoTiffCacheKey:
    path: Path
    display_scale_percent: int
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

    def set_cache_dir(self, cache_dir: str | Path | None) -> None:
        next_dir = Path(cache_dir) if cache_dir else None
        if next_dir == self.cache_dir:
            return
        self._items.clear()
        self.cache_dir = next_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, path: str | Path, display_scale_percent: int) -> GeoTiffInfo | None:
        key = self._key(path, display_scale_percent)
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

    def put(self, info: GeoTiffInfo, display_scale_percent: int) -> None:
        key = self._key(info.file_path, display_scale_percent)
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
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def _load_from_disk(self, key: GeoTiffCacheKey) -> GeoTiffInfo | None:
        if self.cache_dir is None:
            return None
        stem = self._cache_stem(key)
        metadata_path = self.cache_dir / f"{stem}.json"
        preview_path = self.cache_dir / f"{stem}.png"
        tile_dir = self.cache_dir / f"{stem}_tiles"
        if not metadata_path.exists() or not preview_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            tile_levels = tuple(
                GeoTiffTileLevel(
                    index=int(level["index"]),
                    width=int(level["width"]),
                    height=int(level["height"]),
                    cols=int(level["cols"]),
                    rows=int(level["rows"]),
                    scale=int(level["scale"]),
                )
                for level in payload.get("tile_levels", [])
            )
            if not tile_levels or not tile_dir.exists():
                return None
            if (
                Path(payload["file_path"]).resolve(strict=True) != key.path
                or int(payload["display_scale_percent"]) != key.display_scale_percent
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
                tile_dir=tile_dir,
                tile_size=int(payload.get("tile_size", 512)),
                tile_levels=tile_levels,
                metadata_details={
                    str(key): str(value)
                    for key, value in payload.get("metadata_details", {}).items()
                },
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _store_on_disk(self, key: GeoTiffCacheKey, info: GeoTiffInfo) -> None:
        if self.cache_dir is None or not info.preview_path.exists():
            return
        stem = self._cache_stem(key)
        preview_path = self.cache_dir / f"{stem}.png"
        metadata_path = self.cache_dir / f"{stem}.json"
        tile_dir = self.cache_dir / f"{stem}_tiles"
        try:
            if info.preview_path.resolve() != preview_path.resolve():
                shutil.copy2(info.preview_path, preview_path)
            if info.tile_dir is not None and info.tile_dir.exists():
                if info.tile_dir.resolve() != tile_dir.resolve():
                    if tile_dir.exists():
                        shutil.rmtree(tile_dir, ignore_errors=True)
                    shutil.copytree(info.tile_dir, tile_dir)
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
                "tile_size": info.tile_size,
                "tile_levels": [
                    {
                        "index": level.index,
                        "width": level.width,
                        "height": level.height,
                        "cols": level.cols,
                        "rows": level.rows,
                        "scale": level.scale,
                    }
                    for level in info.tile_levels
                ],
                "metadata_details": info.metadata_details,
                "display_scale_percent": key.display_scale_percent,
                "mtime_ns": key.mtime_ns,
                "size": key.size,
            }
            metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            return

    @staticmethod
    def _cache_stem(key: GeoTiffCacheKey) -> str:
        raw = f"{key.path}|{key.display_scale_percent}|{key.mtime_ns}|{key.size}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _key(path: str | Path, display_scale_percent: int) -> GeoTiffCacheKey | None:
        file_path = Path(path).expanduser()
        try:
            resolved = file_path.resolve(strict=True)
            stat = resolved.stat()
        except OSError:
            return None
        return GeoTiffCacheKey(
            path=resolved,
            display_scale_percent=int(display_scale_percent),
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
        )
