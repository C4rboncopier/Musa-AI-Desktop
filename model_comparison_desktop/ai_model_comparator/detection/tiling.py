from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class TileSpec:
    index: int
    x: int
    y: int
    width: int
    height: int

    @property
    def name(self) -> str:
        return f"tile_{self.index:05d}"


def generate_overlapping_tiles(
    image_width: int,
    image_height: int,
    tile_size: int = 512,
    overlap: int = 128,
) -> list[TileSpec]:
    """Generate SAHI-style overlapping windows with full edge coverage."""

    if tile_size <= 0:
        raise ValueError("tile_size must be greater than zero.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be zero or less than tile_size.")

    stride = tile_size - overlap
    xs = _axis_starts(image_width, tile_size, stride)
    ys = _axis_starts(image_height, tile_size, stride)

    tiles: list[TileSpec] = []
    index = 1
    for y in ys:
        for x in xs:
            tiles.append(
                TileSpec(
                    index=index,
                    x=x,
                    y=y,
                    width=min(tile_size, image_width - x),
                    height=min(tile_size, image_height - y),
                )
            )
            index += 1
    return tiles


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = [0]
    count = int(math.ceil((length - tile_size) / stride)) + 1
    for idx in range(1, count):
        starts.append(min(idx * stride, length - tile_size))
    return sorted(set(starts))

