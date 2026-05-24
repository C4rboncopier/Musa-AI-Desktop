from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CLASSES = ("full_leaf", "cut_leaf", "black_sigatoka", "panama")


@dataclass
class DetectionPoint:
    id: str
    source: str
    class_name: str
    x: float
    y: float
    confidence: float | None = None
    description: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceRun:
    source: str
    image_path: str
    image_width: int
    image_height: int
    duration_ms: int
    points: list[DetectionPoint]
    model_name: str = ""
    raw_response: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    tile_count: int = 0
    failed_tiles: int = 0
    raw_detection_count: int = 0
    deduped_detection_count: int = 0
    cancelled: bool = False

    @property
    def counts(self) -> dict[str, int]:
        counts = {key: 0 for key in CLASSES}
        for point in self.points:
            if point.class_name in counts:
                counts[point.class_name] += 1
        counts["total"] = len(self.points)
        return counts

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["counts"] = self.counts
        return payload


@dataclass
class ComparisonMetrics:
    match_radius_px: float
    per_class: dict[str, dict[str, float | int]]
    mean_match_distance_px: float | None
    overall_similarity: float

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return cleaned[:80] or "image"
