from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DetectionRecord:
    id: str
    image_name: str
    class_name: str
    latitude: float
    longitude: float
    confidence: float
    pixel_x: float
    pixel_y: float
    health: str | None = None
    source: str = "ai"
    duplicate_count: int = 1
    related_leaf_id: str | None = None
    layer_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectionRecord":
        allowed = cls.__dataclass_fields__.keys()
        clean = {key: payload[key] for key in allowed if key in payload}
        clean.setdefault("layer_keys", [clean.get("class_name", "unknown")])
        clean.setdefault("metadata", {})
        return cls(**clean)


@dataclass(slots=True)
class ModelResult:
    id: str
    name: str
    version: str
    color: str
    description: str
    records: list[DetectionRecord]

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def average_confidence(self) -> float:
        if not self.records:
            return 0.0
        return sum(record.confidence for record in self.records) / len(self.records)

    def counts_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.class_name] = counts.get(record.class_name, 0) + 1
        return counts


@dataclass(slots=True)
class ComparisonDataset:
    image_name: str
    image_width: int
    image_height: int
    models: list[ModelResult]

    def model_by_id(self, model_id: str) -> ModelResult | None:
        return next((model for model in self.models if model.id == model_id), None)

