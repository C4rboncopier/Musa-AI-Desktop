from __future__ import annotations

import math

from ai_model_comparator.data.models import DetectionRecord, ModelResult


def merge_overlapping_records(records: list[DetectionRecord], distance_px: float = 48.0) -> list[DetectionRecord]:
    """Merge near-duplicate records created by overlapping tiles."""

    merged: list[DetectionRecord] = []
    for record in sorted(records, key=lambda item: item.confidence, reverse=True):
        match = next(
            (
                existing
                for existing in merged
                if existing.class_name == record.class_name
                and _distance(existing.pixel_x, existing.pixel_y, record.pixel_x, record.pixel_y) <= distance_px
            ),
            None,
        )
        if match is None:
            merged.append(record)
            continue
        match.duplicate_count += record.duplicate_count
        match.confidence = max(match.confidence, record.confidence)
        match.metadata.setdefault("merged_ids", []).append(record.id)
    return merged


def refresh_model_records(model: ModelResult, records: list[DetectionRecord]) -> None:
    model.records.clear()
    model.records.extend(records)


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)

