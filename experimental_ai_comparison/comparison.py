from __future__ import annotations

import math

from .models import CLASSES, ComparisonMetrics, DetectionPoint


def compare_points(
    gemini_points: list[DetectionPoint],
    musa_points: list[DetectionPoint],
    *,
    radius_px: float = 40.0,
) -> ComparisonMetrics:
    per_class: dict[str, dict[str, float | int]] = {}
    distances: list[float] = []
    matched_total = 0
    denom_total = 0

    for class_name in CLASSES:
        left = [p for p in gemini_points if p.class_name == class_name]
        right = [p for p in musa_points if p.class_name == class_name]
        matches, class_distances = _greedy_matches(left, right, radius_px)
        distances.extend(class_distances)
        denom = max(len(left), len(right), 1)
        similarity = matches / denom
        per_class[class_name] = {
            "gemini_count": len(left),
            "musa_count": len(right),
            "matched_count": matches,
            "similarity": round(similarity, 4),
            "mean_distance_px": round(sum(class_distances) / len(class_distances), 3)
            if class_distances
            else 0,
        }
        matched_total += matches
        denom_total += max(len(left), len(right))

    return ComparisonMetrics(
        match_radius_px=radius_px,
        per_class=per_class,
        mean_match_distance_px=round(sum(distances) / len(distances), 3) if distances else None,
        overall_similarity=round(matched_total / denom_total, 4) if denom_total else 0.0,
    )


def _greedy_matches(
    left: list[DetectionPoint],
    right: list[DetectionPoint],
    radius_px: float,
) -> tuple[int, list[float]]:
    candidates: list[tuple[float, int, int]] = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            distance = math.hypot(a.x - b.x, a.y - b.y)
            if distance <= radius_px:
                candidates.append((distance, i, j))
    candidates.sort(key=lambda item: item[0])

    used_left: set[int] = set()
    used_right: set[int] = set()
    distances: list[float] = []
    for distance, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        distances.append(distance)
    return len(distances), distances

