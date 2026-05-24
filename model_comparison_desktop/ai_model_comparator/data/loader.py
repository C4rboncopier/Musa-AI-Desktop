from __future__ import annotations

import json
from pathlib import Path

from ai_model_comparator.data.models import ComparisonDataset, DetectionRecord, ModelResult


DEFAULT_MODELS = [
    {
        "id": "gemini",
        "name": "Gemini Vision",
        "version": "API provider",
        "color": "#38bdf8",
        "description": "External Gemini API image analysis result set.",
    },
    {
        "id": "trained_pipeline",
        "name": "Musa AI Trained YOLO Pipeline",
        "version": "two-stage local pipeline",
        "color": "#22c55e",
        "description": "YOLOv8-seg leaf model plus YOLOv8 disease model used as one pipeline.",
    },
]


def empty_dataset(image_name: str = "", image_width: int = 0, image_height: int = 0) -> ComparisonDataset:
    return ComparisonDataset(
        image_name=image_name,
        image_width=image_width,
        image_height=image_height,
        models=[
            ModelResult(
                id=item["id"],
                name=item["name"],
                version=item["version"],
                color=item["color"],
                description=item["description"],
                records=[],
            )
            for item in DEFAULT_MODELS
        ],
    )


def load_dataset(path: Path | None = None) -> ComparisonDataset:
    if path is None:
        return empty_dataset()
    return load_dataset_file(path)


def load_dataset_file(path: Path) -> ComparisonDataset:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    models = []
    for model_payload in payload["models"]:
        records = [DetectionRecord.from_dict(item) for item in model_payload["records"]]
        models.append(
            ModelResult(
                id=model_payload["id"],
                name=model_payload["name"],
                version=model_payload["version"],
                color=model_payload["color"],
                description=model_payload.get("description", ""),
                records=records,
            )
        )

    image_payload = payload.get("image", {})
    return ComparisonDataset(
        image_name=image_payload.get("name", path.name),
        image_width=int(image_payload.get("width", 0) or 0),
        image_height=int(image_payload.get("height", 0) or 0),
        models=models,
    )
