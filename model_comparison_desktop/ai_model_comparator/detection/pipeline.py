from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image

from ai_model_comparator.data.models import ComparisonDataset
from ai_model_comparator.data.settings_store import AppConfig
from ai_model_comparator.detection.postprocess import merge_overlapping_records, refresh_model_records
from ai_model_comparator.detection.providers import DetectionContext, GeminiDetectionProvider, MusaYoloPipelineProvider
from ai_model_comparator.detection.tiling import generate_overlapping_tiles


ProgressCallback = Callable[[int, int, str, str, str], None]


def run_detection_pipeline(
    image_path: Path,
    dataset: ComparisonDataset,
    config: AppConfig,
    gemini_api_key: str,
    *,
    tile_size: int = 512,
    overlap: int = 128,
    mapping_mode: str = "gemini",
    progress: ProgressCallback | None = None,
) -> ComparisonDataset:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    dataset.image_name = image_path.name
    dataset.image_width = width
    dataset.image_height = height

    tiles = generate_overlapping_tiles(width, height, tile_size=tile_size, overlap=overlap)
    provider_id, provider = _provider_for_mode(mapping_mode, config, gemini_api_key)
    providers = [(provider_id, provider)]

    total_steps = len(tiles) * len(providers)
    completed = 0
    records_by_model = {model_id: [] for model_id, _provider in providers}

    for tile in tiles:
        tile_image = image.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height))
        context = DetectionContext(
            image_name=image_path.name,
            image_width=width,
            image_height=height,
        )
        for model_id, provider in providers:
            model = dataset.model_by_id(model_id)
            label = model.name if model else model_id
            if progress:
                progress(completed, total_steps, model_id, tile.name, f"{label}: analyzing {tile.name}")
            records_by_model[model_id].extend(provider.detect_tile(tile_image, tile, context))
            completed += 1
            if progress:
                progress(completed, total_steps, model_id, tile.name, f"{label}: completed {tile.name}")

    for model_id, records in records_by_model.items():
        model = dataset.model_by_id(model_id)
        if model:
            refresh_model_records(model, merge_overlapping_records(records))

    return dataset


def _provider_for_mode(mapping_mode: str, config: AppConfig, gemini_api_key: str):
    if mapping_mode == "trained_pipeline":
        return "trained_pipeline", MusaYoloPipelineProvider(config.leaf_model_path, config.disease_model_path)
    return "gemini", GeminiDetectionProvider(gemini_api_key)
