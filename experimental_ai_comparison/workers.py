from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from .gemini_provider import GeminiVisionProvider
from .models import InferenceRun
from .musa_inference import MusaYoloPointDetector


class GeminiWorker(QThread):
    progress = pyqtSignal(int, int, int, int, int, int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        image_path: str | Path,
        api_key: str,
        model_name: str,
        max_tiles: int | None = None,
        allowed_classes: tuple[str, ...] = ("full_leaf", "cut_leaf", "black_sigatoka", "panama"),
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.api_key = api_key
        self.model_name = model_name
        self.max_tiles = max_tiles
        self.allowed_classes = allowed_classes

    def run(self) -> None:
        try:
            provider = GeminiVisionProvider(self.api_key, self.model_name)
            result: InferenceRun = provider.detect_points(
                self.image_path,
                max_tiles=self.max_tiles,
                allowed_classes=self.allowed_classes,
                progress_callback=self._on_progress,
                should_stop=self.isInterruptionRequested,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_progress(self, tile_index: int, tile_count: int, bounds: tuple[int, int, int, int], message: str) -> None:
        self.progress.emit(tile_index, tile_count, bounds[0], bounds[1], bounds[2], bounds[3], message)


class MusaWorker(QThread):
    progress = pyqtSignal(int, int, int, int, int, int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        image_path: str | Path,
        leaf_model_path: str | Path,
        disease_model_path: str | Path,
        confidence: float,
        device: str,
        include_unmatched_disease: bool,
        run_leaf_model: bool,
        run_disease_model: bool,
        max_tiles: int | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.leaf_model_path = leaf_model_path
        self.disease_model_path = disease_model_path
        self.confidence = confidence
        self.device = device
        self.include_unmatched_disease = include_unmatched_disease
        self.run_leaf_model = run_leaf_model
        self.run_disease_model = run_disease_model
        self.max_tiles = max_tiles

    def run(self) -> None:
        try:
            detector = MusaYoloPointDetector(
                self.leaf_model_path,
                self.disease_model_path,
                confidence=self.confidence,
                device=None if self.device == "auto" else self.device,
                include_unmatched_disease=self.include_unmatched_disease,
                run_leaf_model=self.run_leaf_model,
                run_disease_model=self.run_disease_model,
                max_tiles=self.max_tiles,
            )
            result: InferenceRun = detector.detect_points(
                self.image_path,
                progress_callback=self._on_progress,
                should_stop=self.isInterruptionRequested,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_progress(self, tile_index: int, tile_count: int, bounds: tuple[int, int, int, int], message: str) -> None:
        self.progress.emit(tile_index, tile_count, bounds[0], bounds[1], bounds[2], bounds[3], message)
