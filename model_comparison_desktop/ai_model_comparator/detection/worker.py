from __future__ import annotations

from pathlib import Path
from time import perf_counter

from PyQt6.QtCore import QThread, pyqtSignal

from ai_model_comparator.data.models import ComparisonDataset
from ai_model_comparator.data.settings_store import AppConfig
from ai_model_comparator.detection.pipeline import run_detection_pipeline


class DetectionWorker(QThread):
    progressChanged = pyqtSignal(int, int, str)
    providerProgressChanged = pyqtSignal(int, int, str, str, str)
    detectionFinished = pyqtSignal(object, float, str)
    detectionFailed = pyqtSignal(str)

    def __init__(
        self,
        image_path: Path,
        dataset: ComparisonDataset,
        config: AppConfig,
        gemini_api_key: str,
        mapping_mode: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.dataset = dataset
        self.config = config
        self.gemini_api_key = gemini_api_key
        self.mapping_mode = mapping_mode

    def run(self) -> None:
        started = perf_counter()
        try:
            dataset = run_detection_pipeline(
                self.image_path,
                self.dataset,
                self.config,
                self.gemini_api_key,
                mapping_mode=self.mapping_mode,
                progress=self._emit_progress,
            )
        except Exception as exc:
            self.detectionFailed.emit(str(exc))
            return
        self.detectionFinished.emit(dataset, perf_counter() - started, self.mapping_mode)

    def _emit_progress(self, current: int, total: int, provider_id: str, tile_name: str, message: str) -> None:
        self.progressChanged.emit(current, total, message)
        self.providerProgressChanged.emit(current, total, provider_id, tile_name, message)
