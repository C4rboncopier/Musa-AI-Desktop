"""Background worker thread for GeoTIFF loading.

Runs load_geotiff_for_leaflet on a QThread so the main GUI
stays fully responsive during processing.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from .detection import MappingResult, run_funnel_mapping_geotiff
from .geotiff import DEFAULT_PREVIEW_SCALE, GeoTiffError, GeoTiffInfo, load_geotiff_for_leaflet
from .hardware import detect_hardware


class GeoTiffCancelled(RuntimeError):
    """Raised inside the worker when the user cancels a GeoTIFF import."""


class GeoTiffWorker(QThread):
    """Worker thread that loads a GeoTIFF and emits progress signals.

    Signals:
        progress(int, str): Emitted at each processing stage with
            (percent 0-100, human-readable stage message).
        finished(GeoTiffInfo): Emitted when loading completes
            successfully, carrying the result.
        failed(str): Emitted when loading raises a GeoTiffError,
            carrying the error message string.
    """

    progress: pyqtSignal = pyqtSignal(int, str)
    finished: pyqtSignal = pyqtSignal(object)   # GeoTiffInfo
    failed: pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        path: str | Path,
        preview_scale: float = DEFAULT_PREVIEW_SCALE,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._preview_scale = preview_scale

    def run(self) -> None:
        """Entry point executed on the worker thread."""
        try:
            info = load_geotiff_for_leaflet(
                self._path,
                preview_scale=self._preview_scale,
                progress_callback=self._on_progress,
            )
            self.finished.emit(info)
        except GeoTiffCancelled:
            self.failed.emit("GeoTIFF import cancelled.")
        except GeoTiffError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:                         # pragma: no cover
            self.failed.emit(f"Unexpected error: {exc}")

    def _on_progress(self, percent: int, message: str) -> None:
        """Forward progress reports from the pipeline to the UI thread."""
        if self.isInterruptionRequested():
            raise GeoTiffCancelled()
        self.progress.emit(percent, message)


class AiGeotiffMappingWorker(QThread):
    """Worker thread that runs the two-model leaf disease funnel on a GeoTIFF."""

    progress: pyqtSignal = pyqtSignal(int, str)
    scan_box: pyqtSignal = pyqtSignal(float, float, float, float)
    finished: pyqtSignal = pyqtSignal(object)   # MappingResult
    failed: pyqtSignal = pyqtSignal(str)

    def __init__(
        self,
        geotiff_path: str | Path,
        leaf_model_path: str | Path,
        disease_model_path: str | Path,
        output_dir: str | Path | None = None,
        device: str | int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._geotiff_path = geotiff_path
        self._leaf_model_path = leaf_model_path
        self._disease_model_path = disease_model_path
        self._output_dir = output_dir
        self._device = device

    def run(self) -> None:
        try:
            result = run_funnel_mapping_geotiff(
                self._geotiff_path,
                self._leaf_model_path,
                self._disease_model_path,
                output_dir=self._output_dir,
                device=self._device,
                progress_callback=self._on_progress,
                should_stop=self.isInterruptionRequested,
                scan_callback=self._on_scan,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress.emit(percent, message)

    def _on_scan(self, lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> None:
        self.scan_box.emit(lat_min, lon_min, lat_max, lon_max)


class HardwareCheckWorker(QThread):
    """Worker thread that runs hardware diagnostics with progress feedback."""

    progress: pyqtSignal = pyqtSignal(int, str, str)
    finished: pyqtSignal = pyqtSignal(object)   # HardwareStatus
    failed: pyqtSignal = pyqtSignal(str)

    def run(self) -> None:
        try:
            status = detect_hardware(progress_callback=self._on_progress)
            self.finished.emit(status)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"Hardware diagnostics failed: {exc}")

    def _on_progress(self, percent: int, message: str, level: str) -> None:
        self.progress.emit(percent, message, level)
