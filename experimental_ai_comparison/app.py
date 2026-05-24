from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QSettings, Qt, QUrl
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .comparison import compare_points
from .gemini_provider import DEFAULT_GEMINI_MODEL, estimate_gemini_flash_cost
from .image_metadata import ImageGeoMetadata, extract_image_metadata, metadata_bounds
from .map_bridge import ExperimentMapBridge
from .models import CLASSES, InferenceRun, safe_stem
from .workers import GeminiWorker, MusaWorker


class ExperimentalComparisonWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Musa AI Experimental VLM vs YOLO Comparison")
        self.resize(1500, 900)
        self.setMinimumSize(1120, 720)

        self.image_path: Path | None = None
        self.image_width = 0
        self.image_height = 0
        self.image_metadata: ImageGeoMetadata | None = None
        self.gemini_run: InferenceRun | None = None
        self.musa_run: InferenceRun | None = None
        self.logs: list[str] = []
        self.gemini_worker: GeminiWorker | None = None
        self.musa_worker: MusaWorker | None = None
        self.progress_dialog: QProgressDialog | None = None
        self.progress_by_label: dict[str, tuple[int, int, str]] = {}
        self.benchmark_started_at: float | None = None
        self.map_ready = False

        self.bridge = ExperimentMapBridge()
        self.channel = QWebChannel(self)
        self.channel.registerObject("experimentMapBridge", self.bridge)
        self.bridge.mapReady.connect(self._on_map_ready)

        self._build_ui()
        self._load_saved_settings()
        self._connect_settings_persistence()
        self._apply_theme()
        self._load_map()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = self._build_sidebar()
        map_panel = self._build_map_panel()
        layout.addWidget(sidebar)
        layout.addWidget(map_panel, 1)
        self.setCentralWidget(root)

    def _build_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setFixedWidth(430)
        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll)

        content = QWidget()
        content.setObjectName("sidebarContent")
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Experimental AI Comparison")
        title.setObjectName("title")
        subtitle = QLabel("Single-image benchmark: Gemini Vision points vs MUSA-AI YOLO points")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        layout.addWidget(self._input_card())
        layout.addWidget(self._run_card())
        layout.addWidget(self._metrics_card())
        layout.addStretch(1)
        return panel

    def _input_card(self) -> QWidget:
        card = _card("Inputs")
        layout = card.layout()

        self.image_input = _readonly_line("No image selected")
        image_btn = QPushButton("Browse Image")
        image_btn.clicked.connect(self._browse_image)
        layout.addWidget(_labeled_row("RGB image", self.image_input, image_btn))

        self.leaf_model_input = _readonly_line("No leaf model selected")
        leaf_btn = QPushButton("Leaf Model")
        leaf_btn.clicked.connect(lambda: self._browse_file(self.leaf_model_input, "YOLO weights (*.pt);;All files (*.*)"))
        layout.addWidget(_labeled_row("YOLOv8-seg leaf", self.leaf_model_input, leaf_btn))

        self.disease_model_input = _readonly_line("No disease model selected")
        disease_btn = QPushButton("Disease Model")
        disease_btn.clicked.connect(lambda: self._browse_file(self.disease_model_input, "YOLO weights (*.pt);;All files (*.*)"))
        layout.addWidget(_labeled_row("YOLOv8 disease", self.disease_model_input, disease_btn))

        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key_input.setPlaceholderText("Paste Gemini API key")
        layout.addWidget(_labeled_row("Gemini API key", self.gemini_key_input))

        self.gemini_model_input = QLineEdit(DEFAULT_GEMINI_MODEL)
        layout.addWidget(_labeled_row("Gemini model", self.gemini_model_input))

        footprint_row = QWidget()
        footprint_layout = QGridLayout(footprint_row)
        footprint_layout.setContentsMargins(0, 0, 0, 0)
        footprint_layout.setHorizontalSpacing(8)
        self.ground_width_input = QLineEdit()
        self.ground_width_input.setPlaceholderText("auto")
        self.ground_height_input = QLineEdit()
        self.ground_height_input.setPlaceholderText("auto")
        apply_scale_btn = QPushButton("Apply Scale")
        apply_scale_btn.clicked.connect(self._send_image_to_map)
        footprint_layout.addWidget(QLabel("Manual footprint meters"), 0, 0, 1, 3)
        footprint_layout.addWidget(self.ground_width_input, 1, 0)
        footprint_layout.addWidget(self.ground_height_input, 1, 1)
        footprint_layout.addWidget(apply_scale_btn, 1, 2)
        layout.addWidget(footprint_row)

        return card

    def _run_card(self) -> QWidget:
        card = _card("Run Benchmark")
        layout = card.layout()

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(1, 99)
        self.confidence_slider.setValue(35)
        self.confidence_value = QLabel("35%")
        self.confidence_slider.valueChanged.connect(lambda value: self.confidence_value.setText(f"{value}%"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Local YOLO confidence"))
        row.addWidget(self.confidence_value)
        layout.addLayout(row)
        layout.addWidget(self.confidence_slider)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU 0", "0")
        layout.addWidget(_labeled_row("Local device", self.device_combo))

        self.max_tiles_input = QSpinBox()
        self.max_tiles_input.setRange(0, 100000)
        self.max_tiles_input.setValue(0)
        self.max_tiles_input.setSpecialValueText("No limit")
        self.max_tiles_input.setSuffix(" tiles")
        layout.addWidget(_labeled_row("Temporary tile limit", self.max_tiles_input))

        self.include_unmatched_disease = QCheckBox("Keep disease points outside leaves")
        self.include_unmatched_disease.setChecked(True)
        layout.addWidget(self.include_unmatched_disease)

        self.local_model_mode = QComboBox()
        self.local_model_mode.addItem("Leaf + Disease", "both")
        self.local_model_mode.addItem("Leaf only", "leaf")
        self.local_model_mode.addItem("Disease only", "disease")
        layout.addWidget(_labeled_row("Local model mode", self.local_model_mode))

        buttons = QHBoxLayout()
        self.run_gemini_btn = QPushButton("Run Gemini")
        self.run_musa_btn = QPushButton("Run MUSA-AI")
        self.run_both_btn = QPushButton("Run Both")
        for button in (self.run_gemini_btn, self.run_musa_btn, self.run_both_btn):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            buttons.addWidget(button)
        self.run_gemini_btn.clicked.connect(self._run_gemini)
        self.run_musa_btn.clicked.connect(self._run_musa)
        self.run_both_btn.clicked.connect(self._run_both)
        layout.addLayout(buttons)

        layer_row = QHBoxLayout()
        self.show_gemini = QCheckBox("Gemini layer")
        self.show_gemini.setChecked(True)
        self.show_musa = QCheckBox("MUSA layer")
        self.show_musa.setChecked(True)
        self.show_gemini.toggled.connect(lambda checked: self._run_js(f"setSourceVisible('gemini', {str(checked).lower()});"))
        self.show_musa.toggled.connect(lambda checked: self._run_js(f"setSourceVisible('musa', {str(checked).lower()});"))
        layer_row.addWidget(self.show_gemini)
        layer_row.addWidget(self.show_musa)
        layout.addLayout(layer_row)

        export_btn = QPushButton("Export Comparison JSON")
        export_btn.clicked.connect(self._export_json)
        layout.addWidget(export_btn)
        return card

    def _metrics_card(self) -> QWidget:
        card = _card("Metrics and Logs")
        layout = card.layout()
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(280)
        self.summary.setPlaceholderText("Benchmark counts, timing, cost estimate, and similarity metrics appear here.")
        layout.addWidget(self.summary)
        return card

    def _build_map_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("mapPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.map_view = QWebEngineView()
        self.map_view.page().setWebChannel(self.channel)
        settings = self.map_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        layout.addWidget(self.map_view)
        return panel

    def _load_map(self) -> None:
        path = Path(__file__).resolve().parent / "map_view.html"
        self.map_view.load(QUrl.fromLocalFile(str(path)))

    def _on_map_ready(self) -> None:
        self.map_ready = True
        if self.image_path:
            self._send_image_to_map()
            self._send_points_to_map()

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select one RGB image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)",
        )
        if not path:
            return
        self.image_path = Path(path)
        self.image_metadata = extract_image_metadata(self.image_path)
        self.image_width = self.image_metadata.width
        self.image_height = self.image_metadata.height
        self.image_input.setText(str(self.image_path))
        self.gemini_run = None
        self.musa_run = None
        self._send_image_to_map()
        self._send_points_to_map()
        self._refresh_summary()

    def _browse_file(self, target: QLineEdit, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", file_filter)
        if path:
            target.setText(path)

    def _settings(self) -> QSettings:
        return QSettings("Musa-AI", "ExperimentalComparison")

    def _load_saved_settings(self) -> None:
        settings = self._settings()
        leaf_model = str(settings.value("leaf_model_path", "") or "")
        disease_model = str(settings.value("disease_model_path", "") or "")
        gemini_key = str(settings.value("gemini_api_key", "") or "")
        gemini_model = str(settings.value("gemini_model", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL)

        if leaf_model:
            self.leaf_model_input.setText(leaf_model)
        if disease_model:
            self.disease_model_input.setText(disease_model)
        if gemini_key:
            self.gemini_key_input.setText(gemini_key)
        self.gemini_model_input.setText(gemini_model)

        local_mode = str(settings.value("local_model_mode", "both") or "both")
        index = self.local_model_mode.findData(local_mode)
        if index >= 0:
            self.local_model_mode.setCurrentIndex(index)

        device = str(settings.value("device", "auto") or "auto")
        index = self.device_combo.findData(device)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)

        self.confidence_slider.setValue(_safe_int(settings.value("confidence", 35), 35))
        self.max_tiles_input.setValue(max(0, _safe_int(settings.value("max_tiles", 0), 0)))
        self.include_unmatched_disease.setChecked(_safe_bool(settings.value("include_unmatched_disease", True), True))

    def _connect_settings_persistence(self) -> None:
        self.leaf_model_input.textChanged.connect(self._save_settings)
        self.disease_model_input.textChanged.connect(self._save_settings)
        self.gemini_key_input.textChanged.connect(self._save_settings)
        self.gemini_model_input.textChanged.connect(self._save_settings)
        self.local_model_mode.currentIndexChanged.connect(self._save_settings)
        self.device_combo.currentIndexChanged.connect(self._save_settings)
        self.confidence_slider.valueChanged.connect(self._save_settings)
        self.max_tiles_input.valueChanged.connect(self._save_settings)
        self.include_unmatched_disease.toggled.connect(self._save_settings)

    def _save_settings(self, *args) -> None:
        settings = self._settings()
        settings.setValue("leaf_model_path", _persistable_path_text(self.leaf_model_input.text()))
        settings.setValue("disease_model_path", _persistable_path_text(self.disease_model_input.text()))
        settings.setValue("gemini_api_key", self.gemini_key_input.text().strip())
        settings.setValue("gemini_model", self.gemini_model_input.text().strip() or DEFAULT_GEMINI_MODEL)
        settings.setValue("local_model_mode", self.local_model_mode.currentData())
        settings.setValue("device", self.device_combo.currentData())
        settings.setValue("confidence", self.confidence_slider.value())
        settings.setValue("max_tiles", self.max_tiles_input.value())
        settings.setValue("include_unmatched_disease", self.include_unmatched_disease.isChecked())
        settings.sync()

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    def _run_both(self) -> None:
        if self._busy():
            return
        if not self._require_image():
            return
        api_key = self.gemini_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing Gemini API key", "Paste your Gemini API key before running Gemini.")
            return
        leaf = self.leaf_model_input.text().strip()
        disease = self.disease_model_input.text().strip()
        if not self._validate_local_model_paths(leaf, disease):
            return

        self.gemini_worker = self._make_gemini_worker(api_key)
        self.musa_worker = self._make_musa_worker(leaf, disease)
        self._start_progress_dialog("Gemini + MUSA-AI")
        self.benchmark_started_at = time.perf_counter()
        self.gemini_worker.start()
        self.musa_worker.start()
        self._append_log("Gemini and MUSA-AI tiled mapping started simultaneously.")
        self._set_buttons_enabled(False)

    def _run_gemini(self) -> None:
        if self._busy():
            return
        if not self._require_image():
            return
        api_key = self.gemini_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing Gemini API key", "Paste your Gemini API key before running Gemini.")
            return
        self.gemini_worker = self._make_gemini_worker(api_key)
        self._start_progress_dialog("Gemini")
        self.benchmark_started_at = time.perf_counter()
        self.gemini_worker.start()
        self._append_log("Gemini tiled mapping started.")
        self._set_buttons_enabled(False)

    def _run_musa(self) -> None:
        if self._busy():
            return
        if not self._require_image():
            return
        leaf = self.leaf_model_input.text().strip()
        disease = self.disease_model_input.text().strip()
        if not self._validate_local_model_paths(leaf, disease):
            return
        self.musa_worker = self._make_musa_worker(leaf, disease)
        self._start_progress_dialog("MUSA-AI")
        self.benchmark_started_at = time.perf_counter()
        self.musa_worker.start()
        self._append_log("MUSA-AI tiled local inference started.")
        self._set_buttons_enabled(False)

    def _make_gemini_worker(self, api_key: str) -> GeminiWorker:
        worker = GeminiWorker(
            self.image_path,
            api_key,
            self.gemini_model_input.text().strip(),
            self._max_tile_limit(),
            self._selected_detection_classes(),
            self,
        )
        worker.progress.connect(lambda *args: self._on_tile_progress("Gemini", *args))
        worker.finished.connect(self._on_gemini_finished)
        worker.failed.connect(lambda error: self._on_worker_failed("Gemini", error))
        return worker

    def _make_musa_worker(self, leaf: str, disease: str) -> MusaWorker:
        worker = MusaWorker(
            self.image_path,
            leaf,
            disease,
            self.confidence_slider.value() / 100.0,
            str(self.device_combo.currentData()),
            self.include_unmatched_disease.isChecked(),
            self._run_leaf_model(),
            self._run_disease_model(),
            self._max_tile_limit(),
            self,
        )
        worker.progress.connect(lambda *args: self._on_tile_progress("MUSA-AI", *args))
        worker.finished.connect(self._on_musa_finished)
        worker.failed.connect(lambda error: self._on_worker_failed("MUSA-AI", error))
        return worker

    def _run_leaf_model(self) -> bool:
        return str(self.local_model_mode.currentData()) in {"both", "leaf"}

    def _run_disease_model(self) -> bool:
        return str(self.local_model_mode.currentData()) in {"both", "disease"}

    def _selected_detection_classes(self) -> tuple[str, ...]:
        mode = str(self.local_model_mode.currentData())
        if mode == "leaf":
            return ("full_leaf", "cut_leaf")
        if mode == "disease":
            return ("black_sigatoka", "panama")
        return CLASSES

    def _max_tile_limit(self) -> int | None:
        value = self.max_tiles_input.value()
        return value if value > 0 else None

    def _validate_local_model_paths(self, leaf: str, disease: str) -> bool:
        missing = []
        if self._run_leaf_model() and not Path(leaf).exists():
            missing.append("leaf")
        if self._run_disease_model() and not Path(disease).exists():
            missing.append("disease")
        if missing:
            QMessageBox.warning(
                self,
                "Missing YOLO models",
                "Select valid " + " and ".join(missing) + " model weights first.",
            )
            return False
        return True

    def _on_gemini_finished(self, result_obj: object) -> None:
        self.gemini_run = result_obj  # type: ignore[assignment]
        self.gemini_worker = None
        status = "cancelled" if self.gemini_run.cancelled else "finished"
        self._mark_progress_done("Gemini", f"Gemini: {'Cancelled / partial result' if self.gemini_run.cancelled else 'Done'}")
        self._append_log(f"Gemini {status} with {len(self.gemini_run.points)} point(s).")
        self._send_points_to_map()
        self._refresh_summary()
        self._finish_progress_dialog_if_idle()
        self._set_buttons_enabled(True)

    def _on_musa_finished(self, result_obj: object) -> None:
        self.musa_run = result_obj  # type: ignore[assignment]
        self.musa_worker = None
        status = "cancelled" if self.musa_run.cancelled else "finished"
        self._mark_progress_done("MUSA-AI", f"MUSA-AI: {'Cancelled / partial result' if self.musa_run.cancelled else 'Done'}")
        self._append_log(f"MUSA-AI {status} with {len(self.musa_run.points)} point(s).")
        self._send_points_to_map()
        self._refresh_summary()
        self._finish_progress_dialog_if_idle()
        self._set_buttons_enabled(True)

    def _on_worker_failed(self, label: str, error: str) -> None:
        self._append_log(f"{label} failed: {error}")
        if label == "Gemini":
            self.gemini_worker = None
        if label == "MUSA-AI":
            self.musa_worker = None
        self._mark_progress_done(label, f"{label}: Failed")
        self._finish_progress_dialog_if_idle()
        QMessageBox.critical(self, f"{label} failed", error)
        self._set_buttons_enabled(True)

    def _start_progress_dialog(self, label: str) -> None:
        if self.progress_dialog is not None:
            self.progress_dialog.close()
        self.progress_by_label = {}
        self.progress_dialog = QProgressDialog(f"{label} mapping is starting...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle(f"Running {label} Mapping")
        self.progress_dialog.canceled.connect(self._cancel_active_workers)
        self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setValue(0)
        if label == "Gemini + MUSA-AI":
            self.progress_by_label["Gemini"] = (0, 1, "Gemini: Waiting to start")
            self.progress_by_label["MUSA-AI"] = (0, 1, "MUSA-AI: Waiting to start")
            self._update_progress_dialog()
        self.progress_dialog.show()

    def _cancel_active_workers(self) -> None:
        if self.gemini_worker is not None and self.gemini_worker.isRunning():
            self.gemini_worker.requestInterruption()
        if self.musa_worker is not None and self.musa_worker.isRunning():
            self.musa_worker.requestInterruption()
        self._append_log("Cancellation requested. Current tile/request may finish before workers stop.")
        if self.progress_dialog is not None:
            self.progress_dialog.setLabelText("Cancelling mapping. Partial results will be displayed when workers stop.")

    def _on_tile_progress(
        self,
        label: str,
        tile_index: int,
        tile_count: int,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        message: str,
    ) -> None:
        percent = int(tile_index / max(1, tile_count) * 100)
        text = f"{label}: {message}\nTile {tile_index}/{tile_count} | px ({x0}, {y0}) to ({x1}, {y1})"
        self.progress_by_label[label] = (tile_index, tile_count, text)
        self._update_progress_dialog()
        self._run_js(
            "setScanBoxPixel("
            f"{json.dumps(self._source_key(label))}, {x0}, {y0}, {x1}, {y1}, "
            f"{json.dumps(label + ' tile ' + str(tile_index) + '/' + str(tile_count))});"
        )
        if tile_index == 1 or tile_index == tile_count or tile_index % 10 == 0:
            self._append_log(text.replace("\n", " - "))

    def _source_key(self, label: str) -> str:
        return "musa" if label == "MUSA-AI" else label.lower()

    def _mark_progress_done(self, label: str, text: str) -> None:
        self.progress_by_label[label] = (1, 1, text)
        self._run_js(f"clearScanBox({json.dumps(self._source_key(label))});")
        self._update_progress_dialog()

    def _update_progress_dialog(self) -> None:
        if self.progress_dialog is None or not self.progress_by_label:
            return
        combined = "\n\n".join(value[2] for value in self.progress_by_label.values())
        avg_percent = int(
            sum(value[0] / max(1, value[1]) for value in self.progress_by_label.values())
            / len(self.progress_by_label)
            * 100
        )
        self.progress_dialog.setLabelText(combined)
        self.progress_dialog.setValue(avg_percent)

    def _finish_progress_dialog_if_idle(self) -> None:
        if self._busy():
            return
        self._run_js("clearScanBoxes();")
        if self.progress_dialog is not None:
            self.progress_dialog.setValue(100)
            self.progress_dialog.close()
            self.progress_dialog = None
        if self.benchmark_started_at is not None:
            elapsed = time.perf_counter() - self.benchmark_started_at
            self._append_log(f"Total benchmark wall-clock time: {elapsed:.2f}s")
            self.benchmark_started_at = None

    def _set_buttons_enabled(self, enabled: bool) -> None:
        enabled = enabled and not self._busy()
        for button in (self.run_gemini_btn, self.run_musa_btn, self.run_both_btn):
            button.setEnabled(enabled)

    def _busy(self) -> bool:
        return bool(
            (self.gemini_worker is not None and self.gemini_worker.isRunning())
            or (self.musa_worker is not None and self.musa_worker.isRunning())
        )

    def _require_image(self) -> bool:
        if self.image_path is None or not self.image_path.exists():
            QMessageBox.warning(self, "Missing image", "Select one RGB image first.")
            return False
        return True

    def _send_image_to_map(self) -> None:
        if not self.map_ready or not self.image_path:
            return
        payload = {
            "imageUrl": QUrl.fromLocalFile(str(self.image_path)).toString(),
            "width": self.image_width,
            "height": self.image_height,
        }
        if self.image_metadata is not None:
            payload["metadata"] = self.image_metadata.to_payload()
            bounds = self._manual_bounds() or metadata_bounds(self.image_metadata)
            if bounds is not None:
                payload["bounds"] = bounds
        self._run_js(f"setImage({json.dumps(payload)});")
        self._send_points_to_map()

    def _manual_bounds(self) -> dict[str, float] | None:
        if self.image_metadata is None:
            return None
        if self.image_metadata.latitude is None or self.image_metadata.longitude is None:
            return None
        width_m = _safe_float_text(self.ground_width_input.text())
        height_m = _safe_float_text(self.ground_height_input.text())
        if width_m is None or height_m is None or width_m <= 0 or height_m <= 0:
            return None
        lat_delta = height_m / 111_320.0
        lon_delta = width_m / max(1e-9, 111_320.0 * math.cos(math.radians(self.image_metadata.latitude)))
        return {
            "south": self.image_metadata.latitude - lat_delta / 2.0,
            "west": self.image_metadata.longitude - lon_delta / 2.0,
            "north": self.image_metadata.latitude + lat_delta / 2.0,
            "east": self.image_metadata.longitude + lon_delta / 2.0,
        }

    def _send_points_to_map(self) -> None:
        if not self.map_ready:
            return
        points = []
        if self.gemini_run:
            points.extend(point.to_payload() for point in self.gemini_run.points)
        if self.musa_run:
            points.extend(point.to_payload() for point in self.musa_run.points)
        self._run_js(f"setDetections({json.dumps({'points': points})});")

    def _refresh_summary(self) -> None:
        lines: list[str] = []
        if self.image_path:
            lines.append(f"Image: {self.image_path.name}")
            lines.append(f"Size: {self.image_width:,} x {self.image_height:,} px")
            if self.image_metadata:
                lines.append(f"Bit depth: {self.image_metadata.bit_depth}")
                if self.image_metadata.latitude is not None and self.image_metadata.longitude is not None:
                    lines.append(
                        f"GPS center: {self.image_metadata.latitude:.8f}, {self.image_metadata.longitude:.8f}"
                    )
                else:
                    lines.append("GPS center: not found")
                if self.image_metadata.relative_altitude_m is not None:
                    lines.append(f"Relative altitude: {self.image_metadata.relative_altitude_m:.2f} m")
                if self.image_metadata.altitude_m is not None:
                    lines.append(f"GPS altitude: {self.image_metadata.altitude_m:.2f} m")
                if self.image_metadata.focal_length_mm is not None:
                    lines.append(f"Focal length: {self.image_metadata.focal_length_mm:.2f} mm")
                if self.image_metadata.focal_length_35mm is not None:
                    lines.append(f"35mm focal length: {self.image_metadata.focal_length_35mm:.2f} mm")
                if self.image_metadata.direction_degrees is not None:
                    lines.append(f"Image direction/yaw: {self.image_metadata.direction_degrees:.2f} deg")
                if self.image_metadata.ground_width_m is not None and self.image_metadata.ground_height_m is not None:
                    lines.append(
                        f"Estimated footprint: {self.image_metadata.ground_width_m:.2f} x "
                        f"{self.image_metadata.ground_height_m:.2f} m"
                    )
                else:
                    lines.append("Estimated footprint: unavailable; map uses 120 m fallback or manual scale")
            lines.append("")
        for run in (self.gemini_run, self.musa_run):
            if run is None:
                continue
            lines.append(f"{run.source.upper()} ({run.model_name or 'model'})")
            lines.append(f"Status: {'cancelled / partial' if run.cancelled else 'completed'}")
            lines.append(f"Total model time spent: {run.duration_ms / 1000:.2f}s")
            lines.append(f"Tiles processed: {run.tile_count - run.failed_tiles}/{run.tile_count}")
            lines.append(f"Failed tiles: {run.failed_tiles}")
            if run.tile_count:
                lines.append(f"Avg time/tile: {run.duration_ms / max(1, run.tile_count):.1f} ms")
            lines.append(f"Raw detections before dedupe: {run.raw_detection_count}")
            lines.append(f"Detections after dedupe: {run.deduped_detection_count}")
            for class_name in CLASSES:
                lines.append(f"  {class_name}: {run.counts.get(class_name, 0)}")
            lines.append(f"  total: {run.counts.get('total', 0)}")
            if run.source == "gemini":
                cost = estimate_gemini_flash_cost(run.usage)
                if run.usage:
                    lines.append(f"  usage: {json.dumps(run.usage)}")
                if cost is not None:
                    lines.append(f"  rough Gemini Flash token estimate: ${cost:.6f}")
            if run.warnings:
                lines.append(f"  warnings: {len(run.warnings)}")
                for warning in run.warnings[:3]:
                    lines.append(f"    - {warning}")
            lines.append("")

        if self.gemini_run and self.musa_run:
            metrics = compare_points(self.gemini_run.points, self.musa_run.points)
            lines.append("COMPARISON")
            lines.append(f"Overall similarity within {metrics.match_radius_px:.0f}px: {metrics.overall_similarity:.3f}")
            if metrics.mean_match_distance_px is not None:
                lines.append(f"Mean matched distance: {metrics.mean_match_distance_px:.2f}px")
            for class_name, values in metrics.per_class.items():
                lines.append(
                    f"  {class_name}: Gemini {values['gemini_count']} | "
                    f"MUSA {values['musa_count']} | matched {values['matched_count']} | "
                    f"similarity {values['similarity']}"
                )
        if self.logs:
            lines.append("")
            lines.append("LOGS")
            lines.extend(f"- {entry}" for entry in self.logs[-12:])
        self.summary.setPlainText("\n".join(lines))

    def _append_log(self, message: str) -> None:
        self.logs.append(message)
        self._refresh_summary()

    def _export_json(self) -> None:
        if not self.gemini_run and not self.musa_run:
            QMessageBox.information(self, "No results", "Run at least one detector before exporting.")
            return
        default_name = f"{safe_stem(self.image_path or 'image')}_vlm_vs_musa_comparison.json"
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Export comparison JSON",
            default_name,
            "JSON (*.json);;All files (*.*)",
        )
        if not target:
            return
        payload = {
            "image": str(self.image_path) if self.image_path else "",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_metadata": self.image_metadata.to_payload() if self.image_metadata else None,
            "gemini": self.gemini_run.to_payload() if self.gemini_run else None,
            "musa": self.musa_run.to_payload() if self.musa_run else None,
            "comparison": compare_points(self.gemini_run.points, self.musa_run.points).to_payload()
            if self.gemini_run and self.musa_run
            else None,
        }
        Path(target).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Export complete", f"Saved comparison JSON:\n{target}")

    def _run_js(self, script: str) -> None:
        self.map_view.page().runJavaScript(script)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1120; color: #dbeafe; font-family: Segoe UI; font-size: 10pt; }
            #sidebar { background: #0f172a; border-right: 1px solid #1e293b; }
            #mapPanel { background: #020617; }
            #title { color: #f8fafc; font-size: 18pt; font-weight: 700; }
            #subtitle { color: #94a3b8; }
            #card { background: #111827; border: 1px solid #263244; border-radius: 8px; }
            #cardTitle { color: #f8fafc; font-size: 11pt; font-weight: 700; }
            QLabel { color: #cbd5e1; }
            QLineEdit, QTextEdit, QComboBox {
                background: #020617;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #e2e8f0;
                padding: 7px;
            }
            QPushButton {
                background: #1d4ed8;
                border: 0;
                border-radius: 6px;
                color: #ffffff;
                font-weight: 600;
                padding: 8px 10px;
            }
            QPushButton:hover { background: #2563eb; }
            QPushButton:disabled { background: #334155; color: #94a3b8; }
            QCheckBox { color: #cbd5e1; }
            QSlider::groove:horizontal { height: 5px; background: #334155; border-radius: 2px; }
            QSlider::handle:horizontal { background: #60a5fa; width: 14px; margin: -5px 0; border-radius: 7px; }
            """
        )


def _card(title: str) -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)
    label = QLabel(title)
    label.setObjectName("cardTitle")
    layout.addWidget(label)
    return frame


def _readonly_line(text: str) -> QLineEdit:
    line = QLineEdit(text)
    line.setReadOnly(True)
    return line


def _labeled_row(label: str, widget: QWidget, button: QPushButton | None = None) -> QWidget:
    row_widget = QWidget()
    layout = QGridLayout(row_widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(8)
    label_widget = QLabel(label)
    layout.addWidget(label_widget, 0, 0, 1, 2)
    layout.addWidget(widget, 1, 0)
    if button is not None:
        layout.addWidget(button, 1, 1)
    return row_widget


def _safe_float_text(value: str) -> float | None:
    try:
        text = value.strip()
        if not text:
            return None
        return float(text)
    except ValueError:
        return None


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _persistable_path_text(value: str) -> str:
    text = value.strip()
    if not text or text.lower().startswith("no "):
        return ""
    return text


def run() -> None:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Musa AI Experimental Comparison")
    window = ExperimentalComparisonWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
