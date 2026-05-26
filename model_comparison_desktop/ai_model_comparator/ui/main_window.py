from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ai_model_comparator.data.loader import empty_dataset, load_dataset_file
from ai_model_comparator.data.models import ComparisonDataset
from ai_model_comparator.data.settings_store import AppConfig, SettingsStore
from ai_model_comparator.detection.worker import DetectionWorker
from ai_model_comparator.ui.image_viewer import ImageViewer, MarkerPayload
from ai_model_comparator.ui.loading_dialog import MappingLoadingDialog
from ai_model_comparator.ui.panels import SidePanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Musa AI Model Comparison")

        self.settings_store = SettingsStore()
        self.config: AppConfig = self.settings_store.load_config()
        self.dataset: ComparisonDataset = empty_dataset()
        self.image_path: Path | None = None
        self.visible_model_ids = {model.id for model in self.dataset.models}
        self.selected_model_id = self.dataset.models[0].id if self.dataset.models else ""
        self.worker: DetectionWorker | None = None
        self.loading_dialog: MappingLoadingDialog | None = None

        self.viewer = ImageViewer()
        self.side_panel = SidePanel()
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setObjectName("toolbarMetric")
        self.image_label = QLabel("No image loaded")
        self.image_label.setObjectName("toolbarMetric")
        self.progress = QProgressBar()
        self.progress.setObjectName("pipelineProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setFixedWidth(260)
        self.mapping_mode_combo = QComboBox()
        self.mapping_mode_combo.setObjectName("mappingModeCombo")
        self.mapping_mode_combo.addItem("Gemini Vision", "gemini")
        self.mapping_mode_combo.addItem("Musa AI YOLO Pipeline", "trained_pipeline")

        self._build_toolbar()
        self._build_layout()
        self._connect_signals()
        self._load_initial_state()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main toolbar")
        toolbar.setMovable(False)
        toolbar.setObjectName("mainToolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        open_action = QAction("Open Image", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        load_results_action = QAction("Load Results", self)
        load_results_action.triggered.connect(self.load_results_file)
        toolbar.addAction(load_results_action)

        toolbar.addWidget(self.mapping_mode_combo)

        self.run_detection_action = QAction("Run Detection", self)
        self.run_detection_action.triggered.connect(self.run_detection)
        toolbar.addAction(self.run_detection_action)

        fit_action = QAction("Fit", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self.viewer.fit_to_view)
        toolbar.addAction(fit_action)

        reset_action = QAction("Reset", self)
        reset_action.setShortcut(QKeySequence("Ctrl+R"))
        reset_action.triggered.connect(self.viewer.reset_zoom)
        toolbar.addAction(reset_action)

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self.viewer.zoom_in)
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self.viewer.zoom_out)
        toolbar.addAction(zoom_out_action)

        toolbar.addSeparator()

        show_selected = QAction("Selected Only", self)
        show_selected.triggered.connect(self.show_selected_model_only)
        toolbar.addAction(show_selected)

        show_all = QAction("Show All", self)
        show_all.triggered.connect(self.show_all_models)
        toolbar.addAction(show_all)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(self.progress)
        toolbar.addWidget(self.image_label)
        toolbar.addWidget(self.zoom_label)

    def _build_layout(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(12)

        header = QWidget()
        header.setObjectName("pageHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        title_box = QVBoxLayout()
        title = QLabel("AI Model Detection Comparison")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Compare Gemini against the Musa AI two-model YOLO pipeline on one high-resolution image.")
        subtitle.setObjectName("pageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("headerSummary")
        header_layout.addWidget(self.summary_label)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.viewer)
        splitter.addWidget(self.side_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1000, 430])
        root.addWidget(splitter, 1)

        hint = QLabel("Open a real 4K image, load real results, then use mouse wheel to zoom and drag to pan.")
        hint.setObjectName("footerHint")
        root.addWidget(hint)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.viewer.markerSelected.connect(self._handle_marker_selected)
        self.viewer.zoomChanged.connect(self._handle_zoom_changed)
        self.side_panel.visibilityChanged.connect(self._handle_model_visibility_changed)
        self.side_panel.selectedModelChanged.connect(self._handle_selected_model_changed)
        self.side_panel.configSaved.connect(self._handle_config_saved)

    def _load_initial_state(self) -> None:
        self.viewer.clear()
        self._apply_config_to_models()
        self._refresh_panels()
        self._refresh_markers()
        self._refresh_summary()

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 4K image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not path:
            return
        if not self.viewer.load_image(path):
            QMessageBox.warning(self, "Unsupported Image", "The selected image could not be loaded.")
            return
        self.image_path = Path(path)
        self.image_label.setText(self.image_path.name)
        self._refresh_markers()
        self._refresh_summary()

    def load_results_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load detection comparison results",
            str(Path.home()),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.dataset = load_dataset_file(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid results file", f"The selected JSON file could not be loaded:\n{exc}")
            return
        self.visible_model_ids = {model.id for model in self.dataset.models}
        self.selected_model_id = self.dataset.models[0].id if self.dataset.models else ""
        self._apply_config_to_models()
        self._refresh_panels()
        self._refresh_markers()
        self._refresh_summary()

    def run_detection(self) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Detection running", "A detection run is already in progress.")
            return
        if self.image_path is None:
            QMessageBox.warning(self, "Image required", "Open a 4K image before running detection.")
            return
        gemini_key = self.settings_store.load_gemini_api_key()
        mapping_mode = self.mapping_mode_combo.currentData() or "gemini"
        missing = []
        if mapping_mode == "gemini" and not gemini_key:
            missing.append("Gemini API key")
        if mapping_mode == "trained_pipeline" and not self.config.leaf_model_path:
            missing.append("YOLOv8-seg leaf model path")
        if mapping_mode == "trained_pipeline" and not self.config.disease_model_path:
            missing.append("YOLOv8 disease model path")
        if missing:
            QMessageBox.warning(
                self,
                "Inputs required",
                "Save these inputs before running detection:\n" + "\n".join(missing),
            )
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_detection_action.setEnabled(False)
        self.summary_label.setText("Detection pipeline starting...")
        mapping_label = self.mapping_mode_combo.currentText()
        self.loading_dialog = MappingLoadingDialog(self.image_path.name, mapping_label, self)
        self.loading_dialog.set_progress(0, "Preparing overlapping tiles")
        self.loading_dialog.show()

        self.worker = DetectionWorker(
            self.image_path,
            empty_dataset(self.image_path.name),
            self.config,
            gemini_key,
            mapping_mode,
            parent=self,
        )
        self.worker.progressChanged.connect(self._handle_detection_progress)
        self.worker.providerProgressChanged.connect(self._handle_provider_detection_progress)
        self.worker.detectionFinished.connect(self._handle_detection_finished)
        self.worker.detectionFailed.connect(self._handle_detection_failed)
        self.worker.finished.connect(lambda: self.run_detection_action.setEnabled(True))
        self.worker.start()

    def show_selected_model_only(self) -> None:
        if self.selected_model_id:
            self.visible_model_ids = {self.selected_model_id}
            self._refresh_panels()
            self._refresh_markers()
            self._refresh_summary()

    def show_all_models(self) -> None:
        self.visible_model_ids = {model.id for model in self.dataset.models}
        self._refresh_panels()
        self._refresh_markers()
        self._refresh_summary()

    def _handle_model_visibility_changed(self, model_id: str, visible: bool) -> None:
        if visible:
            self.visible_model_ids.add(model_id)
        else:
            self.visible_model_ids.discard(model_id)
        self._refresh_panels()
        self._refresh_markers()
        self._refresh_summary()

    def _handle_selected_model_changed(self, model_id: str) -> None:
        self.selected_model_id = model_id
        self._refresh_panels()

    def _handle_config_saved(self, gemini_api_key: str, leaf_model_path: str, disease_model_path: str) -> None:
        self.settings_store.save_model_paths(leaf_model_path, disease_model_path)
        if gemini_api_key == "__CLEAR_GEMINI_KEY__":
            self.settings_store.save_gemini_api_key("")
        elif gemini_api_key or not self.config.gemini_api_key_configured:
            self.settings_store.save_gemini_api_key(gemini_api_key)
        self.config = self.settings_store.load_config()
        self._apply_config_to_models()
        self._refresh_panels()
        self._refresh_summary()
        QMessageBox.information(self, "Inputs saved", "Gemini key and model paths were saved locally.")

    def _handle_marker_selected(self, payload: MarkerPayload) -> None:
        self.selected_model_id = payload.model.id
        self.side_panel.set_detection(payload.model, payload.record)
        self._refresh_panels()

    def _handle_zoom_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"Zoom: {zoom * 100:.0f}%")

    def _handle_detection_progress(self, current: int, total: int, message: str) -> None:
        percent = int((current / max(total, 1)) * 100)
        self.progress.setValue(percent)
        self.summary_label.setText(message)
        if self.loading_dialog:
            self.loading_dialog.set_progress(percent, message)

    def _handle_provider_detection_progress(
        self,
        current: int,
        total: int,
        provider_id: str,
        tile_name: str,
        message: str,
    ) -> None:
        if self.loading_dialog:
            self.loading_dialog.set_provider_tile(provider_id, tile_name, message)

    def _handle_detection_finished(self, dataset: ComparisonDataset, elapsed_seconds: float, mapping_mode: str) -> None:
        self.dataset = dataset
        self.visible_model_ids = {model.id for model in self.dataset.models}
        self.selected_model_id = self.dataset.models[0].id if self.dataset.models else ""
        self._apply_config_to_models()
        self._refresh_panels()
        self._refresh_markers()
        self._refresh_summary()
        self.progress.setValue(100)
        if self.loading_dialog:
            self.loading_dialog.finish()
            self.loading_dialog = None
        model = self.dataset.model_by_id(mapping_mode)
        counts = model.counts_by_class() if model else {}
        count_lines = "\n".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "No detections"
        QMessageBox.information(
            self,
            "Detection complete",
            f"{self._mapping_label(mapping_mode)} finished in {self._format_elapsed(elapsed_seconds)}.\n\n{count_lines}",
        )

    def _handle_detection_failed(self, error_message: str) -> None:
        self.progress.setVisible(False)
        if self.loading_dialog:
            self.loading_dialog.fail(error_message)
            self.loading_dialog = None
        QMessageBox.critical(self, "Detection failed", error_message)
        self._refresh_summary()

    def _refresh_markers(self) -> None:
        self.viewer.set_markers(self.dataset.models, self.visible_model_ids)

    def _refresh_panels(self) -> None:
        self.side_panel.set_config(self.config)
        self.side_panel.set_models(self.dataset.models, self.visible_model_ids, self.selected_model_id)

    def _refresh_summary(self) -> None:
        visible_count = sum(model.count for model in self.dataset.models if model.id in self.visible_model_ids)
        image_status = self.image_path.name if self.image_path else "No image"
        self.summary_label.setText(
            f"{image_status} | {len(self.visible_model_ids)} visible models | {visible_count} visible detections"
        )

    def _apply_config_to_models(self) -> None:
        gemini = self.dataset.model_by_id("gemini")
        trained = self.dataset.model_by_id("trained_pipeline")
        if gemini:
            gemini.version = "configured" if self.config.gemini_api_key_configured else "not configured"
            gemini.description = "Existing Gemini model prompted to detect all four target classes."
        if trained:
            leaf_name = Path(self.config.leaf_model_path).name if self.config.leaf_model_path else "leaf model missing"
            disease_name = Path(self.config.disease_model_path).name if self.config.disease_model_path else "disease model missing"
            trained.version = f"{leaf_name} + {disease_name}"
            trained.description = "YOLOv8-seg for full/cut leaves plus YOLOv8 for black_sigatoka/panama."

    def _mapping_label(self, mapping_mode: str) -> str:
        return "Musa AI YOLO Pipeline" if mapping_mode == "trained_pipeline" else "Gemini Vision"

    def _format_elapsed(self, seconds: float) -> str:
        minutes, remainder = divmod(int(seconds), 60)
        if minutes:
            return f"{minutes}m {remainder}s"
        return f"{seconds:.1f}s"
