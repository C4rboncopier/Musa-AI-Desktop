from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.models import ProjectBundle
from ..geotiff import GeoTiffInfo
from .widgets import ASSET_LABELS, MODEL_LABELS, AssetRow, MetricCard, card, meta_row, section_label


class WorkspacePage(QWidget):
    backRequested = pyqtSignal()
    importGeoTiffRequested = pyqtSignal()
    imageFolderRequested = pyqtSignal()
    mrkFileRequested = pyqtSignal()
    outputFolderRequested = pyqtSignal()
    leafModelRequested = pyqtSignal()
    diseaseModelRequested = pyqtSignal()
    runMappingRequested = pyqtSignal()
    pinCheckerRequested = pyqtSignal()
    clearPinRequested = pyqtSignal()
    fitMapRequested = pyqtSignal()
    resetViewRequested = pyqtSignal()
    zoomInRequested = pyqtSignal()
    zoomOutRequested = pyqtSignal()
    overlayVisibilityChanged = pyqtSignal(bool)
    overlayOpacityChanged = pyqtSignal(int)
    detectionOpacityChanged = pyqtSignal(int)
    detectionLayerChanged = pyqtSignal(str, bool)
    dataSourceChanged = pyqtSignal(int)
    resolutionChanged = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.current_bundle: ProjectBundle | None = None
        self.asset_list_layout: QVBoxLayout | None = None
        self.metadata_labels: dict[str, QLabel] = {}
        self.count_labels: dict[str, QLabel] = {}
        self.layer_toggles: dict[str, QCheckBox] = {}
        self._resolution_options: list[tuple[str, int]] = [
            ("Maximum (65 MP, GPU optimized)", 65_000_000),
            ("High (40 MP)", 40_000_000),
            ("Medium (20 MP)", 20_000_000),
            ("Low (10 MP)", 10_000_000),
            ("Minimum (5 MP)", 5_000_000),
        ]
        self._resolution_index = 0
        self._syncing_resolution = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        topbar = QFrame()
        topbar.setObjectName("workspaceTopbar")
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(14, 10, 14, 10)
        topbar_layout.setSpacing(10)

        self.back_btn = QPushButton("Dashboard")
        self.back_btn.setObjectName("navButton")
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.project_title = QLabel("No Project")
        self.project_title.setObjectName("workspaceTitle")
        self.project_subtitle = QLabel("Select or create a project to begin")
        self.project_subtitle.setObjectName("pageSubtitle")
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_col.addWidget(self.project_title)
        title_col.addWidget(self.project_subtitle)

        self.run_btn = QPushButton("Run AI Mapping")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_btn = QPushButton("Import GeoTIFF")
        self.import_btn.setObjectName("secondaryButton")
        self.import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        topbar_layout.addWidget(self.back_btn)
        topbar_layout.addLayout(title_col, 1)
        topbar_layout.addWidget(self.import_btn)
        topbar_layout.addWidget(self.run_btn)
        root.addWidget(topbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("workspaceSplitter")
        splitter.setHandleWidth(1)

        left = self._build_left_panel()
        center = self._build_map_panel()
        right = self._build_right_panel()
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setSizes([300, 850, 360])
        splitter.setCollapsible(1, False)
        root.addWidget(splitter, 1)

        self._connect_ui()

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("workspaceSidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        layout.addWidget(section_label("Project Explorer"))
        project_card = card("Session")
        pc_layout = project_card.layout()
        self.created_label = QLabel("--")
        self.created_label.setObjectName("metaValue")
        self.modified_label = QLabel("--")
        self.modified_label.setObjectName("metaValue")
        self.output_label = QLabel("--")
        self.output_label.setObjectName("metaValue")
        self.output_label.setWordWrap(True)
        for title, value in [
            ("Created", self.created_label),
            ("Modified", self.modified_label),
            ("Export", self.output_label),
        ]:
            row = QHBoxLayout()
            key = QLabel(title.upper())
            key.setObjectName("metaKey")
            row.addWidget(key)
            row.addWidget(value, 1, Qt.AlignmentFlag.AlignRight)
            pc_layout.addLayout(row)
        output_btn = QPushButton("Change Folder")
        output_btn.setObjectName("secondaryButton")
        output_btn.clicked.connect(lambda _=False: self.outputFolderRequested.emit())
        pc_layout.addWidget(output_btn)
        layout.addWidget(project_card)

        layout.addWidget(section_label("Layers"))
        layer_card = card("Map Layers")
        lc_layout = layer_card.layout()
        self.visibility_toggle = QCheckBox("Orthomosaic")
        self.visibility_toggle.setChecked(True)
        self.visibility_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        lc_layout.addWidget(self.visibility_toggle)

        overlay_row = QHBoxLayout()
        overlay_label = QLabel("GeoTIFF opacity")
        overlay_label.setObjectName("bodyText")
        self.opacity_value = QLabel("78%")
        self.opacity_value.setObjectName("opacityValue")
        overlay_row.addWidget(overlay_label)
        overlay_row.addWidget(self.opacity_value, 1, Qt.AlignmentFlag.AlignRight)
        lc_layout.addLayout(overlay_row)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(78)
        lc_layout.addWidget(self.opacity_slider)

        ai_row = QHBoxLayout()
        ai_label = QLabel("Detection opacity")
        ai_label.setObjectName("bodyText")
        self.ai_opacity_value = QLabel("100%")
        self.ai_opacity_value.setObjectName("opacityValue")
        ai_row.addWidget(ai_label)
        ai_row.addWidget(self.ai_opacity_value, 1, Qt.AlignmentFlag.AlignRight)
        lc_layout.addLayout(ai_row)
        self.ai_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.ai_opacity_slider.setRange(0, 100)
        self.ai_opacity_slider.setValue(100)
        lc_layout.addWidget(self.ai_opacity_slider)

        layer_specs = [
            ("cut_leaf", "Cut leaf points"),
            ("black_sigatoka", "Black sigatoka points"),
            ("panama", "Panama points"),
        ]
        for key, label in layer_specs:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
            self.layer_toggles[key] = checkbox
            lc_layout.addWidget(checkbox)
        layout.addWidget(layer_card)

        counts = card("Detection Counts")
        counts_layout = counts.layout()
        for label, key in [
            ("Full", "full_leaf"),
            ("Cut", "cut_leaf"),
            ("Healthy", "healthy_leaf"),
            ("Diseased", "diseased_leaf"),
            ("Sigatoka", "black_sigatoka"),
            ("Panama", "panama"),
        ]:
            row, value = meta_row(label, "--")
            counts_layout.addWidget(row)
            self.count_labels[key] = value
        layout.addWidget(counts)

        layout.addStretch(1)
        return panel

    def _build_map_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("mapShell")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        controls = QFrame()
        controls.setObjectName("mapToolbar")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setSpacing(8)
        for text, signal in [
            ("+", self.zoomInRequested),
            ("-", self.zoomOutRequested),
            ("Fit", self.fitMapRequested),
            ("Reset", self.resetViewRequested),
        ]:
            btn = QToolButton()
            btn.setText(text)
            btn.setObjectName("mapToolButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, sig=signal: sig.emit())
            controls_layout.addWidget(btn)
        controls_layout.addStretch(1)
        self.coord_label = QLabel("Cursor: --, --")
        self.coord_label.setObjectName("mapToolbarLabel")
        self.zoom_label = QLabel("Zoom: --")
        self.zoom_label.setObjectName("mapToolbarLabel")
        controls_layout.addWidget(self.coord_label)
        controls_layout.addWidget(self.zoom_label)
        layout.addWidget(controls)

        self.map_view = QWebEngineView(panel)
        self.map_view.setObjectName("mapView")
        self.map_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self.map_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
        self.map_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalStorageEnabled, True
        )
        layout.addWidget(self.map_view, 1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("workspaceInspector")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("inspectorTabs")
        self.tabs.addTab(self._assets_tab(), "Assets")
        self.tabs.addTab(self._ai_tab(), "AI")
        self.tabs.addTab(self._metadata_tab(), "Metadata")
        self.tabs.addTab(self._logs_tab(), "Console")
        layout.addWidget(self.tabs)
        return panel

    def _assets_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)
        actions = QGridLayout()
        actions.setSpacing(8)
        for text, signal, row, col in [
            ("Link GeoTIFF", self.importGeoTiffRequested, 0, 0),
            ("Image Folder", self.imageFolderRequested, 0, 1),
            ("MRK File", self.mrkFileRequested, 1, 0),
            ("Leaf Model", self.leafModelRequested, 1, 1),
            ("Disease Model", self.diseaseModelRequested, 2, 0),
        ]:
            btn = QPushButton(text)
            btn.setObjectName("secondaryButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, sig=signal: sig.emit())
            actions.addWidget(btn, row, col)
        layout.addLayout(actions)

        resolution = card("GeoTIFF Display Resolution")
        res_layout = resolution.layout()
        self.geotiff_resolution_combo = self._make_resolution_combo()
        res_layout.addWidget(self.geotiff_resolution_combo)
        self.render_resolution_label = QLabel("Rendering: Maximum (65 MP, GPU optimized)")
        self.render_resolution_label.setObjectName("metaValue")
        self.render_resolution_label.setWordWrap(True)
        self.processing_resolution_label = QLabel("Processing: Native GeoTIFF resolution")
        self.processing_resolution_label.setObjectName("metaValue")
        self.processing_resolution_label.setWordWrap(True)
        self.downsample_label = QLabel("Downsampling: Auto, constrained by selected preview budget")
        self.downsample_label.setObjectName("metaValue")
        self.downsample_label.setWordWrap(True)
        res_layout.addWidget(self.render_resolution_label)
        res_layout.addWidget(self.processing_resolution_label)
        res_layout.addWidget(self.downsample_label)
        layout.addWidget(resolution)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.asset_list_layout = QVBoxLayout(content)
        self.asset_list_layout.setContentsMargins(0, 0, 0, 0)
        self.asset_list_layout.setSpacing(8)
        self.asset_list_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        return tab

    def _ai_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)

        workflow = card("AI Mapping Workflow")
        w_layout = workflow.layout()
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItem("Source: linked image folder")
        self.data_source_combo.addItem("Source: current GeoTIFF")
        w_layout.addWidget(self.data_source_combo)

        self.ai_processing_status = QLabel("AI resolution: Native GeoTIFF pixels")
        self.ai_processing_status.setObjectName("metaValue")
        self.ai_processing_status.setWordWrap(True)
        w_layout.addWidget(self.ai_processing_status)

        self.folder_status = QLabel("Image folder: --")
        self.folder_status.setObjectName("metaValue")
        self.folder_status.setWordWrap(True)
        self.mrk_status = QLabel("MRK file: --")
        self.mrk_status.setObjectName("metaValue")
        self.mrk_status.setWordWrap(True)
        self.output_status = QLabel("Export folder: --")
        self.output_status.setObjectName("metaValue")
        self.output_status.setWordWrap(True)
        self.leaf_model_status = QLabel("Leaf model: --")
        self.leaf_model_status.setObjectName("metaValue")
        self.leaf_model_status.setWordWrap(True)
        self.disease_model_status = QLabel("Disease model: --")
        self.disease_model_status.setObjectName("metaValue")
        self.disease_model_status.setWordWrap(True)
        w_layout.addWidget(self.folder_status)
        w_layout.addWidget(self.mrk_status)
        w_layout.addWidget(self.output_status)
        w_layout.addWidget(self.leaf_model_status)
        w_layout.addWidget(self.disease_model_status)
        output_btn = QPushButton("Change Folder")
        output_btn.setObjectName("secondaryButton")
        output_btn.clicked.connect(lambda _=False: self.outputFolderRequested.emit())
        w_layout.addWidget(output_btn)
        run_btn = QPushButton("Run AI Mapping")
        run_btn.setObjectName("primaryButton")
        run_btn.clicked.connect(lambda _=False: self.runMappingRequested.emit())
        w_layout.addWidget(run_btn)
        layout.addWidget(workflow)

        checker = card("Coordinate QA")
        c_layout = checker.layout()
        qa_row = QHBoxLayout()
        pin_btn = QPushButton("Open Image")
        pin_btn.setObjectName("secondaryButton")
        clear_btn = QPushButton("Clear Pin")
        clear_btn.setObjectName("secondaryButton")
        pin_btn.clicked.connect(lambda _=False: self.pinCheckerRequested.emit())
        clear_btn.clicked.connect(lambda _=False: self.clearPinRequested.emit())
        qa_row.addWidget(pin_btn)
        qa_row.addWidget(clear_btn)
        c_layout.addLayout(qa_row)
        layout.addWidget(checker)
        layout.addStretch(1)
        return tab

    def _metadata_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)
        metadata = card("GeoTIFF Metadata")
        md_layout = metadata.layout()
        fields = [
            "File",
            "Pixel Resolution",
            "Spatial Resolution",
            "CRS",
            "Band Count",
            "Source Extent",
            "Map Extent",
            "Geotransform",
            "Pixel Size",
            "Rendering Preview",
        ]
        for field in fields:
            row, value = meta_row(field, "--")
            md_layout.addWidget(row)
            self.metadata_labels[field] = value
        layout.addWidget(metadata)
        layout.addStretch(1)
        return tab

    def _logs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)
        self.log_output = QTextEdit()
        self.log_output.setObjectName("logConsole")
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Processing messages and session recovery notes appear here.")
        layout.addWidget(self.log_output, 1)
        return tab

    def _connect_ui(self) -> None:
        self.back_btn.clicked.connect(lambda _=False: self.backRequested.emit())
        self.import_btn.clicked.connect(lambda _=False: self.importGeoTiffRequested.emit())
        self.run_btn.clicked.connect(lambda _=False: self.runMappingRequested.emit())
        self.visibility_toggle.toggled.connect(self.overlayVisibilityChanged.emit)
        self.opacity_slider.valueChanged.connect(self._overlay_opacity_changed)
        self.ai_opacity_slider.valueChanged.connect(self._detection_opacity_changed)
        self.data_source_combo.currentIndexChanged.connect(self.dataSourceChanged.emit)
        self.geotiff_resolution_combo.currentIndexChanged.connect(self._on_resolution_combo_changed)
        for key, checkbox in self.layer_toggles.items():
            checkbox.toggled.connect(
                lambda checked, layer_key=key: self.detectionLayerChanged.emit(layer_key, checked)
            )

    def load_map(self) -> None:
        html_path = Path(__file__).resolve().parent.parent / "map_view.html"
        self.map_view.load(QUrl.fromLocalFile(str(html_path)))

    def set_web_channel(self, channel) -> None:
        self.map_view.page().setWebChannel(channel)

    def set_project(self, bundle: ProjectBundle) -> None:
        self.current_bundle = bundle
        project = bundle.project
        self.project_title.setText(project.name)
        self.project_subtitle.setText(project.description or "Drone mapping project")
        self.created_label.setText(project.created_at.replace("T", " ")[:16])
        self.modified_label.setText(project.modified_at.replace("T", " ")[:16])
        output_text = self._output_text(project.output_dir)
        self.output_label.setText(output_text)
        self.folder_status.setText(f"Image folder: {self._asset_name(bundle.image_folder_path)}")
        mrk_asset = bundle.first_asset("mrk_file")
        self.mrk_status.setText(f"MRK file: {mrk_asset.display_name if mrk_asset else '--'}")
        self.output_status.setText(f"Export folder: {output_text}")
        leaf = bundle.first_model("leaf")
        disease = bundle.first_model("disease")
        self.leaf_model_status.setText(f"Leaf model: {leaf.display_name if leaf else '--'}")
        self.disease_model_status.setText(f"Disease model: {disease.display_name if disease else '--'}")
        self.render_assets(bundle)

    def render_assets(self, bundle: ProjectBundle) -> None:
        if self.asset_list_layout is None:
            return
        while self.asset_list_layout.count() > 1:
            item = self.asset_list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        rows = []
        for asset in bundle.assets:
            title = ASSET_LABELS.get(asset.asset_type, asset.asset_type.replace("_", " ").title())
            rows.append(AssetRow(title, asset.path, asset.exists, asset.modified_at.replace("T", " ")[:16]))
        for model in bundle.models:
            title = MODEL_LABELS.get(model.role, model.role.title())
            rows.append(AssetRow(title, model.path, model.exists, model.modified_at.replace("T", " ")[:16]))
        for result in bundle.results[:4]:
            rows.append(AssetRow("Analysis Result", result.json_path, result.exists, result.created_at.replace("T", " ")[:16]))

        if not rows:
            hint = QLabel("No assets linked yet.")
            hint.setObjectName("bodyText")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rows.append(hint)
        for row in rows:
            self.asset_list_layout.insertWidget(self.asset_list_layout.count() - 1, row)

    def set_geotiff_metadata(self, info: GeoTiffInfo) -> None:
        self.metadata_labels["File"].setText(info.file_name)
        self.metadata_labels["Pixel Resolution"].setText(info.pixel_resolution_label)
        self.metadata_labels["Spatial Resolution"].setText(info.spatial_resolution_label)
        self.metadata_labels["CRS"].setText(f"{info.source_crs_authority}\n{info.source_crs}")
        self.metadata_labels["Band Count"].setText(f"{info.band_count} band(s)")
        sw, ss, se, sn = info.bounds_source
        self.metadata_labels["Source Extent"].setText(f"W {sw:.8f}\nS {ss:.8f}\nE {se:.8f}\nN {sn:.8f}")
        self.metadata_labels["Map Extent"].setText(
            f"W {info.west:.8f}\nS {info.south:.8f}\nE {info.east:.8f}\nN {info.north:.8f}"
        )
        self.metadata_labels["Geotransform"].setText(", ".join(f"{v:.6f}" for v in info.transform))
        self.metadata_labels["Pixel Size"].setText(f"X {info.pixel_size_x:.8f}\nY {info.pixel_size_y:.8f}")
        self.metadata_labels["Rendering Preview"].setText(info.preview_resolution_label)
        native_text = f"AI resolution: Native {info.pixel_resolution_label}"
        self.processing_resolution_label.setText(f"Processing: Native {info.pixel_resolution_label}")
        self.ai_processing_status.setText(native_text)

    def clear_geotiff_metadata(self) -> None:
        for label in self.metadata_labels.values():
            label.setText("--")
        if hasattr(self, "processing_resolution_label"):
            self.processing_resolution_label.setText("Processing: Native GeoTIFF resolution")
        if hasattr(self, "ai_processing_status"):
            self.ai_processing_status.setText("AI resolution: Native GeoTIFF pixels")

    def update_counts(self, counts: dict[str, int]) -> None:
        for key, label in self.count_labels.items():
            label.setText(f"{counts.get(key, 0):,}")

    def set_coordinates(self, latitude: float, longitude: float) -> None:
        self.coord_label.setText(f"Cursor: {latitude:.7f}, {longitude:.7f}")

    def set_zoom(self, zoom: int) -> None:
        self.zoom_label.setText(f"Zoom: {zoom}")

    def append_log(self, message: str) -> None:
        self.log_output.append(message)

    def selected_resolution_pixels(self) -> int:
        if 0 <= self._resolution_index < len(self._resolution_options):
            return self._resolution_options[self._resolution_index][1]
        return 65_000_000

    def data_source_index(self) -> int:
        return self.data_source_combo.currentIndex()

    def _overlay_opacity_changed(self, value: int) -> None:
        self.opacity_value.setText(f"{value}%")
        self.overlayOpacityChanged.emit(value)

    def _detection_opacity_changed(self, value: int) -> None:
        self.ai_opacity_value.setText(f"{value}%")
        self.detectionOpacityChanged.emit(value)

    def _make_resolution_combo(self) -> QComboBox:
        combo = QComboBox()
        for label, _ in self._resolution_options:
            combo.addItem(label)
        combo.setCurrentIndex(self._resolution_index)
        return combo

    def _on_resolution_combo_changed(self, index: int) -> None:
        if self._syncing_resolution:
            return
        self._set_resolution_index(index)
        self.resolutionChanged.emit(index)

    def _set_resolution_index(self, index: int) -> None:
        if not 0 <= index < len(self._resolution_options):
            return
        self._resolution_index = index
        self._syncing_resolution = True
        try:
            if self.geotiff_resolution_combo.currentIndex() != index:
                self.geotiff_resolution_combo.setCurrentIndex(index)
        finally:
            self._syncing_resolution = False
        self._update_resolution_labels()

    def _update_resolution_labels(self) -> None:
        label, pixels = self._resolution_options[self._resolution_index]
        megapixels = pixels / 1_000_000
        render_text = f"Rendering: {label}"
        downsample_text = f"Downsampling: Auto cap at {megapixels:.0f} MP for responsive display"
        if hasattr(self, "render_resolution_label"):
            self.render_resolution_label.setText(render_text)
        if hasattr(self, "downsample_label"):
            self.downsample_label.setText(downsample_text)

    @staticmethod
    def _output_text(path: str) -> str:
        if not path:
            return "--"
        output_path = Path(path)
        if output_path.exists():
            return str(output_path)
        return f"Missing: {output_path}"

    @staticmethod
    def _asset_name(path: str) -> str:
        return Path(path).name if path else "--"
