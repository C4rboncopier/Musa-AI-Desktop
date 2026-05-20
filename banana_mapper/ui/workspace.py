from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
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
from ..core.output_manager import OutputFile
from ..geotiff import GeoTiffInfo
from .widgets import ASSET_LABELS, AssetRow, MetricCard, card, meta_row, section_label


class OutputFileRow(QFrame):
    openRequested = pyqtSignal(str)
    revealRequested = pyqtSignal(str)
    exportRequested = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)

    def __init__(self, item: OutputFile, parent=None) -> None:
        super().__init__(parent)
        self.item = item
        self.setObjectName("outputRow")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(5)

        name = QLabel(self.item.path.name)
        name.setObjectName("outputTitle")
        name.setWordWrap(True)
        rel_path = QLabel(self.item.relative_path)
        rel_path.setObjectName("outputPath")
        rel_path.setWordWrap(True)
        details = QLabel(f"{self.item.file_type}  |  {self.item.size_label}  |  {self.item.created_at}")
        details.setObjectName("outputMeta")

        actions = QHBoxLayout()
        actions.setSpacing(5)
        for text, signal in [
            ("Open", self.openRequested),
            ("Reveal", self.revealRequested),
            ("Export", self.exportRequested),
            ("Delete", self.deleteRequested),
        ]:
            btn = QPushButton(text)
            btn.setObjectName("miniActionButton" if text != "Delete" else "miniDangerButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, sig=signal: sig.emit(str(self.item.path)))
            actions.addWidget(btn)

        layout.addWidget(name, 0, 0, 1, 2)
        layout.addWidget(rel_path, 1, 0, 1, 2)
        layout.addWidget(details, 2, 0)
        layout.addLayout(actions, 3, 0, 1, 2)


class WorkspacePage(QWidget):
    backRequested = pyqtSignal()
    importGeoTiffRequested = pyqtSignal()
    projectOutputOpenRequested = pyqtSignal()
    outputsRefreshRequested = pyqtSignal()
    outputOpenRequested = pyqtSignal(str)
    outputRevealRequested = pyqtSignal(str)
    outputExportRequested = pyqtSignal(str)
    outputDeleteRequested = pyqtSignal(str)
    runMappingRequested = pyqtSignal()
    fitMapRequested = pyqtSignal()
    resetViewRequested = pyqtSignal()
    zoomInRequested = pyqtSignal()
    zoomOutRequested = pyqtSignal()
    baseMapChanged = pyqtSignal(str)
    overlayVisibilityChanged = pyqtSignal(bool)
    overlayOpacityChanged = pyqtSignal(int)
    detectionOpacityChanged = pyqtSignal(int)
    detectionLayerChanged = pyqtSignal(str, bool)
    resolutionChanged = pyqtSignal(int)
    projectExplorerVisibilityChanged = pyqtSignal(bool)
    inspectorVisibilityChanged = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.current_bundle: ProjectBundle | None = None
        self.asset_list_layout: QVBoxLayout | None = None
        self.output_list_layout: QVBoxLayout | None = None
        self.metadata_labels: dict[str, QLabel] = {}
        self.count_labels: dict[str, QLabel] = {}
        self.layer_toggles: dict[str, QCheckBox] = {}
        self.splitter: QSplitter | None = None
        self.left_panel: QWidget | None = None
        self.right_panel: QWidget | None = None
        self._left_panel_width = 300
        self._right_panel_width = 360
        self._project_explorer_visible = True
        self._inspector_visible = True
        self._resolution_options: list[tuple[str, int, float]] = [
            ("Default Resolution - 100% (Original Resolution)", 100, 1.0),
            ("75% Scaled", 75, 0.75),
            ("50% Scaled", 50, 0.50),
            ("25% Scaled", 25, 0.25),
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
        topbar_layout.addWidget(self.back_btn)
        topbar_layout.addLayout(title_col, 1)
        topbar_layout.addWidget(self.run_btn)
        root.addWidget(topbar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("workspaceSplitter")
        self.splitter.setHandleWidth(1)

        self.left_panel = self._build_left_panel()
        center = self._build_map_panel()
        self.right_panel = self._build_right_panel()
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(center)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([self._left_panel_width, 850, self._right_panel_width])
        self.splitter.setCollapsible(0, True)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, True)
        self.splitter.splitterMoved.connect(self._on_splitter_moved)
        root.addWidget(self.splitter, 1)

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
        self.export_name_label = QLabel("No folder selected")
        self.export_name_label.setObjectName("exportPathName")
        self.export_name_label.setWordWrap(True)
        self.export_name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.export_location_label = QLabel("Managed automatically")
        self.export_location_label.setObjectName("exportPathLocation")
        self.export_location_label.setWordWrap(True)
        self.export_location_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        for title, value in [
            ("Created", self.created_label),
            ("Modified", self.modified_label),
        ]:
            row = QHBoxLayout()
            key = QLabel(title.upper())
            key.setObjectName("metaKey")
            row.addWidget(key)
            row.addWidget(value, 1, Qt.AlignmentFlag.AlignRight)
            pc_layout.addLayout(row)

        export_header = QHBoxLayout()
        export_header.setSpacing(8)
        export_key = QLabel("EXPORT")
        export_key.setObjectName("metaKey")
        export_header.addWidget(export_key)
        export_header.addStretch(1)
        output_btn = QPushButton("Open")
        output_btn.setObjectName("compactButton")
        output_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        output_btn.clicked.connect(lambda _=False: self.projectOutputOpenRequested.emit())
        export_header.addWidget(output_btn)

        export_summary = QFrame()
        export_summary.setObjectName("exportPathSummary")
        export_summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        export_summary_layout = QVBoxLayout(export_summary)
        export_summary_layout.setContentsMargins(10, 8, 10, 8)
        export_summary_layout.setSpacing(3)
        export_summary_layout.addWidget(self.export_name_label)
        export_summary_layout.addWidget(self.export_location_label)

        export_block = QVBoxLayout()
        export_block.setSpacing(6)
        export_block.addLayout(export_header)
        export_block.addWidget(export_summary)
        pc_layout.addLayout(export_block)
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
        self.base_map_combo = QComboBox()
        self.base_map_combo.setObjectName("mapBaseCombo")
        self.base_map_combo.addItem("OpenStreetMap", "osm")
        self.base_map_combo.addItem("Google Satellite", "google_satellite")
        self.base_map_combo.setFixedWidth(178)
        controls_layout.addWidget(self.base_map_combo)
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
        settings = self.map_view.settings()
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalStorageEnabled, True
        )
        self._set_web_engine_attribute(settings, "WebGLEnabled", True)
        self._set_web_engine_attribute(settings, "Accelerated2dCanvasEnabled", True)
        self._set_web_engine_attribute(settings, "ScrollAnimatorEnabled", False)
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
        self.tabs.addTab(self._outputs_tab(), "Outputs")
        self.tabs.addTab(self._ai_tab(), "AI")
        self.tabs.addTab(self._metadata_tab(), "Metadata")
        self.tabs.addTab(self._logs_tab(), "Console")
        layout.addWidget(self.tabs)
        return panel

    def _assets_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("inspectorTabPage")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(12)

        resolution = card("GeoTIFF Display")
        res_layout = resolution.layout()
        helper = QLabel("Choose the display scale before importing or refreshing the orthomosaic preview.")
        helper.setObjectName("bodyText")
        helper.setWordWrap(True)
        res_layout.addWidget(helper)
        self.geotiff_resolution_combo = self._make_resolution_combo()
        res_layout.addWidget(self.geotiff_resolution_combo)
        self.import_geotiff_btn = QPushButton("Import GeoTIFF")
        self.import_geotiff_btn.setObjectName("primaryButton")
        self.import_geotiff_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        res_layout.addWidget(self.import_geotiff_btn)
        self.render_resolution_label = QLabel("Display scale: Default Resolution - 100% (Original Resolution)")
        self.render_resolution_label.setObjectName("metaValue")
        self.render_resolution_label.setWordWrap(True)
        self.processing_resolution_label = QLabel("Processing: Native GeoTIFF resolution")
        self.processing_resolution_label.setObjectName("metaValue")
        self.processing_resolution_label.setWordWrap(True)
        self.downsample_label = QLabel("Preview scaling: original display grid")
        self.downsample_label.setObjectName("metaValue")
        self.downsample_label.setWordWrap(True)
        res_layout.addWidget(self.render_resolution_label)
        res_layout.addWidget(self.processing_resolution_label)
        res_layout.addWidget(self.downsample_label)
        layout.addWidget(resolution)

        layout.addWidget(section_label("Linked Assets"))
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

    def _outputs_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("inspectorTabPage")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Project Output Manager")
        title.setObjectName("panelTitle")
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("compactButton")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(lambda _=False: self.outputsRefreshRequested.emit())
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.output_list_layout = QVBoxLayout(content)
        self.output_list_layout.setContentsMargins(0, 0, 0, 0)
        self.output_list_layout.setSpacing(8)
        self.output_list_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        return tab

    def _ai_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("inspectorTabPage")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)

        workflow = card("AI Mapping Workflow")
        w_layout = workflow.layout()
        self.geotiff_source_status = QLabel("Source: no GeoTIFF imported")
        self.geotiff_source_status.setObjectName("metaValue")
        self.geotiff_source_status.setWordWrap(True)
        self.ai_processing_status = QLabel("AI resolution: Native GeoTIFF pixels")
        self.ai_processing_status.setObjectName("metaValue")
        self.ai_processing_status.setWordWrap(True)
        w_layout.addWidget(self.geotiff_source_status)
        w_layout.addWidget(self.ai_processing_status)

        self.output_status = QLabel("Output: system-managed project folder")
        self.output_status.setObjectName("metaValue")
        self.output_status.setWordWrap(True)
        self.ai_model_status = QLabel("AI models: managed in AI Models")
        self.ai_model_status.setObjectName("metaValue")
        self.ai_model_status.setWordWrap(True)
        w_layout.addWidget(self.output_status)
        w_layout.addWidget(self.ai_model_status)
        run_btn = QPushButton("Run AI Mapping")
        run_btn.setObjectName("primaryButton")
        run_btn.clicked.connect(lambda _=False: self.runMappingRequested.emit())
        w_layout.addWidget(run_btn)
        layout.addWidget(workflow)
        layout.addStretch(1)
        return tab

    def _metadata_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("inspectorTabPage")
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
            "Elevation / Altitude",
            "Camera",
            "Capture Time",
            "Sensor Settings",
            "Flight / GPS",
            "Source Extent",
            "Map Extent",
            "Geotransform",
            "Pixel Size",
            "Rendering Preview",
            "Raster Details",
            "Storage",
            "Band Details",
            "Metadata Tags",
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
        tab.setObjectName("inspectorTabPage")
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
        self.import_geotiff_btn.clicked.connect(lambda _=False: self.importGeoTiffRequested.emit())
        self.run_btn.clicked.connect(lambda _=False: self.runMappingRequested.emit())
        self.visibility_toggle.toggled.connect(self.overlayVisibilityChanged.emit)
        self.opacity_slider.valueChanged.connect(self._overlay_opacity_changed)
        self.ai_opacity_slider.valueChanged.connect(self._detection_opacity_changed)
        self.base_map_combo.currentIndexChanged.connect(self._on_base_map_changed)
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

    @staticmethod
    def _set_web_engine_attribute(settings, attribute_name: str, enabled: bool) -> None:
        attribute = getattr(QWebEngineSettings.WebAttribute, attribute_name, None)
        if attribute is not None:
            settings.setAttribute(attribute, enabled)

    def set_project(self, bundle: ProjectBundle) -> None:
        self.current_bundle = bundle
        project = bundle.project
        self.project_title.setText(project.name)
        self.project_subtitle.setText(project.description or "Drone mapping project")
        self.created_label.setText(project.created_at.replace("T", " ")[:16])
        self.modified_label.setText(project.modified_at.replace("T", " ")[:16])
        output_text = self._output_text(project.output_dir)
        export_name, export_location = self._export_display_parts(output_text)
        self.export_name_label.setText(export_name)
        self.export_location_label.setText(export_location)
        self.export_name_label.setToolTip(output_text)
        self.export_location_label.setToolTip(output_text)
        geotiff = bundle.first_asset("geotiff")
        self.geotiff_source_status.setText(
            f"Source: {geotiff.display_name if geotiff and geotiff.exists else 'no GeoTIFF imported'}"
        )
        self.output_status.setText(f"Output: system-managed at {output_text}")
        leaf = bundle.first_model("leaf")
        disease = bundle.first_model("disease")
        self.ai_model_status.setText(
            "AI models: configured"
            if leaf is not None and disease is not None and leaf.exists and disease.exists
            else "AI models: configure in AI Models"
        )
        self.render_assets(bundle)

    def set_outputs(self, outputs: list[OutputFile]) -> None:
        if self.output_list_layout is None:
            return
        while self.output_list_layout.count() > 1:
            item = self.output_list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not outputs:
            empty = QLabel("No generated outputs yet.")
            empty.setObjectName("bodyText")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.output_list_layout.insertWidget(0, empty)
            return

        for output in outputs:
            row = OutputFileRow(output)
            row.openRequested.connect(self.outputOpenRequested.emit)
            row.revealRequested.connect(self.outputRevealRequested.emit)
            row.exportRequested.connect(self.outputExportRequested.emit)
            row.deleteRequested.connect(self.outputDeleteRequested.emit)
            self.output_list_layout.insertWidget(self.output_list_layout.count() - 1, row)

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
            if asset.asset_type in {"image_folder", "mrk_file"}:
                continue
            title = ASSET_LABELS.get(asset.asset_type, asset.asset_type.replace("_", " ").title())
            rows.append(AssetRow(title, asset.path, asset.exists, asset.modified_at.replace("T", " ")[:16]))
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
        for field in [
            "Elevation / Altitude",
            "Camera",
            "Capture Time",
            "Sensor Settings",
            "Flight / GPS",
            "Raster Details",
            "Storage",
            "Band Details",
            "Metadata Tags",
        ]:
            self.metadata_labels[field].setText(info.metadata_details.get(field, "--"))
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

    def set_base_map(self, base_map: str) -> None:
        for index in range(self.base_map_combo.count()):
            if self.base_map_combo.itemData(index) == base_map:
                self.base_map_combo.blockSignals(True)
                self.base_map_combo.setCurrentIndex(index)
                self.base_map_combo.blockSignals(False)
                return

    def set_project_explorer_visible(self, visible: bool) -> None:
        self._set_side_panel_visible(self.left_panel, 0, visible)

    def set_inspector_visible(self, visible: bool) -> None:
        self._set_side_panel_visible(self.right_panel, 2, visible)

    def is_project_explorer_visible(self) -> bool:
        return self._project_explorer_visible

    def is_inspector_visible(self) -> bool:
        return self._inspector_visible

    def append_log(self, message: str) -> None:
        self.log_output.append(message)

    def selected_resolution_scale(self) -> float:
        if 0 <= self._resolution_index < len(self._resolution_options):
            return self._resolution_options[self._resolution_index][2]
        return 1.0

    def selected_resolution_percent(self) -> int:
        if 0 <= self._resolution_index < len(self._resolution_options):
            return self._resolution_options[self._resolution_index][1]
        return 100

    def set_resolution_index(self, index: int) -> None:
        self._set_resolution_index(index)

    def _overlay_opacity_changed(self, value: int) -> None:
        self.opacity_value.setText(f"{value}%")
        self.overlayOpacityChanged.emit(value)

    def _detection_opacity_changed(self, value: int) -> None:
        self.ai_opacity_value.setText(f"{value}%")
        self.detectionOpacityChanged.emit(value)

    def _on_base_map_changed(self, index: int) -> None:
        value = self.base_map_combo.itemData(index)
        if value:
            self.baseMapChanged.emit(str(value))

    def _make_resolution_combo(self) -> QComboBox:
        combo = QComboBox()
        for label, _, _ in self._resolution_options:
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
        label, percent, _ = self._resolution_options[self._resolution_index]
        render_text = f"Display scale: {label}"
        downsample_text = (
            "Preview scaling: original display grid"
            if percent == 100
            else f"Preview scaling: {percent}% of the original display grid"
        )
        if hasattr(self, "render_resolution_label"):
            self.render_resolution_label.setText(render_text)
        if hasattr(self, "downsample_label"):
            self.downsample_label.setText(downsample_text)

    def _set_side_panel_visible(self, panel: QWidget | None, index: int, visible: bool) -> None:
        if panel is None or self.splitter is None:
            return
        current = self._project_explorer_visible if index == 0 else self._inspector_visible
        if current == visible:
            return
        sizes = self.splitter.sizes()
        if not visible and index < len(sizes) and sizes[index] > 32:
            if index == 0:
                self._left_panel_width = sizes[index]
            elif index == 2:
                self._right_panel_width = sizes[index]
        if index == 0:
            self._project_explorer_visible = visible
        elif index == 2:
            self._inspector_visible = visible
        panel.setVisible(visible)
        self._restore_splitter_sizes()
        self._refresh_map_layout()
        self._emit_side_panel_visibility(index, visible)

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        if self.splitter is None:
            return
        sizes = self.splitter.sizes()
        if len(sizes) < 3:
            return
        self._sync_side_panel_visibility(0, sizes[0] > 24)
        self._sync_side_panel_visibility(2, sizes[2] > 24)

    def _sync_side_panel_visibility(self, index: int, visible: bool) -> None:
        if index == 0:
            if self._project_explorer_visible == visible:
                if visible and self.splitter:
                    self._left_panel_width = max(self.splitter.sizes()[0], self._left_panel_width)
                return
            self._project_explorer_visible = visible
            if visible and self.splitter:
                self._left_panel_width = max(self.splitter.sizes()[0], self._left_panel_width)
        elif index == 2:
            if self._inspector_visible == visible:
                if visible and self.splitter:
                    self._right_panel_width = max(self.splitter.sizes()[2], self._right_panel_width)
                return
            self._inspector_visible = visible
            if visible and self.splitter:
                self._right_panel_width = max(self.splitter.sizes()[2], self._right_panel_width)
        else:
            return
        self._emit_side_panel_visibility(index, visible)
        self._refresh_map_layout()

    def _emit_side_panel_visibility(self, index: int, visible: bool) -> None:
        if index == 0:
            self.projectExplorerVisibilityChanged.emit(visible)
        elif index == 2:
            self.inspectorVisibilityChanged.emit(visible)

    def _restore_splitter_sizes(self) -> None:
        if self.splitter is None:
            return
        total = max(self.splitter.width(), sum(self.splitter.sizes()), 900)
        left = self._left_panel_width if self.is_project_explorer_visible() else 0
        right = self._right_panel_width if self.is_inspector_visible() else 0
        center = max(420, total - left - right)
        self.splitter.setSizes([left, center, right])

    def _refresh_map_layout(self) -> None:
        script = "if (typeof map !== 'undefined' && map) setTimeout(function() { map.invalidateSize(true); }, 60);"
        QTimer.singleShot(80, lambda: self.map_view.page().runJavaScript(script))

    @staticmethod
    def _output_text(path: str) -> str:
        if not path:
            return "--"
        output_path = Path(path)
        if output_path.exists():
            return str(output_path)
        return f"Missing: {output_path}"

    @staticmethod
    def _export_display_parts(path_text: str) -> tuple[str, str]:
        if path_text == "--":
            return "No output folder", "Managed automatically"

        prefix = ""
        raw_path = path_text
        if path_text.startswith("Missing: "):
            prefix = "Missing: "
            raw_path = path_text.removeprefix("Missing: ")

        normalized = raw_path.replace("/", "\\")
        parts = [part for part in normalized.split("\\") if part]
        if not parts:
            return f"{prefix}{raw_path}", ""

        leaf = parts[-1]
        parents = parts[:-1]
        if not parents:
            return f"{prefix}{leaf}", normalized

        if len(parents) <= 2:
            context = "\\".join(parents)
        elif parents[0].endswith(":"):
            context = f"{parents[0]}\\...\\{parents[-1]}"
        elif normalized.startswith("\\\\") and len(parents) >= 3:
            context = f"\\\\{parents[0]}\\{parents[1]}\\...\\{parents[-1]}"
        else:
            context = f"{parents[0]}\\...\\{parents[-1]}"
        return f"{prefix}{leaf}", context

    @staticmethod
    def _asset_name(path: str) -> str:
        return Path(path).name if path else "--"
