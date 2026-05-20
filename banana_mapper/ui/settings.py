from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..hardware import HardwareStatus
from .widgets import StatusPill, card, meta_row


class SettingsPage(QWidget):
    backRequested = pyqtSignal()
    devicePreferenceChanged = pyqtSignal(str)
    refreshHardwareRequested = pyqtSignal()
    openPytorchGuideRequested = pyqtSignal()
    openNvidiaDriverRequested = pyqtSignal()
    clearCacheRequested = pyqtSignal()
    openOutputRootRequested = pyqtSignal()
    openDatabaseFolderRequested = pyqtSignal()
    googleMapsApiKeyChanged = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._syncing_device = False
        self._hardware: HardwareStatus | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        topbar = QFrame()
        topbar.setObjectName("workspaceTopbar")
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(18, 12, 18, 12)
        topbar_layout.setSpacing(12)

        back_btn = QPushButton("Dashboard")
        back_btn.setObjectName("navButton")
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.clicked.connect(lambda _=False: self.backRequested.emit())

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("Settings")
        title.setObjectName("workspaceTitle")
        subtitle = QLabel("Hardware, processing, storage, and project maintenance")
        subtitle.setObjectName("pageSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        topbar_layout.addWidget(back_btn)
        topbar_layout.addLayout(title_col, 1)
        root.addWidget(topbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("dashboardContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)

        layout.addWidget(self._build_hardware_card())
        layout.addWidget(self._build_processing_card())
        layout.addWidget(self._build_maps_card())
        layout.addWidget(self._build_setup_card())
        layout.addWidget(self._build_storage_card())
        layout.addStretch(1)

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _build_hardware_card(self) -> QFrame:
        frame = card("Hardware Check & Testing")
        layout = frame.layout()

        summary = QHBoxLayout()
        summary.setSpacing(10)
        self.hardware_state = StatusPill("Checking", "neutral")
        self.hardware_title = QLabel("Detecting hardware resources")
        self.hardware_title.setObjectName("panelTitle")
        self.hardware_detail = QLabel("Run a hardware check to see whether GPU acceleration is available.")
        self.hardware_detail.setObjectName("bodyText")
        self.hardware_detail.setWordWrap(True)
        text_col = QVBoxLayout()
        text_col.setSpacing(3)
        text_col.addWidget(self.hardware_title)
        text_col.addWidget(self.hardware_detail)
        summary.addWidget(self.hardware_state, 0, Qt.AlignmentFlag.AlignTop)
        summary.addLayout(text_col, 1)
        layout.addLayout(summary)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.cpu_value = self._add_hardware_tile(grid, 0, 0, "CPU", "Checking", "neutral", "--")
        self.ram_value = self._add_hardware_tile(grid, 0, 1, "RAM", "Checking", "neutral", "--")
        self.gpu_value = self._add_hardware_tile(grid, 1, 0, "GPU", "Checking", "neutral", "--")
        self.torch_value = self._add_hardware_tile(grid, 1, 1, "PyTorch", "Checking", "neutral", "--")
        self.cuda_value = self._add_hardware_tile(grid, 2, 0, "CUDA", "Checking", "neutral", "--")
        self.device_value = self._add_hardware_tile(grid, 2, 1, "Device", "Pending", "neutral", "--")
        layout.addLayout(grid)

        self.hardware_progress = QProgressBar()
        self.hardware_progress.setObjectName("hardwareProgress")
        self.hardware_progress.setRange(0, 100)
        self.hardware_progress.setValue(0)
        self.hardware_progress.setTextVisible(True)
        layout.addWidget(self.hardware_progress)

        self.hardware_log = QTextEdit()
        self.hardware_log.setObjectName("diagnosticLog")
        self.hardware_log.setReadOnly(True)
        self.hardware_log.setFixedHeight(128)
        self.hardware_log.setPlaceholderText("Diagnostic output appears here when a hardware check runs.")
        layout.addWidget(self.hardware_log)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.refresh_btn = QPushButton("Run Hardware Check")
        self.refresh_btn.setObjectName("secondaryButton")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(lambda _=False: self.refreshHardwareRequested.emit())
        actions.addWidget(self.refresh_btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        return frame

    def _build_processing_card(self) -> QFrame:
        frame = card("AI Processing")
        layout = frame.layout()
        row = QHBoxLayout()
        row.setSpacing(12)
        label_col = QVBoxLayout()
        label_col.setSpacing(3)
        title = QLabel("Processing device")
        title.setObjectName("panelTitle")
        body = QLabel("Choose the hardware used for YOLO inference. GPU is selected automatically when CUDA is ready.")
        body.setObjectName("bodyText")
        body.setWordWrap(True)
        label_col.addWidget(title)
        label_col.addWidget(body)
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        row.addLayout(label_col, 1)
        row.addWidget(self.device_combo, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)
        return frame

    def _build_setup_card(self) -> QFrame:
        frame = card("Guided GPU Setup")
        layout = frame.layout()
        self.setup_steps = QLabel("Hardware setup guidance will appear after the check finishes.")
        self.setup_steps.setObjectName("bodyText")
        self.setup_steps.setWordWrap(True)
        layout.addWidget(self.setup_steps)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        pytorch_btn = QPushButton("Open PyTorch GPU Setup")
        pytorch_btn.setObjectName("secondaryButton")
        pytorch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        pytorch_btn.clicked.connect(lambda _=False: self.openPytorchGuideRequested.emit())
        nvidia_btn = QPushButton("Open NVIDIA Driver Page")
        nvidia_btn.setObjectName("secondaryButton")
        nvidia_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        nvidia_btn.clicked.connect(lambda _=False: self.openNvidiaDriverRequested.emit())
        actions.addWidget(pytorch_btn)
        actions.addWidget(nvidia_btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        return frame

    def _build_maps_card(self) -> QFrame:
        frame = card("Map Services")
        layout = frame.layout()

        row = QHBoxLayout()
        row.setSpacing(12)
        label_col = QVBoxLayout()
        label_col.setSpacing(3)
        title = QLabel("Google Satellite basemap")
        title.setObjectName("panelTitle")
        body = QLabel("Enter your own Google Maps Platform API key to enable the Google Satellite basemap.")
        body.setObjectName("bodyText")
        body.setWordWrap(True)
        label_col.addWidget(title)
        label_col.addWidget(body)
        self.google_maps_state = StatusPill("Not set", "neutral")
        row.addLayout(label_col, 1)
        row.addWidget(self.google_maps_state, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)

        key_row = QHBoxLayout()
        key_row.setSpacing(8)
        self.google_maps_key_edit = QLineEdit()
        self.google_maps_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.google_maps_key_edit.setPlaceholderText("Paste Google Maps Platform API key")
        save_btn = QPushButton("Save Key")
        save_btn.setObjectName("secondaryButton")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(lambda _=False: self.googleMapsApiKeyChanged.emit(self.google_maps_key_edit.text().strip()))
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("secondaryButton")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(lambda _=False: self.googleMapsApiKeyChanged.emit(""))
        key_row.addWidget(self.google_maps_key_edit, 1)
        key_row.addWidget(save_btn)
        key_row.addWidget(clear_btn)
        layout.addLayout(key_row)

        hint = QLabel("The key is stored locally for this user and is not included in source control or installers.")
        hint.setObjectName("infoLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return frame

    def _build_storage_card(self) -> QFrame:
        frame = card("Storage & Synchronization")
        layout = frame.layout()
        self.database_row, self.database_value = meta_row("Database", "--")
        self.output_row, self.output_value = meta_row("Output Root", "--")
        self.cache_row, self.cache_value = meta_row("GeoTIFF Cache", "--")
        self.sync_row, self.sync_value = meta_row("Project Sync", "Watching current project outputs")
        layout.addWidget(self.database_row)
        layout.addWidget(self.output_row)
        layout.addWidget(self.cache_row)
        layout.addWidget(self.sync_row)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        output_btn = QPushButton("Open Output Root")
        output_btn.setObjectName("secondaryButton")
        output_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        output_btn.clicked.connect(lambda _=False: self.openOutputRootRequested.emit())
        db_btn = QPushButton("Open Database Folder")
        db_btn.setObjectName("secondaryButton")
        db_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        db_btn.clicked.connect(lambda _=False: self.openDatabaseFolderRequested.emit())
        cache_btn = QPushButton("Clear Project Cache")
        cache_btn.setObjectName("secondaryButton")
        cache_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cache_btn.clicked.connect(lambda _=False: self.clearCacheRequested.emit())
        actions.addWidget(output_btn)
        actions.addWidget(db_btn)
        actions.addWidget(cache_btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        return frame

    def set_hardware_status(self, status: HardwareStatus, preference: str) -> None:
        self._hardware = status
        if status.has_compatible_gpu:
            self.hardware_state.setText("Available")
            self.hardware_state.set_tone("ok")
        else:
            self.hardware_state.setText("Missing")
            self.hardware_state.set_tone("danger")
        self.hardware_title.setText(status.issue_title)
        self.hardware_detail.setText(status.issue_detail)

        cpu_detail = f"{status.cpu_name}\n{status.cpu_cores} cores / {status.cpu_threads} threads"
        if status.cpu_clock != "Unavailable":
            cpu_detail += f" | {status.cpu_clock}"
        self._set_status_row(self.cpu_value, "Available", "ok", cpu_detail)
        self._set_status_row(
            self.ram_value,
            "Available",
            "ok",
            f"{status.ram_total} total\n{status.ram_available} available",
        )
        gpu = status.gpus[0] if status.gpus else None
        if gpu and gpu.status == "Available":
            self._set_status_row(self.gpu_value, "Available", "ok", f"{gpu.name}\n{gpu.detail}")
        elif gpu and gpu.status == "Installed":
            detail = f"{gpu.name}\n{gpu.detail}; CUDA unavailable" if gpu.detail else f"{gpu.name}\nCUDA unavailable"
            self._set_status_row(self.gpu_value, "Installed", "neutral", detail)
        else:
            self._set_status_row(self.gpu_value, "Missing", "danger", "No CUDA-ready GPU detected")

        torch_tone = "ok" if status.torch_installed else "danger"
        torch_state = "Installed" if status.torch_installed else "Missing"
        self._set_status_row(self.torch_value, torch_state, torch_tone, status.torch_version)

        cuda_tone = "ok" if status.cuda_available else "danger"
        cuda_state = "Available" if status.cuda_available else "Missing"
        self._set_status_row(self.cuda_value, cuda_state, cuda_tone, status.cuda_version)
        self._set_status_row(
            self.device_value,
            "Selected",
            "ok" if preference == "gpu" else "neutral",
            status.device_label if preference == "gpu" and status.has_compatible_gpu else "CPU",
        )

        steps = status.setup_steps or ("No action needed. GPU acceleration is ready for AI mapping.",)
        self.setup_steps.setText("\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1)))
        self.hardware_progress.setValue(100)
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("Run Hardware Check")
        self.set_device_preference(preference, status)

    def begin_hardware_check(self) -> None:
        self.hardware_state.setText("Running")
        self.hardware_state.set_tone("neutral")
        self.hardware_title.setText("Running hardware diagnostics")
        self.hardware_detail.setText("Checking processor, memory, GPU, PyTorch, and CUDA availability.")
        self.hardware_progress.setValue(0)
        self.hardware_log.clear()
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("Checking...")
        for value in (
            self.cpu_value,
            self.ram_value,
            self.gpu_value,
            self.torch_value,
            self.cuda_value,
            self.device_value,
        ):
            self._set_status_row(value, "Checking", "neutral", "--")

    def update_hardware_check_progress(self, percent: int, message: str, level: str = "info") -> None:
        self.hardware_progress.setValue(max(0, min(100, percent)))
        prefix = {"ok": "OK", "warning": "WARN", "error": "ERROR"}.get(level, "INFO")
        self.hardware_log.append(f"[{prefix}] {message}")
        self.hardware_detail.setText(message)

    def fail_hardware_check(self, error_message: str) -> None:
        self.hardware_state.setText("Error")
        self.hardware_state.set_tone("danger")
        self.hardware_title.setText("Hardware check failed")
        self.hardware_detail.setText(error_message)
        self.hardware_progress.setValue(0)
        self.hardware_log.append(f"[ERROR] {error_message}")
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("Run Hardware Check")

    def set_device_preference(self, preference: str, status: HardwareStatus | None = None) -> None:
        status = status or self._hardware
        effective = preference
        if effective == "gpu" and (status is None or not status.has_compatible_gpu):
            effective = "cpu"

        self._syncing_device = True
        try:
            self.device_combo.clear()
            if status is not None and status.has_compatible_gpu:
                self.device_combo.addItem(status.device_label, "gpu")
            self.device_combo.addItem("CPU", "cpu")
            for index in range(self.device_combo.count()):
                if self.device_combo.itemData(index) == effective:
                    self.device_combo.setCurrentIndex(index)
                    break
        finally:
            self._syncing_device = False

    def set_storage_paths(self, database_path: str | Path, output_root: str | Path, cache_path: str | Path | None) -> None:
        self.database_value.setText(str(database_path))
        self.output_value.setText(str(output_root))
        self.cache_value.setText(str(cache_path) if cache_path else "Open a project to activate its GeoTIFF preview cache.")

    def set_sync_status(self, text: str) -> None:
        self.sync_value.setText(text)

    def set_google_maps_api_key_configured(self, configured: bool) -> None:
        self.google_maps_state.setText("Configured" if configured else "Not set")
        self.google_maps_state.set_tone("ok" if configured else "neutral")
        self.google_maps_key_edit.clear()
        self.google_maps_key_edit.setPlaceholderText(
            "Google Maps API key saved locally" if configured else "Paste Google Maps Platform API key"
        )

    def _on_device_changed(self, _index: int) -> None:
        if self._syncing_device:
            return
        value = self.device_combo.currentData()
        if value:
            self.devicePreferenceChanged.emit(str(value))

    @staticmethod
    def _add_hardware_tile(grid: QGridLayout, row: int, column: int, label: str, state: str, tone: str, detail: str) -> QLabel:
        tile = QFrame()
        tile.setObjectName("hardwareInfoCard")
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(10, 9, 10, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(8)
        key = QLabel(label.upper())
        key.setObjectName("hardwareLabel")
        pill = StatusPill(state, tone)
        value = QLabel(detail)
        value.setObjectName("hardwareValue")
        value.setWordWrap(True)
        header.addWidget(key)
        header.addStretch(1)
        header.addWidget(pill)
        layout.addLayout(header)
        layout.addWidget(value, 1)
        grid.addWidget(tile, row, column)
        value._status_pill = pill  # type: ignore[attr-defined]
        return value

    @staticmethod
    def _set_status_row(value_label: QLabel, state: str, tone: str, detail: str) -> None:
        pill = getattr(value_label, "_status_pill", None)
        if pill is not None:
            pill.setText(state)
            pill.set_tone(tone)
        value_label.setText(detail)
