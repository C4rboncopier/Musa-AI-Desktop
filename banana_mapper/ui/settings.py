from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
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
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        self.cpu_value = self._add_status_row(grid, 0, "CPU", "Checking", "neutral", "--")
        self.gpu_value = self._add_status_row(grid, 1, "GPU", "Checking", "neutral", "--")
        self.torch_value = self._add_status_row(grid, 2, "PyTorch", "Checking", "neutral", "--")
        self.cuda_value = self._add_status_row(grid, 3, "CUDA", "Checking", "neutral", "--")
        layout.addLayout(grid)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        refresh_btn = QPushButton("Run Hardware Check")
        refresh_btn.setObjectName("secondaryButton")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(lambda _=False: self.refreshHardwareRequested.emit())
        actions.addWidget(refresh_btn)
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

        self._set_status_row(self.cpu_value, "Available", "ok", f"{status.cpu_name} ({status.cpu_cores} cores)")
        gpu = status.gpus[0] if status.gpus else None
        if gpu and gpu.status == "Available":
            self._set_status_row(self.gpu_value, "Available", "ok", f"{gpu.name} - {gpu.detail}")
        elif gpu and gpu.status == "Installed":
            self._set_status_row(self.gpu_value, "Installed", "neutral", f"{gpu.name} - {gpu.detail}; CUDA unavailable")
        else:
            self._set_status_row(self.gpu_value, "Missing", "danger", "No CUDA-ready GPU detected")

        torch_tone = "ok" if status.torch_installed else "danger"
        torch_state = "Installed" if status.torch_installed else "Missing"
        self._set_status_row(self.torch_value, torch_state, torch_tone, status.torch_version)

        cuda_tone = "ok" if status.cuda_available else "danger"
        cuda_state = "Available" if status.cuda_available else "Missing"
        self._set_status_row(self.cuda_value, cuda_state, cuda_tone, status.cuda_version)

        steps = status.setup_steps or ("No action needed. GPU acceleration is ready for AI mapping.",)
        self.setup_steps.setText("\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1)))
        self.set_device_preference(preference, status)

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

    def _on_device_changed(self, _index: int) -> None:
        if self._syncing_device:
            return
        value = self.device_combo.currentData()
        if value:
            self.devicePreferenceChanged.emit(str(value))

    @staticmethod
    def _add_status_row(grid: QGridLayout, row: int, label: str, state: str, tone: str, detail: str) -> QLabel:
        key = QLabel(label.upper())
        key.setObjectName("metaKey")
        pill = StatusPill(state, tone)
        value = QLabel(detail)
        value.setObjectName("metaValue")
        value.setWordWrap(True)
        grid.addWidget(key, row, 0)
        grid.addWidget(pill, row, 1)
        grid.addWidget(value, row, 2)
        value._status_pill = pill  # type: ignore[attr-defined]
        return value

    @staticmethod
    def _set_status_row(value_label: QLabel, state: str, tone: str, detail: str) -> None:
        pill = getattr(value_label, "_status_pill", None)
        if pill is not None:
            pill.setText(state)
            pill.set_tone(tone)
        value_label.setText(detail)
