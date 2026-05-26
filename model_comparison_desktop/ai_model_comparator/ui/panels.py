from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ai_model_comparator.data.models import DetectionRecord, ModelResult
from ai_model_comparator.data.settings_store import AppConfig


class ConfigurationPanel(QWidget):
    configSaved = pyqtSignal(str, str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Model Inputs")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        body = QFrame()
        body.setObjectName("settingsCard")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 12, 14, 12)
        body_layout.setSpacing(10)

        key_label = QLabel("Gemini API key")
        key_label.setObjectName("detailKey")
        self.gemini_key_edit = QLineEdit()
        self.gemini_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key_edit.setPlaceholderText("Paste Gemini API key")
        self.key_status = QLabel("No Gemini key saved")
        self.key_status.setObjectName("mutedText")
        self._clear_key_requested = False
        body_layout.addWidget(key_label)
        body_layout.addWidget(self.gemini_key_edit)
        body_layout.addWidget(self.key_status)

        self.leaf_model_edit = self._path_row(
            body_layout,
            "YOLOv8-seg leaf model",
            "Select full_leaf / cut_leaf segmentation model",
        )
        self.disease_model_edit = self._path_row(
            body_layout,
            "YOLOv8 disease model",
            "Select black_sigatoka / panama detection model",
        )

        note = QLabel(
            "Only file paths and a local Gemini secret file are stored. Detection execution can be added later without changing the viewer."
        )
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        body_layout.addWidget(note)

        actions = QHBoxLayout()
        actions.addStretch(1)
        clear_key = QPushButton("Clear Key")
        clear_key.setObjectName("secondaryButton")
        clear_key.clicked.connect(self._clear_key)
        save = QPushButton("Save Inputs")
        save.setObjectName("primaryButton")
        save.clicked.connect(self._emit_save)
        actions.addWidget(clear_key)
        actions.addWidget(save)
        body_layout.addLayout(actions)

        layout.addWidget(body)

    def set_config(self, config: AppConfig) -> None:
        self.gemini_key_edit.setText("")
        self._clear_key_requested = False
        self.leaf_model_edit.setText(config.leaf_model_path)
        self.disease_model_edit.setText(config.disease_model_path)
        status = "Gemini key saved locally" if config.gemini_api_key_configured else "No Gemini key saved"
        self.key_status.setText(status)

    def _path_row(self, parent_layout: QVBoxLayout, label_text: str, placeholder: str) -> QLineEdit:
        label = QLabel(label_text)
        label.setObjectName("detailKey")
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        browse = QPushButton("Browse")
        browse.setObjectName("secondaryButton")
        browse.clicked.connect(lambda _=False, target=edit: self._select_model_path(target))

        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        parent_layout.addWidget(label)
        parent_layout.addLayout(row)
        return edit

    def _select_model_path(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select trained model",
            target.text().strip(),
            "Model files (*.pt *.onnx);;All files (*.*)",
        )
        if path:
            target.setText(path)

    def _clear_key(self) -> None:
        self._clear_key_requested = True
        self.gemini_key_edit.setText("")
        self.key_status.setText("Gemini key will be cleared on save")

    def _emit_save(self) -> None:
        missing = []
        for label, value in [
            ("YOLOv8-seg leaf model", self.leaf_model_edit.text().strip()),
            ("YOLOv8 disease model", self.disease_model_edit.text().strip()),
        ]:
            if value and not Path(value).exists():
                missing.append(label)
        if missing:
            QMessageBox.warning(
                self,
                "Model file not found",
                "These selected model paths do not exist:\n" + "\n".join(missing),
            )
            return
        api_key = "__CLEAR_GEMINI_KEY__" if self._clear_key_requested else self.gemini_key_edit.text().strip()
        self.configSaved.emit(
            api_key,
            self.leaf_model_edit.text().strip(),
            self.disease_model_edit.text().strip(),
        )


class ModelCard(QFrame):
    visibilityChanged = pyqtSignal(str, bool)
    selected = pyqtSignal(str)

    def __init__(self, model: ModelResult, checked: bool, selected: bool, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.setObjectName("modelCard")
        self.setProperty("selected", selected)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.radio = QRadioButton(model.name)
        self.radio.setChecked(selected)
        self.radio.toggled.connect(lambda active: active and self.selected.emit(model.id))
        top.addWidget(self.radio, 1)

        self.toggle = QCheckBox("Visible")
        self.toggle.setChecked(checked)
        self.toggle.toggled.connect(lambda value: self.visibilityChanged.emit(model.id, value))
        top.addWidget(self.toggle)
        layout.addLayout(top)

        meta = QLabel(f"{model.version} | {model.description}")
        meta.setObjectName("mutedText")
        meta.setWordWrap(True)
        layout.addWidget(meta)

        stats = QGridLayout()
        stats.setHorizontalSpacing(14)
        stats.setVerticalSpacing(6)
        self._add_stat(stats, 0, "Detections", str(model.count))
        self._add_stat(stats, 1, "Avg. confidence", f"{model.average_confidence:.1%}")
        layout.addLayout(stats)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        class_counts = model.counts_by_class()
        if class_counts:
            for class_name, count in class_counts.items():
                chip = QLabel(f"{class_name}: {count}")
                chip.setObjectName("classChip")
                chip.setStyleSheet(f"border-color: {QColor(model.color).name()};")
                chips.addWidget(chip)
        else:
            empty = QLabel("No detections loaded")
            empty.setObjectName("mutedText")
            chips.addWidget(empty)
        chips.addStretch(1)
        layout.addLayout(chips)

    def _add_stat(self, layout: QGridLayout, column: int, label: str, value: str) -> None:
        wrapper = QVBoxLayout()
        key = QLabel(label)
        key.setObjectName("metricLabel")
        val = QLabel(value)
        val.setObjectName("metricValue")
        wrapper.addWidget(key)
        wrapper.addWidget(val)
        layout.addLayout(wrapper, 0, column)


class ModelComparisonPanel(QWidget):
    visibilityChanged = pyqtSignal(str, bool)
    selectedModelChanged = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._models: list[ModelResult] = []
        self._visible_ids: set[str] = set()
        self._selected_id = ""
        self._button_group = QButtonGroup(self)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)

        title = QLabel("Model Comparison")
        title.setObjectName("panelTitle")
        self.layout.addWidget(title)

        self.cards_container = QVBoxLayout()
        self.cards_container.setSpacing(10)
        self.layout.addLayout(self.cards_container)
        self.layout.addStretch(1)

    def set_models(self, models: list[ModelResult], visible_ids: set[str], selected_id: str) -> None:
        self._models = models
        self._visible_ids = visible_ids
        self._selected_id = selected_id
        self._rebuild_cards()

    def _rebuild_cards(self) -> None:
        while self.cards_container.count():
            item = self.cards_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self._button_group = QButtonGroup(self)
        for model in self._models:
            card = ModelCard(model, model.id in self._visible_ids, model.id == self._selected_id)
            card.visibilityChanged.connect(self.visibilityChanged.emit)
            card.selected.connect(self.selectedModelChanged.emit)
            self._button_group.addButton(card.radio)
            self.cards_container.addWidget(card)


class DetectionDetailsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("Selected Detection")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.tabs = QTabWidget()
        self.summary = QWidget()
        self.raw = QWidget()
        self.tabs.addTab(self.summary, "Summary")
        self.tabs.addTab(self.raw, "Metadata")
        layout.addWidget(self.tabs, 1)

        self.summary_layout = QGridLayout(self.summary)
        self.summary_layout.setContentsMargins(12, 12, 12, 12)
        self.summary_layout.setVerticalSpacing(9)
        self.summary_layout.setHorizontalSpacing(12)

        self.raw_layout = QVBoxLayout(self.raw)
        self.raw_layout.setContentsMargins(12, 12, 12, 12)
        self.raw_label = QLabel("Click a marker to inspect its data.")
        self.raw_label.setObjectName("monoText")
        self.raw_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.raw_label.setWordWrap(True)
        self.raw_layout.addWidget(self.raw_label)

        self.set_empty()

    def set_empty(self) -> None:
        self._clear_summary()
        empty = QLabel("No marker selected")
        empty.setObjectName("emptyState")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_layout.addWidget(empty, 0, 0, 1, 2)
        self.raw_label.setText("Click a marker to inspect its data.")

    def set_detection(self, model: ModelResult, record: DetectionRecord) -> None:
        self._clear_summary()
        rows = [
            ("Model", model.name),
            ("Model version", model.version),
            ("Detection ID", record.id),
            ("Disease / class", record.class_name),
            ("Health", record.health or "-"),
            ("Confidence", f"{record.confidence:.2%}"),
            ("Latitude", f"{record.latitude:.7f}"),
            ("Longitude", f"{record.longitude:.7f}"),
            ("Pixel", f"{record.pixel_x:.1f}, {record.pixel_y:.1f}"),
            ("Source", record.source),
            ("Duplicates", str(record.duplicate_count)),
            ("Related leaf", record.related_leaf_id or "-"),
            ("Layers", ", ".join(record.layer_keys)),
        ]
        for row, (key, value) in enumerate(rows):
            key_label = QLabel(key)
            key_label.setObjectName("detailKey")
            value_label = QLabel(value)
            value_label.setObjectName("detailValue")
            value_label.setWordWrap(True)
            self.summary_layout.addWidget(key_label, row, 0)
            self.summary_layout.addWidget(value_label, row, 1)

        raw_payload = {
            "model": {"id": model.id, "name": model.name, "version": model.version},
            "record": {
                "id": record.id,
                "image_name": record.image_name,
                "class_name": record.class_name,
                "latitude": record.latitude,
                "longitude": record.longitude,
                "confidence": record.confidence,
                "pixel_x": record.pixel_x,
                "pixel_y": record.pixel_y,
                "health": record.health,
                "source": record.source,
                "duplicate_count": record.duplicate_count,
                "related_leaf_id": record.related_leaf_id,
                "layer_keys": record.layer_keys,
                "metadata": record.metadata,
            },
        }
        self.raw_label.setText(json.dumps(raw_payload, indent=2))

    def _clear_summary(self) -> None:
        while self.summary_layout.count():
            item = self.summary_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()


class SidePanel(QWidget):
    visibilityChanged = pyqtSignal(str, bool)
    selectedModelChanged = pyqtSignal(str)
    configSaved = pyqtSignal(str, str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(380)
        self.setMaximumWidth(500)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self.config_panel = ConfigurationPanel()
        self.model_panel = ModelComparisonPanel()
        self.details_panel = DetectionDetailsPanel()
        self.config_panel.configSaved.connect(self.configSaved.emit)
        self.model_panel.visibilityChanged.connect(self.visibilityChanged.emit)
        self.model_panel.selectedModelChanged.connect(self.selectedModelChanged.emit)

        layout.addWidget(self.config_panel)
        layout.addWidget(self.model_panel)
        layout.addWidget(self.details_panel)
        layout.addStretch(1)
        root.addWidget(scroll)

    def set_models(self, models: list[ModelResult], visible_ids: set[str], selected_id: str) -> None:
        self.model_panel.set_models(models, visible_ids, selected_id)

    def set_config(self, config: AppConfig) -> None:
        self.config_panel.set_config(config)

    def set_detection(self, model: ModelResult, record: DetectionRecord) -> None:
        self.details_panel.set_detection(model, record)
