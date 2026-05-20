from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


ASSET_LABELS = {
    "geotiff": "GeoTIFF",
    "output_dir": "Output Folder",
}

def section_label(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("sectionHeader")
    return label


def card(title: str = "") -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 14)
    layout.setSpacing(8)
    if title:
        heading = QLabel(title.upper())
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
    return frame


def separator() -> QFrame:
    sep = QFrame()
    sep.setObjectName("separator")
    sep.setFrameShape(QFrame.Shape.HLine)
    return sep


def meta_row(key: str, value: str = "--") -> tuple[QWidget, QLabel]:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    key_label = QLabel(key.upper())
    key_label.setObjectName("metaKey")
    value_label = QLabel(value)
    value_label.setObjectName("metaValue")
    value_label.setWordWrap(True)
    value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(key_label, 0)
    layout.addWidget(value_label, 1)
    return row, value_label


class StatusPill(QLabel):
    def __init__(self, text: str = "Ready", tone: str = "neutral", parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("statusPill")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        self.setProperty("tone", tone)
        self.style().unpolish(self)
        self.style().polish(self)


class AssetRow(QFrame):
    def __init__(
        self,
        title: str,
        path: str,
        exists: bool,
        subtitle: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("assetRow")
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)

        state = StatusPill("Linked" if exists else "Missing", "ok" if exists else "danger")
        name = QLabel(title)
        name.setObjectName("assetTitle")
        name.setWordWrap(True)
        path_label = QLabel(_compact_path(path))
        path_label.setObjectName("assetPath")
        path_label.setToolTip(path)
        path_label.setWordWrap(True)
        detail = QLabel(subtitle)
        detail.setObjectName("infoLabel")
        detail.setVisible(bool(subtitle))

        layout.addWidget(name, 0, 0)
        layout.addWidget(state, 0, 1, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(path_label, 1, 0, 1, 2)
        layout.addWidget(detail, 2, 0, 1, 2)


class EmptyState(QWidget):
    def __init__(self, title: str, body: str, button_text: str = "", parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 28, 18, 28)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setObjectName("emptyTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body_label = QLabel(body)
        body_label.setObjectName("bodyText")
        body_label.setWordWrap(True)
        body_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        layout.addWidget(title_label)
        layout.addWidget(body_label)
        if button_text:
            self.button = QPushButton(button_text)
            self.button.setObjectName("primaryButton")
            self.button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            layout.addWidget(self.button, 0, Qt.AlignmentFlag.AlignCenter)
        else:
            self.button = None
        layout.addStretch(1)


class MetricCard(QFrame):
    def __init__(self, label: str, value: str = "--", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        title = QLabel(label.upper())
        title.setObjectName("metaKey")
        layout.addWidget(self.value_label)
        layout.addWidget(title)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


def _compact_path(path: str, max_chars: int = 86) -> str:
    if not path:
        return "--"
    text = str(path)
    if len(text) <= max_chars:
        return text
    p = Path(text)
    return f"...{p.parent.name}\\{p.name}" if p.parent.name else f"...{p.name}"
