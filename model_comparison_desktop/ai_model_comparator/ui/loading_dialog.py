from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QDialog, QFrame, QGridLayout, QLabel, QProgressBar, QVBoxLayout


class MappingLoadingDialog(QDialog):
    def __init__(self, image_name: str, mapping_label: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Running AI Mapping")
        self.setModal(True)
        self.setFixedSize(540, 270)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )
        self._dot_count = 0
        self._base_stage = "Initializing"

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(12)

        title = QLabel("Running AI Mapping")
        title.setObjectName("loaderTitle")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        root.addWidget(title)

        image = QLabel(image_name)
        image.setObjectName("loaderImage")
        image.setWordWrap(True)
        root.addWidget(image)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedHeight(9)
        self.bar.setTextVisible(False)
        root.addWidget(self.bar)

        self.stage = QLabel("Initializing")
        self.stage.setObjectName("loaderStage")
        self.stage.setWordWrap(True)
        root.addWidget(self.stage)

        self.percent = QLabel("0%")
        self.percent.setObjectName("loaderPercent")
        self.percent.setAlignment(Qt.AlignmentFlag.AlignRight)
        root.addWidget(self.percent)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("loaderSeparator")
        root.addWidget(separator)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.current_tile = self._add_provider_row(grid, 0, mapping_label)
        root.addLayout(grid)

        self.timer = QTimer(self)
        self.timer.setInterval(420)
        self.timer.timeout.connect(self._pulse)
        self.timer.start()

        self.setStyleSheet(
            """
            QDialog {
                background: #111827;
                border: 1px solid #243044;
                border-radius: 10px;
            }
            QLabel#loaderTitle {
                color: #f8fafc;
                font-weight: 800;
            }
            QLabel#loaderImage,
            QLabel#loaderStage {
                color: #cbd5e1;
            }
            QLabel#loaderPercent,
            QLabel#loaderProviderValue {
                color: #bae6fd;
                font-weight: 800;
            }
            QLabel#loaderProviderKey {
                color: #94a3b8;
                font-size: 11px;
                font-weight: 800;
                text-transform: uppercase;
            }
            QProgressBar {
                background: #0b1120;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #22c55e);
                border-radius: 4px;
            }
            #loaderSeparator {
                background: #243044;
                max-height: 1px;
                min-height: 1px;
                border: none;
            }
            """
        )

    def set_progress(self, percent: int, message: str) -> None:
        self.bar.setValue(percent)
        self.percent.setText(f"{percent}%")
        self._base_stage = message.rstrip(".")
        self.stage.setText(self._base_stage)
        self._dot_count = 0

    def set_provider_tile(self, provider_id: str, tile_name: str, message: str) -> None:
        text = f"{tile_name} | {message.split(':', 1)[-1].strip()}"
        self.current_tile.setText(text)

    def finish(self) -> None:
        self.timer.stop()
        self.bar.setValue(100)
        self.percent.setText("100%")
        self.stage.setText("Done")
        self.accept()

    def fail(self, message: str) -> None:
        self.timer.stop()
        self.stage.setText(message)
        self.reject()

    def _add_provider_row(self, grid: QGridLayout, row: int, label: str) -> QLabel:
        key = QLabel(label)
        key.setObjectName("loaderProviderKey")
        value = QLabel("Waiting")
        value.setObjectName("loaderProviderValue")
        value.setWordWrap(True)
        grid.addWidget(key, row, 0)
        grid.addWidget(value, row, 1)
        return value

    def _pulse(self) -> None:
        self._dot_count = (self._dot_count + 1) % 4
        self.stage.setText(self._base_stage + ("." * self._dot_count))
