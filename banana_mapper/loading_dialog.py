"""Loading dialog shown while a GeoTIFF is being processed.

A theme-aware, modal progress popup with a determinate progress bar,
a stage message label, an animated pulse strip, and a Cancel button.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from .themes import THEMES, ThemeColors


class LoadingDialog(QDialog):
    """Modal progress dialog displayed during GeoTIFF import.

    Call ``set_progress(percent, message)`` from the main thread
    (connected to the worker's ``progress`` signal) to update the UI.
    """

    cancel_requested = pyqtSignal()

    def __init__(
        self,
        file_name: str,
        theme_name: str,
        parent=None,
        title: str = "Importing GeoTIFF",
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(480, 220)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )

        self._cancelled = False
        self._theme_name = theme_name
        self._build_ui(file_name, title)
        self._apply_theme(theme_name)

        # Pulse animation timer (updates a dot counter in the subtitle)
        self._dot_count = 0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(400)
        self._pulse_timer.timeout.connect(self._pulse_dots)
        self._pulse_timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, file_name: str, title_text: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(0)

        # --- Title ---
        title = QLabel(title_text)
        title.setObjectName("dlgTitle")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        root.addWidget(title)

        root.addSpacing(4)

        # --- File name ---
        self._file_label = QLabel(file_name)
        self._file_label.setObjectName("dlgFile")
        self._file_label.setWordWrap(True)
        self._file_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        root.addWidget(self._file_label)

        root.addSpacing(18)

        # --- Progress bar ---
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        root.addWidget(self._bar)

        root.addSpacing(10)

        # --- Stage message ---
        self._stage_label = QLabel("Initializing")
        self._stage_label.setObjectName("dlgStage")
        self._stage_label.setWordWrap(True)
        root.addWidget(self._stage_label)

        root.addSpacing(4)

        # --- Percent indicator ---
        self._pct_label = QLabel("0%")
        self._pct_label.setObjectName("dlgPct")
        self._pct_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        root.addWidget(self._pct_label)

        root.addSpacing(16)

        # --- Separator ---
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("dlgSep")
        root.addWidget(sep)

        root.addSpacing(12)

        # --- Cancel button ---
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("dlgCancel")
        self._cancel_btn.setFixedWidth(90)
        self._cancel_btn.setFixedHeight(34)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Progress API (called from main thread via signal)
    # ------------------------------------------------------------------

    def set_progress(self, percent: int, message: str) -> None:
        """Update the progress bar and stage message."""
        self._bar.setValue(percent)
        self._pct_label.setText(f"{percent}%")
        # Strip trailing ellipsis for the stage label; the dot animation adds its own
        self._stage_label.setText(message.rstrip(".").rstrip())
        self._dot_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pulse_dots(self) -> None:
        """Append animated dots to the stage message."""
        self._dot_count = (self._dot_count + 1) % 4
        dots = "." * self._dot_count
        base = self._stage_label.text().rstrip(".")
        self._stage_label.setText(base + dots)

    def _on_cancel(self) -> None:
        self._cancelled = True
        self._pulse_timer.stop()
        self.cancel_requested.emit()
        self.reject()

    @property
    def was_cancelled(self) -> bool:
        return self._cancelled

    def finish(self) -> None:
        """Call from the main thread when the worker finishes successfully."""
        self._pulse_timer.stop()
        self._bar.setValue(100)
        self._pct_label.setText("100%")
        self._stage_label.setText("Done")
        self.accept()

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def _apply_theme(self, theme_name: str) -> None:
        theme = THEMES.get(theme_name, next(iter(THEMES.values())))
        self.setStyleSheet(self._build_stylesheet(theme))

    @staticmethod
    def _build_stylesheet(t: ThemeColors) -> str:
        return f"""
            QDialog {{
                background: {t.sidebar_bg};
                border: 1px solid {t.card_border};
                border-radius: 10px;
            }}
            #dlgTitle {{
                color: {t.title_fg};
                font-size: 14px;
                font-weight: 700;
            }}
            #dlgFile {{
                color: {t.meta_key_fg};
                font-size: 11px;
                font-weight: 600;
            }}
            QProgressBar {{
                background: {t.slider_groove};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {t.primary_bg},
                    stop:1 {t.primary_hover_bg}
                );
                border-radius: 4px;
            }}
            #dlgStage {{
                color: {t.body_fg};
                font-size: 11px;
            }}
            #dlgPct {{
                color: {t.meta_key_fg};
                font-size: 12px;
                font-weight: 700;
            }}
            #dlgSep {{
                background: {t.separator_color};
                max-height: 1px;
                min-height: 1px;
                border: none;
            }}
            #dlgCancel {{
                background: {t.secondary_bg};
                color: {t.secondary_fg};
                border: 1px solid {t.secondary_border};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }}
            #dlgCancel:hover {{
                background: {t.secondary_hover_bg};
                border-color: {t.secondary_hover_border};
            }}
        """
