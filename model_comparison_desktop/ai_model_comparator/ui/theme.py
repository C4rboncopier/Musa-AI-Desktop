from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(
        """
        QMainWindow {
            background: #0b1120;
        }

        QWidget {
            color: #e5edf4;
            background: transparent;
        }

        QToolBar#mainToolbar {
            min-height: 48px;
            spacing: 8px;
            padding: 7px 12px;
            background: #111827;
            border: 0;
            border-bottom: 1px solid #1f2a3a;
        }

        QToolBar#mainToolbar QToolButton {
            min-height: 31px;
            padding: 4px 11px;
            color: #d8e2ec;
            background: #182234;
            border: 1px solid #273449;
            border-radius: 7px;
            font-weight: 700;
        }

        QToolBar#mainToolbar QToolButton:hover {
            background: #203047;
            border-color: #38bdf8;
        }

        QLabel#toolbarMetric,
        QLabel#headerSummary {
            padding: 7px 11px;
            color: #bae6fd;
            background: #0f2537;
            border: 1px solid #164e63;
            border-radius: 7px;
            font-weight: 800;
        }

        QWidget#pageHeader {
            background: #111827;
            border: 1px solid #243044;
            border-radius: 8px;
        }

        QLabel#pageTitle {
            color: #f8fafc;
            font-size: 22px;
            font-weight: 800;
        }

        QLabel#pageSubtitle,
        QLabel#footerHint,
        QLabel#mutedText {
            color: #94a3b8;
        }

        QGraphicsView {
            background: #050816;
            border: 1px solid #243044;
            border-radius: 8px;
        }

        QScrollArea {
            background: transparent;
            border: 0;
        }

        QFrame#modelCard,
        QFrame#settingsCard {
            background: #111827;
            border: 1px solid #243044;
            border-radius: 8px;
        }

        QFrame#modelCard[selected="true"] {
            border: 2px solid #38bdf8;
            background: #102033;
        }

        QLabel#panelTitle {
            color: #f8fafc;
            font-size: 16px;
            font-weight: 800;
        }

        QRadioButton,
        QCheckBox {
            color: #e5edf4;
            font-weight: 700;
        }

        QLineEdit {
            min-height: 34px;
            padding: 0 10px;
            color: #f8fafc;
            background: #0b1120;
            border: 1px solid #273449;
            border-radius: 7px;
            selection-background-color: #0ea5e9;
        }

        QComboBox#mappingModeCombo {
            min-height: 31px;
            padding: 0 10px;
            color: #d8e2ec;
            background: #182234;
            border: 1px solid #273449;
            border-radius: 7px;
            font-weight: 700;
        }

        QComboBox#mappingModeCombo:hover {
            border-color: #38bdf8;
        }

        QLineEdit:focus {
            border-color: #38bdf8;
        }

        QPushButton#primaryButton,
        QPushButton#secondaryButton {
            min-height: 34px;
            padding: 0 12px;
            border-radius: 7px;
            font-weight: 800;
        }

        QPushButton#primaryButton {
            color: #031018;
            background: #38bdf8;
            border: 1px solid #38bdf8;
        }

        QPushButton#primaryButton:hover {
            background: #7dd3fc;
        }

        QPushButton#secondaryButton {
            color: #d8e2ec;
            background: #182234;
            border: 1px solid #273449;
        }

        QPushButton#secondaryButton:hover {
            border-color: #38bdf8;
        }

        QLabel#metricLabel,
        QLabel#detailKey {
            color: #94a3b8;
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
        }

        QLabel#metricValue {
            color: #f8fafc;
            font-size: 21px;
            font-weight: 800;
        }

        QLabel#classChip {
            padding: 4px 8px;
            color: #e5edf4;
            background: #0b1120;
            border: 1px solid #273449;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 800;
        }

        QTabWidget::pane {
            background: #111827;
            border: 1px solid #243044;
            border-radius: 8px;
        }

        QTabBar::tab {
            padding: 7px 12px;
            margin-right: 4px;
            color: #94a3b8;
            background: #182234;
            border: 1px solid #273449;
            border-bottom: 0;
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
            font-weight: 700;
        }

        QTabBar::tab:selected {
            color: #f8fafc;
            background: #111827;
            border-color: #38bdf8;
        }

        QLabel#detailValue {
            color: #e5edf4;
            font-weight: 700;
        }

        QLabel#monoText {
            color: #d8e2ec;
            font-family: Consolas, "Cascadia Mono", monospace;
            font-size: 12px;
        }

        QLabel#emptyState {
            min-height: 160px;
            color: #94a3b8;
            background: #0b1120;
            border: 1px dashed #334155;
            border-radius: 8px;
            font-weight: 700;
        }

        QSplitter::handle {
            background: #0b1120;
            width: 10px;
        }

        QProgressBar#pipelineProgress {
            min-height: 24px;
            color: #e0f2fe;
            background: #0b1120;
            border: 1px solid #273449;
            border-radius: 7px;
            text-align: center;
            font-weight: 800;
        }

        QProgressBar#pipelineProgress::chunk {
            background: #38bdf8;
            border-radius: 6px;
        }
        """
    )
