from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from ai_model_comparator.ui.main_window import MainWindow
from ai_model_comparator.ui.theme import apply_theme


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Musa AI Model Comparison")
    app.setOrganizationName("Musa AI")
    apply_theme(app)

    window = MainWindow()
    window.resize(1440, 900)
    window.show()
    return app.exec()

