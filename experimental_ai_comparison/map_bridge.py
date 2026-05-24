from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class ExperimentMapBridge(QObject):
    mapReady = pyqtSignal()

    @pyqtSlot()
    def reportMapReady(self) -> None:
        self.mapReady.emit()

