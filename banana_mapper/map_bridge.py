from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class MapBridge(QObject):
    coordinatesChanged = pyqtSignal(float, float)
    zoomChanged = pyqtSignal(int)
    mapReady = pyqtSignal()

    @pyqtSlot()
    def reportMapReady(self) -> None:
        self.mapReady.emit()

    @pyqtSlot(float, float)
    def reportCoordinates(self, latitude: float, longitude: float) -> None:
        self.coordinatesChanged.emit(latitude, longitude)

    @pyqtSlot(int)
    def reportZoom(self, zoom: int) -> None:
        self.zoomChanged.emit(zoom)

    scanBox = pyqtSignal(float, float, float, float)

    @pyqtSlot(float, float, float, float)
    def reportScanBox(self, lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> None:
        self.scanBox.emit(lat_min, lon_min, lat_max, lon_max)
