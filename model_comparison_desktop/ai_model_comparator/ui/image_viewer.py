from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsObject, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ai_model_comparator.data.models import DetectionRecord, ModelResult


DISEASE_COLORS = {
    "black_sigatoka": "#facc15",
    "panama": "#ef4444",
    "healthy_leaf": "#22c55e",
    "diseased_leaf": "#f97316",
}


@dataclass(slots=True)
class MarkerPayload:
    model: ModelResult
    record: DetectionRecord


class DetectionMarkerItem(QGraphicsObject):
    selected = pyqtSignal(object)

    def __init__(self, payload: MarkerPayload, parent=None) -> None:
        super().__init__(parent)
        self.payload = payload
        self.radius = 8.0
        self.setPos(QPointF(payload.record.pixel_x, payload.record.pixel_y))
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setToolTip(
            f"{payload.model.name}\n"
            f"{payload.record.class_name}\n"
            f"Confidence: {payload.record.confidence:.2%}"
        )

    def boundingRect(self) -> QRectF:
        pad = 4.0
        size = (self.radius + pad) * 2
        return QRectF(-self.radius - pad, -self.radius - pad, size, size)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        disease_color = QColor(DISEASE_COLORS.get(self.payload.record.class_name, "#38bdf8"))
        model_color = QColor(self.payload.model.color)

        painter.setPen(QPen(QColor(255, 255, 255, 220), 5))
        painter.setBrush(disease_color)
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

        painter.setPen(QPen(model_color, 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(0, 0), self.radius + 4, self.radius + 4)

    def mousePressEvent(self, event) -> None:
        self.selected.emit(self.payload)
        event.accept()

    def hoverEnterEvent(self, event) -> None:
        self.radius = 11.0
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.radius = 8.0
        self.update()
        super().hoverLeaveEvent(event)


class ImageViewer(QGraphicsView):
    markerSelected = pyqtSignal(object)
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._marker_items: list[DetectionMarkerItem] = []
        self._zoom = 1.0

        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#111827"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    def clear(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._marker_items.clear()
        self.resetTransform()
        self._zoom = 1.0
        self.zoomChanged.emit(self._zoom)

    def load_image(self, image_path: str) -> bool:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return False

        self._scene.clear()
        self._marker_items.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.resetTransform()
        self._zoom = 1.0
        self.fit_to_view()
        return True

    def set_markers(self, models: list[ModelResult], visible_model_ids: set[str]) -> None:
        for item in self._marker_items:
            self._scene.removeItem(item)
        self._marker_items.clear()

        for model in models:
            if model.id not in visible_model_ids:
                continue
            for record in model.records:
                marker = DetectionMarkerItem(MarkerPayload(model=model, record=record))
                marker.selected.connect(self.markerSelected.emit)
                marker.setZValue(10)
                self._scene.addItem(marker)
                self._marker_items.append(marker)

    def fit_to_view(self) -> None:
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self.zoomChanged.emit(self._zoom)

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        self.zoomChanged.emit(self._zoom)

    def zoom_in(self) -> None:
        self._apply_zoom(1.18)

    def zoom_out(self) -> None:
        self._apply_zoom(1 / 1.18)

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def _apply_zoom(self, factor: float) -> None:
        next_zoom = self._zoom * factor
        if not 0.05 <= next_zoom <= 16:
            return
        self.scale(factor, factor)
        self._zoom = next_zoom
        self.zoomChanged.emit(self._zoom)
