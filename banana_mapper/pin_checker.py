from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImageReader, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QLabel, QMessageBox, QVBoxLayout
)

from .detection import extract_drone_metadata, pixel_to_lat_lon


class ImageClickLabel(QLabel):
    clicked_coord = pyqtSignal(float, float)  # original_x, original_y

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_pixmap = None
        self.original_size = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(1, 1)
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_image(self, pixmap: QPixmap, orig_size: tuple[int, int]) -> None:
        self.original_pixmap = pixmap
        self.original_size = orig_size
        self.update_scaled_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.original_pixmap:
            self.update_scaled_pixmap()

    def update_scaled_pixmap(self) -> None:
        if not self.original_pixmap:
            return
        scaled = self.original_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def mousePressEvent(self, event) -> None:
        if not self.original_pixmap or not self.original_size:
            return

        pixmap = self.pixmap()
        if not pixmap:
            return

        lbl_w = self.width()
        lbl_h = self.height()
        pix_w = pixmap.width()
        pix_h = pixmap.height()

        offset_x = (lbl_w - pix_w) / 2
        offset_y = (lbl_h - pix_h) / 2

        click_x = event.pos().x() - offset_x
        click_y = event.pos().y() - offset_y

        if 0 <= click_x <= pix_w and 0 <= click_y <= pix_h:
            scale_x = self.original_size[0] / pix_w
            scale_y = self.original_size[1] / pix_h
            orig_x = click_x * scale_x
            orig_y = click_y * scale_y
            self.clicked_coord.emit(orig_x, orig_y)


class PinCheckerDialog(QDialog):
    pin_requested = pyqtSignal(float, float, str)  # lat, lon, filename

    def __init__(self, image_path: str, mrk_path: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.image_path = Path(image_path)
        self.mrk_path = Path(mrk_path) if mrk_path else None
        self.setWindowTitle(f"Pin Location Checker - {self.image_path.name}")
        self.resize(1000, 700)

        self.metadata = extract_drone_metadata(self.image_path, self.mrk_path)
        if not self.metadata:
            QMessageBox.warning(
                self,
                "Metadata Missing",
                "Could not extract valid GPS or camera metadata from this image.",
            )
            # Cannot reject() in __init__ safely and have it stop execution of caller,
            # so we let it render but it won't be usable, or caller handles it.
            # Usually we use a static method or check before creating.
            
        self._build_ui()
        self._load_image()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel(
            "Click anywhere on the image to pin its location on the GeoTIFF map."
        )
        header.setObjectName("bodyText")
        header.setWordWrap(True)
        layout.addWidget(header)

        if getattr(self.metadata, "used_rtk", False):
            source = self.mrk_path.name if self.mrk_path else "nearby .MRK"
            rtk_status = f"Active ({source})"
        elif self.mrk_path:
            rtk_status = f"Linked but no matching image record ({self.mrk_path.name})"
        else:
            rtk_status = "Inactive (using EXIF)"
        status_label = QLabel(f"RTK Precision: {rtk_status}")
        status_label.setStyleSheet(
            "color: #10b981; font-weight: bold;"
            if getattr(self.metadata, "used_rtk", False)
            else "color: #ef4444; font-weight: bold;"
        )
        layout.addWidget(status_label)

        checker = QLabel(self._mrk_checker_text())
        checker.setObjectName("bodyText")
        checker.setWordWrap(True)
        checker.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        checker.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #334155; "
            "border-radius: 4px; color: #cbd5e1; padding: 8px;"
        )
        layout.addWidget(checker)
        self.mrk_checker_label = checker

        self.click_status_label = QLabel("Click status: waiting for image click")
        self.click_status_label.setObjectName("bodyText")
        self.click_status_label.setWordWrap(True)
        self.click_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.click_status_label)

        self.image_label = ImageClickLabel()
        self.image_label.setStyleSheet(
            "background-color: #020617; border: 1px solid #334155; border-radius: 4px;"
        )
        self.image_label.clicked_coord.connect(self._on_image_clicked)
        layout.addWidget(self.image_label, 1)

    def _load_image(self) -> None:
        reader = QImageReader(str(self.image_path))
        reader.setAutoTransform(True)
        orig_size = reader.size()

        # Scale down if too large to fit in memory easily (e.g., 50MP drone image)
        target_size = orig_size
        if orig_size.width() > 2048 or orig_size.height() > 2048:
            target_size = orig_size.scaled(2048, 2048, Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(target_size)

        img = reader.read()
        if img.isNull():
            QMessageBox.critical(
                self, "Error", f"Failed to load image:\n{reader.errorString()}"
            )
            return

        pixmap = QPixmap.fromImage(img)
        self.image_label.set_image(pixmap, (orig_size.width(), orig_size.height()))

    def _on_image_clicked(self, x: float, y: float) -> None:
        if not self.metadata:
            QMessageBox.warning(
                self,
                "No Metadata",
                "This image has no GPS metadata, so coordinates cannot be calculated.",
            )
            return

        lat, lon = pixel_to_lat_lon(x, y, self.metadata)
        source = "MRK-matched camera coordinate" if self.metadata.used_rtk else "EXIF/XMP camera coordinate"
        self.click_status_label.setText(
            "Click status: "
            f"x={x:.1f}, y={y:.1f} -> lat={lat:.8f}, lon={lon:.8f} "
            f"using {source}"
        )
        self.pin_requested.emit(lat, lon, self.image_path.name)

    def _mrk_checker_text(self) -> str:
        if self.metadata is None:
            linked = str(self.mrk_path) if self.mrk_path else "None"
            return f"MRK checker: metadata unavailable\nLinked MRK: {linked}\nUsed for this image: NO"

        linked = str(self.mrk_path) if self.mrk_path else "None"
        if self.metadata.used_rtk:
            return (
                "MRK checker: ACTIVE\n"
                f"Linked MRK: {linked}\n"
                f"Matched MRK: {self.metadata.rtk_source or linked}\n"
                f"Image sequence: {self.metadata.rtk_sequence}\n"
                f"MRK camera coordinate: {self.metadata.rtk_latitude:.8f}, {self.metadata.rtk_longitude:.8f}\n"
                "Used for this image: YES"
            )

        reason = "No linked MRK file" if self.mrk_path is None else "No matching MRK row for this image sequence"
        return (
            "MRK checker: NOT USED\n"
            f"Linked MRK: {linked}\n"
            f"Reason: {reason}\n"
            f"Fallback coordinate: {self.metadata.latitude:.8f}, {self.metadata.longitude:.8f}\n"
            "Used for this image: NO"
        )
