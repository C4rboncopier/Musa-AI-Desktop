from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class NewProjectDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(520)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        title = QLabel("Create Project")
        title.setObjectName("dialogTitle")
        subtitle = QLabel("Projects keep GeoTIFFs, image folders, AI models, exports, and analysis metadata isolated.")
        subtitle.setObjectName("bodyText")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Subang banana block survey")
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Purpose, plantation block, flight notes, or operator comments")
        self.description_edit.setFixedHeight(92)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Optional export folder")

        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("secondaryButton")
        browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_btn.clicked.connect(self._select_output_dir)
        output_row = QHBoxLayout()
        output_row.setContentsMargins(0, 0, 0, 0)
        output_row.setSpacing(8)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse_btn)

        form.addRow("Name", self.name_edit)
        form.addRow("Description", self.description_edit)
        form.addRow("Export folder", output_row)
        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondaryButton")
        cancel_btn.clicked.connect(self.reject)
        create_btn = QPushButton("Create Project")
        create_btn.setObjectName("primaryButton")
        create_btn.clicked.connect(self._validate_and_accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(create_btn)
        root.addLayout(buttons)

    @property
    def project_name(self) -> str:
        return self.name_edit.text().strip()

    @property
    def description(self) -> str:
        return self.description_edit.toPlainText().strip()

    @property
    def output_dir(self) -> str:
        return self.output_edit.text().strip()

    def _select_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select export folder", "")
        if folder:
            self.output_edit.setText(folder)

    def _validate_and_accept(self) -> None:
        if not self.project_name:
            QMessageBox.warning(self, "Project name required", "Enter a project name before creating it.")
            return
        if self.output_dir and not Path(self.output_dir).exists():
            QMessageBox.warning(self, "Folder not found", "The selected export folder does not exist.")
            return
        self.accept()


class EditProjectDialog(QDialog):
    def __init__(
        self,
        name: str,
        description: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Project")
        self.setMinimumWidth(520)
        self.setModal(True)
        self._build_ui(name, description)

    def _build_ui(self, name: str, description: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        title = QLabel("Edit Project Details")
        title.setObjectName("dialogTitle")
        subtitle = QLabel("Update the project title and description stored in the local database.")
        subtitle.setObjectName("bodyText")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Project title")
        self.description_edit = QTextEdit()
        self.description_edit.setPlainText(description)
        self.description_edit.setPlaceholderText("Project description")
        self.description_edit.setFixedHeight(112)

        form.addRow("Title", self.name_edit)
        form.addRow("Description", self.description_edit)
        root.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondaryButton")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Changes")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._validate_and_accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        root.addLayout(buttons)

    @property
    def project_name(self) -> str:
        return self.name_edit.text().strip()

    @property
    def description(self) -> str:
        return self.description_edit.toPlainText().strip()

    def _validate_and_accept(self) -> None:
        if not self.project_name:
            QMessageBox.warning(self, "Project title required", "Enter a project title before saving.")
            return
        self.accept()


class PreferredModelsDialog(QDialog):
    def __init__(
        self,
        leaf_model_path: str = "",
        disease_model_path: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage AI Models")
        self.setMinimumWidth(620)
        self.setModal(True)
        self._build_ui(leaf_model_path, disease_model_path)

    def _build_ui(self, leaf_model_path: str, disease_model_path: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        title = QLabel("Preferred AI Models")
        title.setObjectName("dialogTitle")
        subtitle = QLabel(
            "Choose the default YOLO model files. New projects will automatically use these paths."
        )
        subtitle.setObjectName("bodyText")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        leaf_label = QLabel("Leaf model")
        leaf_label.setObjectName("metaKey")
        disease_label = QLabel("Disease model")
        disease_label.setObjectName("metaKey")

        self.leaf_edit = QLineEdit(leaf_model_path)
        self.leaf_edit.setPlaceholderText("YOLOv8 segmentation model for full_leaf / cut_leaf")
        self.disease_edit = QLineEdit(disease_model_path)
        self.disease_edit.setPlaceholderText("YOLOv8 disease model for black_sigatoka / panama")

        leaf_browse = QPushButton("Browse")
        leaf_browse.setObjectName("secondaryButton")
        leaf_browse.clicked.connect(
            lambda _=False: self._select_model(self.leaf_edit, "Open preferred leaf model")
        )
        disease_browse = QPushButton("Browse")
        disease_browse.setObjectName("secondaryButton")
        disease_browse.clicked.connect(
            lambda _=False: self._select_model(self.disease_edit, "Open preferred disease model")
        )
        leaf_clear = QPushButton("Clear")
        leaf_clear.setObjectName("secondaryButton")
        leaf_clear.clicked.connect(self.leaf_edit.clear)
        disease_clear = QPushButton("Clear")
        disease_clear.setObjectName("secondaryButton")
        disease_clear.clicked.connect(self.disease_edit.clear)

        grid.addWidget(leaf_label, 0, 0)
        grid.addWidget(self.leaf_edit, 0, 1)
        grid.addWidget(leaf_browse, 0, 2)
        grid.addWidget(leaf_clear, 0, 3)
        grid.addWidget(disease_label, 1, 0)
        grid.addWidget(self.disease_edit, 1, 1)
        grid.addWidget(disease_browse, 1, 2)
        grid.addWidget(disease_clear, 1, 3)
        root.addLayout(grid)

        note = QLabel(
            "The app stores only these filepaths. Keep the .pt files in a stable local folder."
        )
        note.setObjectName("bodyText")
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondaryButton")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Models")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._validate_and_accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        root.addLayout(buttons)

    @property
    def leaf_model_path(self) -> str:
        return self.leaf_edit.text().strip()

    @property
    def disease_model_path(self) -> str:
        return self.disease_edit.text().strip()

    def _select_model(self, target: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            target.text().strip(),
            "YOLO weights (*.pt);;All files (*.*)",
        )
        if path:
            target.setText(path)

    def _validate_and_accept(self) -> None:
        missing = []
        for label, value in [
            ("Leaf model", self.leaf_model_path),
            ("Disease model", self.disease_model_path),
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
        self.accept()
