from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.models import ProjectBundle
from .widgets import EmptyState, MetricCard, StatusPill


class ProjectListHeader(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("projectListHeader")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(12)
        for text, width, stretch in [
            ("Project", 0, 4),
            ("Status", 120, 0),
            ("Assets", 82, 0),
            ("Results", 82, 0),
            ("Detections", 104, 0),
            ("Modified", 130, 0),
            ("Actions", 180, 0),
        ]:
            label = QLabel(text.upper())
            label.setObjectName("projectListHeaderText")
            if width:
                label.setFixedWidth(width)
            layout.addWidget(label, stretch)


class ProjectListRow(QFrame):
    openRequested = pyqtSignal(str)
    editRequested = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)

    def __init__(self, bundle: ProjectBundle, parent=None) -> None:
        super().__init__(parent)
        self.bundle = bundle
        self.setObjectName("projectListRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_ui()

    def _build_ui(self) -> None:
        project = self.bundle.project
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        identity = QVBoxLayout()
        identity.setSpacing(3)
        title = QLabel(project.name)
        title.setObjectName("projectListTitle")
        title.setWordWrap(True)
        description = QLabel(project.description or "No description")
        description.setObjectName("projectListDescription")
        description.setWordWrap(False)
        description.setToolTip(project.description)
        identity.addWidget(title)
        identity.addWidget(description)
        layout.addLayout(identity, 4)

        state = StatusPill(
            "Ready" if self.bundle.missing_path_count == 0 else f"{self.bundle.missing_path_count} missing",
            "ok" if self.bundle.missing_path_count == 0 else "danger",
        )
        state.setFixedWidth(120)
        layout.addWidget(state, 0, Qt.AlignmentFlag.AlignVCenter)

        for value, tooltip in [
            (str(len(self.bundle.assets) + len(self.bundle.models)), "Linked assets and models"),
            (str(len(self.bundle.results)), "Saved analysis runs"),
            (f"{self.bundle.detection_total:,}", "Latest detection total"),
        ]:
            metric = QLabel(value)
            metric.setObjectName("projectListMetric")
            metric.setAlignment(Qt.AlignmentFlag.AlignCenter)
            metric.setToolTip(tooltip)
            metric.setFixedWidth(82 if tooltip != "Latest detection total" else 104)
            layout.addWidget(metric)

        modified = QLabel(project.modified_at.replace("T", " ")[:16])
        modified.setObjectName("projectListMeta")
        modified.setFixedWidth(130)
        layout.addWidget(modified, 0, Qt.AlignmentFlag.AlignVCenter)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        open_btn = QPushButton("Open")
        open_btn.setObjectName("tableActionButton")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        edit_btn = QPushButton("Edit")
        edit_btn.setObjectName("tableActionButton")
        edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("tableDangerButton")
        delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.clicked.connect(lambda _=False: self.openRequested.emit(project.id))
        edit_btn.clicked.connect(lambda _=False: self.editRequested.emit(project.id))
        delete_btn.clicked.connect(lambda _=False: self.deleteRequested.emit(project.id))
        actions.addWidget(open_btn)
        actions.addWidget(edit_btn)
        actions.addWidget(delete_btn)
        action_host = QWidget()
        action_host.setFixedWidth(180)
        action_host.setLayout(actions)
        layout.addWidget(action_host, 0, Qt.AlignmentFlag.AlignVCenter)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.openRequested.emit(self.bundle.project.id)
        super().mouseReleaseEvent(event)


class DashboardPage(QWidget):
    createRequested = pyqtSignal()
    projectOpenRequested = pyqtSignal(str)
    projectEditRequested = pyqtSignal(str)
    projectDeleteRequested = pyqtSignal(str)
    refreshRequested = pyqtSignal()
    settingsRequested = pyqtSignal()
    modelManagerRequested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._bundles: list[ProjectBundle] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        rail = QFrame()
        rail.setObjectName("navRail")
        rail.setFixedWidth(250)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(18, 20, 18, 18)
        rail_layout.setSpacing(10)

        app_title = QLabel("Musa AI")
        app_title.setObjectName("appTitle")
        app_subtitle = QLabel("GEOSPATIAL AI PLATFORM")
        app_subtitle.setObjectName("appSubtitle")
        rail_layout.addWidget(app_title)
        rail_layout.addWidget(app_subtitle)
        rail_layout.addSpacing(18)

        self.new_btn = self._nav_button("New Project", primary=True)
        self.models_btn = self._nav_button("AI Models")
        self.settings_btn = self._nav_button("Settings")
        self.refresh_btn = self._nav_button("Refresh")
        rail_layout.addWidget(self.new_btn)
        rail_layout.addWidget(self.models_btn)
        rail_layout.addWidget(self.settings_btn)
        rail_layout.addStretch(1)
        rail_layout.addWidget(self.refresh_btn)

        self.new_btn.clicked.connect(lambda _=False: self.createRequested.emit())
        self.models_btn.clicked.connect(lambda _=False: self.modelManagerRequested.emit())
        self.settings_btn.clicked.connect(lambda _=False: self.settingsRequested.emit())
        self.refresh_btn.clicked.connect(lambda _=False: self.refreshRequested.emit())

        content = QWidget()
        content.setObjectName("dashboardContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(28, 24, 28, 24)
        content_layout.setSpacing(18)

        header = QHBoxLayout()
        heading_col = QVBoxLayout()
        heading_col.setSpacing(4)
        title = QLabel("Project Dashboard")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Create projects, relink local assets, and launch AI mapping workflows from one hub.")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        heading_col.addWidget(title)
        heading_col.addWidget(subtitle)
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("searchBox")
        self.search_edit.setPlaceholderText("Search projects")
        self.search_edit.textChanged.connect(self._apply_filter)
        header.addLayout(heading_col, 1)
        header.addWidget(self.search_edit, 0, Qt.AlignmentFlag.AlignBottom)
        content_layout.addLayout(header)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.project_metric = MetricCard("Projects", "0")
        self.asset_metric = MetricCard("Linked Assets", "0")
        self.result_metric = MetricCard("Analysis Runs", "0")
        self.missing_metric = MetricCard("Missing Paths", "0")
        for metric in [self.project_metric, self.asset_metric, self.result_metric, self.missing_metric]:
            metrics.addWidget(metric)
        content_layout.addLayout(metrics)

        projects_header = QHBoxLayout()
        projects_title = QLabel("Recent Projects")
        projects_title.setObjectName("panelTitle")
        projects_header.addWidget(projects_title)
        projects_header.addStretch(1)
        content_layout.addLayout(projects_header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.project_container = QWidget()
        self.project_list = QVBoxLayout(self.project_container)
        self.project_list.setContentsMargins(0, 0, 0, 0)
        self.project_list.setSpacing(8)
        self.scroll.setWidget(self.project_container)
        content_layout.addWidget(self.scroll, 1)

        root.addWidget(rail)
        root.addWidget(content, 1)

    def set_projects(self, bundles: list[ProjectBundle]) -> None:
        self._bundles = bundles
        self.project_metric.set_value(str(len(bundles)))
        self.asset_metric.set_value(str(sum(len(b.assets) + len(b.models) for b in bundles)))
        self.result_metric.set_value(str(sum(len(b.results) for b in bundles)))
        self.missing_metric.set_value(str(sum(b.missing_path_count for b in bundles)))
        self._render_projects(bundles)

    def _apply_filter(self) -> None:
        query = self.search_edit.text().strip().lower()
        if not query:
            self._render_projects(self._bundles)
            return
        filtered = [
            bundle for bundle in self._bundles
            if query in bundle.project.name.lower()
            or query in bundle.project.description.lower()
        ]
        self._render_projects(filtered)

    def _render_projects(self, bundles: list[ProjectBundle]) -> None:
        while self.project_list.count():
            item = self.project_list.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not bundles:
            empty = EmptyState(
                "No Projects Yet",
                "Create a project to connect GeoTIFFs, AI models, and exports.",
                "Create Project",
            )
            if empty.button:
                empty.button.clicked.connect(lambda _=False: self.createRequested.emit())
            self.project_list.addWidget(empty)
            return

        self.project_list.addWidget(ProjectListHeader())
        for bundle in bundles:
            row_widget = ProjectListRow(bundle)
            row_widget.openRequested.connect(self.projectOpenRequested.emit)
            row_widget.editRequested.connect(self.projectEditRequested.emit)
            row_widget.deleteRequested.connect(self.projectDeleteRequested.emit)
            self.project_list.addWidget(row_widget)
        self.project_list.addStretch(1)

    @staticmethod
    def _nav_button(text: str, primary: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("primaryButton" if primary else "navButton")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return btn
