from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_WEB_ENGINE_GPU_FLAGS = (
    "--ignore-gpu-blocklist",
    "--enable-gpu-rasterization",
    "--enable-zero-copy",
    "--enable-native-gpu-memory-buffers",
    "--enable-accelerated-2d-canvas",
)


def _configure_web_engine_gpu() -> None:
    current = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    tokens = current.split()
    for flag in _WEB_ENGINE_GPU_FLAGS:
        if flag not in tokens:
            tokens.append(flag)
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(tokens)


_configure_web_engine_gpu()

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QActionGroup, QDesktopServices, QFont
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
)

from .core.database import ProjectRepository
from .core.geotiff_cache import GeoTiffSessionCache
from .core.models import ProjectBundle
from .core.output_manager import ProjectOutputManager
from .detection import DetectionRecord, MappingResult
from .geotiff import GeoTiffInfo
from .hardware import NVIDIA_DRIVER_URL, PYTORCH_SETUP_URL, HardwareStatus, detect_hardware, inference_device_arg
from .loading_dialog import LoadingDialog
from .map_bridge import MapBridge
from .themes import DEFAULT_THEME, THEMES, generate_stylesheet
from .ui.dashboard import DashboardPage
from .ui.dialogs import EditProjectDialog, NewProjectDialog, PreferredModelsDialog
from .ui.settings import SettingsPage
from .ui.workspace import WorkspacePage
from .worker import AiGeotiffMappingWorker, GeoTiffWorker, HardwareCheckWorker
from .csv_map_importer import parse_csv_coordinates, CsvImportError


BASE_MAPS = {"osm", "google_satellite"}


class MainWindow(QMainWindow):
    """Application shell for the project-based geospatial AI workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Musa AI")
        self.resize(1500, 920)
        self.setMinimumSize(1100, 720)

        self.repo = ProjectRepository()
        self.output_manager = ProjectOutputManager()
        self.hardware_status: HardwareStatus = detect_hardware()
        self.processing_device_preference = self._initial_processing_device_preference()
        self.current_bundle: ProjectBundle | None = None
        self.current_geotiff: GeoTiffInfo | None = None
        self.latest_mapping_result: MappingResult | None = None
        self.imported_csv_records: list[dict] = []
        self.geotiff_cache = GeoTiffSessionCache(
            max_items=4,
            cache_dir=None,
        )

        self.map_ready = False
        self.pending_geotiff: GeoTiffInfo | None = None
        self.pending_mapping: MappingResult | None = None
        self._active_worker: GeoTiffWorker | None = None
        self._active_mapping_worker: AiGeotiffMappingWorker | None = None
        self._active_hardware_worker: HardwareCheckWorker | None = None
        self._loading_dialog: LoadingDialog | None = None
        self._mapping_dialog: LoadingDialog | None = None
        self._project_open_dialog: LoadingDialog | None = None
        self._integrity_timer = QTimer(self)
        self._startup_hardware_warning_shown = False
        self._current_theme_name = self.repo.get_preference("theme", DEFAULT_THEME)
        if self._current_theme_name not in THEMES:
            self._current_theme_name = DEFAULT_THEME
        self._current_base_map = str(self.repo.get_preference("base_map", "osm") or "osm")
        if self._current_base_map not in BASE_MAPS:
            self._current_base_map = "osm"
        self._google_maps_api_key = _load_google_maps_api_key(self._google_maps_secret_path())
        if self._current_base_map.startswith("google_") and not self._google_maps_api_key:
            self._current_base_map = "osm"

        self.bridge = MapBridge()
        self.web_channel = QWebChannel(self)
        self.web_channel.registerObject("mapBridge", self.bridge)

        self._build_actions()
        self._build_pages()
        self._build_status_bar()
        self._connect_signals()
        self._apply_theme(self._current_theme_name)
        self.workspace.set_base_map(self._current_base_map)
        self._ensure_all_project_output_dirs()
        self._cleanup_removed_feature_assets()
        self._update_settings_page()
        self.refresh_dashboard()
        self.workspace.load_map()
        self._start_integrity_monitor()
        QTimer.singleShot(900, self._show_startup_hardware_warning_if_needed)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_actions(self) -> None:
        self.new_project_action = QAction("New Project", self)
        self.new_project_action.setShortcut("Ctrl+N")
        self.new_project_action.triggered.connect(self.create_project)

        self.dashboard_action = QAction("Dashboard", self)
        self.dashboard_action.setShortcut("Ctrl+D")
        self.dashboard_action.triggered.connect(self.show_dashboard)

        self.settings_action = QAction("Settings", self)
        self.settings_action.setShortcut("Ctrl+,")
        self.settings_action.triggered.connect(self.show_settings)

        self.fit_action = QAction("Fit to Orthomosaic", self)
        self.fit_action.setShortcut("Ctrl+F")
        self.fit_action.triggered.connect(self.fit_overlay)

        self.project_explorer_action = QAction("Project Explorer", self)
        self.project_explorer_action.setCheckable(True)
        self.project_explorer_action.setChecked(True)

        self.inspector_action = QAction("Inspector Panel", self)
        self.inspector_action.setCheckable(True)
        self.inspector_action.setChecked(True)

        self.exit_action = QAction("Exit", self)
        self.exit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.dashboard_action)
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.fit_action)
        view_menu.addAction("Reset View", self.reset_view)
        view_menu.addSeparator()
        view_menu.addAction(self.project_explorer_action)
        view_menu.addAction(self.inspector_action)

        theme_menu = self.menuBar().addMenu("Theme")
        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.setExclusive(True)
        for theme_name in THEMES:
            action = QAction(theme_name, self)
            action.setCheckable(True)
            action.setChecked(theme_name == self._current_theme_name)
            action.triggered.connect(lambda _=False, name=theme_name: self._apply_theme(name))
            self.theme_action_group.addAction(action)
            theme_menu.addAction(action)

    def _build_pages(self) -> None:
        self.stack = QStackedWidget(self)
        self.dashboard = DashboardPage()
        self.workspace = WorkspacePage()
        self.settings = SettingsPage()
        self.workspace.set_web_channel(self.web_channel)
        self.stack.addWidget(self.dashboard)
        self.stack.addWidget(self.workspace)
        self.stack.addWidget(self.settings)
        self.setCentralWidget(self.stack)

    def _build_status_bar(self) -> None:
        status = QStatusBar(self)
        status.setObjectName("statusBar")
        self.setStatusBar(status)
        self.statusBar().showMessage("Ready.", 2500)

    def _connect_signals(self) -> None:
        self.dashboard.createRequested.connect(self.create_project)
        self.dashboard.projectOpenRequested.connect(self.open_project)
        self.dashboard.projectEditRequested.connect(self.edit_project)
        self.dashboard.projectDeleteRequested.connect(self.delete_project)
        self.dashboard.refreshRequested.connect(self.refresh_dashboard)
        self.dashboard.settingsRequested.connect(self.show_settings)
        self.dashboard.modelManagerRequested.connect(self.open_model_manager)

        self.settings.backRequested.connect(self.show_dashboard)
        self.settings.devicePreferenceChanged.connect(self._set_processing_device_preference)
        self.settings.refreshHardwareRequested.connect(self.refresh_hardware_status)
        self.settings.openPytorchGuideRequested.connect(lambda: QDesktopServices.openUrl(QUrl(PYTORCH_SETUP_URL)))
        self.settings.openNvidiaDriverRequested.connect(lambda: QDesktopServices.openUrl(QUrl(NVIDIA_DRIVER_URL)))
        self.settings.clearCacheRequested.connect(self.clear_project_cache)
        self.settings.openOutputRootRequested.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_manager.root_dir)))
        )
        self.settings.openDatabaseFolderRequested.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.repo.db_path.parent)))
        )
        self.settings.googleMapsApiKeyChanged.connect(self._set_google_maps_api_key)

        self.workspace.backRequested.connect(self.show_dashboard)
        self.workspace.importGeoTiffRequested.connect(self.open_geotiff)
        self.workspace.importCsvRequested.connect(self.import_csv_file)
        self.workspace.drawOnMapRequested.connect(self.draw_csv_markers)
        self.workspace.runMappingRequested.connect(self.run_ai_mapping)
        self.workspace.fitMapRequested.connect(self.fit_overlay)
        self.workspace.resetViewRequested.connect(self.reset_view)
        self.workspace.zoomInRequested.connect(lambda: self._run_js("zoomIn();"))
        self.workspace.zoomOutRequested.connect(lambda: self._run_js("zoomOut();"))
        self.workspace.baseMapChanged.connect(self._set_base_map)
        self.workspace.overlayVisibilityChanged.connect(self._set_overlay_visible)
        self.workspace.overlayOpacityChanged.connect(self._set_overlay_opacity)
        self.workspace.detectionOpacityChanged.connect(self._set_detection_opacity)
        self.workspace.detectionLayerChanged.connect(self._set_detection_layer_visible)
        self.workspace.detectionStyleChanged.connect(self._set_detection_style)
        self.workspace.resolutionChanged.connect(self._on_resolution_changed)
        self.workspace.projectOutputOpenRequested.connect(self.open_project_output_folder)
        self.workspace.outputsRefreshRequested.connect(self.refresh_project_outputs)
        self.workspace.outputOpenRequested.connect(self.open_output_file)
        self.workspace.outputRevealRequested.connect(self.reveal_output_file)
        self.workspace.outputExportRequested.connect(self.export_output_file)
        self.workspace.outputDeleteRequested.connect(self.delete_output_file)
        self.project_explorer_action.toggled.connect(self.workspace.set_project_explorer_visible)
        self.inspector_action.toggled.connect(self.workspace.set_inspector_visible)
        self.workspace.projectExplorerVisibilityChanged.connect(
            lambda visible: self._sync_view_action(self.project_explorer_action, visible)
        )
        self.workspace.inspectorVisibilityChanged.connect(
            lambda visible: self._sync_view_action(self.inspector_action, visible)
        )

        self.bridge.coordinatesChanged.connect(self._update_coordinates)
        self.bridge.zoomChanged.connect(self._update_zoom)
        self.bridge.mapReady.connect(self._handle_map_ready)

    def _initial_processing_device_preference(self) -> str:
        saved = str(self.repo.get_preference("processing_device", "") or "")
        if saved in {"cpu", "gpu"}:
            if saved == "gpu" and not self.hardware_status.has_compatible_gpu:
                self.repo.set_preference("processing_device", "cpu")
                return "cpu"
            return saved
        preference = "gpu" if self.hardware_status.has_compatible_gpu else "cpu"
        self.repo.set_preference("processing_device", preference)
        return preference

    def _set_processing_device_preference(self, preference: str) -> None:
        if preference == "gpu" and not self.hardware_status.has_compatible_gpu:
            preference = "cpu"
            self.statusBar().showMessage("GPU is not available; AI mapping will use CPU.", 4500)
        if preference not in {"cpu", "gpu"}:
            return
        self.processing_device_preference = preference
        self.repo.set_preference("processing_device", preference)
        self._update_settings_page()
        label = "GPU" if preference == "gpu" else "CPU"
        self.statusBar().showMessage(f"AI processing device set to {label}.", 3500)

    def refresh_hardware_status(self) -> None:
        if self._active_hardware_worker is not None and self._active_hardware_worker.isRunning():
            return
        self.settings.begin_hardware_check()
        self.statusBar().showMessage("Hardware check started.", 2500)
        self._active_hardware_worker = HardwareCheckWorker(self)
        self._active_hardware_worker.progress.connect(self._on_hardware_check_progress)
        self._active_hardware_worker.finished.connect(self._on_hardware_check_finished)
        self._active_hardware_worker.failed.connect(self._on_hardware_check_failed)
        self._active_hardware_worker.start()

    def _on_hardware_check_progress(self, percent: int, message: str, level: str) -> None:
        if self.sender() is not self._active_hardware_worker:
            return
        self.settings.update_hardware_check_progress(percent, message, level)
        self.statusBar().showMessage(message, 2000)

    def _on_hardware_check_finished(self, status_obj: object) -> None:
        if self.sender() is not self._active_hardware_worker:
            return
        self.hardware_status = status_obj  # type: ignore[assignment]
        if self.processing_device_preference == "gpu" and not self.hardware_status.has_compatible_gpu:
            self.processing_device_preference = "cpu"
            self.repo.set_preference("processing_device", "cpu")
        elif self.processing_device_preference not in {"cpu", "gpu"}:
            self.processing_device_preference = "gpu" if self.hardware_status.has_compatible_gpu else "cpu"
            self.repo.set_preference("processing_device", self.processing_device_preference)
        self._active_hardware_worker = None
        self._update_settings_page()
        self.statusBar().showMessage("Hardware check complete.", 3500)

    def _on_hardware_check_failed(self, error_message: str) -> None:
        if self.sender() is not self._active_hardware_worker:
            return
        self._active_hardware_worker = None
        self.settings.fail_hardware_check(error_message)
        self.statusBar().showMessage("Hardware check failed.", 4500)

    def _update_settings_page(self) -> None:
        if not hasattr(self, "settings"):
            return
        self.settings.set_hardware_status(self.hardware_status, self.processing_device_preference)
        self.settings.set_storage_paths(
            self.repo.db_path,
            self.output_manager.root_dir,
            self.geotiff_cache.cache_dir,
        )
        self.settings.set_google_maps_api_key_configured(bool(self._google_maps_api_key))

    def _selected_inference_device(self) -> str:
        return inference_device_arg(self.processing_device_preference, self.hardware_status)

    def _show_startup_hardware_warning_if_needed(self) -> None:
        if self._startup_hardware_warning_shown or self.hardware_status.has_compatible_gpu:
            return
        self._startup_hardware_warning_shown = True
        steps = "\n".join(f"{index}. {step}" for index, step in enumerate(self.hardware_status.setup_steps, start=1))
        box = QMessageBox(self)
        box.setWindowTitle("GPU Acceleration Not Ready")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            "AI mapping will use CPU processing on this computer.\n\n"
            f"{self.hardware_status.issue_detail}\n\n"
            "Recommended setup:\n"
            f"{steps}"
        )
        settings_btn = box.addButton("Open Settings", QMessageBox.ButtonRole.AcceptRole)
        guide_btn = box.addButton("Open Setup Guide", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        clicked = box.clickedButton()
        if clicked is settings_btn:
            self.show_settings()
        elif clicked is guide_btn:
            QDesktopServices.openUrl(QUrl(PYTORCH_SETUP_URL))

    def _start_integrity_monitor(self) -> None:
        self._integrity_timer.setInterval(5000)
        self._integrity_timer.timeout.connect(self._sync_current_project_integrity)
        self._integrity_timer.start()

    # ------------------------------------------------------------------
    # Dashboard and project workflow
    # ------------------------------------------------------------------

    def refresh_dashboard(self) -> None:
        bundles = [self.repo.get_bundle(project.id) for project in self.repo.list_projects()]
        self.dashboard.set_projects(bundles)

    def show_dashboard(self) -> None:
        self.refresh_dashboard()
        self.stack.setCurrentWidget(self.dashboard)
        self.statusBar().showMessage("Dashboard ready.", 2500)

    def create_project(self) -> None:
        dialog = NewProjectDialog(self)
        dialog.setStyleSheet(generate_stylesheet(THEMES[self._current_theme_name]))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        project = self.repo.create_project(
            dialog.project_name,
            dialog.description,
            "",
            settings={"theme": self._current_theme_name, "autosave": True},
        )
        output_dir = self._ensure_project_output_dir(project.id, project.name)
        self._set_project_geotiff_cache(output_dir)
        self._apply_preferred_models_to_project(project.id)
        self.refresh_dashboard()
        self.open_project(project.id)

    def edit_project(self, project_id: str) -> None:
        try:
            project = self.repo.get_project(project_id)
        except KeyError:
            QMessageBox.warning(self, "Project not found", "This project no longer exists.")
            self.refresh_dashboard()
            return

        dialog = EditProjectDialog(project.name, project.description, parent=self)
        dialog.setStyleSheet(generate_stylesheet(THEMES[self._current_theme_name]))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.repo.update_project(
            project_id,
            name=dialog.project_name,
            description=dialog.description,
        )
        output_dir = self._ensure_project_output_dir(project_id, dialog.project_name)
        if self.current_bundle is not None and self.current_bundle.project.id == project_id:
            self._set_project_geotiff_cache(output_dir)
        if self.current_bundle is not None and self.current_bundle.project.id == project_id:
            self.current_bundle = self.repo.get_bundle(project_id)
            self.workspace.set_project(self.current_bundle)
            self.refresh_project_outputs()
            self.workspace.append_log("Project title and description updated.")
        self.refresh_dashboard()
        self.statusBar().showMessage("Project details updated.", 3500)

    def delete_project(self, project_id: str) -> None:
        try:
            project = self.repo.get_project(project_id)
        except KeyError:
            QMessageBox.warning(self, "Project not found", "This project no longer exists.")
            self.refresh_dashboard()
            return

        output_dir = Path(project.output_dir) if project.output_dir else None
        output_label = str(output_dir) if output_dir else "No project folder is recorded."
        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle("Delete Project")
        confirm.setText(f"Delete this project?\n\n{project.name}")
        confirm.setInformativeText(
            "Choose whether to remove only the local database record or also delete the project folder.\n\n"
            f"Project folder:\n{output_label}"
        )
        database_only_btn = confirm.addButton("Delete Project Only", QMessageBox.ButtonRole.AcceptRole)
        delete_folder_btn = confirm.addButton("Delete Project and Folder", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = confirm.addButton(QMessageBox.StandardButton.Cancel)
        if output_dir is None or not output_dir.exists():
            delete_folder_btn.setEnabled(False)
            delete_folder_btn.setToolTip("The recorded project folder does not exist.")
        confirm.setDefaultButton(cancel_btn)
        confirm.exec()

        clicked = confirm.clickedButton()
        if clicked is cancel_btn:
            return
        delete_project_folder = clicked is delete_folder_btn
        if clicked is not database_only_btn and clicked is not delete_folder_btn:
            return

        if delete_project_folder:
            folder_answer = QMessageBox.question(
                self,
                "Delete Project Folder",
                "Permanently delete this project folder and all files inside it?\n\n"
                f"{output_dir}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if folder_answer != QMessageBox.StandardButton.Yes:
                return

        deleting_current = self.current_bundle is not None and self.current_bundle.project.id == project_id
        if deleting_current:
            self._cancel_geotiff_worker("Project deleted; active GeoTIFF restore/import cancelled.")
            if self._active_mapping_worker is not None and self._active_mapping_worker.isRunning():
                self._active_mapping_worker.requestInterruption()
            self._active_mapping_worker = None
            self._clear_map_view()
            self.workspace.clear_geotiff_metadata()
            self.workspace.update_counts({})
            self.current_bundle = None
            self.current_geotiff = None
            self.latest_mapping_result = None
            self.pending_geotiff = None
            self.pending_mapping = None

        if delete_project_folder and output_dir is not None:
            try:
                self._delete_project_folder(output_dir)
            except OSError as exc:
                QMessageBox.critical(
                    self,
                    "Unable to delete project folder",
                    f"The project was not deleted because the folder could not be removed.\n\n{exc}",
                )
                self.refresh_dashboard()
                return

        self.repo.delete_project(project_id)
        self.refresh_dashboard()
        self.stack.setCurrentWidget(self.dashboard)
        message = "Project and folder deleted." if delete_project_folder else "Project deleted from the local database."
        self.statusBar().showMessage(message, 4000)

    def _delete_project_folder(self, folder: Path) -> None:
        folder = folder.resolve()
        root = self.output_manager.root_dir.resolve()
        if folder == root or folder.parent == folder:
            raise OSError(f"Refusing to delete unsafe project folder: {folder}")
        if not folder.exists():
            return
        if not folder.is_dir():
            raise OSError(f"Recorded project folder is not a folder: {folder}")
        shutil.rmtree(folder)

    def _preferred_model_paths(self) -> dict[str, str]:
        return {
            "leaf": self.repo.get_preference("preferred_leaf_model_path", "") or "",
            "disease": self.repo.get_preference("preferred_disease_model_path", "") or "",
        }

    def _apply_preferred_models_to_project(self, project_id: str) -> None:
        preferred = self._preferred_model_paths()
        for role, path in preferred.items():
            if path and Path(path).exists():
                self.repo.upsert_model(project_id, role, path, label=Path(path).name)

    def _refresh_saved_models_for_mapping(self, bundle: ProjectBundle) -> ProjectBundle | None:
        preferred = self._preferred_model_paths()
        missing = {
            role: path
            for role, path in preferred.items()
            if path and not Path(path).exists()
        }
        if missing:
            labels = {
                "leaf": "Leaf model",
                "disease": "Disease model",
            }
            details = "\n".join(
                f"{labels.get(role, role.title())}: {path}"
                for role, path in missing.items()
            )
            QMessageBox.warning(
                self,
                "Saved AI model missing",
                "The saved preferred model path no longer exists. "
                "Choose a valid model before running AI mapping.\n\n"
                f"{details}",
            )
            return None

        updated_roles: list[str] = []
        for role, path in preferred.items():
            if not path:
                continue
            model_path = str(Path(path))
            current = bundle.first_model(role)
            if current is None or current.path != model_path:
                updated_roles.append(role)
                self.repo.upsert_model(bundle.project.id, role, model_path, label=Path(model_path).name)

        if updated_roles:
            role_labels = ", ".join(role.replace("_", " ") for role in updated_roles)
            self.workspace.append_log(f"Loaded current saved AI model setting(s): {role_labels}.")
            return self._reload_current_bundle() or self.repo.get_bundle(bundle.project.id)
        return bundle

    def open_project(self, project_id: str) -> None:
        if self._active_worker is not None and self._active_worker.isRunning():
            QMessageBox.information(
                self,
                "Project is busy",
                "Wait for the current GeoTIFF operation to finish before opening another project.",
            )
            return

        self.repo.touch_project(project_id)
        project = self.repo.get_project(project_id)
        output_dir = self._ensure_project_output_dir(project.id, project.name)
        self._set_project_geotiff_cache(output_dir)
        self.current_bundle = self.repo.get_bundle(project_id)
        self._apply_project_resolution(project_id)
        self.current_geotiff = None
        self.latest_mapping_result = None
        self.pending_geotiff = None
        self.pending_mapping = None
        self._clear_map_view()
        self.workspace.clear_geotiff_metadata()
        self.workspace.update_counts({})

        self._project_open_dialog = LoadingDialog(
            self.current_bundle.project.name,
            theme_name=self._current_theme_name,
            parent=self,
            title="Opening Project",
        )
        dialog = self._project_open_dialog
        dialog.set_progress(8, "Loading project metadata...")

        self.workspace.set_project(self.current_bundle)
        self.refresh_project_outputs()
        self.stack.setCurrentWidget(self.workspace)
        self.workspace.append_log(f"Opened project {self.current_bundle.project.name}.")
        self._log_missing_paths(self.current_bundle)

        geotiff_asset = self.current_bundle.first_asset("geotiff")
        if geotiff_asset is not None and geotiff_asset.exists:
            dialog.set_progress(18, "Restoring linked GeoTIFF...")
            cached = self.geotiff_cache.get(
                geotiff_asset.path,
                self.workspace.selected_resolution_percent(),
            )
            if cached is not None:
                QTimer.singleShot(60, lambda info=cached: self._finish_project_open_with_geotiff(info, cached=True))
                dialog.exec()
                if dialog.was_cancelled:
                    self._project_open_dialog = None
                    self.statusBar().showMessage("Project opening cancelled.", 3000)
                return

            self._active_worker = GeoTiffWorker(
                geotiff_asset.path,
                preview_scale=self.workspace.selected_resolution_scale(),
                parent=self,
            )
            dialog.cancel_requested.connect(
                lambda: self._cancel_geotiff_worker("Project GeoTIFF restore cancelled.")
            )
            self._active_worker.progress.connect(self._on_project_open_progress)
            self._active_worker.finished.connect(self._on_project_geotiff_finished)
            self._active_worker.failed.connect(self._on_project_geotiff_failed)
            self._active_worker.start()
            dialog.exec()
            if dialog.was_cancelled and self._active_worker is not None:
                self._active_worker.requestInterruption()
            return

        if geotiff_asset is not None and not geotiff_asset.exists:
            self.workspace.append_log(f"Linked GeoTIFF is missing: {geotiff_asset.path}")

        QTimer.singleShot(60, self._finish_project_open_without_geotiff)
        dialog.exec()
        if dialog.was_cancelled:
            self._project_open_dialog = None
            self.statusBar().showMessage("Project opening cancelled.", 3000)

    def _finish_project_open_with_geotiff(self, info: GeoTiffInfo, *, cached: bool = False) -> None:
        if self.current_bundle is None or self._project_open_dialog is None:
            return
        if cached:
            self._project_open_dialog.set_progress(54, "Reusing cached GeoTIFF preview...")
        self._apply_geotiff_info(
            info,
            log_message=(
                f"Reused cached project GeoTIFF: {info.file_name}"
                if cached
                else f"Restored project GeoTIFF: {info.file_name}"
            ),
        )
        self._project_open_dialog.set_progress(90, "Restoring saved analysis results...")
        if self.current_bundle is not None:
            self._recover_latest_analysis(self.current_bundle)
        self._project_open_dialog.set_progress(100, "Project ready.")
        self._project_open_dialog.finish()
        self._project_open_dialog = None
        self._schedule_overlay_fit_to_screen()
        self.refresh_dashboard()
        self.statusBar().showMessage("Project loaded with linked GeoTIFF and saved results.", 5000)

    def _finish_project_open_without_geotiff(self) -> None:
        if self.current_bundle is None or self._project_open_dialog is None:
            return
        self._project_open_dialog.set_progress(72, "Restoring saved analysis results...")
        self._recover_latest_analysis(self.current_bundle)
        self._project_open_dialog.set_progress(100, "Project ready.")
        self._project_open_dialog.finish()
        self._project_open_dialog = None
        self.statusBar().showMessage("Project loaded. Import a GeoTIFF to start mapping.", 3500)

    def _on_project_open_progress(self, percent: int, message: str) -> None:
        if self.sender() is not self._active_worker:
            return
        if self._project_open_dialog is None:
            return
        adjusted = min(86, 18 + int(percent * 0.68))
        self._project_open_dialog.set_progress(adjusted, message)

    def _on_project_geotiff_finished(self, info_obj: object) -> None:
        if self.sender() is not self._active_worker:
            return
        if self._active_worker is not None and self._active_worker.isInterruptionRequested():
            self._active_worker = None
            if self._project_open_dialog is not None:
                self._project_open_dialog.reject()
                self._project_open_dialog = None
            self.statusBar().showMessage("Project opening cancelled.", 3000)
            return

        self._active_worker = None
        info: GeoTiffInfo = info_obj  # type: ignore[assignment]
        self.geotiff_cache.put(info, self.workspace.selected_resolution_percent())
        self._finish_project_open_with_geotiff(info)

    def _on_project_geotiff_failed(self, error_message: str) -> None:
        if self.sender() is not self._active_worker:
            return
        self._active_worker = None
        if "cancelled" in error_message.lower():
            if self._project_open_dialog is not None:
                self._project_open_dialog.reject()
                self._project_open_dialog = None
            self.statusBar().showMessage("Project GeoTIFF restore cancelled.", 3000)
            return
        if self._project_open_dialog is not None:
            self._project_open_dialog.set_progress(82, "GeoTIFF restore failed; checking saved results...")
        self.workspace.append_log(f"Project GeoTIFF restore failed: {error_message}")
        if self.current_bundle is not None:
            self._recover_latest_analysis(self.current_bundle)
        if self._project_open_dialog is not None:
            self._project_open_dialog.finish()
            self._project_open_dialog = None
        QMessageBox.warning(
            self,
            "GeoTIFF restore failed",
            "The project opened, but the linked GeoTIFF could not be restored.\n\n"
            f"{error_message}",
        )
        self.statusBar().showMessage("Project loaded, but GeoTIFF restore failed.", 5000)

    def _reload_current_bundle(self) -> ProjectBundle | None:
        if self.current_bundle is None:
            return None
        output_dir = self._ensure_project_output_dir(
            self.current_bundle.project.id,
            self.current_bundle.project.name,
            self.current_bundle.project.output_dir,
        )
        self._set_project_geotiff_cache(output_dir)
        self.current_bundle = self.repo.get_bundle(self.current_bundle.project.id)
        self.workspace.set_project(self.current_bundle)
        self.refresh_project_outputs()
        self.refresh_dashboard()
        return self.current_bundle

    def _apply_geotiff_info(self, info: GeoTiffInfo, *, log_message: str = "") -> None:
        self.current_geotiff = info
        self.workspace.set_geotiff_metadata(info)
        self._send_overlay_to_map(info)

        if self.current_bundle is not None:
            self.repo.upsert_asset(
                self.current_bundle.project.id,
                "geotiff",
                info.file_path,
                label=info.file_name,
                metadata=self._geotiff_metadata(info),
            )
            self.current_bundle = self.repo.get_bundle(self.current_bundle.project.id)
            self.workspace.set_project(self.current_bundle)
            self.refresh_project_outputs()

        if log_message:
            self.workspace.append_log(log_message)

    @staticmethod
    def _geotiff_metadata(info: GeoTiffInfo) -> dict:
        return {
            "width": info.width,
            "height": info.height,
            "band_count": info.band_count,
            "crs": info.source_crs_authority,
            "source_crs": info.source_crs,
            "bounds_source": info.bounds_source,
            "bounds_wgs84": info.bounds_wgs84,
            "pixel_size_x": info.pixel_size_x,
            "pixel_size_y": info.pixel_size_y,
            "spatial_resolution": info.spatial_resolution_label,
            "preview_width": info.preview_width,
            "preview_height": info.preview_height,
            "metadata_details": info.metadata_details,
        }

    def _on_resolution_changed(self, index: int) -> None:
        if self.current_bundle is None:
            return
        label = self.workspace.geotiff_resolution_combo.itemText(index)
        self.repo.set_config(
            self.current_bundle.project.id,
            "resolution",
            {
                "index": index,
                "display_scale_percent": self.workspace.selected_resolution_percent(),
                "display_scale": self.workspace.selected_resolution_scale(),
                "ai_processing": "native_geotiff_resolution",
                "label": label,
            },
        )
        self.statusBar().showMessage("Display resolution saved. AI analysis remains native resolution.", 3500)
        if self.current_geotiff is not None:
            self._reload_current_geotiff_for_display_scale()

    def _apply_project_resolution(self, project_id: str) -> None:
        config = self.repo.get_config(project_id, "resolution", {})
        index = int(config.get("index", 0) or 0) if isinstance(config, dict) else 0
        index = max(0, min(index, 3))
        self.workspace.set_resolution_index(index)

    def _cleanup_removed_feature_assets(self) -> None:
        for project in self.repo.list_projects():
            removed_types = {
                asset.asset_type
                for asset in self.repo.list_assets(project.id)
                if asset.asset_type in {"image_folder", "mrk_file"}
            }
            for asset_type in removed_types:
                self.repo.remove_asset(project.id, asset_type)
            last_mapping = self.repo.get_config(project.id, "last_mapping", {})
            if isinstance(last_mapping, dict) and last_mapping.get("source") == "image_folder":
                self.repo.remove_config(project.id, "last_mapping")

    def _reload_current_geotiff_for_display_scale(self) -> None:
        if self.current_geotiff is None:
            return
        if self._active_worker is not None and self._active_worker.isRunning():
            self.statusBar().showMessage("Wait for the current GeoTIFF operation to finish.", 3500)
            return

        path = self.current_geotiff.file_path
        cached = self.geotiff_cache.get(path, self.workspace.selected_resolution_percent())
        if cached is not None:
            self._apply_geotiff_info(cached, log_message=f"Updated GeoTIFF display scale: {cached.file_name}")
            self.statusBar().showMessage("GeoTIFF display scale updated from cache.", 3500)
            return

        self._loading_dialog = LoadingDialog(
            file_name=path.name,
            theme_name=self._current_theme_name,
            parent=self,
            title="Updating GeoTIFF Display",
        )
        dialog = self._loading_dialog
        dialog.cancel_requested.connect(
            lambda: self._cancel_geotiff_worker("GeoTIFF display update cancelled.")
        )
        self._active_worker = GeoTiffWorker(
            path,
            preview_scale=self.workspace.selected_resolution_scale(),
            parent=self,
        )
        self._active_worker.progress.connect(self._on_load_progress)
        self._active_worker.finished.connect(self._on_load_finished)
        self._active_worker.failed.connect(self._on_load_failed)
        self._active_worker.start()
        self.workspace.append_log(
            f"Updating GeoTIFF display scale to {self.workspace.selected_resolution_percent()}%: {path}"
        )
        dialog.exec()
        if dialog.was_cancelled and self._active_worker is not None:
            self._active_worker.requestInterruption()

    def _require_project(self) -> ProjectBundle | None:
        if self.current_bundle is not None:
            output_dir = self._ensure_project_output_dir(
                self.current_bundle.project.id,
                self.current_bundle.project.name,
                self.current_bundle.project.output_dir,
            )
            self._set_project_geotiff_cache(output_dir)
            self.current_bundle = self.repo.get_bundle(self.current_bundle.project.id)
            return self.current_bundle
        QMessageBox.information(
            self,
            "Project required",
            "Create or open a project before importing a GeoTIFF or running AI mapping.",
        )
        self.show_dashboard()
        return None

    def _ensure_all_project_output_dirs(self) -> None:
        for project in self.repo.list_projects():
            self._ensure_project_output_dir(project.id, project.name, project.output_dir)

    def _ensure_project_output_dir(
        self,
        project_id: str,
        project_name: str,
        existing_output_dir: str = "",
    ) -> Path:
        if existing_output_dir:
            output_dir = Path(existing_output_dir)
            if self.output_manager.is_managed_path(output_dir):
                output_dir = self.output_manager.ensure_project_dir_at(output_dir)
            else:
                output_dir = self.output_manager.ensure_project_dir(project_id, project_name)
        else:
            output_dir = self.output_manager.ensure_project_dir(project_id, project_name)
        if str(output_dir) != existing_output_dir:
            self._save_project_output_dir(project_id, output_dir)
        return output_dir

    def _current_output_dir(self) -> Path | None:
        bundle = self._require_project()
        if bundle is None:
            return None
        output_dir = self._ensure_project_output_dir(
            bundle.project.id,
            bundle.project.name,
            bundle.project.output_dir,
        )
        self._set_project_geotiff_cache(output_dir)
        return output_dir

    def _set_project_geotiff_cache(self, project_output_dir: str | Path) -> None:
        cache_dir = Path(project_output_dir) / "cache" / "geotiff"
        self.geotiff_cache.set_cache_dir(cache_dir)
        self._update_settings_page()

    def _log_missing_paths(self, bundle: ProjectBundle) -> None:
        missing = []
        for asset in bundle.assets:
            if not asset.exists:
                missing.append(asset.path)
        for model in bundle.models:
            if not model.exists:
                missing.append(model.path)
        self._ensure_project_output_dir(bundle.project.id, bundle.project.name, bundle.project.output_dir)
        if missing:
            self.workspace.append_log(f"Missing local paths detected: {len(missing)}")
            for path in missing[:8]:
                self.workspace.append_log(f"  - {path}")

    def _recover_latest_analysis(self, bundle: ProjectBundle) -> None:
        latest = next((result for result in bundle.results if result.exists), None)
        if latest is None:
            return
        try:
            payload = json.loads(Path(latest.json_path).read_text(encoding="utf-8"))
            records = [_record_from_payload(item) for item in payload.get("records", [])]
            counts = payload.get("counts", {})
            warnings = payload.get("warnings", [])
            self.latest_mapping_result = MappingResult(
                records=records,
                counts=counts,
                json_path=latest.json_path,
                csv_path=latest.csv_path,
                xlsx_path=str(latest.summary.get("xlsx_path", "")),
                processed_images=int(latest.summary.get("processed_images", 0) or 0),
                skipped_images=int(latest.summary.get("skipped_images", 0) or 0),
                warnings=list(warnings),
                qa_json_path=str(latest.summary.get("qa_json_path", "")),
            )
            self.workspace.update_counts(counts)
            self._send_detections_to_map(self.latest_mapping_result)
            self.workspace.append_log(f"Recovered latest analysis: {Path(latest.json_path).name}")
        except Exception as exc:
            self.workspace.append_log(f"Session recovery skipped: {exc}")

    # ------------------------------------------------------------------
    # Asset management
    # ------------------------------------------------------------------

    def open_geotiff(self) -> None:
        bundle = self._require_project()
        if bundle is None:
            return
        if self._active_worker is not None and self._active_worker.isRunning():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open stitched drone GeoTIFF",
            bundle.geotiff_path or "",
            "GeoTIFF files (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return

        cached = self.geotiff_cache.get(path, self.workspace.selected_resolution_percent())
        if cached is not None:
            self._apply_geotiff_info(cached, log_message=f"Reused cached GeoTIFF: {cached.file_name}")
            self.statusBar().showMessage("GeoTIFF restored from session cache.", 3500)
            return

        file_name = Path(path).name
        self._loading_dialog = LoadingDialog(
            file_name=file_name,
            theme_name=self._current_theme_name,
            parent=self,
            title="Importing GeoTIFF",
        )
        dialog = self._loading_dialog
        dialog.cancel_requested.connect(
            lambda: self._cancel_geotiff_worker("GeoTIFF import cancelled.")
        )

        self._active_worker = GeoTiffWorker(
            path,
            preview_scale=self.workspace.selected_resolution_scale(),
            parent=self,
        )
        self._active_worker.progress.connect(self._on_load_progress)
        self._active_worker.finished.connect(self._on_load_finished)
        self._active_worker.failed.connect(self._on_load_failed)
        self._active_worker.start()
        self.workspace.append_log(f"Started GeoTIFF import: {path}")
        dialog.exec()
        if dialog.was_cancelled and self._active_worker is not None:
            self._active_worker.requestInterruption()

    def _on_load_progress(self, percent: int, message: str) -> None:
        if self.sender() is not self._active_worker:
            return
        if self._loading_dialog is not None:
            self._loading_dialog.set_progress(percent, message)

    def _on_load_finished(self, info_obj: object) -> None:
        if self.sender() is not self._active_worker:
            return
        if self._loading_dialog is not None:
            self._loading_dialog.finish()
        if self._active_worker is not None and self._active_worker.isInterruptionRequested():
            self._active_worker = None
            self._loading_dialog = None
            self.statusBar().showMessage("GeoTIFF import cancelled.", 3000)
            return

        self._active_worker = None
        self._loading_dialog = None
        info: GeoTiffInfo = info_obj  # type: ignore[assignment]
        self.geotiff_cache.put(info, self.workspace.selected_resolution_percent())
        self._apply_geotiff_info(info, log_message=f"GeoTIFF overlay loaded: {info.file_name}")
        self.statusBar().showMessage("GeoTIFF overlay loaded and autosaved to project.", 5000)

    def _on_load_failed(self, error_message: str) -> None:
        if self.sender() is not self._active_worker:
            return
        if self._loading_dialog is not None:
            self._loading_dialog.reject()
        self._active_worker = None
        self._loading_dialog = None
        if "cancelled" in error_message.lower():
            self.statusBar().showMessage("GeoTIFF import cancelled.", 3000)
            return
        QMessageBox.critical(self, "Unable to load GeoTIFF", error_message)
        self.workspace.append_log(f"GeoTIFF import failed: {error_message}")
        self.statusBar().showMessage("GeoTIFF import failed.", 5000)

    def _cancel_geotiff_worker(self, message: str) -> None:
        worker = self._active_worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
        self._active_worker = None
        if self._loading_dialog is not None:
            self._loading_dialog = None
        if self._project_open_dialog is not None:
            self._project_open_dialog = None
        self.workspace.append_log(message)
        self.statusBar().showMessage(message, 3000)

    def import_csv_file(self) -> None:
        bundle = self._require_project()
        if bundle is None:
            return
        
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Coordinates CSV",
            "",
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return

        try:
            records = parse_csv_coordinates(path)
            self.imported_csv_records = records
            self.workspace.set_draw_button_enabled(True)
            self.workspace.append_log(f"Successfully imported {len(records)} coordinate(s) from {Path(path).name}")
            self.statusBar().showMessage(f"Imported {len(records)} coordinate(s) from CSV.", 5000)
        except CsvImportError as exc:
            self.imported_csv_records = []
            self.workspace.set_draw_button_enabled(False)
            self.workspace.append_log(f"CSV import failed: {exc}")
            QMessageBox.critical(self, "CSV Import Error", str(exc))
            self.statusBar().showMessage("CSV import failed.", 5000)

    def draw_csv_markers(self) -> None:
        if not self.imported_csv_records:
            return

        records = []
        counts = {"black_sigatoka": 0, "panama": 0}
        
        for idx, item in enumerate(self.imported_csv_records):
            name = item["name"]
            name_lower = name.lower()
            if "panama" in name_lower or "fusarium" in name_lower or "wilt" in name_lower:
                class_name = "panama"
            elif "sigatoka" in name_lower or "black" in name_lower:
                class_name = "black_sigatoka"
            else:
                class_name = "black_sigatoka"

            r = DetectionRecord(
                id=f"csv-import-{idx}",
                image_name=name,
                class_name=class_name,
                latitude=item["latitude"],
                longitude=item["longitude"],
                confidence=1.0,
                pixel_x=0.0,
                pixel_y=0.0,
                health="diseased",
                source="csv",
                layer_keys=[class_name],
            )
            records.append(r)
            counts[class_name] = counts.get(class_name, 0) + 1

        result = MappingResult(
            records=records,
            counts=counts,
            json_path="",
            csv_path="",
            xlsx_path="",
            processed_images=0,
            skipped_images=0,
            warnings=[],
        )
        self.workspace.update_counts(counts)
        self._send_detections_to_map(result)
        self.workspace.append_log(
            f"Plotted {len(records)} marker(s) on map "
            f"(Black Sigatoka: {counts.get('black_sigatoka', 0)}, Panama: {counts.get('panama', 0)})"
        )
        self.statusBar().showMessage("CSV coordinates drawn on map.", 5000)

    def open_project_output_folder(self) -> None:
        output_dir = self._current_output_dir()
        if output_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))
        self.statusBar().showMessage("Project output folder opened.", 2500)

    def refresh_project_outputs(self) -> None:
        if self.current_bundle is None:
            self.workspace.set_outputs([])
            return
        output_dir = self._ensure_project_output_dir(
            self.current_bundle.project.id,
            self.current_bundle.project.name,
            self.current_bundle.project.output_dir,
        )
        self.workspace.set_outputs(self.output_manager.list_outputs_at(output_dir))
        self._update_settings_page()

    def _sync_current_project_integrity(self, notify: bool = False) -> bool:
        if self.current_bundle is None:
            if hasattr(self, "settings"):
                self.settings.set_sync_status("No project open")
            return False
        if self._active_mapping_worker is not None and self._active_mapping_worker.isRunning():
            return False

        project_id = self.current_bundle.project.id
        try:
            bundle = self.repo.get_bundle(project_id)
        except KeyError:
            return False

        removed_results: list[str] = []
        for result in bundle.results:
            missing_paths = [path for path in self._result_paths(result) if not path.exists()]
            if not missing_paths:
                continue
            removed_results.append(Path(result.json_path).name or f"result {result.id}")
            self.repo.remove_analysis_result(project_id, result.id)

        removed_assets: list[str] = []
        geotiff_removed = False
        for asset in bundle.assets:
            if asset.asset_type == "output_dir":
                continue
            if asset.exists:
                continue
            removed_assets.append(asset.display_name)
            if asset.asset_type == "geotiff":
                geotiff_removed = True
            self.repo.remove_asset(project_id, asset.asset_type)

        removed_models: list[str] = []
        for model in bundle.models:
            if model.exists:
                continue
            removed_models.append(model.display_name)
            self.repo.remove_model(project_id, model.role)

        if not removed_results and not removed_assets and not removed_models:
            self.refresh_project_outputs()
            self.settings.set_sync_status("Watching current project outputs")
            return False

        self.current_bundle = self.repo.get_bundle(project_id)
        self.workspace.set_project(self.current_bundle)
        self.refresh_project_outputs()
        self.refresh_dashboard()

        if removed_results:
            self._clear_mapping_result_view()
            self._recover_latest_analysis(self.current_bundle)

        if geotiff_removed:
            self.current_geotiff = None
            self.workspace.clear_geotiff_metadata()
            if self.map_ready:
                self._run_js("clearOverlay();")

        parts = []
        if removed_results:
            parts.append(f"{len(removed_results)} stale analysis result(s)")
        if removed_assets:
            parts.append(f"{len(removed_assets)} missing asset link(s)")
        if removed_models:
            parts.append(f"{len(removed_models)} missing model link(s)")
        message = "Project synchronized: removed " + " and ".join(parts) + "."
        self.workspace.append_log(message)
        self.settings.set_sync_status(message)
        self.statusBar().showMessage(message, 6500 if notify else 4500)
        return True

    def _clear_mapping_result_view(self) -> None:
        self.latest_mapping_result = None
        self.workspace.update_counts({})
        if self.map_ready:
            self._run_js("clearDetections(); setScanBox(0, 0, 0, 0);")

    @staticmethod
    def _result_paths(result) -> list[Path]:
        paths = []
        for raw in [
            result.json_path,
            result.csv_path,
            result.summary.get("xlsx_path", ""),
        ]:
            if raw:
                paths.append(Path(raw))
        return paths

    def open_output_file(self, path: str) -> None:
        if not self._validate_managed_output(path):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def reveal_output_file(self, path: str) -> None:
        if not self._validate_managed_output(path):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    def export_output_file(self, path: str) -> None:
        if not self._validate_managed_output(path):
            return
        source = Path(path)
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Export generated output",
            source.name,
            "All files (*.*)",
        )
        if not target:
            return
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", f"Unable to export the selected file.\n\n{exc}")
            return
        self.statusBar().showMessage("Output file exported.", 3000)

    def delete_output_file(self, path: str) -> None:
        if not self._validate_managed_output(path):
            return
        file_path = Path(path)
        answer = QMessageBox.question(
            self,
            "Delete Output File",
            f"Delete this generated output?\n\n{file_path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            file_path.unlink()
        except OSError as exc:
            QMessageBox.warning(self, "Delete failed", f"Unable to delete the output file.\n\n{exc}")
            return
        synced = self._sync_current_project_integrity(notify=True)
        self.refresh_project_outputs()
        if not synced:
            self.statusBar().showMessage("Output file deleted.", 2500)

    def _validate_managed_output(self, path: str | Path) -> bool:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            QMessageBox.warning(self, "Output not found", "The selected output file no longer exists.")
            self.refresh_project_outputs()
            return False
        if not self.output_manager.is_managed_path(file_path):
            QMessageBox.warning(self, "Blocked", "Only system-managed project outputs can be opened here.")
            return False
        return True

    # ------------------------------------------------------------------
    # AI mapping
    # ------------------------------------------------------------------

    def run_ai_mapping(self) -> None:
        bundle = self._require_project()
        if bundle is None:
            return
        if self._active_mapping_worker is not None and self._active_mapping_worker.isRunning():
            return

        bundle = self._refresh_saved_models_for_mapping(bundle)
        if bundle is None:
            return

        leaf = bundle.first_model("leaf")
        disease = bundle.first_model("disease")
        if leaf is None or disease is None or not leaf.exists or not disease.exists:
            QMessageBox.warning(
                self,
                "Missing AI models",
                "Link valid leaf and disease YOLO model files before running AI mapping.",
            )
            return

        if self.current_geotiff is None:
            QMessageBox.warning(
                self,
                "GeoTIFF required",
                "Import the project GeoTIFF into the map before scanning it.",
            )
            return
        source_path = self.current_geotiff.file_path
        target_name = self.current_geotiff.file_name

        project_output_dir = self._ensure_project_output_dir(
            bundle.project.id,
            bundle.project.name,
            bundle.project.output_dir,
        )
        output_dir = self.output_manager.ensure_mapping_run_dir_at(project_output_dir)
        bundle = self._reload_current_bundle() or bundle
        self.repo.set_config(
            bundle.project.id,
            "last_mapping",
            {
                "source": "geotiff",
                "output_dir": str(output_dir),
            },
        )

        self._mapping_dialog = LoadingDialog(
            target_name,
            theme_name=self._current_theme_name,
            parent=self,
            title="Running AI Mapping",
        )
        dialog = self._mapping_dialog
        device = self._selected_inference_device()
        device_label = "GPU" if device != "cpu" else "CPU"
        mapping_options = self.workspace.mapping_options()

        self._active_mapping_worker = AiGeotiffMappingWorker(
            source_path,
            leaf.path,
            disease.path,
            output_dir=output_dir,
            device=device,
            mapping_options=mapping_options,
            parent=self,
        )
        self._active_mapping_worker.scan_box.connect(self._on_scan_box)
        self.fit_overlay()
        self.workspace.append_log(f"GeoTIFF AI analysis will use native raster resolution on {device_label}.")
        self.workspace.append_log(
            "AI thresholds: "
            f"leaf {int(float(mapping_options['leaf_confidence']) * 100)}%, "
            f"disease {int(float(mapping_options['disease_confidence']) * 100)}%."
        )
        if mapping_options.get("qa_enabled"):
            self.workspace.append_log("QA diagnostics enabled for this mapping run.")
        self._active_mapping_worker.progress.connect(self._on_mapping_progress)
        self._active_mapping_worker.finished.connect(self._on_mapping_finished)
        self._active_mapping_worker.failed.connect(self._on_mapping_failed)
        self._active_mapping_worker.start()
        self.workspace.append_log(f"AI mapping started on {device_label}. Output: {output_dir}")
        dialog.exec()
        if dialog.was_cancelled and self._active_mapping_worker is not None:
            self._active_mapping_worker.requestInterruption()

    def _save_project_output_dir(self, project_id: str, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        self.repo.update_project(project_id, output_dir=str(output_path))
        self.repo.upsert_asset(project_id, "output_dir", str(output_path), label="Output folder")

    def _on_mapping_progress(self, percent: int, message: str) -> None:
        if self.sender() is not self._active_mapping_worker:
            return
        if self._mapping_dialog is not None:
            self._mapping_dialog.set_progress(percent, message)

    def _on_mapping_finished(self, result_obj: object) -> None:
        if self.sender() is not self._active_mapping_worker:
            return
        if self._mapping_dialog is not None:
            self._mapping_dialog.finish()
        if self._active_mapping_worker is not None and self._active_mapping_worker.isInterruptionRequested():
            self._active_mapping_worker = None
            self._mapping_dialog = None
            self.statusBar().showMessage("AI mapping cancelled.", 3000)
            return

        self._active_mapping_worker = None
        self._mapping_dialog = None
        result: MappingResult = result_obj  # type: ignore[assignment]
        self.latest_mapping_result = result
        self.workspace.update_counts(result.counts)
        self._send_detections_to_map(result)

        if self.current_bundle is not None:
            self.repo.save_analysis_result(
                self.current_bundle.project.id,
                "banana_disease_mapping",
                result.json_path,
                result.csv_path,
                {
                    "counts": result.counts,
                    "processed_images": result.processed_images,
                    "skipped_images": result.skipped_images,
                    "warning_count": len(result.warnings),
                    "xlsx_path": result.xlsx_path,
                    "qa_json_path": result.qa_json_path,
                },
            )
            self._reload_current_bundle()
            self.refresh_project_outputs()
        skipped = f", {result.skipped_images} skipped" if result.skipped_images else ""
        self.workspace.append_log(
            f"AI mapping complete: {result.processed_images} processed{skipped}. "
            f"JSON: {result.json_path} | Excel: {result.xlsx_path}"
        )
        if result.qa_json_path:
            self.workspace.append_log(f"QA diagnostics: {result.qa_json_path}")
        self.statusBar().showMessage("AI mapping complete and saved to project.", 7000)
        if result.warnings:
            QMessageBox.information(
                self,
                "AI mapping finished with notes",
                "Mapping completed, but some images or tiles produced warnings. "
                f"See {Path(result.json_path).name} for details.",
            )

    def _on_mapping_failed(self, error_message: str) -> None:
        if self.sender() is not self._active_mapping_worker:
            return
        if self._mapping_dialog is not None:
            self._mapping_dialog.reject()
        self._active_mapping_worker = None
        self._mapping_dialog = None
        QMessageBox.critical(self, "AI mapping failed", error_message)
        self.workspace.append_log(f"AI mapping failed: {error_message}")
        self.statusBar().showMessage("AI mapping failed.", 5000)

    def _on_scan_box(self, lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> None:
        self._run_js(f"setScanBox({lat_min}, {lon_min}, {lat_max}, {lon_max});")

    # ------------------------------------------------------------------
    # Map integration
    # ------------------------------------------------------------------

    def _clear_map_view(self) -> None:
        if not self.map_ready:
            return
        self._run_js("clearOverlay(); clearDetections(); setScanBox(0, 0, 0, 0);")

    def _handle_map_ready(self) -> None:
        self.map_ready = True
        self._push_theme_to_map()
        self._push_base_map_to_map()
        if self.pending_geotiff:
            info = self.pending_geotiff
            self.pending_geotiff = None
            self._send_overlay_to_map(info)
        if self.pending_mapping:
            result = self.pending_mapping
            self.pending_mapping = None
            self._send_detections_to_map(result)
        self.statusBar().showMessage("Map engine ready.", 2500)

    def _send_overlay_to_map(self, info: GeoTiffInfo) -> None:
        if not self.map_ready:
            self.pending_geotiff = info
            return
        payload = {
            "fileName": info.file_name,
            "imageUrl": QUrl.fromLocalFile(str(info.preview_path)).toString(),
            "bounds": {
                "west": info.west,
                "south": info.south,
                "east": info.east,
                "north": info.north,
            },
        }
        if info.tile_dir is not None and info.tile_levels:
            base_url = QUrl.fromLocalFile(str(info.tile_dir)).toString()
            if not base_url.endswith("/"):
                base_url += "/"
            payload["tiles"] = {
                "baseUrl": base_url,
                "tileSize": info.tile_size,
                "levels": [
                    {
                        "index": level.index,
                        "width": level.width,
                        "height": level.height,
                        "cols": level.cols,
                        "rows": level.rows,
                        "scale": level.scale,
                    }
                    for level in info.tile_levels
                ],
            }
        self._run_js(f"setOverlay({json.dumps(payload)});")

    def _send_detections_to_map(self, result: MappingResult) -> None:
        if not self.map_ready:
            self.pending_mapping = result
            return
        payload = result.to_map_payload()
        visibility = {
            "full_leaf": False,
            "healthy_leaf": True,
            "diseased_leaf": True,
        }
        visibility.update(
            {
                key: checkbox.isChecked()
                for key, checkbox in self.workspace.layer_toggles.items()
            }
        )
        payload["visibility"] = visibility
        payload["styles"] = {
            key: checkbox.isChecked()
            for key, checkbox in self.workspace.style_toggles.items()
        }
        payload["opacity"] = self.workspace.ai_opacity_slider.value() / 100
        self._run_js(f"setDetectionData({json.dumps(payload)});")

    def fit_overlay(self) -> None:
        self._run_js("if (typeof fitToOverlay === 'function') fitToOverlay();")

    def _schedule_overlay_fit_to_screen(self) -> None:
        if self.current_geotiff is None:
            return
        for delay in (0, 150, 400, 800, 1400, 2200):
            QTimer.singleShot(
                delay,
                lambda: self._run_js("if (typeof fitToOverlay === 'function') fitToOverlay(false);"),
            )

    def reset_view(self) -> None:
        self._run_js("if (typeof resetView === 'function') resetView();")

    def _set_overlay_visible(self, checked: bool) -> None:
        self._run_js(f"setOverlayVisible({str(checked).lower()});")

    def _set_overlay_opacity(self, value: int) -> None:
        self._run_js(f"setOverlayOpacity({value / 100:.2f});")

    def _set_detection_layer_visible(self, layer_key: str, checked: bool) -> None:
        self._run_js(
            f"setDetectionLayerVisible({json.dumps(layer_key)}, {str(checked).lower()});"
        )

    def _set_detection_style(self, style_key: str, checked: bool) -> None:
        self._run_js(f"setDetectionStyle({json.dumps(style_key)}, {str(checked).lower()});")

    def _set_detection_opacity(self, value: int) -> None:
        self._run_js(f"setDetectionOpacity({value / 100:.2f});")

    def _set_base_map(self, base_map: str) -> None:
        if base_map not in BASE_MAPS:
            return
        if base_map.startswith("google_") and not self._google_maps_api_key:
            self.workspace.set_base_map(self._current_base_map)
            self.statusBar().showMessage("Add your Google Maps API key in Settings to enable Google Satellite.", 6000)
            QMessageBox.information(
                self,
                "Google Maps API key required",
                "Google Satellite needs your own Google Maps Platform API key.\n\n"
                "Open Settings, paste your key under Map Services, then select Google Satellite again.",
            )
            return
        self._current_base_map = base_map
        self.repo.set_preference("base_map", base_map)
        self._push_base_map_to_map()
        labels = {
            "osm": "OpenStreetMap",
            "google_satellite": "Google Satellite",
        }
        label = labels.get(base_map, base_map)
        self.statusBar().showMessage(f"Basemap set to {label}.", 2500)

    def _push_base_map_to_map(self) -> None:
        if hasattr(self, "workspace"):
            self.workspace.set_base_map(self._current_base_map)
        if not self.map_ready:
            return
        payload = {
            "name": self._current_base_map,
            "googleApiKey": self._google_maps_api_key if self._current_base_map.startswith("google_") else "",
        }
        self._run_js(f"setBaseMap({json.dumps(payload)});")

    def _set_google_maps_api_key(self, api_key: str) -> None:
        self._google_maps_api_key = api_key.strip()
        _save_google_maps_api_key(self._google_maps_secret_path(), self._google_maps_api_key)
        self.settings.set_google_maps_api_key_configured(bool(self._google_maps_api_key))
        if self._current_base_map.startswith("google_") and not self._google_maps_api_key:
            self._current_base_map = "osm"
            self.repo.set_preference("base_map", "osm")
        self._push_base_map_to_map()
        message = "Google Maps API key saved locally." if self._google_maps_api_key else "Google Maps API key cleared."
        self.statusBar().showMessage(message, 4000)

    def _google_maps_secret_path(self) -> Path:
        return self.repo.db_path.parent / "google_maps.env"

    def _run_js(self, script: str) -> None:
        self.workspace.map_view.page().runJavaScript(script)

    def _update_coordinates(self, latitude: float, longitude: float) -> None:
        self.workspace.set_coordinates(latitude, longitude)

    def _update_zoom(self, zoom: int) -> None:
        self.workspace.set_zoom(zoom)

    # ------------------------------------------------------------------
    # Secondary navigation
    # ------------------------------------------------------------------

    def open_model_manager(self) -> None:
        preferred = self._preferred_model_paths()
        dialog = PreferredModelsDialog(
            preferred.get("leaf", ""),
            preferred.get("disease", ""),
            parent=self,
        )
        dialog.setStyleSheet(generate_stylesheet(THEMES[self._current_theme_name]))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.repo.set_preference("preferred_leaf_model_path", dialog.leaf_model_path)
        self.repo.set_preference("preferred_disease_model_path", dialog.disease_model_path)

        if self.current_bundle is not None:
            self._apply_preferred_models_to_project(self.current_bundle.project.id)
            self._reload_current_bundle()
            self.workspace.append_log("Preferred AI models saved and applied to this project.")

        self.show_dashboard()
        self.statusBar().showMessage("Preferred AI models saved.", 3500)

    def show_settings(self) -> None:
        self.refresh_hardware_status()
        self._update_settings_page()
        self.stack.setCurrentWidget(self.settings)
        self.statusBar().showMessage("Settings ready.", 2500)

    def clear_project_cache(self) -> None:
        cache_path = self.geotiff_cache.cache_dir
        if cache_path is None:
            self.statusBar().showMessage("Open a project to activate its GeoTIFF cache.", 3500)
            return

        confirm = QMessageBox.question(
            self,
            "Clear Project Cache",
            "Clear cached GeoTIFF previews for this project?\n\nOriginal GeoTIFFs, project records, models, and generated outputs will not be deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.geotiff_cache.clear()
        self.refresh_project_outputs()
        self._update_settings_page()
        self.statusBar().showMessage("Project GeoTIFF cache cleared.", 4000)

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def _apply_theme(self, theme_name: str) -> None:
        if theme_name not in THEMES:
            return
        self._current_theme_name = theme_name
        app = QApplication.instance()
        if app:
            app.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(generate_stylesheet(THEMES[theme_name]))
        self.repo.set_preference("theme", theme_name)
        self._push_theme_to_map()
        for action in self.menuBar().findChildren(QAction):
            if action.text() in THEMES:
                action.setChecked(action.text() == theme_name)

    def _push_theme_to_map(self) -> None:
        if not self.map_ready:
            return
        theme = THEMES.get(self._current_theme_name)
        if not theme:
            return
        payload = {
            "mapBg": theme.map_bg,
            "ctrlBg": theme.leaflet_ctrl_bg,
            "ctrlFg": theme.leaflet_ctrl_fg,
            "attrBg": theme.leaflet_attr_bg,
            "attrFg": theme.leaflet_attr_fg,
            "tooltipBg": theme.tooltip_bg,
            "tooltipBorder": theme.tooltip_border,
            "tooltipFg": theme.tooltip_fg,
            "extentColor": theme.extent_color,
        }
        self._run_js(f"applyTheme({json.dumps(payload)});")

    @staticmethod
    def _sync_view_action(action: QAction, checked: bool) -> None:
        if action.isChecked() == checked:
            return
        action.blockSignals(True)
        action.setChecked(checked)
        action.blockSignals(False)


def _record_from_payload(payload: dict) -> DetectionRecord:
    allowed = set(DetectionRecord.__dataclass_fields__.keys())
    clean = {key: value for key, value in payload.items() if key in allowed}
    if clean.get("layer_keys") is None:
        clean["layer_keys"] = []
    return DetectionRecord(**clean)


def _load_google_maps_api_key(secret_path: Path) -> str:
    for name in ("GOOGLE_MAPS_API_KEY", "MUSA_GOOGLE_MAPS_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if not secret_path.exists():
        return ""
    try:
        for line in secret_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() in {"GOOGLE_MAPS_API_KEY", "MUSA_GOOGLE_MAPS_API_KEY"}:
                return value.strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""


def _save_google_maps_api_key(secret_path: Path, api_key: str) -> None:
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if not api_key:
        secret_path.unlink(missing_ok=True)
        return
    secret_path.write_text(f"GOOGLE_MAPS_API_KEY={api_key}\n", encoding="utf-8")


def run() -> None:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    _configure_web_engine_gpu()
    app = QApplication(sys.argv)
    app.setApplicationName("Musa AI")
    app.setOrganizationName("Drone GIS")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
