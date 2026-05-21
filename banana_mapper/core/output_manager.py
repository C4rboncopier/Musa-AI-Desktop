from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


OUTPUT_ROOT_NAME = "SystemOutput"
PROJECT_SUBDIRS = ("ai_analysis", "logs", "exports", "cache", "cache/geotiff", "reports")


@dataclass(frozen=True)
class OutputFile:
    path: Path
    relative_path: str
    file_type: str
    created_at: str
    size_label: str
    status: str = "Ready"


class ProjectOutputManager:
    """Owns system-managed output folders for projects."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            root_dir = os.environ.get("MUSA_AI_OUTPUT_HOME", Path.cwd() / OUTPUT_ROOT_NAME)
        self.root_dir = Path(root_dir)
        self.ensure_system_root()

    def ensure_system_root(self) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._validate_writable(self.root_dir)
        return self.root_dir

    def ensure_project_dir(self, project_id: str, project_name: str) -> Path:
        project_dir = self.project_dir(project_id, project_name)
        return self.ensure_project_dir_at(project_dir)

    def ensure_project_dir_at(self, project_dir: str | Path) -> Path:
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        for name in PROJECT_SUBDIRS:
            (project_dir / name).mkdir(parents=True, exist_ok=True)
        self._validate_writable(project_dir)
        return project_dir

    def project_dir(self, project_id: str, project_name: str) -> Path:
        safe_name = safe_filename(project_name) or "Project"
        suffix = project_id.replace("-", "")[:8] if project_id else "managed"
        return self.root_dir / f"{safe_name}_{suffix}"

    def ensure_mapping_run_dir(self, project_id: str, project_name: str) -> Path:
        project_dir = self.ensure_project_dir(project_id, project_name)
        return self.ensure_mapping_run_dir_at(project_dir)

    def ensure_mapping_run_dir_at(self, project_dir: str | Path) -> Path:
        project_dir = self.ensure_project_dir_at(project_dir)
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = project_dir / "ai_analysis" / base
        counter = 2
        while run_dir.exists():
            run_dir = project_dir / "ai_analysis" / f"{base}_{counter}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        self._validate_writable(run_dir)
        return run_dir

    def list_outputs(self, project_id: str, project_name: str) -> list[OutputFile]:
        project_dir = self.ensure_project_dir(project_id, project_name)
        return self.list_outputs_at(project_dir)

    def list_outputs_at(self, project_dir: str | Path) -> list[OutputFile]:
        project_dir = self.ensure_project_dir_at(project_dir)
        records: list[OutputFile] = []
        for path in project_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(project_dir)
            if relative.parts and relative.parts[0] == "cache":
                continue
            if relative.parts and relative.parts[0] == "coordinate_qa":
                continue
            if (
                len(relative.parts) >= 2
                and relative.parts[0] == "qa_diagnostics"
                and relative.parts[1] == "disease_crops"
            ):
                continue
            stat = path.stat()
            records.append(
                OutputFile(
                    path=path,
                    relative_path=relative.as_posix(),
                    file_type=_file_type(path),
                    created_at=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    size_label=_size_label(stat.st_size),
                )
            )
        return sorted(records, key=lambda item: item.path.stat().st_mtime, reverse=True)

    def is_managed_path(self, path: str | Path) -> bool:
        try:
            Path(path).resolve().relative_to(self.root_dir.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _validate_writable(path: Path) -> None:
        probe = path / ".write_test"
        try:
            probe.write_text("ok", encoding="utf-8")
        finally:
            if probe.exists():
                probe.unlink()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("._ ")
    return cleaned[:80]


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "json":
        return "JSON"
    if suffix == "csv":
        return "CSV"
    if suffix == "xlsx":
        return "Excel"
    if suffix in {"log", "txt"}:
        return "Log"
    if suffix in {"png", "jpg", "jpeg", "tif", "tiff"}:
        return "Image"
    return suffix.upper() if suffix else "File"


def _size_label(size: int) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
