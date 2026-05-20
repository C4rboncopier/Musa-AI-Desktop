from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

from .models import (
    AiModelRecord,
    AnalysisResultRecord,
    ProjectAsset,
    ProjectBundle,
    ProjectRecord,
    utc_now_iso,
)


DB_FILENAME = "banana_mapper.sqlite3"


def default_database_path() -> Path:
    """Return the local application database path.

    The app stores metadata in a small SQLite database beside the application
    workspace by default. The location can be overridden for deployment or
    tests with ``BANANA_MAPPER_HOME``.
    """

    import os

    root = Path(os.environ.get("BANANA_MAPPER_HOME", Path.cwd() / ".banana_mapper"))
    root.mkdir(parents=True, exist_ok=True)
    return root / DB_FILENAME


class ProjectRepository:
    """SQLite-backed project metadata store.

    Only file paths, settings, and analysis metadata are persisted here. Drone
    imagery, GeoTIFFs, model weights, and exports remain on the local filesystem.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_database_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connect().close()
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def migrate(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    output_dir TEXT NOT NULL DEFAULT '',
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    last_opened_at TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS project_assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    UNIQUE(project_id, asset_type),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ai_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    path TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    UNIQUE(project_id, role),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    result_type TEXT NOT NULL,
                    json_path TEXT NOT NULL,
                    csv_path TEXT NOT NULL,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS processing_configs (
                    project_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, key),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS app_preferences (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_projects_recent
                    ON projects(last_opened_at DESC, modified_at DESC);
                CREATE INDEX IF NOT EXISTS idx_assets_project
                    ON project_assets(project_id);
                CREATE INDEX IF NOT EXISTS idx_models_project
                    ON ai_models(project_id);
                CREATE INDEX IF NOT EXISTS idx_results_project
                    ON analysis_results(project_id, created_at DESC);
                """
            )

    def create_project(
        self,
        name: str,
        description: str = "",
        output_dir: str = "",
        settings: dict[str, Any] | None = None,
    ) -> ProjectRecord:
        now = utc_now_iso()
        project_id = str(uuid.uuid4())
        clean_name = name.strip() or "Untitled Project"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO projects (
                    id, name, description, output_dir, settings_json,
                    created_at, modified_at, last_opened_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    clean_name,
                    description.strip(),
                    output_dir.strip(),
                    _json(settings or {}),
                    now,
                    now,
                    now,
                ),
            )
        return self.get_project(project_id)

    def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        output_dir: str | None = None,
        settings: dict[str, Any] | None = None,
        touch: bool = True,
    ) -> None:
        current = self.get_project(project_id)
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE projects
                SET name = ?, description = ?, output_dir = ?,
                    settings_json = ?, modified_at = ?
                WHERE id = ?
                """,
                (
                    current.name if name is None else name.strip(),
                    current.description if description is None else description.strip(),
                    current.output_dir if output_dir is None else output_dir.strip(),
                    _json(current.settings if settings is None else settings),
                    now if touch else current.modified_at,
                    project_id,
                ),
            )

    def delete_project(self, project_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    def touch_project(self, project_id: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE projects SET last_opened_at = ?, modified_at = ? WHERE id = ?",
                (now, now, project_id),
            )

    def list_projects(self, search: str = "") -> list[ProjectRecord]:
        pattern = f"%{search.strip()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM projects
                WHERE ? = '%%'
                   OR name LIKE ?
                   OR description LIKE ?
                ORDER BY COALESCE(NULLIF(last_opened_at, ''), modified_at) DESC
                """,
                (pattern, pattern, pattern),
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def recent_projects(self, limit: int = 6) -> list[ProjectRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM projects
                ORDER BY COALESCE(NULLIF(last_opened_at, ''), modified_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def get_project(self, project_id: str) -> ProjectRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"Project not found: {project_id}")
        return self._project_from_row(row)

    def get_bundle(self, project_id: str) -> ProjectBundle:
        project = self.get_project(project_id)
        return ProjectBundle(
            project=project,
            assets=self.list_assets(project_id),
            models=self.list_models(project_id),
            results=self.list_results(project_id),
        )

    def upsert_asset(
        self,
        project_id: str,
        asset_type: str,
        path: str | Path,
        *,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ProjectAsset:
        now = utc_now_iso()
        norm_path = str(Path(path))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO project_assets (
                    project_id, asset_type, path, label, metadata_json,
                    created_at, modified_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, asset_type) DO UPDATE SET
                    path = excluded.path,
                    label = excluded.label,
                    metadata_json = excluded.metadata_json,
                    modified_at = excluded.modified_at
                """,
                (
                    project_id,
                    asset_type,
                    norm_path,
                    label or Path(norm_path).name,
                    _json(metadata or {}),
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE projects SET modified_at = ? WHERE id = ?",
                (now, project_id),
            )
        return next(asset for asset in self.list_assets(project_id) if asset.asset_type == asset_type)

    def remove_asset(self, project_id: str, asset_type: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM project_assets WHERE project_id = ? AND asset_type = ?",
                (project_id, asset_type),
            )
            conn.execute("UPDATE projects SET modified_at = ? WHERE id = ?", (now, project_id))

    def list_assets(self, project_id: str) -> list[ProjectAsset]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM project_assets
                WHERE project_id = ?
                ORDER BY asset_type ASC, modified_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [self._asset_from_row(row) for row in rows]

    def upsert_model(
        self,
        project_id: str,
        role: str,
        path: str | Path,
        *,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AiModelRecord:
        now = utc_now_iso()
        norm_path = str(Path(path))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_models (
                    project_id, role, path, label, metadata_json,
                    created_at, modified_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, role) DO UPDATE SET
                    path = excluded.path,
                    label = excluded.label,
                    metadata_json = excluded.metadata_json,
                    modified_at = excluded.modified_at
                """,
                (
                    project_id,
                    role,
                    norm_path,
                    label or Path(norm_path).name,
                    _json(metadata or {}),
                    now,
                    now,
                ),
            )
            conn.execute("UPDATE projects SET modified_at = ? WHERE id = ?", (now, project_id))
        return next(model for model in self.list_models(project_id) if model.role == role)

    def list_models(self, project_id: str) -> list[AiModelRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ai_models
                WHERE project_id = ?
                ORDER BY role ASC, modified_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [self._model_from_row(row) for row in rows]

    def remove_model(self, project_id: str, role: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM ai_models WHERE project_id = ? AND role = ?",
                (project_id, role),
            )
            conn.execute("UPDATE projects SET modified_at = ? WHERE id = ?", (now, project_id))

    def save_analysis_result(
        self,
        project_id: str,
        result_type: str,
        json_path: str | Path,
        csv_path: str | Path,
        summary: dict[str, Any],
    ) -> AnalysisResultRecord:
        now = utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO analysis_results (
                    project_id, result_type, json_path, csv_path, summary_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    result_type,
                    str(Path(json_path)),
                    str(Path(csv_path)) if csv_path else "",
                    _json(summary),
                    now,
                ),
            )
            conn.execute("UPDATE projects SET modified_at = ? WHERE id = ?", (now, project_id))
            row_id = int(cur.lastrowid)
        return next(result for result in self.list_results(project_id) if result.id == row_id)

    def list_results(self, project_id: str) -> list[AnalysisResultRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM analysis_results
                WHERE project_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (project_id,),
            ).fetchall()
        return [self._result_from_row(row) for row in rows]

    def remove_analysis_result(self, project_id: str, result_id: int) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM analysis_results WHERE project_id = ? AND id = ?",
                (project_id, result_id),
            )
            conn.execute("UPDATE projects SET modified_at = ? WHERE id = ?", (now, project_id))

    def set_config(self, project_id: str, key: str, value: Any) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processing_configs(project_id, key, value_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(project_id, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (project_id, key, _json(value), now),
            )
            conn.execute("UPDATE projects SET modified_at = ? WHERE id = ?", (now, project_id))

    def get_config(self, project_id: str, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM processing_configs WHERE project_id = ? AND key = ?",
                (project_id, key),
            ).fetchone()
        return _loads(row["value_json"], default) if row else default

    def remove_config(self, project_id: str, key: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM processing_configs WHERE project_id = ? AND key = ?",
                (project_id, key),
            )
            conn.execute("UPDATE projects SET modified_at = ? WHERE id = ?", (now, project_id))

    def set_preference(self, key: str, value: Any) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_preferences(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, _json(value), now),
            )

    def get_preference(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM app_preferences WHERE key = ?",
                (key,),
            ).fetchone()
        return _loads(row["value_json"], default) if row else default

    def validate_project_paths(self, project_id: str) -> dict[str, bool]:
        bundle = self.get_bundle(project_id)
        status: dict[str, bool] = {}
        for asset in bundle.assets:
            status[f"asset:{asset.asset_type}"] = asset.exists
        for model in bundle.models:
            status[f"model:{model.role}"] = model.exists
        for result in bundle.results:
            status[f"result:{result.id}"] = result.exists
        if bundle.project.output_dir:
            status["project:output_dir"] = Path(bundle.project.output_dir).exists()
        return status

    def _project_from_row(self, row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            output_dir=row["output_dir"],
            settings=_loads(row["settings_json"], {}),
            created_at=row["created_at"],
            modified_at=row["modified_at"],
            last_opened_at=row["last_opened_at"],
        )

    def _asset_from_row(self, row: sqlite3.Row) -> ProjectAsset:
        return ProjectAsset(
            id=int(row["id"]),
            project_id=row["project_id"],
            asset_type=row["asset_type"],
            path=row["path"],
            label=row["label"],
            exists=Path(row["path"]).exists(),
            metadata=_loads(row["metadata_json"], {}),
            created_at=row["created_at"],
            modified_at=row["modified_at"],
        )

    def _model_from_row(self, row: sqlite3.Row) -> AiModelRecord:
        return AiModelRecord(
            id=int(row["id"]),
            project_id=row["project_id"],
            role=row["role"],
            path=row["path"],
            label=row["label"],
            exists=Path(row["path"]).exists(),
            metadata=_loads(row["metadata_json"], {}),
            created_at=row["created_at"],
            modified_at=row["modified_at"],
        )

    def _result_from_row(self, row: sqlite3.Row) -> AnalysisResultRecord:
        json_path = row["json_path"]
        csv_path = row["csv_path"]
        summary = _loads(row["summary_json"], {})
        xlsx_path = str(summary.get("xlsx_path", "") or "")
        exists = bool(json_path and Path(json_path).exists()) and (
            not csv_path or Path(csv_path).exists()
        ) and (
            not xlsx_path or Path(xlsx_path).exists()
        )
        return AnalysisResultRecord(
            id=int(row["id"]),
            project_id=row["project_id"],
            result_type=row["result_type"],
            json_path=json_path,
            csv_path=csv_path,
            summary=summary,
            created_at=row["created_at"],
            exists=exists,
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _loads(value: str, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return default
