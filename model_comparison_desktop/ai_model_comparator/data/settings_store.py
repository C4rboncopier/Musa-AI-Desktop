from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DB_FILENAME = "model_comparator.sqlite3"
SECRET_FILENAME = "gemini.env"


@dataclass(slots=True)
class AppConfig:
    leaf_model_path: str = ""
    disease_model_path: str = ""
    gemini_api_key_configured: bool = False


def app_home() -> Path:
    configured = os.environ.get("MUSA_MODEL_COMPARATOR_HOME", "").strip()
    if configured:
        root = Path(configured)
    elif getattr(sys, "frozen", False):
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Musa AI Model Comparator"
    else:
        root = Path.cwd() / ".model_comparator"
    root.mkdir(parents=True, exist_ok=True)
    return root


class SettingsStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or app_home()
        self.db_path = self.root / DB_FILENAME
        self.secret_path = self.root / SECRET_FILENAME
        self.root.mkdir(parents=True, exist_ok=True)
        self._connect().close()
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def migrate(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_preferences (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get_preference(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM app_preferences WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except Exception:
            return default

    def set_preference(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_preferences(key, value_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(value, ensure_ascii=True, sort_keys=True)),
            )

    def load_config(self) -> AppConfig:
        return AppConfig(
            leaf_model_path=str(
                self.get_preference("leaf_model_path", "")
                or self.get_preference("trained_model_a_path", "")
                or ""
            ),
            disease_model_path=str(
                self.get_preference("disease_model_path", "")
                or self.get_preference("trained_model_b_path", "")
                or ""
            ),
            gemini_api_key_configured=bool(self.load_gemini_api_key()),
        )

    def save_model_paths(self, leaf_model_path: str, disease_model_path: str) -> None:
        self.set_preference("leaf_model_path", leaf_model_path.strip())
        self.set_preference("disease_model_path", disease_model_path.strip())

    def load_gemini_api_key(self) -> str:
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "MUSA_GEMINI_API_KEY"):
            value = os.environ.get(name, "").strip()
            if value:
                return value
        if not self.secret_path.exists():
            return ""
        try:
            for line in self.secret_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() in {"GEMINI_API_KEY", "GOOGLE_API_KEY", "MUSA_GEMINI_API_KEY"}:
                    return value.strip().strip('"').strip("'")
        except OSError:
            return ""
        return ""

    def save_gemini_api_key(self, api_key: str) -> None:
        api_key = api_key.strip()
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        if not api_key:
            self.secret_path.unlink(missing_ok=True)
            return
        self.secret_path.write_text(f"GEMINI_API_KEY={api_key}\n", encoding="utf-8")
