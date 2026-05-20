from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for persisted metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ProjectRecord:
    id: str
    name: str
    description: str
    created_at: str
    modified_at: str
    output_dir: str = ""
    last_opened_at: str = ""
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectAsset:
    id: int
    project_id: str
    asset_type: str
    path: str
    label: str
    exists: bool
    metadata: dict[str, Any]
    created_at: str
    modified_at: str

    @property
    def display_name(self) -> str:
        return self.label or Path(self.path).name or self.asset_type


@dataclass(frozen=True)
class AiModelRecord:
    id: int
    project_id: str
    role: str
    path: str
    label: str
    exists: bool
    metadata: dict[str, Any]
    created_at: str
    modified_at: str

    @property
    def display_name(self) -> str:
        return self.label or Path(self.path).name or self.role


@dataclass(frozen=True)
class AnalysisResultRecord:
    id: int
    project_id: str
    result_type: str
    json_path: str
    csv_path: str
    summary: dict[str, Any]
    created_at: str
    exists: bool


@dataclass(frozen=True)
class ProjectBundle:
    project: ProjectRecord
    assets: list[ProjectAsset]
    models: list[AiModelRecord]
    results: list[AnalysisResultRecord]

    def first_asset(self, asset_type: str) -> ProjectAsset | None:
        return next((asset for asset in self.assets if asset.asset_type == asset_type), None)

    def first_model(self, role: str) -> AiModelRecord | None:
        return next((model for model in self.models if model.role == role), None)

    @property
    def geotiff_path(self) -> str:
        asset = self.first_asset("geotiff")
        return asset.path if asset else ""

    @property
    def output_dir(self) -> str:
        return self.project.output_dir

    @property
    def missing_path_count(self) -> int:
        missing_assets = sum(1 for asset in self.assets if not asset.exists)
        missing_models = sum(1 for model in self.models if not model.exists)
        missing_results = sum(1 for result in self.results if not result.exists)
        return missing_assets + missing_models + missing_results

    @property
    def detection_total(self) -> int:
        if not self.results:
            return 0
        latest = self.results[0]
        counts = latest.summary.get("counts", {})
        return int(counts.get("total", 0) or 0)
