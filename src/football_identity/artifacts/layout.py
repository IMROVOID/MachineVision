"""Artifact directory layout and path resolution for Football Identity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class RunArtifactLayout:
    runs_root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.run_dir / "config.resolved.yaml"

    @property
    def video_fingerprint_path(self) -> Path:
        return self.run_dir / "video_fingerprint.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def infrastructure_log_path(self) -> Path:
        return self.logs_dir / "infrastructure.jsonl"

    @property
    def pid01_detection_dir(self) -> Path:
        return self.run_dir / "pid01_detection"

    @property
    def detection_manifest_path(self) -> Path:
        return self.pid01_detection_dir / "detection_manifest.json"

    @property
    def chunks_path(self) -> Path:
        return self.pid01_detection_dir / "chunks.json"

    @property
    def base_dir(self) -> Path:
        return self.pid01_detection_dir / "base"

    @property
    def audit_dir(self) -> Path:
        return self.pid01_detection_dir / "audit"

    @property
    def repair_dir(self) -> Path:
        return self.pid01_detection_dir / "repair"

    @property
    def previews_dir(self) -> Path:
        return self.pid01_detection_dir / "previews"

    @property
    def metrics_dir(self) -> Path:
        return self.pid01_detection_dir / "metrics"

    def get_chunk_parquet_path(self, source: str, chunk_id: int | str) -> Path:
        """Returns the canonical final Parquet path for a given source and chunk."""
        if source == "BASE_15FPS":
            return self.base_dir / f"chunk-{int(chunk_id):05d}.parquet"
        elif source == "AUDIT_TILE_1FPS":
            return self.audit_dir / f"chunk-{int(chunk_id):05d}.parquet"
        elif source == "REPAIR_30FPS":
            return self.repair_dir / f"repair-{chunk_id}.parquet"
        else:
            raise ValueError(f"Unknown source partition type: '{source}'")

    def ensure_directories(self) -> None:
        """Creates the full directory hierarchy for the run."""
        for d in [
            self.run_dir,
            self.logs_dir,
            self.pid01_detection_dir,
            self.base_dir,
            self.audit_dir,
            self.repair_dir,
            self.previews_dir,
            self.metrics_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
