"""Unit tests for RunArtifactLayout."""

from pathlib import Path
from football_identity.artifacts.layout import RunArtifactLayout


def test_artifact_layout_structure(tmp_path):
    layout = RunArtifactLayout(runs_root=tmp_path, run_id="match_2026_001")
    layout.ensure_directories()

    assert layout.run_dir.exists()
    assert layout.logs_dir.exists()
    assert layout.pid01_detection_dir.exists()
    assert layout.base_dir.exists()
    assert layout.audit_dir.exists()
    assert layout.repair_dir.exists()
    assert layout.previews_dir.exists()
    assert layout.metrics_dir.exists()

    # Paths check
    assert layout.get_chunk_parquet_path("BASE_15FPS", 0) == layout.base_dir / "chunk-00000.parquet"
    assert layout.get_chunk_parquet_path("BASE_15FPS", 42) == layout.base_dir / "chunk-00042.parquet"
    assert layout.get_chunk_parquet_path("AUDIT_TILE_1FPS", 0) == layout.audit_dir / "chunk-00000.parquet"
    assert layout.get_chunk_parquet_path("REPAIR_30FPS", "int_123") == layout.repair_dir / "repair-int_123.parquet"
