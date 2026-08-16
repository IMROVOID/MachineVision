"""Unit tests for Atomic Publication and Failure Recovery."""

import pytest
import pyarrow as pa
from unittest.mock import patch

from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    DETECTION_SCHEMA_VERSION,
    ValidationError,
    compute_bottom_center,
)
from football_identity.artifacts.parquet_writer import write_detection_partition_atomic


def test_atomic_publication_on_validation_failure(tmp_path):
    # Construct an invalid table (inverted box x1 >= x2)
    invalid_rows = [{
        "schema_version": DETECTION_SCHEMA_VERSION,
        "run_id": "run_001",
        "chunk_id": 0,
        "frame_id": 0,
        "timestamp_ms": 0,
        "detection_index": 0,
        "x1": 500.0,
        "y1": 100.0,
        "x2": 200.0,  # Invalid: x2 < x1
        "y2": 300.0,
        "bottom_center_x": 350.0,
        "bottom_center_y": 300.0,
        "confidence": 0.9,
        "class_label": "player",
        "source": "BASE_15FPS",
        "tile_id": None,
        "config_id": "cfg_123",
    }]
    table = pa.Table.from_pylist(invalid_rows, schema=DETECTION_PYARROW_SCHEMA)
    target_file = tmp_path / "failed_chunk.parquet"

    with pytest.raises(ValidationError):
        write_detection_partition_atomic(table, target_file, source_width=3840.0, source_height=2160.0)

    # Must NOT exist as published file
    assert not target_file.exists()

    # No leftover temporary files
    tmp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(tmp_files) == 0


def test_atomic_publication_on_disk_error(tmp_path):
    valid_rows = [{
        "schema_version": DETECTION_SCHEMA_VERSION,
        "run_id": "run_001",
        "chunk_id": 0,
        "frame_id": 0,
        "timestamp_ms": 0,
        "detection_index": 0,
        "x1": 100.0,
        "y1": 100.0,
        "x2": 200.0,
        "y2": 300.0,
        "bottom_center_x": 150.0,
        "bottom_center_y": 300.0,
        "confidence": 0.9,
        "class_label": "player",
        "source": "BASE_15FPS",
        "tile_id": None,
        "config_id": "cfg_123",
    }]
    table = pa.Table.from_pylist(valid_rows, schema=DETECTION_PYARROW_SCHEMA)
    target_file = tmp_path / "crashed_chunk.parquet"

    # Simulate crash during atomic rename
    with patch("os.replace", side_effect=OSError("Simulated disk I/O failure")):
        with pytest.raises(OSError, match="Simulated disk I/O failure"):
            write_detection_partition_atomic(table, target_file, source_width=3840.0, source_height=2160.0)

    assert not target_file.exists()
