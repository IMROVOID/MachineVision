"""Unit and integration tests for Parquet writer and streaming reader."""

import pytest
import pyarrow as pa

from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    DETECTION_SCHEMA_VERSION,
    compute_bottom_center,
)
from football_identity.artifacts.parquet_writer import write_detection_partition_atomic
from football_identity.artifacts.parquet_reader import (
    read_partition_table,
    stream_partition_batches,
)


def make_test_table(num_rows: int = 100, base_frame: int = 0) -> pa.Table:
    rows = []
    for i in range(num_rows):
        f_id = base_frame + i // 5
        d_idx = i % 5
        x1 = float(100 + d_idx * 50)
        y1 = float(200 + d_idx * 30)
        x2 = float(x1 + 80.0)
        y2 = float(y1 + 180.0)
        bc_x, bc_y = compute_bottom_center(x1, y1, x2, y2)
        conf = 0.1 + (i % 10) * 0.09
        rows.append({
            "schema_version": DETECTION_SCHEMA_VERSION,
            "run_id": "run_001",
            "chunk_id": 0,
            "frame_id": f_id,
            "timestamp_ms": int(f_id * 33.333),
            "detection_index": d_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "bottom_center_x": bc_x,
            "bottom_center_y": bc_y,
            "confidence": conf,
            "class_label": "player",
            "source": "BASE_15FPS",
            "tile_id": None,
            "config_id": "cfg_test_123",
        })
    return pa.Table.from_pylist(rows, schema=DETECTION_PYARROW_SCHEMA)


def test_parquet_write_and_read_roundtrip(tmp_path):
    table = make_test_table(100)
    out_file = tmp_path / "chunk-00000.parquet"

    row_count, file_size, sha256_hex = write_detection_partition_atomic(
        table, out_file, source_width=3840.0, source_height=2160.0
    )

    assert row_count == 100
    assert file_size > 0
    assert len(sha256_hex) == 64
    assert out_file.exists()

    # Read full table
    read_table = read_partition_table(out_file)
    assert read_table.num_rows == 100
    assert read_table.schema.names == DETECTION_PYARROW_SCHEMA.names


def test_parquet_streaming_batches_and_filtering(tmp_path):
    table = make_test_table(200, base_frame=100)
    out_file = tmp_path / "chunk-00001.parquet"

    write_detection_partition_atomic(table, out_file, source_width=3840.0, source_height=2160.0)

    # Stream in batches of 50
    batches = list(stream_partition_batches(out_file, batch_size=50))
    assert len(batches) == 4
    total_streamed = sum(b.num_rows for b in batches)
    assert total_streamed == 200

    # Stream with min_confidence filter
    high_conf_batches = list(stream_partition_batches(out_file, min_confidence=0.5))
    total_high_conf = sum(b.num_rows for b in high_conf_batches)
    assert 0 < total_high_conf < 200

    # Stream with frame range filter
    frame_batches = list(stream_partition_batches(out_file, start_frame=100, end_frame=105))
    total_framed = sum(b.num_rows for b in frame_batches)
    assert total_framed == 30  # 6 frames * 5 detections/frame
