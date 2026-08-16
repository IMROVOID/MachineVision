"""Unit tests for RunStateEngine execution state transitions."""

import pytest
import pyarrow as pa
from pathlib import Path

from football_identity.artifacts.layout import RunArtifactLayout
from football_identity.artifacts.parquet_writer import write_detection_partition_atomic
from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    DETECTION_SCHEMA_VERSION,
    compute_bottom_center,
)
from football_identity.contracts.video import VideoFingerprint, VIDEO_FINGERPRINT_SCHEMA_NAME, VIDEO_FINGERPRINT_SCHEMA_VERSION
from football_identity.runtime.state_engine import RunStateEngine


@pytest.fixture
def mock_video_fp():
    return VideoFingerprint(
        schema_name=VIDEO_FINGERPRINT_SCHEMA_NAME,
        schema_version=VIDEO_FINGERPRINT_SCHEMA_VERSION,
        source_path_recorded="/test/match.mp4",
        file_size_bytes=1000000,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        width=3840,
        height=2160,
        nominal_fps_num=30,
        nominal_fps_den=1,
        duration_ms=20000,
        frame_count_reported=600,
        container="mp4",
        video_codec="h264",
    )


@pytest.fixture
def mock_config():
    return {
        "model_identifier": "yolov8x_4k",
        "model_weight_digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        "backend": "tensorrt",
        "input_resolution": [1920, 1080],
        "fps_policies": {"base_fps": 15, "audit_fps": 1, "repair_fps": 30},
        "batch_size": 8,
        "confidence_thresholds": {"storage": 0.1, "high": 0.5},
        "preprocessing_policy": {"letterbox": True, "color_space": "RGB"},
        "tile_policy": {"tile_w": 1920, "tile_h": 1080, "overlap": 0.2},
        "schema_version": 1,
        "implementation_version": "0.1.0",
    }


def make_dummy_parquet(target_path: Path, chunk_id: int, start_frame: int, end_frame: int) -> tuple[int, int, str]:
    rows = []
    for f in range(start_frame, end_frame):
        bc_x, bc_y = compute_bottom_center(100.0, 200.0, 200.0, 500.0)
        rows.append({
            "schema_version": DETECTION_SCHEMA_VERSION,
            "run_id": "test_run_01",
            "chunk_id": chunk_id,
            "frame_id": f,
            "timestamp_ms": int(f * 33.333),
            "detection_index": 0,
            "x1": 100.0,
            "y1": 200.0,
            "x2": 200.0,
            "y2": 500.0,
            "bottom_center_x": bc_x,
            "bottom_center_y": bc_y,
            "confidence": 0.9,
            "class_label": "player",
            "source": "BASE_15FPS",
            "tile_id": None,
            "config_id": "cfg_test",
        })
    table = pa.Table.from_pylist(rows, schema=DETECTION_PYARROW_SCHEMA)
    return write_detection_partition_atomic(table, target_path, 3840.0, 2160.0)


def test_state_engine_lifecycle(tmp_path, mock_video_fp, mock_config):
    layout = RunArtifactLayout(runs_root=tmp_path, run_id="test_run_01")
    engine = RunStateEngine(layout, mock_config, mock_video_fp)

    # 1. Initialize fresh run with 10s chunks (300 frames per chunk -> 2 chunks for 600 frames)
    all_chunks, pending = engine.initialize_or_resume(chunk_duration_sec=10.0)
    assert len(all_chunks) == 2
    assert len(pending) == 2
    assert engine.manifest.state == "IN_PROGRESS"

    # 2. Process Chunk 0
    c0 = engine.mark_chunk_running(chunk_id=0, source="BASE_15FPS")
    assert c0.state == "RUNNING"

    c0_path = layout.get_chunk_parquet_path("BASE_15FPS", 0)
    row_count, file_size, checksum = make_dummy_parquet(c0_path, 0, 0, 300)
    engine.mark_chunk_completed(0, "BASE_15FPS", "pid01_detection/base/chunk-00000.parquet", row_count, checksum)

    # 3. Process Chunk 1
    c1 = engine.mark_chunk_running(chunk_id=1, source="BASE_15FPS")
    c1_path = layout.get_chunk_parquet_path("BASE_15FPS", 1)
    row_count, file_size, checksum = make_dummy_parquet(c1_path, 1, 300, 600)
    engine.mark_chunk_completed(1, "BASE_15FPS", "pid01_detection/base/chunk-00001.parquet", row_count, checksum)

    # 4. Finalize
    final_state = engine.finalize_run()
    assert final_state == "COMPLETED"
    assert engine.manifest.state == "COMPLETED"
    assert len(engine.manifest.partitions) == 2
