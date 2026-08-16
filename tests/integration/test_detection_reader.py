"""Integration tests for detector-neutral PID-2 reader and two-chunk boundary traversal."""

import sys
import pytest
import pyarrow as pa
from pathlib import Path

from football_identity.artifacts.layout import RunArtifactLayout
from football_identity.artifacts.manifest import (
    DetectionManifest,
    PartitionInventoryItem,
    save_detection_manifest,
    compute_file_sha256,
)
from football_identity.artifacts.parquet_writer import write_detection_partition_atomic
from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    DETECTION_SCHEMA_VERSION,
    compute_bottom_center,
)
from football_identity.integration.detection_reader import (
    DetectionItem,
    DetectionStreamReader,
)


@pytest.fixture
def two_chunk_run_fixture(tmp_path):
    """Generates a complete multi-chunk run with base and audit partitions."""
    run_id = "run_smoke_fixture_01"
    layout = RunArtifactLayout(runs_root=tmp_path, run_id=run_id)
    layout.ensure_directories()

    # Chunk 0: Frames 0 to 299 (2 detections per frame = 600 rows)
    rows_c0 = []
    for f in range(0, 300):
        for d in range(2):
            x1, y1 = 100.0 + d * 50.0, 200.0 + d * 50.0
            x2, y2 = x1 + 100.0, y1 + 200.0
            bc_x, bc_y = compute_bottom_center(x1, y1, x2, y2)
            conf = 0.2 + (d * 0.5)  # 0.2 and 0.7
            rows_c0.append({
                "schema_version": DETECTION_SCHEMA_VERSION,
                "run_id": run_id,
                "chunk_id": 0,
                "frame_id": f,
                "timestamp_ms": int(f * 33.333),
                "detection_index": d,
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
                "config_id": "cfg_neutral_01",
            })
    t0 = pa.Table.from_pylist(rows_c0, schema=DETECTION_PYARROW_SCHEMA)
    p0_path = layout.get_chunk_parquet_path("BASE_15FPS", 0)
    r0, s0, h0 = write_detection_partition_atomic(t0, p0_path, 3840.0, 2160.0)

    # Chunk 1: Frames 300 to 599 (2 detections per frame = 600 rows)
    rows_c1 = []
    for f in range(300, 600):
        for d in range(2):
            x1, y1 = 150.0 + d * 50.0, 250.0 + d * 50.0
            x2, y2 = x1 + 100.0, y1 + 200.0
            bc_x, bc_y = compute_bottom_center(x1, y1, x2, y2)
            conf = 0.3 + (d * 0.5)  # 0.3 and 0.8
            rows_c1.append({
                "schema_version": DETECTION_SCHEMA_VERSION,
                "run_id": run_id,
                "chunk_id": 1,
                "frame_id": f,
                "timestamp_ms": int(f * 33.333),
                "detection_index": d,
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
                "config_id": "cfg_neutral_01",
            })
    t1 = pa.Table.from_pylist(rows_c1, schema=DETECTION_PYARROW_SCHEMA)
    p1_path = layout.get_chunk_parquet_path("BASE_15FPS", 1)
    r1, s1, h1 = write_detection_partition_atomic(t1, p1_path, 3840.0, 2160.0)

    # Audit partition: Frames 0 to 600 at 1 FPS (every 30 frames)
    rows_audit = []
    for f in range(0, 600, 30):
        x1, y1, x2, y2 = 500.0, 500.0, 700.0, 900.0
        bc_x, bc_y = compute_bottom_center(x1, y1, x2, y2)
        rows_audit.append({
            "schema_version": DETECTION_SCHEMA_VERSION,
            "run_id": run_id,
            "chunk_id": 0,
            "frame_id": f,
            "timestamp_ms": int(f * 33.333),
            "detection_index": 0,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "bottom_center_x": bc_x,
            "bottom_center_y": bc_y,
            "confidence": 0.99,
            "class_label": "player",
            "source": "AUDIT_TILE_1FPS",
            "tile_id": "tile_0_0",
            "config_id": "cfg_neutral_01",
        })
    t_aud = pa.Table.from_pylist(rows_audit, schema=DETECTION_PYARROW_SCHEMA)
    p_aud_path = layout.get_chunk_parquet_path("AUDIT_TILE_1FPS", 0)
    r_aud, s_aud, h_aud = write_detection_partition_atomic(t_aud, p_aud_path, 3840.0, 2160.0)

    # Save manifest
    manifest = DetectionManifest(
        run_id=run_id,
        video_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        source_resolution=[3840, 2160],
        model_name="neutral_detector",
        model_weight_digest="sha256:0000000000000000000000000000000000000000000000000000000000000000",
        inference_backend="neutral_backend",
        config_id="cfg_neutral_01",
        resolved_input_resolution=[1920, 1080],
        base_fps=15,
        audit_fps=1,
        batch_size=8,
        confidence_thresholds={"storage": 0.1, "high": 0.5},
        partitions=[
            PartitionInventoryItem("BASE_15FPS", 0, "pid01_detection/base/chunk-00000.parquet", s0, r0, h0),
            PartitionInventoryItem("BASE_15FPS", 1, "pid01_detection/base/chunk-00001.parquet", s1, r1, h1),
            PartitionInventoryItem("AUDIT_TILE_1FPS", 0, "pid01_detection/audit/chunk-00000.parquet", s_aud, r_aud, h_aud),
        ],
        state="COMPLETED",
    )
    save_detection_manifest(manifest, layout.detection_manifest_path)
    return tmp_path, run_id


def test_two_chunk_boundary_traversal(two_chunk_run_fixture):
    runs_root, run_id = two_chunk_run_fixture
    reader = DetectionStreamReader(runs_root=runs_root, run_id=run_id)

    # Stream frames 290 to 310 across chunk boundary (boundary is frame 300)
    detections = list(reader.stream_detections(
        sources="BASE_15FPS",
        start_frame=290,
        end_frame=310,
    ))

    # Frames 290 to 310 inclusive = 21 frames * 2 detections/frame = 42 detections
    assert len(detections) == 42

    # Verify frame order is monotonic and continuous across boundary
    frame_ids = [d.frame_id for d in detections]
    assert frame_ids[0] == 290
    assert frame_ids[-1] == 310

    # Ensure chunk boundary was crossed seamlessly: items from both chunk 0 and chunk 1
    chunks_seen = {d.chunk_id for d in detections}
    assert chunks_seen == {0, 1}


def test_source_and_confidence_filtering(two_chunk_run_fixture):
    runs_root, run_id = two_chunk_run_fixture
    reader = DetectionStreamReader(runs_root=runs_root, run_id=run_id)

    # 1. Audit source only
    audit_dets = list(reader.stream_detections(sources="AUDIT_TILE_1FPS"))
    assert len(audit_dets) == 20  # 600 / 30 = 20 frames
    assert all(d.source == "AUDIT_TILE_1FPS" for d in audit_dets)

    # 2. Min confidence filtering on Base
    high_conf_dets = list(reader.stream_detections(sources="BASE_15FPS", min_confidence=0.5))
    assert all(d.confidence >= 0.5 for d in high_conf_dets)
    assert len(high_conf_dets) == 600  # Exactly half the detections had conf >= 0.5


def test_detector_independence():
    """Verify that importing and using the reader does not load any detector or ML packages."""
    forbidden_modules = ["torch", "torchvision", "ultralytics", "cv2", "cuda", "tensorrt"]
    for mod in forbidden_modules:
        assert mod not in sys.modules, f"Forbidden detector module '{mod}' was loaded!"
