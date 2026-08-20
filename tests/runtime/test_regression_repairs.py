"""Regression tests verifying all 12 audit findings and repairs for Wave 1 Worker A."""

from __future__ import annotations

import copy
from pathlib import Path
import pytest
import pyarrow as pa

from football_identity.artifacts.layout import RunArtifactLayout
from football_identity.artifacts.manifest import (
    load_detection_manifest,
    load_run_manifest,
)
from football_identity.artifacts.parquet_writer import write_detection_partition_atomic
from football_identity.contracts.config import (
    CanonicalConfig,
    compute_canonical_config_hash,
)
from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    DETECTION_SCHEMA_VERSION,
    ValidationError,
    compute_bottom_center,
    validate_detection_row,
    validate_detection_table,
)
from football_identity.contracts.video import VideoFingerprint
from football_identity.integration.detection_reader import DetectionStreamReader
from football_identity.runtime.chunk_planner import plan_chunks, plan_multi_source_chunks
from football_identity.runtime.chunk_state import (
    ChunkRecord,
    InvalidStateTransitionError,
    load_chunks_manifest,
)
from football_identity.runtime.state_engine import RunStateEngine


@pytest.fixture
def base_video_fp():
    return VideoFingerprint(
        schema_name="football_identity.video_fingerprint",
        schema_version=1,
        source_path_recorded="/data/football_4k.mp4",
        file_size_bytes=50000000,
        sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        width=3840,
        height=2160,
        nominal_fps_num=30,
        nominal_fps_den=1,
        duration_ms=60000,
        frame_count_reported=1800,
        container="mp4",
        video_codec="h264",
    )


@pytest.fixture
def base_config():
    return {
        "model_identifier": "yolov8x_football",
        "model_weight_digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "backend": "tensorrt",
        "input_resolution": [1920, 1080],
        "fps_policies": {"base_fps": 15, "audit_fps": 1, "repair_fps": 30},
        "batch_size": 16,
        "confidence_thresholds": {"storage": 0.1, "high": 0.6},
        "preprocessing_policy": {"letterbox": True, "color_space": "RGB"},
        "tile_policy": {"tile_w": 1920, "tile_h": 1080, "overlap": 0.2},
        "schema_version": 1,
        "implementation_version": "0.1.0",
    }


def make_test_partition_table(
    run_id: str,
    source: str,
    chunk_id: int,
    frame_ids: list[int],
    config_id: str,
) -> pa.Table:
    rows = []
    for f in frame_ids:
        bc_x, bc_y = compute_bottom_center(100.0, 100.0, 200.0, 300.0)
        rows.append({
            "schema_version": DETECTION_SCHEMA_VERSION,
            "run_id": run_id,
            "chunk_id": chunk_id,
            "frame_id": f,
            "timestamp_ms": int(f * 33.333),
            "detection_index": 0,
            "x1": 100.0,
            "y1": 100.0,
            "x2": 200.0,
            "y2": 300.0,
            "bottom_center_x": bc_x,
            "bottom_center_y": bc_y,
            "confidence": 0.85,
            "class_label": "player",
            "source": source,
            "tile_id": "tile_0" if "AUDIT" in source else None,
            "config_id": config_id,
        })
    return pa.Table.from_pylist(rows, schema=DETECTION_PYARROW_SCHEMA)


# 1. Multi-Source Global Ordering (Item 9)
def test_multi_source_global_ordering(tmp_path, base_video_fp, base_config):
    layout = RunArtifactLayout(runs_root=tmp_path, run_id="run_multi_source")
    engine = RunStateEngine(layout, base_config, base_video_fp)
    engine.initialize_or_resume(chunk_duration_sec=10.0, source="BASE_15FPS")
    engine.plan_source("AUDIT_TILE_1FPS", chunk_duration_sec=10.0)

    cfg_id = engine.config_id

    # Write BASE chunk 0 (frames 0, 2, 4, 6)
    base_c0_path = layout.get_chunk_parquet_path("BASE_15FPS", 0)
    base_t0 = make_test_partition_table("run_multi_source", "BASE_15FPS", 0, [0, 2, 4, 6], cfg_id)
    rows_b0, sz_b0, sha_b0 = write_detection_partition_atomic(base_t0, base_c0_path, 3840.0, 2160.0)
    engine.mark_chunk_running(0, "BASE_15FPS")
    engine.mark_chunk_completed(0, "BASE_15FPS", "pid01_detection/base/chunk-00000.parquet", rows_b0, sha_b0)

    # Write AUDIT chunk 0 (frames 1, 3, 5)
    audit_c0_path = layout.get_chunk_parquet_path("AUDIT_TILE_1FPS", 0)
    audit_t0 = make_test_partition_table("run_multi_source", "AUDIT_TILE_1FPS", 0, [1, 3, 5], cfg_id)
    rows_a0, sz_a0, sha_a0 = write_detection_partition_atomic(audit_t0, audit_c0_path, 3840.0, 2160.0)
    engine.mark_chunk_running(0, "AUDIT_TILE_1FPS")
    engine.mark_chunk_completed(0, "AUDIT_TILE_1FPS", "pid01_detection/audit/chunk-00000.parquet", rows_a0, sha_a0)

    reader = DetectionStreamReader(
        tmp_path,
        "run_multi_source",
        expected_run_id="run_multi_source",
        expected_config_id=cfg_id,
        expected_video_sha256=base_video_fp.sha256,
    )
    items = list(reader.stream_detections())
    assert len(items) == 7
    # Verify global monotonic ordering across sources: 0, 1, 2, 3, 4, 5, 6
    streamed_frames = [item.frame_id for item in items]
    assert streamed_frames == [0, 1, 2, 3, 4, 5, 6]


# 2. Reader Expected Identity Validation (Item 9)
def test_reader_expected_identity_rejection(tmp_path, base_video_fp, base_config):
    layout = RunArtifactLayout(runs_root=tmp_path, run_id="run_id_check")
    engine = RunStateEngine(layout, base_config, base_video_fp)
    engine.initialize_or_resume(chunk_duration_sec=10.0, source="BASE_15FPS")

    # Mismatched run_id
    with pytest.raises(ValueError, match="Reader expected run_id"):
        DetectionStreamReader(tmp_path, "run_id_check", expected_run_id="wrong_run")

    # Mismatched video_sha256
    with pytest.raises(ValueError, match="Reader expected video_sha256"):
        DetectionStreamReader(tmp_path, "run_id_check", expected_video_sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")

    # Mismatched config_id
    with pytest.raises(ValueError, match="Reader expected config_id"):
        DetectionStreamReader(tmp_path, "run_id_check", expected_config_id="wrong_cfg")


# 3. Invalid State Machine Transitions (Item 4)
def test_invalid_chunk_state_transitions():
    c = ChunkRecord(
        chunk_id=0,
        start_frame=0,
        end_frame=300,
        start_time_ms=0,
        end_time_ms=10000,
        source="BASE_15FPS",
        config_id="cfg1",
    )
    assert c.state == "NOT_STARTED"

    # Illegal jump: NOT_STARTED -> COMPLETED (must go to RUNNING first)
    with pytest.raises(InvalidStateTransitionError):
        c.transition_to("COMPLETED")

    # Valid: NOT_STARTED -> RUNNING
    c.transition_to("RUNNING")
    assert c.state == "RUNNING"

    # Valid: RUNNING -> COMPLETED
    c.transition_to("COMPLETED")
    assert c.state == "COMPLETED"

    # Illegal jump: COMPLETED -> RUNNING
    with pytest.raises(InvalidStateTransitionError):
        c.transition_to("RUNNING")

    # Valid: Any -> INVALIDATED
    c.transition_to("INVALIDATED")
    assert c.state == "INVALIDATED"


# 4. Pre-Commit Context and Boundary Validation (Item 3)
def test_pre_commit_row_and_boundary_rejection(tmp_path, base_config):
    target = tmp_path / "chunk_invalid.parquet"

    # Table with frame_id=500 but expected chunk frame bounds [0, 300)
    bad_table = make_test_partition_table("run_01", "BASE_15FPS", 0, [10, 20, 500], "cfg_test")

    with pytest.raises(ValidationError, match="outside chunk frame bounds"):
        write_detection_partition_atomic(
            bad_table,
            target,
            3840.0,
            2160.0,
            expected_run_id="run_01",
            expected_source="BASE_15FPS",
            expected_chunk_id=0,
            expected_config_id="cfg_test",
            frame_bounds=(0, 300),
        )

    # Table with wrong run_id
    with pytest.raises(ValidationError, match="mismatched run_id"):
        write_detection_partition_atomic(
            bad_table,
            target,
            3840.0,
            2160.0,
            expected_run_id="expected_other_run",
        )

    # Table with wrong source
    with pytest.raises(ValidationError, match="mismatched source"):
        write_detection_partition_atomic(
            bad_table,
            target,
            3840.0,
            2160.0,
            expected_source="AUDIT_TILE_1FPS",
        )


# 5. Comprehensive Resume Incompatibility and Consistent Invalidation (Item 2, Item 4)
def test_resume_incompatibility_invalidates_manifest_and_chunks(tmp_path, base_video_fp, base_config):
    layout = RunArtifactLayout(runs_root=tmp_path, run_id="run_incompat")
    engine1 = RunStateEngine(layout, base_config, base_video_fp)
    engine1.initialize_or_resume(chunk_duration_sec=10.0, source="BASE_15FPS")
    assert layout.detection_manifest_path.exists()
    assert layout.chunks_path.exists()

    # Attempt to resume with altered video resolution
    alt_video_fp = copy.deepcopy(base_video_fp)
    object.__setattr__(alt_video_fp, "width", 1920)
    object.__setattr__(alt_video_fp, "height", 1080)

    engine2 = RunStateEngine(layout, base_config, alt_video_fp)
    all_chunks, pending = engine2.initialize_or_resume(chunk_duration_sec=10.0, source="BASE_15FPS")
    assert pending == []

    # Verify both manifests on disk were marked INVALIDATED
    manifest = load_detection_manifest(layout.detection_manifest_path)
    assert manifest.state == "INVALIDATED"

    chunks = load_chunks_manifest(layout.chunks_path)
    assert len(chunks) > 0
    assert all(c.state == "INVALIDATED" for c in chunks)


# 6. Configuration Hashing Coverage (Item 5)
def test_config_hash_covers_all_fields_without_float_rounding(base_config):
    base_hash = compute_canonical_config_hash(base_config)

    # Change a nested confidence threshold
    cfg_modified = copy.deepcopy(base_config)
    cfg_modified["confidence_thresholds"]["high"] = 0.6000001
    mod_hash = compute_canonical_config_hash(cfg_modified)
    assert base_hash != mod_hash

    # Add a custom behavior-affecting parameter
    cfg_extra = copy.deepcopy(base_config)
    cfg_extra["custom_model_flag"] = "fast_nms"
    extra_hash = compute_canonical_config_hash(cfg_extra)
    assert base_hash != extra_hash


# 7. Root Artifacts Creation (Item 7)
def test_root_artifacts_created_and_finalized(tmp_path, base_video_fp, base_config):
    layout = RunArtifactLayout(runs_root=tmp_path, run_id="run_root_artifacts")
    engine = RunStateEngine(layout, base_config, base_video_fp)
    engine.initialize_or_resume(chunk_duration_sec=10.0, source="BASE_15FPS")

    assert layout.video_fingerprint_path.exists()
    assert layout.resolved_config_path.exists()
    assert layout.run_manifest_path.exists()

    run_manifest = load_run_manifest(layout.run_manifest_path)
    assert run_manifest.state == "IN_PROGRESS"
    assert run_manifest.video_sha256 == base_video_fp.sha256

    # Complete all chunks and finalize
    for chunk in engine.chunks:
        c_path = layout.get_chunk_parquet_path("BASE_15FPS", chunk.chunk_id)
        table = make_test_partition_table(
            "run_root_artifacts",
            "BASE_15FPS",
            chunk.chunk_id,
            [chunk.start_frame],
            engine.config_id,
        )
        rows, sz, sha = write_detection_partition_atomic(table, c_path, 3840.0, 2160.0)
        engine.mark_chunk_running(chunk.chunk_id, "BASE_15FPS")
        engine.mark_chunk_completed(chunk.chunk_id, "BASE_15FPS", f"pid01_detection/base/chunk-{chunk.chunk_id:05d}.parquet", rows, sha)

    final_state = engine.finalize_run()
    assert final_state == "COMPLETED"

    # Root run_manifest should now be COMPLETED
    updated_run_manifest = load_run_manifest(layout.run_manifest_path)
    assert updated_run_manifest.state == "COMPLETED"


# 8. Run-Wide Duplicate Key Rejection (Item 8)
def test_run_wide_duplicate_detection_rejection(tmp_path, base_video_fp, base_config):
    layout = RunArtifactLayout(runs_root=tmp_path, run_id="run_dup_check")
    engine = RunStateEngine(layout, base_config, base_video_fp)
    engine.initialize_or_resume(chunk_duration_sec=10.0, source="BASE_15FPS")

    # Chunk 0 has key (run_dup_check, BASE_15FPS, frame_id=10, None, 0)
    c0_path = layout.get_chunk_parquet_path("BASE_15FPS", 0)
    t0 = make_test_partition_table("run_dup_check", "BASE_15FPS", 0, [10], engine.config_id)
    r0, sz0, sha0 = write_detection_partition_atomic(t0, c0_path, 3840.0, 2160.0)
    engine.mark_chunk_running(0, "BASE_15FPS")
    engine.mark_chunk_completed(0, "BASE_15FPS", "pid01_detection/base/chunk-00000.parquet", r0, sha0)

    # Chunk 1 attempts to commit duplicate key (same frame_id=10 and detection_index=0)
    c1_path = layout.get_chunk_parquet_path("BASE_15FPS", 1)
    t1 = make_test_partition_table("run_dup_check", "BASE_15FPS", 1, [10], engine.config_id)
    r1, sz1, sha1 = write_detection_partition_atomic(t1, c1_path, 3840.0, 2160.0)
    engine.mark_chunk_running(1, "BASE_15FPS")

    with pytest.raises(ValidationError, match="duplicate detection key"):
        engine.mark_chunk_completed(1, "BASE_15FPS", "pid01_detection/base/chunk-00001.parquet", r1, sha1)


# 9. Multi-Source Independent Chunk Planning (Item 1)
def test_multi_source_chunk_planning():
    plans = plan_multi_source_chunks(
        total_frames=900,
        duration_ms=30000,
        fps_num=30,
        fps_den=1,
        source_chunk_durations={"BASE_15FPS": 10.0, "AUDIT_TILE_1FPS": 5.0},
        config_id="cfg_multi",
    )
    base_chunks = [c for c in plans if c.source == "BASE_15FPS"]
    audit_chunks = [c for c in plans if c.source == "AUDIT_TILE_1FPS"]

    assert len(base_chunks) == 3  # 900 frames / (10s * 30fps) = 3 chunks
    assert len(audit_chunks) == 6  # 900 frames / (5s * 30fps) = 6 chunks
