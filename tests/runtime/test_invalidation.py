"""Unit tests for Invalidation rules on Video/Config mismatches."""

import copy
import pytest
from football_identity.artifacts.layout import RunArtifactLayout
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


def test_invalidation_on_config_change(tmp_path, mock_video_fp, mock_config):
    layout = RunArtifactLayout(runs_root=tmp_path, run_id="test_run_inval")
    engine1 = RunStateEngine(layout, mock_config, mock_video_fp)
    engine1.initialize_or_resume(chunk_duration_sec=10.0)

    # Now attempt to resume with modified model_identifier
    changed_config = copy.deepcopy(mock_config)
    changed_config["model_identifier"] = "rf_detr_large"

    engine2 = RunStateEngine(layout, changed_config, mock_video_fp)
    all_chunks, pending = engine2.initialize_or_resume(chunk_duration_sec=10.0)

    assert engine2.manifest.state == "INVALIDATED"
    assert len(pending) == 0


def test_invalidation_on_video_mismatch(tmp_path, mock_video_fp, mock_config):
    layout = RunArtifactLayout(runs_root=tmp_path, run_id="test_run_inval_vid")
    engine1 = RunStateEngine(layout, mock_config, mock_video_fp)
    engine1.initialize_or_resume(chunk_duration_sec=10.0)

    # Attempt resume with different video SHA-256
    different_video_fp = VideoFingerprint(
        schema_name=VIDEO_FINGERPRINT_SCHEMA_NAME,
        schema_version=VIDEO_FINGERPRINT_SCHEMA_VERSION,
        source_path_recorded="/test/different_match.mp4",
        file_size_bytes=2000000,
        sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        width=3840,
        height=2160,
        nominal_fps_num=30,
        nominal_fps_den=1,
        duration_ms=20000,
        frame_count_reported=600,
        container="mp4",
        video_codec="h264",
    )

    engine2 = RunStateEngine(layout, mock_config, different_video_fp)
    all_chunks, pending = engine2.initialize_or_resume(chunk_duration_sec=10.0)

    assert engine2.manifest.state == "INVALIDATED"
    assert len(pending) == 0
