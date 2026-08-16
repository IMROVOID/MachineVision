"""Unit tests for Video Fingerprinting contract."""

import hashlib
import json
import pytest

from football_identity.contracts.video import (
    VideoFingerprint,
    compute_video_sha256,
    validate_video_fingerprint,
    VIDEO_FINGERPRINT_SCHEMA_NAME,
    VIDEO_FINGERPRINT_SCHEMA_VERSION,
)


def test_compute_video_sha256(tmp_path):
    video_file = tmp_path / "test_match.mp4"
    dummy_bytes = b"SIMULATED_4K_VIDEO_CONTENT_BYTE_STREAM_12345"
    video_file.write_bytes(dummy_bytes)

    expected_hash = hashlib.sha256(dummy_bytes).hexdigest()
    calc_hash, size = compute_video_sha256(video_file)

    assert calc_hash == expected_hash
    assert size == len(dummy_bytes)


def test_valid_video_fingerprint():
    data = {
        "schema_name": VIDEO_FINGERPRINT_SCHEMA_NAME,
        "schema_version": VIDEO_FINGERPRINT_SCHEMA_VERSION,
        "source_path_recorded": "/data/videos/match1.mp4",
        "file_size_bytes": 1024000,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "width": 3840,
        "height": 2160,
        "nominal_fps_num": 30,
        "nominal_fps_den": 1,
        "duration_ms": 6000000,
        "frame_count_reported": 180000,
        "container": "mp4",
        "video_codec": "h265",
    }
    fp = validate_video_fingerprint(data)
    assert fp.width == 3840
    assert fp.height == 2160
    assert fp.sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_invalid_video_fingerprint_rejections():
    valid_data = {
        "schema_name": VIDEO_FINGERPRINT_SCHEMA_NAME,
        "schema_version": VIDEO_FINGERPRINT_SCHEMA_VERSION,
        "source_path_recorded": "match.mp4",
        "file_size_bytes": 1000,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "width": 3840,
        "height": 2160,
        "nominal_fps_num": 30,
        "nominal_fps_den": 1,
        "duration_ms": 1000,
        "frame_count_reported": 30,
        "container": "mp4",
        "video_codec": "h264",
    }

    # Wrong schema name
    bad1 = dict(valid_data, schema_name="wrong.name")
    with pytest.raises(ValueError, match="Invalid schema_name"):
        validate_video_fingerprint(bad1)

    # Invalid sha256
    bad2 = dict(valid_data, sha256="short_hash")
    with pytest.raises(ValueError, match="Invalid sha256"):
        validate_video_fingerprint(bad2)

    # Invalid resolution
    bad3 = dict(valid_data, width=-100)
    with pytest.raises(ValueError, match="Invalid resolution"):
        validate_video_fingerprint(bad3)
