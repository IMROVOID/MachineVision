"""Unit tests for Detection Manifest and Integrity verification."""

import pytest
from pathlib import Path

from football_identity.artifacts.manifest import (
    DetectionManifest,
    PartitionInventoryItem,
    save_detection_manifest,
    load_detection_manifest,
    verify_manifest_integrity,
    compute_file_sha256,
)


@pytest.fixture
def sample_manifest():
    return DetectionManifest(
        run_id="run_test_001",
        video_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        source_resolution=[3840, 2160],
        model_name="yolov8x_4k",
        model_weight_digest="sha256:1111111111111111111111111111111111111111111111111111111111111111",
        inference_backend="tensorrt",
        config_id="cfg_test_hash",
        resolved_input_resolution=[1920, 1080],
        base_fps=15,
        audit_fps=1,
        batch_size=8,
        confidence_thresholds={"storage": 0.1, "high": 0.5},
        partitions=[],
        state="IN_PROGRESS",
    )


def test_manifest_save_and_load(sample_manifest, tmp_path):
    dest = tmp_path / "detection_manifest.json"
    save_detection_manifest(sample_manifest, dest)

    loaded = load_detection_manifest(dest)
    assert loaded.run_id == sample_manifest.run_id
    assert loaded.video_sha256 == sample_manifest.video_sha256
    assert loaded.state == "IN_PROGRESS"


def test_manifest_state_transitions(sample_manifest, tmp_path):
    dest = tmp_path / "manifest.json"
    for state in ["IN_PROGRESS", "COMPLETED", "FAILED", "INVALIDATED"]:
        sample_manifest.state = state
        save_detection_manifest(sample_manifest, dest)
        loaded = load_detection_manifest(dest)
        assert loaded.state == state

    with pytest.raises(ValueError, match="Invalid manifest state"):
        sample_manifest.state = "BOGUS_STATE"
        save_detection_manifest(sample_manifest, dest)


def test_manifest_integrity_verification(sample_manifest, tmp_path):
    # Create a dummy partition file
    part_file = tmp_path / "base" / "chunk-00000.parquet"
    part_file.parent.mkdir(parents=True, exist_ok=True)
    part_file.write_bytes(b"TEST_PARQUET_BYTES_DUMMY")

    sha256_hex, size = compute_file_sha256(part_file)

    sample_manifest.partitions = [
        PartitionInventoryItem(
            source="BASE_15FPS",
            chunk_id=0,
            relative_path="base/chunk-00000.parquet",
            file_size_bytes=size,
            row_count=100,
            sha256=sha256_hex,
        )
    ]

    # Valid check
    errors = verify_manifest_integrity(sample_manifest, tmp_path)
    assert len(errors) == 0

    # Corrupted content (tampered)
    part_file.write_bytes(b"TAMPERED_BYTES")
    errors_tampered = verify_manifest_integrity(sample_manifest, tmp_path)
    assert len(errors_tampered) > 0
    assert any("Checksum mismatch" in e or "Size mismatch" in e for e in errors_tampered)

    # Missing file check
    part_file.unlink()
    errors_missing = verify_manifest_integrity(sample_manifest, tmp_path)
    assert len(errors_missing) == 1
    assert "Missing partition file" in errors_missing[0]
