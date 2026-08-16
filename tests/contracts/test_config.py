"""Unit tests for Canonical Configuration Hashing."""

import copy
import pytest
import yaml

from football_identity.contracts.config import (
    CanonicalConfig,
    compute_canonical_config_hash,
    load_resolved_config,
)


@pytest.fixture
def base_config_dict():
    return {
        "model_identifier": "yolov8x_4k_custom",
        "model_weight_digest": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
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


def test_canonical_hash_stability(base_config_dict):
    hash1 = compute_canonical_config_hash(base_config_dict)
    hash2 = compute_canonical_config_hash(base_config_dict)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_canonical_hash_key_order_and_yaml_insensitivity(base_config_dict, tmp_path):
    # Reordered keys in dict
    reordered = {k: base_config_dict[k] for k in reversed(list(base_config_dict.keys()))}
    assert compute_canonical_config_hash(base_config_dict) == compute_canonical_config_hash(reordered)

    # Save as YAML with comments/whitespace
    yaml_text = """
    # This is a comment
    backend: "tensorrt"
    batch_size: 8
    confidence_thresholds:
      high: 0.5
      storage: 0.1
    fps_policies:
      audit_fps: 1
      base_fps: 15
      repair_fps: 30
    implementation_version: "0.1.0"
    input_resolution:
      - 1920
      - 1080
    model_identifier: "yolov8x_4k_custom"
    model_weight_digest: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    preprocessing_policy:
      color_space: "RGB"
      letterbox: true
    schema_version: 1
    tile_policy:
      overlap: 0.2
      tile_h: 1080
      tile_w: 1920
    """
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_text, encoding="utf-8")
    loaded = load_resolved_config(yaml_file)
    assert compute_canonical_config_hash(loaded) == compute_canonical_config_hash(base_config_dict)


def test_canonical_hash_sensitivity_to_each_field(base_config_dict):
    base_hash = compute_canonical_config_hash(base_config_dict)

    mutations = [
        ("model_identifier", "rf_detr_large"),
        ("model_weight_digest", "sha256:1111111111111111111111111111111111111111111111111111111111111111"),
        ("backend", "onnxruntime"),
        ("input_resolution", [3840, 2160]),
        ("fps_policies", {"base_fps": 30, "audit_fps": 1, "repair_fps": 30}),
        ("batch_size", 16),
        ("confidence_thresholds", {"storage": 0.2, "high": 0.5}),
        ("preprocessing_policy", {"letterbox": False, "color_space": "RGB"}),
        ("tile_policy", {"tile_w": 1280, "tile_h": 720, "overlap": 0.1}),
        ("schema_version", 2),
        ("implementation_version", "0.2.0"),
    ]

    for key, new_val in mutations:
        mutated = copy.deepcopy(base_config_dict)
        mutated[key] = new_val
        mutated_hash = compute_canonical_config_hash(mutated)
        assert mutated_hash != base_hash, f"Hash did not change when mutating '{key}'"
