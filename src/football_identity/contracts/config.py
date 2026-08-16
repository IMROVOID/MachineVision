"""Configuration and Canonical Identity hashing for Detection pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

CANONICAL_SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "0.1.0"


def _canonicalize_value(val: Any) -> Any:
    """Recursively normalizes Python objects for deterministic JSON serialization."""
    if isinstance(val, Mapping):
        return {k: _canonicalize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [_canonicalize_value(item) for item in val]
    if isinstance(val, float):
        # Round floats to 6 decimal places to prevent platform precision drift
        return round(val, 6)
    return val


@dataclass(frozen=True)
class CanonicalConfig:
    model_identifier: str
    model_weight_digest: str
    backend: str
    input_resolution: list[int]
    fps_policies: dict[str, int]
    batch_size: int
    confidence_thresholds: dict[str, float]
    preprocessing_policy: dict[str, Any]
    tile_policy: dict[str, Any]
    schema_version: int = CANONICAL_SCHEMA_VERSION
    implementation_version: str = IMPLEMENTATION_VERSION

    def to_canonical_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return _canonicalize_value(d)

    def compute_hash(self) -> str:
        canonical_dict = self.to_canonical_dict()
        canonical_json = json.dumps(
            canonical_dict,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_canonical_config_hash(config_data: Union[Mapping[str, Any], CanonicalConfig]) -> str:
    """Computes deterministic SHA-256 hash across canonical configuration keys."""
    if isinstance(config_data, CanonicalConfig):
        return config_data.compute_hash()

    # Extract required fields or normalize dict
    required_keys = [
        "model_identifier",
        "model_weight_digest",
        "backend",
        "input_resolution",
        "fps_policies",
        "batch_size",
        "confidence_thresholds",
        "preprocessing_policy",
        "tile_policy",
    ]
    extracted = {}
    for key in required_keys:
        if key not in config_data:
            raise KeyError(f"Missing required canonical configuration key: '{key}'")
        extracted[key] = config_data[key]

    extracted["schema_version"] = config_data.get("schema_version", CANONICAL_SCHEMA_VERSION)
    extracted["implementation_version"] = config_data.get("implementation_version", IMPLEMENTATION_VERSION)

    canonical_obj = _canonicalize_value(extracted)
    canonical_json = json.dumps(
        canonical_obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def load_resolved_config(config_path_or_dict: Union[str, Path, Mapping[str, Any]]) -> dict[str, Any]:
    """Loads configuration from YAML file path or returns dictionary copy."""
    if isinstance(config_path_or_dict, (str, Path)):
        p = Path(config_path_or_dict)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid YAML config format in {p}")
            return data
    elif isinstance(config_path_or_dict, Mapping):
        return dict(config_path_or_dict)
    else:
        raise TypeError(f"Unsupported config type: {type(config_path_or_dict)}")
