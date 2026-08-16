"""Contracts module for schema definitions and strict data validation."""

from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    DETECTION_SCHEMA_VERSION,
    ALLOWED_SOURCES,
    DetectionRow,
    validate_detection_row,
    validate_detection_table,
    clip_box_to_image,
    compute_bottom_center,
)
from football_identity.contracts.config import (
    compute_canonical_config_hash,
    load_resolved_config,
    CanonicalConfig,
)
from football_identity.contracts.video import (
    VideoFingerprint,
    compute_video_sha256,
    validate_video_fingerprint,
)

__all__ = [
    "DETECTION_PYARROW_SCHEMA",
    "DETECTION_SCHEMA_VERSION",
    "ALLOWED_SOURCES",
    "DetectionRow",
    "validate_detection_row",
    "validate_detection_table",
    "clip_box_to_image",
    "compute_bottom_center",
    "compute_canonical_config_hash",
    "load_resolved_config",
    "CanonicalConfig",
    "VideoFingerprint",
    "compute_video_sha256",
    "validate_video_fingerprint",
]
