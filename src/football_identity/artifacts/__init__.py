"""Artifacts package for storage layout, manifests, atomic writes and streaming reads."""

from football_identity.artifacts.layout import RunArtifactLayout
from football_identity.artifacts.manifest import (
    DetectionManifest,
    PartitionInventoryItem,
    compute_file_sha256,
    save_detection_manifest,
    load_detection_manifest,
    verify_manifest_integrity,
    atomic_write_json,
    ALLOWED_FINAL_STATES,
)
from football_identity.artifacts.parquet_writer import write_detection_partition_atomic
from football_identity.artifacts.parquet_reader import (
    read_partition_table,
    stream_partition_batches,
    PartitionCorruptionError,
)

__all__ = [
    "RunArtifactLayout",
    "DetectionManifest",
    "PartitionInventoryItem",
    "compute_file_sha256",
    "save_detection_manifest",
    "load_detection_manifest",
    "verify_manifest_integrity",
    "atomic_write_json",
    "ALLOWED_FINAL_STATES",
    "write_detection_partition_atomic",
    "read_partition_table",
    "stream_partition_batches",
    "PartitionCorruptionError",
]
