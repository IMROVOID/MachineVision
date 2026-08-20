"""Detection Manifest and Partition Inventory contract."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

import pyarrow as pa
import yaml

MANIFEST_SCHEMA_NAME = "football_identity.detection_manifest"
MANIFEST_SCHEMA_VERSION = 1

ALLOWED_FINAL_STATES = {
    "IN_PROGRESS",
    "COMPLETED",
    "FAILED",
    "INVALIDATED",
}


@dataclass
class PartitionInventoryItem:
    source: str
    chunk_id: Union[int, str]
    relative_path: str
    file_size_bytes: int
    row_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionManifest:
    run_id: str
    video_sha256: str
    source_resolution: list[int]
    model_name: str
    model_weight_digest: str
    inference_backend: str
    config_id: str
    resolved_input_resolution: list[int]
    base_fps: int
    audit_fps: int
    batch_size: int
    confidence_thresholds: dict[str, float]
    artifact_format: str = "parquet"
    compression: str = "SNAPPY"
    code_version: str = "0.1.0"
    dependency_versions: dict[str, str] = field(default_factory=lambda: {
        "pyarrow": pa.__version__,
        "pyyaml": yaml.__version__,
    })
    partitions: list[PartitionInventoryItem] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    state: str = "IN_PROGRESS"
    schema_name: str = MANIFEST_SCHEMA_NAME
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["partitions"] = [p if isinstance(p, dict) else asdict(p) for p in self.partitions]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


RUN_MANIFEST_SCHEMA_NAME = "football_identity.run_manifest"
RUN_MANIFEST_SCHEMA_VERSION = 1


@dataclass
class RunManifest:
    run_id: str
    video_sha256: str
    config_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    state: str = "IN_PROGRESS"
    schema_name: str = RUN_MANIFEST_SCHEMA_NAME
    schema_version: int = RUN_MANIFEST_SCHEMA_VERSION
    detection_manifest_path: str = "pid01_detection/detection_manifest.json"
    video_fingerprint_path: str = "video_fingerprint.json"
    resolved_config_path: str = "config.resolved.yaml"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fsync_dir(dir_path: Path) -> None:
    """Safely attempts to fsync directory descriptor if supported."""
    try:
        if hasattr(os, "O_DIRECTORY"):
            dir_fd = os.open(str(dir_path), os.O_RDONLY | os.O_DIRECTORY)
        else:
            dir_fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):
        pass


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> tuple[str, int]:
    """Computes sha256 checksum and byte size of a file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    hasher = hashlib.sha256()
    size = 0
    with open(p, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            hasher.update(buf)
            size += len(buf)
    return hasher.hexdigest().lower(), size


def atomic_write_json(destination: Union[str, Path], data: Mapping[str, Any]) -> None:
    """Writes JSON atomically: writes to temporary file, flushes, fsyncs, and renames."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.parent / f"{dest.name}.tmp.{uuid.uuid4().hex}"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, dest)
        fsync_dir(dest.parent)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def atomic_write_yaml(destination: Union[str, Path], data: Mapping[str, Any]) -> None:
    """Writes YAML atomically: writes to temporary file, flushes, fsyncs, and renames."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.parent / f"{dest.name}.tmp.{uuid.uuid4().hex}"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(data), f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, dest)
        fsync_dir(dest.parent)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def save_run_manifest(manifest: RunManifest, destination: Union[str, Path]) -> None:
    """Saves root run manifest atomically."""
    if manifest.state not in ALLOWED_FINAL_STATES:
        raise ValueError(f"Invalid manifest state '{manifest.state}'. Allowed: {ALLOWED_FINAL_STATES}")
    atomic_write_json(destination, manifest.to_dict())


def load_run_manifest(manifest_path: Union[str, Path]) -> RunManifest:
    """Loads and validates a root run manifest."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Run manifest not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    return RunManifest(
        run_id=data["run_id"],
        video_sha256=data["video_sha256"],
        config_id=data["config_id"],
        created_at=data.get("created_at", ""),
        completed_at=data.get("completed_at"),
        state=data.get("state", "IN_PROGRESS"),
        schema_name=data.get("schema_name", RUN_MANIFEST_SCHEMA_NAME),
        schema_version=data.get("schema_version", RUN_MANIFEST_SCHEMA_VERSION),
        detection_manifest_path=data.get("detection_manifest_path", "pid01_detection/detection_manifest.json"),
        video_fingerprint_path=data.get("video_fingerprint_path", "video_fingerprint.json"),
        resolved_config_path=data.get("resolved_config_path", "config.resolved.yaml"),
    )


def save_detection_manifest(manifest: DetectionManifest, destination: Union[str, Path]) -> None:
    """Saves detection manifest atomically."""
    if manifest.state not in ALLOWED_FINAL_STATES:
        raise ValueError(f"Invalid manifest state '{manifest.state}'. Allowed: {ALLOWED_FINAL_STATES}")
    atomic_write_json(destination, manifest.to_dict())


def load_detection_manifest(manifest_path: Union[str, Path]) -> DetectionManifest:
    """Loads and validates a detection manifest."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("schema_name") != MANIFEST_SCHEMA_NAME:
        raise ValueError(f"Invalid schema_name: {data.get('schema_name')}")
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Invalid schema_version: {data.get('schema_version')}")

    state = data.get("state", "IN_PROGRESS")
    if state not in ALLOWED_FINAL_STATES:
        raise ValueError(f"Invalid manifest state: {state}")

    raw_partitions = data.get("partitions", [])
    partitions = [
        PartitionInventoryItem(
            source=p_dict["source"],
            chunk_id=p_dict["chunk_id"],
            relative_path=p_dict["relative_path"],
            file_size_bytes=p_dict["file_size_bytes"],
            row_count=p_dict["row_count"],
            sha256=p_dict["sha256"],
        )
        for p_dict in raw_partitions
    ]

    return DetectionManifest(
        run_id=data["run_id"],
        video_sha256=data["video_sha256"],
        source_resolution=data["source_resolution"],
        model_name=data["model_name"],
        model_weight_digest=data["model_weight_digest"],
        inference_backend=data["inference_backend"],
        config_id=data["config_id"],
        resolved_input_resolution=data["resolved_input_resolution"],
        base_fps=data["base_fps"],
        audit_fps=data["audit_fps"],
        batch_size=data["batch_size"],
        confidence_thresholds=data["confidence_thresholds"],
        artifact_format=data.get("artifact_format", "parquet"),
        compression=data.get("compression", "SNAPPY"),
        code_version=data.get("code_version", "0.1.0"),
        dependency_versions=data.get("dependency_versions", {}),
        partitions=partitions,
        created_at=data.get("created_at", ""),
        completed_at=data.get("completed_at"),
        state=state,
        schema_name=data["schema_name"],
        schema_version=data["schema_version"],
    )


def verify_manifest_integrity(
    manifest: DetectionManifest,
    base_dir: Union[str, Path],
) -> list[str]:
    """Verifies that all partitions listed in manifest exist and match their SHA-256 checksums."""
    base_path = Path(base_dir)
    errors = []

    for item in manifest.partitions:
        part_path = base_path / item.relative_path
        if not part_path.exists():
            errors.append(f"Missing partition file: {part_path}")
            continue

        calc_sha, calc_size = compute_file_sha256(part_path)
        if calc_size != item.file_size_bytes:
            errors.append(
                f"Size mismatch for {item.relative_path}: expected {item.file_size_bytes}, got {calc_size}"
            )
        if calc_sha.lower() != item.sha256.lower():
            errors.append(
                f"Checksum mismatch for {item.relative_path}: expected {item.sha256}, got {calc_sha}"
            )

    return errors
