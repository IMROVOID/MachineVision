"""Source Video Fingerprint Contract for Football Identity Wave 1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

VIDEO_FINGERPRINT_SCHEMA_NAME = "football_identity.video_fingerprint"
VIDEO_FINGERPRINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VideoFingerprint:
    source_path_recorded: str
    file_size_bytes: int
    sha256: str
    width: int
    height: int
    nominal_fps_num: int
    nominal_fps_den: int
    duration_ms: int
    frame_count_reported: int
    container: str
    video_codec: str
    schema_name: str = VIDEO_FINGERPRINT_SCHEMA_NAME
    schema_version: int = VIDEO_FINGERPRINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VideoFingerprint:
        return validate_video_fingerprint(data)


def compute_video_sha256(
    video_path: Union[str, Path],
    chunk_size: int = 1024 * 1024,
) -> tuple[str, int]:
    """Computes SHA-256 hash and byte size directly from file contents."""
    p = Path(video_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Source video not found: {p}")

    hasher = hashlib.sha256()
    total_bytes = 0
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            total_bytes += len(chunk)

    return hasher.hexdigest().lower(), total_bytes


def validate_video_fingerprint(data: Mapping[str, Any]) -> VideoFingerprint:
    """Validates video fingerprint metadata structure and types."""
    if data.get("schema_name") != VIDEO_FINGERPRINT_SCHEMA_NAME:
        raise ValueError(
            f"Invalid schema_name: expected '{VIDEO_FINGERPRINT_SCHEMA_NAME}', got '{data.get('schema_name')}'"
        )
    if data.get("schema_version") != VIDEO_FINGERPRINT_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid schema_version: expected {VIDEO_FINGERPRINT_SCHEMA_VERSION}, got {data.get('schema_version')}'"
        )

    sha256 = data.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64 or not all(c in "0123456789abcdefABCDEF" for c in sha256):
        raise ValueError(f"Invalid sha256 hex string: {sha256}")

    file_size_bytes = data.get("file_size_bytes")
    if not isinstance(file_size_bytes, int) or file_size_bytes < 0:
        raise ValueError(f"Invalid file_size_bytes: {file_size_bytes}")

    width = data.get("width")
    height = data.get("height")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        raise ValueError(f"Invalid resolution dimensions: ({width}, {height})")

    fps_num = data.get("nominal_fps_num")
    fps_den = data.get("nominal_fps_den")
    if not isinstance(fps_num, int) or fps_num <= 0 or not isinstance(fps_den, int) or fps_den <= 0:
        raise ValueError(f"Invalid FPS fraction: {fps_num}/{fps_den}")

    return VideoFingerprint(
        schema_name=str(data["schema_name"]),
        schema_version=int(data["schema_version"]),
        source_path_recorded=str(data.get("source_path_recorded", "")),
        file_size_bytes=int(file_size_bytes),
        sha256=sha256.lower(),
        width=int(width),
        height=int(height),
        nominal_fps_num=int(fps_num),
        nominal_fps_den=int(fps_den),
        duration_ms=int(data.get("duration_ms", 0)),
        frame_count_reported=int(data.get("frame_count_reported", 0)),
        container=str(data.get("container", "unknown")),
        video_codec=str(data.get("video_codec", "unknown")),
    )
