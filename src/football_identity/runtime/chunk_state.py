"""Chunk record dataclass and state definitions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from football_identity.artifacts.manifest import atomic_write_json

CHUNK_STATES = {
    "NOT_STARTED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "INVALIDATED",
}

VALID_CHUNK_TRANSITIONS = {
    "NOT_STARTED": {"RUNNING", "INVALIDATED"},
    "RUNNING": {"COMPLETED", "FAILED", "INVALIDATED"},
    "COMPLETED": {"INVALIDATED"},
    "FAILED": {"NOT_STARTED", "RUNNING", "INVALIDATED"},
    "INVALIDATED": set(),
}


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal chunk state transition is attempted."""
    pass


@dataclass
class ChunkRecord:
    chunk_id: int
    start_frame: int
    end_frame: int
    start_time_ms: int
    end_time_ms: int
    source: str
    config_id: str
    attempt_count: int = 0
    state: str = "NOT_STARTED"
    output_path: Optional[str] = None
    row_count: int = 0
    checksum_sha256: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    error_message: Optional[str] = None

    def transition_to(self, new_state: str, allow_force: bool = False) -> None:
        """Enforces legal state machine transitions."""
        if new_state not in CHUNK_STATES:
            raise ValueError(f"Unknown target state: '{new_state}'")
        if not allow_force and new_state != self.state:
            allowed = VALID_CHUNK_TRANSITIONS.get(self.state, set())
            if new_state not in allowed:
                raise InvalidStateTransitionError(
                    f"Illegal chunk state transition from '{self.state}' to '{new_state}' for chunk {self.chunk_id} ({self.source})"
                )
        self.state = new_state

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChunkRecord:
        state = data.get("state", "NOT_STARTED")
        if state not in CHUNK_STATES:
            raise ValueError(f"Invalid chunk state: '{state}'")
        return cls(
            chunk_id=int(data["chunk_id"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            start_time_ms=int(data["start_time_ms"]),
            end_time_ms=int(data["end_time_ms"]),
            source=str(data["source"]),
            config_id=str(data["config_id"]),
            attempt_count=int(data.get("attempt_count", 0)),
            state=state,
            output_path=data.get("output_path"),
            row_count=int(data.get("row_count", 0)),
            checksum_sha256=data.get("checksum_sha256"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            failed_at=data.get("failed_at"),
            error_message=data.get("error_message"),
        )


def save_chunks_manifest(chunks: Sequence[ChunkRecord], destination: Union[str, Path]) -> None:
    """Saves list of chunk records atomically."""
    dest = Path(destination)
    data = {
        "schema_name": "football_identity.chunks_manifest",
        "schema_version": 1,
        "chunk_count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }
    atomic_write_json(dest, data)


def load_chunks_manifest(destination: Union[str, Path]) -> list[ChunkRecord]:
    """Loads chunk records from manifest file."""
    dest = Path(destination)
    if not dest.exists():
        raise FileNotFoundError(f"Chunks manifest not found: {dest}")

    with open(dest, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [ChunkRecord.from_dict(c) for c in data.get("chunks", [])]
