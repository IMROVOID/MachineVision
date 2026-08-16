"""State machine engine managing execution lifecycle, resume, retry, corruption detection, and invalidation."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import pyarrow.parquet as pq

from football_identity.artifacts.layout import RunArtifactLayout
from football_identity.artifacts.manifest import (
    DetectionManifest,
    PartitionInventoryItem,
    compute_file_sha256,
    load_detection_manifest,
    save_detection_manifest,
    verify_manifest_integrity,
)
from football_identity.contracts.config import compute_canonical_config_hash
from football_identity.contracts.video import VideoFingerprint
from football_identity.runtime.chunk_planner import plan_chunks
from football_identity.runtime.chunk_state import (
    ChunkRecord,
    load_chunks_manifest,
    save_chunks_manifest,
)
from football_identity.runtime.logger import InfrastructureLogger


class RunStateEngine:
    def __init__(
        self,
        layout: RunArtifactLayout,
        resolved_config: Mapping[str, Any],
        video_fingerprint: VideoFingerprint,
        logger: Optional[InfrastructureLogger] = None,
    ):
        self.layout = layout
        self.resolved_config = dict(resolved_config)
        self.video_fingerprint = video_fingerprint
        self.config_id = compute_canonical_config_hash(self.resolved_config)
        self.layout.ensure_directories()
        self.logger = logger or InfrastructureLogger(self.layout.infrastructure_log_path)

        self.manifest: Optional[DetectionManifest] = None
        self.chunks: list[ChunkRecord] = []

    def initialize_or_resume(
        self,
        chunk_duration_sec: float,
        source: str = "BASE_15FPS",
    ) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
        """Initializes run plan or resumes from existing manifests with corruption and identity checks.

        Returns:
            tuple[list[ChunkRecord], list[ChunkRecord]]: (all_chunks, pending_executable_chunks)
        """
        manifest_path = self.layout.detection_manifest_path
        chunks_path = self.layout.chunks_path

        # Case 1: Existing run artifacts found -> Attempt resume
        if manifest_path.exists() and chunks_path.exists():
            self.manifest = load_detection_manifest(manifest_path)
            existing_chunks = load_chunks_manifest(chunks_path)

            # Check 1: Video Identity Mismatch
            if self.manifest.video_sha256.lower() != self.video_fingerprint.sha256.lower():
                self.logger.log(
                    "VIDEO_MISMATCH_INVALIDATION",
                    level="ERROR",
                    run_id=self.layout.run_id,
                    expected=self.manifest.video_sha256,
                    actual=self.video_fingerprint.sha256,
                )
                self.invalidate_run("Source video SHA-256 mismatch against existing manifest")
                return self.chunks, []

            # Check 2: Canonical Config Identity Mismatch
            if self.manifest.config_id != self.config_id:
                self.logger.log(
                    "CONFIG_MISMATCH_INVALIDATION",
                    level="ERROR",
                    run_id=self.layout.run_id,
                    expected=self.manifest.config_id,
                    actual=self.config_id,
                )
                self.invalidate_run("Configuration canonical hash mismatch against existing manifest")
                return self.chunks, []

            # Process chunk resume states
            pending_chunks: list[ChunkRecord] = []
            verified_chunks: list[ChunkRecord] = []

            for chunk in existing_chunks:
                if chunk.state == "COMPLETED" and chunk.output_path:
                    parquet_full_path = self.layout.run_dir / chunk.output_path
                    if parquet_full_path.exists():
                        try:
                            # Verify checksum
                            calc_sha, calc_size = compute_file_sha256(parquet_full_path)
                            if (
                                chunk.checksum_sha256
                                and calc_sha.lower() == chunk.checksum_sha256.lower()
                            ):
                                # Valid completed chunk -> REUSE
                                self.logger.log(
                                    "CHUNK_REUSED",
                                    run_id=self.layout.run_id,
                                    chunk_id=chunk.chunk_id,
                                    checksum=calc_sha,
                                )
                                verified_chunks.append(chunk)
                                continue
                            else:
                                self.logger.log(
                                    "CHUNK_CHECKSUM_CORRUPTION",
                                    level="WARNING",
                                    run_id=self.layout.run_id,
                                    chunk_id=chunk.chunk_id,
                                )
                        except Exception as e:
                            self.logger.log(
                                "CHUNK_FILE_READ_ERROR",
                                level="WARNING",
                                run_id=self.layout.run_id,
                                chunk_id=chunk.chunk_id,
                                error=str(e),
                            )

                    # If file missing or corrupted -> Mark for retry
                    chunk.state = "FAILED"
                    chunk.error_message = "Partition file missing or corrupted on disk"

                # If chunk was RUNNING, NOT_STARTED, or FAILED -> Schedule for retry
                chunk.state = "NOT_STARTED"
                chunk.attempt_count += 1
                chunk.error_message = None
                verified_chunks.append(chunk)
                pending_chunks.append(chunk)

            self.chunks = verified_chunks
            save_chunks_manifest(self.chunks, chunks_path)
            return self.chunks, pending_chunks

        # Case 2: Fresh run
        self.chunks = plan_chunks(
            total_frames=self.video_fingerprint.frame_count_reported,
            duration_ms=self.video_fingerprint.duration_ms,
            fps_num=self.video_fingerprint.nominal_fps_num,
            fps_den=self.video_fingerprint.nominal_fps_den,
            chunk_duration_sec=chunk_duration_sec,
            source=source,
            config_id=self.config_id,
        )

        fps_policies = self.resolved_config.get("fps_policies", {"base_fps": 15, "audit_fps": 1, "repair_fps": 30})
        self.manifest = DetectionManifest(
            run_id=self.layout.run_id,
            video_sha256=self.video_fingerprint.sha256,
            source_resolution=[self.video_fingerprint.width, self.video_fingerprint.height],
            model_name=str(self.resolved_config.get("model_identifier", "unknown_model")),
            model_weight_digest=str(self.resolved_config.get("model_weight_digest", "")),
            inference_backend=str(self.resolved_config.get("backend", "unknown")),
            config_id=self.config_id,
            resolved_input_resolution=list(self.resolved_config.get("input_resolution", [1920, 1080])),
            base_fps=int(fps_policies.get("base_fps", 15)),
            audit_fps=int(fps_policies.get("audit_fps", 1)),
            batch_size=int(self.resolved_config.get("batch_size", 8)),
            confidence_thresholds=dict(self.resolved_config.get("confidence_thresholds", {"storage": 0.1, "high": 0.5})),
            state="IN_PROGRESS",
        )

        save_chunks_manifest(self.chunks, chunks_path)
        save_detection_manifest(self.manifest, manifest_path)
        self.logger.log("RUN_INITIALIZED", run_id=self.layout.run_id, total_chunks=len(self.chunks))
        return self.chunks, list(self.chunks)

    def mark_chunk_running(self, chunk_id: int, source: str) -> ChunkRecord:
        """Transitions chunk to RUNNING state."""
        chunk = next((c for c in self.chunks if c.chunk_id == chunk_id and c.source == source), None)
        if not chunk:
            raise KeyError(f"Chunk not found: id={chunk_id}, source={source}")

        chunk.state = "RUNNING"
        chunk.started_at = datetime.now(timezone.utc).isoformat()
        chunk.error_message = None
        save_chunks_manifest(self.chunks, self.layout.chunks_path)
        self.logger.log("CHUNK_STARTED", run_id=self.layout.run_id, chunk_id=chunk_id, source=source)
        return chunk

    def mark_chunk_completed(
        self,
        chunk_id: int,
        source: str,
        output_relative_path: str,
        row_count: int,
        checksum: str,
    ) -> ChunkRecord:
        """Transitions chunk to COMPLETED state and updates manifest inventory."""
        chunk = next((c for c in self.chunks if c.chunk_id == chunk_id and c.source == source), None)
        if not chunk:
            raise KeyError(f"Chunk not found: id={chunk_id}, source={source}")

        chunk.state = "COMPLETED"
        chunk.output_path = output_relative_path
        chunk.row_count = row_count
        chunk.checksum_sha256 = checksum
        chunk.completed_at = datetime.now(timezone.utc).isoformat()
        chunk.error_message = None

        # Update partition in manifest
        if self.manifest:
            # Remove any existing entry for this partition
            self.manifest.partitions = [
                p for p in self.manifest.partitions
                if not (p.source == source and p.chunk_id == chunk_id)
            ]
            full_path = self.layout.run_dir / output_relative_path
            file_size = full_path.stat().st_size if full_path.exists() else 0
            self.manifest.partitions.append(
                PartitionInventoryItem(
                    source=source,
                    chunk_id=chunk_id,
                    relative_path=output_relative_path,
                    file_size_bytes=file_size,
                    row_count=row_count,
                    sha256=checksum,
                )
            )
            save_detection_manifest(self.manifest, self.layout.detection_manifest_path)

        save_chunks_manifest(self.chunks, self.layout.chunks_path)
        self.logger.log(
            "CHUNK_COMPLETED",
            run_id=self.layout.run_id,
            chunk_id=chunk_id,
            source=source,
            rows=row_count,
            checksum=checksum,
        )
        return chunk

    def mark_chunk_failed(self, chunk_id: int, source: str, error_message: str) -> ChunkRecord:
        """Transitions chunk to FAILED state."""
        chunk = next((c for c in self.chunks if c.chunk_id == chunk_id and c.source == source), None)
        if not chunk:
            raise KeyError(f"Chunk not found: id={chunk_id}, source={source}")

        chunk.state = "FAILED"
        chunk.failed_at = datetime.now(timezone.utc).isoformat()
        chunk.error_message = str(error_message)
        save_chunks_manifest(self.chunks, self.layout.chunks_path)
        self.logger.log(
            "CHUNK_FAILED",
            level="ERROR",
            run_id=self.layout.run_id,
            chunk_id=chunk_id,
            source=source,
            error=str(error_message),
        )
        return chunk

    def finalize_run(self) -> str:
        """Validates all chunks and transitions run manifest to COMPLETED or FAILED."""
        if not self.manifest:
            raise ValueError("No active manifest to finalize")

        all_completed = all(c.state == "COMPLETED" for c in self.chunks)
        integrity_errors = verify_manifest_integrity(self.manifest, self.layout.run_dir)

        if all_completed and not integrity_errors:
            self.manifest.state = "COMPLETED"
            self.manifest.completed_at = datetime.now(timezone.utc).isoformat()
            save_detection_manifest(self.manifest, self.layout.detection_manifest_path)
            self.logger.log("RUN_COMPLETED", run_id=self.layout.run_id)
            return "COMPLETED"
        else:
            self.manifest.state = "FAILED"
            self.manifest.completed_at = datetime.now(timezone.utc).isoformat()
            save_detection_manifest(self.manifest, self.layout.detection_manifest_path)
            self.logger.log(
                "RUN_FINALIZATION_FAILED",
                level="ERROR",
                run_id=self.layout.run_id,
                integrity_errors=integrity_errors,
            )
            return "FAILED"

    def invalidate_run(self, reason: str) -> None:
        """Invalidates entire run artifacts due to incompatibility."""
        for c in self.chunks:
            c.state = "INVALIDATED"
            c.error_message = reason

        if self.manifest:
            self.manifest.state = "INVALIDATED"
            save_detection_manifest(self.manifest, self.layout.detection_manifest_path)

        if self.chunks:
            save_chunks_manifest(self.chunks, self.layout.chunks_path)

        self.logger.log(
            "RUN_INVALIDATED",
            level="ERROR",
            run_id=self.layout.run_id,
            reason=reason,
        )
