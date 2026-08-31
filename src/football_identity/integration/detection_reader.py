"""Detector-neutral streaming detection reader for PID-2 downstream consumers."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional, Sequence, Set, Union

import pyarrow as pa
import pyarrow.compute as pc

from football_identity.artifacts.layout import RunArtifactLayout
from football_identity.artifacts.manifest import (
    DetectionManifest,
    load_detection_manifest,
    verify_manifest_integrity,
)
from football_identity.artifacts.parquet_reader import stream_partition_batches
from football_identity.contracts.detection import DETECTION_PYARROW_SCHEMA
from football_identity.runtime.chunk_state import ChunkRecord, load_chunks_manifest


@dataclass(frozen=True)
class DetectionItem:
    """Detector-neutral detection item in source coordinates."""
    frame_id: int
    detection_index: int
    timestamp_ms: int
    x1: float
    y1: float
    x2: float
    y2: float
    bottom_center_x: float
    bottom_center_y: float
    confidence: float
    class_label: str
    source: str
    tile_id: Optional[str]
    chunk_id: int
    run_id: str
    config_id: str
    schema_version: int = 1


class DetectionStreamReader:
    """Streams validated detections across multiple chunk partitions with globally deterministic ordering.
    
    # ponytail: Multi-way heapq.merge stream across partition batch generators.
    # Ceiling: strictly bounded memory for 16 GB RAM with global monotonic order.
    """

    def __init__(
        self,
        runs_root: Union[str, Path],
        run_id: str,
        validate_integrity: bool = True,
        expected_run_id: Optional[str] = None,
        expected_video_sha256: Optional[str] = None,
        expected_config_id: Optional[str] = None,
    ):
        self.layout = RunArtifactLayout(runs_root=Path(runs_root), run_id=run_id)
        if not self.layout.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.layout.run_dir}")

        if not self.layout.detection_manifest_path.exists():
            raise FileNotFoundError(f"Detection manifest not found: {self.layout.detection_manifest_path}")

        self.manifest: DetectionManifest = load_detection_manifest(self.layout.detection_manifest_path)
        if self.manifest.state not in {"COMPLETED", "IN_PROGRESS"}:
            raise ValueError(f"Cannot read detections from run in state '{self.manifest.state}'")

        # Validate consumer expectations if provided
        if expected_run_id is not None and self.manifest.run_id != expected_run_id:
            raise ValueError(
                f"Reader expected run_id '{expected_run_id}', but manifest has '{self.manifest.run_id}'"
            )
        if expected_video_sha256 is not None and self.manifest.video_sha256.lower() != expected_video_sha256.lower():
            raise ValueError(
                f"Reader expected video_sha256 '{expected_video_sha256}', but manifest has '{self.manifest.video_sha256}'"
            )
        if expected_config_id is not None and self.manifest.config_id != expected_config_id:
            raise ValueError(
                f"Reader expected config_id '{expected_config_id}', but manifest has '{self.manifest.config_id}'"
            )

        if validate_integrity:
            errors = verify_manifest_integrity(self.manifest, self.layout.run_dir)
            if errors:
                raise ValueError(f"Manifest integrity verification failed: {errors}")

        self.chunks: list[ChunkRecord] = []
        if self.layout.chunks_path.exists():
            self.chunks = load_chunks_manifest(self.layout.chunks_path)

    @property
    def video_sha256(self) -> str:
        return self.manifest.video_sha256

    @property
    def config_id(self) -> str:
        return self.manifest.config_id

    @property
    def source_resolution(self) -> tuple[int, int]:
        return (self.manifest.source_resolution[0], self.manifest.source_resolution[1])

    def _partition_item_generator(
        self,
        part_full_path: Path,
        batch_size: int,
        columns: Optional[Sequence[str]],
        min_confidence: Optional[float],
        max_confidence: Optional[float],
        start_frame: Optional[int],
        end_frame: Optional[int],
    ) -> Generator[DetectionItem, None, None]:
        """Streams DetectionItem objects from a single partition."""
        for batch in stream_partition_batches(
            file_path=part_full_path,
            batch_size=batch_size,
            columns=columns,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            start_frame=start_frame,
            end_frame=end_frame,
        ):
            d = batch.to_pydict()
            num_rows = batch.num_rows
            schema_versions = d.get("schema_version", [1] * num_rows)
            for i in range(num_rows):
                yield DetectionItem(
                    frame_id=d["frame_id"][i],
                    detection_index=d["detection_index"][i],
                    timestamp_ms=d["timestamp_ms"][i],
                    x1=d["x1"][i],
                    y1=d["y1"][i],
                    x2=d["x2"][i],
                    y2=d["y2"][i],
                    bottom_center_x=d["bottom_center_x"][i],
                    bottom_center_y=d["bottom_center_y"][i],
                    confidence=d["confidence"][i],
                    class_label=d["class_label"][i],
                    source=d["source"][i],
                    tile_id=d["tile_id"][i],
                    chunk_id=d["chunk_id"][i],
                    run_id=d["run_id"][i],
                    config_id=d["config_id"][i],
                    schema_version=schema_versions[i],
                )

    def stream_detections(
        self,
        sources: Optional[Union[str, Sequence[str]]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        batch_size: int = 5000,
    ) -> Generator[DetectionItem, None, None]:
        """Yields individual DetectionItem instances strictly ordered globally by (frame_id, timestamp_ms, detection_index, source)."""
        target_sources: Optional[Set[str]] = None
        if sources is not None:
            if isinstance(sources, str):
                target_sources = {sources}
            else:
                target_sources = set(sources)

        valid_partitions = [
            p for p in self.manifest.partitions
            if target_sources is None or p.source in target_sources
        ]

        if not valid_partitions:
            return

        generators = []
        for partition in valid_partitions:
            part_full_path = self.layout.run_dir / partition.relative_path
            if not part_full_path.exists():
                raise FileNotFoundError(f"Partition file missing: {part_full_path}")

            gen = self._partition_item_generator(
                part_full_path=part_full_path,
                batch_size=batch_size,
                columns=None,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            generators.append(gen)

        # Merge across all generators using global sort key
        for item in heapq.merge(
            *generators,
            key=lambda x: (x.frame_id, x.timestamp_ms, x.detection_index, x.source)
        ):
            yield item

    def stream_batches(
        self,
        sources: Optional[Union[str, Sequence[str]]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        batch_size: int = 5000,
        columns: Optional[Sequence[str]] = None,
    ) -> Generator[pa.RecordBatch, None, None]:
        """Streams globally sorted PyArrow RecordBatches across chunks and sources."""
        buffer_rows: list[dict[str, Any]] = []

        target_schema = DETECTION_PYARROW_SCHEMA
        if columns is not None:
            target_schema = pa.schema([DETECTION_PYARROW_SCHEMA.field(c) for c in columns])

        for item in self.stream_detections(
            sources=sources,
            start_frame=start_frame,
            end_frame=end_frame,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            batch_size=batch_size,
        ):
            row_dict = {
                "schema_version": item.schema_version,
                "run_id": item.run_id,
                "chunk_id": item.chunk_id,
                "frame_id": item.frame_id,
                "timestamp_ms": item.timestamp_ms,
                "detection_index": item.detection_index,
                "x1": item.x1,
                "y1": item.y1,
                "x2": item.x2,
                "y2": item.y2,
                "bottom_center_x": item.bottom_center_x,
                "bottom_center_y": item.bottom_center_y,
                "confidence": item.confidence,
                "class_label": item.class_label,
                "source": item.source,
                "tile_id": item.tile_id,
                "config_id": item.config_id,
            }
            if columns is not None:
                row_dict = {k: v for k, v in row_dict.items() if k in columns}
            buffer_rows.append(row_dict)

            if len(buffer_rows) >= batch_size:
                table = pa.Table.from_pylist(buffer_rows, schema=target_schema)
                for batch in table.to_batches():
                    yield batch
                buffer_rows.clear()

        if buffer_rows:
            table = pa.Table.from_pylist(buffer_rows, schema=target_schema)
            for batch in table.to_batches():
                yield batch
