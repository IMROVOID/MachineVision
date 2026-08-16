"""Detector-neutral streaming detection reader for PID-2 downstream consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, List, Optional, Sequence, Set, Union

import pyarrow as pa
import pyarrow.compute as pc

from football_identity.artifacts.layout import RunArtifactLayout
from football_identity.artifacts.manifest import (
    DetectionManifest,
    load_detection_manifest,
    verify_manifest_integrity,
)
from football_identity.artifacts.parquet_reader import stream_partition_batches
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


class DetectionStreamReader:
    """Streams validated detections across multiple chunk partitions without high memory overhead.
    
    # ponytail: Single-process streaming generator reading chunk-by-chunk batches.
    # Ceiling: strictly bounded memory for 16 GB RAM. Upgrade path: background pre-fetch thread pool if reader becomes downstream bottleneck.
    """

    def __init__(
        self,
        runs_root: Union[str, Path],
        run_id: str,
        validate_integrity: bool = True,
    ):
        self.layout = RunArtifactLayout(runs_root=Path(runs_root), run_id=run_id)
        if not self.layout.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.layout.run_dir}")

        if not self.layout.detection_manifest_path.exists():
            raise FileNotFoundError(f"Detection manifest not found: {self.layout.detection_manifest_path}")

        self.manifest: DetectionManifest = load_detection_manifest(self.layout.detection_manifest_path)
        if self.manifest.state not in {"COMPLETED", "IN_PROGRESS"}:
            raise ValueError(f"Cannot read detections from run in state '{self.manifest.state}'")

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
        """Streams filtered PyArrow RecordBatches across chunks in monotonic frame order."""
        target_sources: Optional[Set[str]] = None
        if sources is not None:
            if isinstance(sources, str):
                target_sources = {sources}
            else:
                target_sources = set(sources)

        # Identify all relevant completed partition files
        # Sort partitions by source and chunk_id
        valid_partitions = [
            p for p in self.manifest.partitions
            if target_sources is None or p.source in target_sources
        ]

        # Order partitions by chunk_id
        def get_sort_key(p):
            try:
                return (p.source, int(p.chunk_id))
            except ValueError:
                return (p.source, str(p.chunk_id))

        sorted_partitions = sorted(valid_partitions, key=get_sort_key)

        for partition in sorted_partitions:
            part_full_path = self.layout.run_dir / partition.relative_path
            if not part_full_path.exists():
                raise FileNotFoundError(f"Partition file missing: {part_full_path}")

            # Stream batches from this partition file
            for batch in stream_partition_batches(
                file_path=part_full_path,
                batch_size=batch_size,
                columns=columns,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                start_frame=start_frame,
                end_frame=end_frame,
            ):
                if batch.num_rows > 0:
                    yield batch

    def stream_detections(
        self,
        sources: Optional[Union[str, Sequence[str]]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        batch_size: int = 5000,
    ) -> Generator[DetectionItem, None, None]:
        """Yields individual DetectionItem instances strictly ordered by frame_id and detection_index."""
        for batch in self.stream_batches(
            sources=sources,
            start_frame=start_frame,
            end_frame=end_frame,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            batch_size=batch_size,
        ):
            # Convert RecordBatch to dictionaries
            d_dict = batch.to_pydict()
            num_rows = batch.num_rows

            for i in range(num_rows):
                yield DetectionItem(
                    frame_id=d_dict["frame_id"][i],
                    detection_index=d_dict["detection_index"][i],
                    timestamp_ms=d_dict["timestamp_ms"][i],
                    x1=d_dict["x1"][i],
                    y1=d_dict["y1"][i],
                    x2=d_dict["x2"][i],
                    y2=d_dict["y2"][i],
                    bottom_center_x=d_dict["bottom_center_x"][i],
                    bottom_center_y=d_dict["bottom_center_y"][i],
                    confidence=d_dict["confidence"][i],
                    class_label=d_dict["class_label"][i],
                    source=d_dict["source"][i],
                    tile_id=d_dict["tile_id"][i],
                    chunk_id=d_dict["chunk_id"][i],
                    run_id=d_dict["run_id"][i],
                    config_id=d_dict["config_id"][i],
                )
