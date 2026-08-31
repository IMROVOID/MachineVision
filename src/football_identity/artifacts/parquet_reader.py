"""Streaming Parquet reader for Detection partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional, Sequence, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    validate_detection_table,
)


class PartitionCorruptionError(Exception):
    """Raised when a partition file is corrupt or unreadable."""
    pass


def read_partition_table(
    file_path: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
) -> pa.Table:
    """Reads full partition Parquet table with validation."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Partition file not found: {p}")

    try:
        table = pq.read_table(p, columns=columns)
        return table
    except Exception as e:
        raise PartitionCorruptionError(f"Failed to read parquet partition {p}: {e}") from e


def stream_partition_batches(
    file_path: Union[str, Path],
    batch_size: int = 10000,
    columns: Optional[Sequence[str]] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Generator[pa.RecordBatch, None, None]:
    """Streams RecordBatches from a single Parquet file applying row-level predicate filters.

    Memory bounded: reads chunk by chunk from disk.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Partition file not found: {p}")

    try:
        parquet_file = pq.ParquetFile(p)
    except Exception as e:
        raise PartitionCorruptionError(f"Corrupted or invalid parquet file {p}: {e}") from e

    # Ensure predicate columns are loaded even if not in projection
    required_filter_cols: set[str] = set()
    if min_confidence is not None or max_confidence is not None:
        required_filter_cols.add("confidence")
    if start_frame is not None or end_frame is not None:
        required_filter_cols.add("frame_id")

    read_columns = None
    needs_projection = False
    if columns is not None:
        read_columns = list(dict.fromkeys(list(columns) + list(required_filter_cols)))
        if set(read_columns) != set(columns):
            needs_projection = True

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
        filtered_batch = batch

        # Build mask if filtering
        mask = None
        if min_confidence is not None:
            c_mask = pc.greater_equal(filtered_batch.column("confidence"), min_confidence)
            mask = c_mask if mask is None else pc.and_(mask, c_mask)

        if max_confidence is not None:
            c_mask = pc.less_equal(filtered_batch.column("confidence"), max_confidence)
            mask = c_mask if mask is None else pc.and_(mask, c_mask)

        if start_frame is not None:
            f_mask = pc.greater_equal(filtered_batch.column("frame_id"), start_frame)
            mask = f_mask if mask is None else pc.and_(mask, f_mask)

        if end_frame is not None:
            f_mask = pc.less_equal(filtered_batch.column("frame_id"), end_frame)
            mask = f_mask if mask is None else pc.and_(mask, f_mask)

        if mask is not None:
            filtered_batch = pc.filter(filtered_batch, mask)

        if needs_projection and columns is not None:
            filtered_batch = filtered_batch.select(list(columns))

        if filtered_batch.num_rows > 0:
            yield filtered_batch
