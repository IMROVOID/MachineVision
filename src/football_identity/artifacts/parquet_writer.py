"""Atomic Parquet writer for Detection partitions."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

from football_identity.artifacts.manifest import compute_file_sha256, fsync_dir
from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    validate_detection_table,
)


def write_detection_partition_atomic(
    table: pa.Table,
    target_path: Union[str, Path],
    source_width: float,
    source_height: float,
    compression: str = "SNAPPY",
    tolerance: float = 1e-3,
    expected_run_id: Optional[str] = None,
    expected_source: Optional[str] = None,
    expected_config_id: Optional[str] = None,
    expected_chunk_id: Optional[int] = None,
    frame_bounds: Optional[tuple[int, int]] = None,
) -> tuple[int, int, str]:
    """Writes a detection table to Parquet atomically with strict validation and crash-safety.

    # ponytail: Local filesystem atomic rename (os.replace) with fsync.
    # Ceiling: single host / NVMe storage. Upgrade path: cloud object store two-phase commit if moved to distributed storage.

    Steps:
    1. Validate input table in-memory against schema, bounds, and expected chunk context.
    2. Write Parquet data to a temporary file in target directory.
    3. Flush and fsync temporary file to persistent storage.
    4. Read back table from disk and re-validate against schema and context.
    5. Independently compute row count, SHA-256 and byte size.
    6. Atomically publish (rename) to final target path and fsync parent directory.

    Returns:
        tuple[int, int, str]: (row_count, file_size_bytes, sha256_checksum)
    """
    dest = Path(target_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 1. Pre-validation in-memory
    validate_detection_table(
        table,
        source_width,
        source_height,
        tolerance=tolerance,
        expected_run_id=expected_run_id,
        expected_source=expected_source,
        expected_config_id=expected_config_id,
        expected_chunk_id=expected_chunk_id,
        frame_bounds=frame_bounds,
    )

    temp_path = dest.parent / f"{dest.name}.tmp.{uuid.uuid4().hex}"
    try:
        # Cast/ensure schema matches before writing
        table_to_write = table.cast(DETECTION_PYARROW_SCHEMA) if table.schema != DETECTION_PYARROW_SCHEMA else table
        
        # 2. Write to temporary file
        pq.write_table(
            table_to_write,
            temp_path,
            compression=compression,
            use_dictionary=True,
        )

        # 3. Flush & fsync file to disk
        with open(temp_path, "a+b") as f:
            f.flush()
            os.fsync(f.fileno())

        # 4. Read back and re-verify from disk
        read_back_table = pq.read_table(temp_path)
        validate_detection_table(
            read_back_table,
            source_width,
            source_height,
            tolerance=tolerance,
            expected_run_id=expected_run_id,
            expected_source=expected_source,
            expected_config_id=expected_config_id,
            expected_chunk_id=expected_chunk_id,
            frame_bounds=frame_bounds,
        )

        row_count = read_back_table.num_rows
        if row_count != table.num_rows:
            raise ValueError(
                f"Row count mismatch on disk verification: written {table.num_rows} vs read {row_count}"
            )

        # 5. Compute SHA-256 and byte size independently from disk
        sha256_sum, file_size = compute_file_sha256(temp_path)

        # 6. Atomically rename to final target and fsync directory
        os.replace(temp_path, dest)
        fsync_dir(dest.parent)

        return row_count, file_size, sha256_sum
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
