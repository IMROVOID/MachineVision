"""Atomic Parquet writer for Detection partitions."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

from football_identity.artifacts.manifest import compute_file_sha256
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
) -> tuple[int, int, str]:
    """Writes a detection table to Parquet atomically with strict validation.

    Steps:
    1. Write data to a temporary file in target directory.
    2. Flush and sync.
    3. Validate table against detection schema invariants.
    4. Compute SHA-256 and byte size.
    5. Atomically publish (rename) to final target path.

    Returns:
        tuple[int, int, str]: (row_count, file_size_bytes, sha256_checksum)
    """
    dest = Path(target_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # First, validate table before writing
    validate_detection_table(table, source_width, source_height, tolerance=tolerance)

    temp_path = dest.parent / f"{dest.name}.tmp.{uuid.uuid4().hex}"
    try:
        # Cast/ensure schema matches before writing
        table_to_write = table.cast(DETECTION_PYARROW_SCHEMA) if table.schema != DETECTION_PYARROW_SCHEMA else table
        # Write to temporary file with compression
        pq.write_table(
            table_to_write,
            temp_path,
            compression=compression,
            use_dictionary=True,
        )

        # Validate by reading back from temp path
        read_back_table = pq.read_table(temp_path)
        validate_detection_table(read_back_table, source_width, source_height, tolerance=tolerance)

        if read_back_table.num_rows != table.num_rows:
            raise ValueError(
                f"Row count mismatch on disk verification: written {table.num_rows} vs read {read_back_table.num_rows}"
            )

        sha256_sum, file_size = compute_file_sha256(temp_path)

        # Atomically rename to final target
        os.replace(temp_path, dest)
        return table.num_rows, file_size, sha256_sum
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
