"""Synthetic detection fixture generators for testing and validation."""

from __future__ import annotations

from typing import Optional, Sequence
import pyarrow as pa

from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    DETECTION_SCHEMA_VERSION,
    DetectionRow,
    compute_bottom_center,
)


def create_test_detection_row(
    run_id: str = "run_test_001",
    chunk_id: int = 0,
    frame_id: int = 0,
    timestamp_ms: int = 0,
    detection_index: int = 0,
    x1: float = 100.0,
    y1: float = 200.0,
    x2: float = 200.0,
    y2: float = 500.0,
    confidence: float = 0.95,
    class_label: str = "player",
    source: str = "BASE_15FPS",
    tile_id: Optional[str] = None,
    config_id: str = "cfg_test_001",
) -> DetectionRow:
    """Constructs a validated single DetectionRow."""
    bc_x, bc_y = compute_bottom_center(x1, y1, x2, y2)
    return DetectionRow(
        schema_version=DETECTION_SCHEMA_VERSION,
        run_id=run_id,
        chunk_id=chunk_id,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        detection_index=detection_index,
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        bottom_center_x=float(bc_x),
        bottom_center_y=float(bc_y),
        confidence=float(confidence),
        class_label=class_label,
        source=source,
        tile_id=tile_id,
        config_id=config_id,
    )


def create_test_detection_table(
    num_rows: int = 100,
    base_frame: int = 0,
    run_id: str = "run_test_001",
    chunk_id: int = 0,
    source: str = "BASE_15FPS",
    config_id: str = "cfg_test_001",
    tile_id: Optional[str] = None,
) -> pa.Table:
    """Constructs a valid PyArrow Table conforming strictly to DETECTION_PYARROW_SCHEMA."""
    rows = []
    for i in range(num_rows):
        f_id = base_frame + i // 5
        d_idx = i % 5
        x1 = float(100 + d_idx * 50)
        y1 = float(200 + d_idx * 30)
        x2 = float(x1 + 80.0)
        y2 = float(y1 + 180.0)
        bc_x, bc_y = compute_bottom_center(x1, y1, x2, y2)
        conf = 0.1 + (i % 10) * 0.09
        rows.append({
            "schema_version": DETECTION_SCHEMA_VERSION,
            "run_id": run_id,
            "chunk_id": chunk_id,
            "frame_id": f_id,
            "timestamp_ms": int(f_id * 33.333),
            "detection_index": d_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "bottom_center_x": bc_x,
            "bottom_center_y": bc_y,
            "confidence": conf,
            "class_label": "player",
            "source": source,
            "tile_id": tile_id,
            "config_id": config_id,
        })
    return pa.Table.from_pylist(rows, schema=DETECTION_PYARROW_SCHEMA)


def create_test_multichunk_partitions(
    run_id: str = "run_multichunk_test",
    config_id: str = "cfg_test_001",
    frames_per_chunk: int = 300,
    num_chunks: int = 2,
    source: str = "BASE_15FPS",
) -> list[pa.Table]:
    """Generates multiple contiguous chunk partitions for boundary and streaming tests."""
    tables = []
    for c_id in range(num_chunks):
        start_frame = c_id * frames_per_chunk
        t = create_test_detection_table(
            num_rows=frames_per_chunk * 2,
            base_frame=start_frame,
            run_id=run_id,
            chunk_id=c_id,
            source=source,
            config_id=config_id,
        )
        tables.append(t)
    return tables
