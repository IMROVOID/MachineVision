"""Detection schema and validation contract for Football Identity Wave 1."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Set

import pyarrow as pa
import pyarrow.compute as pc

DETECTION_SCHEMA_VERSION: int = 1

ALLOWED_SOURCES: Set[str] = {
    "BASE_15FPS",
    "AUDIT_TILE_1FPS",
    "REPAIR_30FPS",
}

DETECTION_PYARROW_SCHEMA = pa.schema([
    pa.field("schema_version", pa.int16(), nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("chunk_id", pa.int32(), nullable=False),
    pa.field("frame_id", pa.int64(), nullable=False),
    pa.field("timestamp_ms", pa.int64(), nullable=False),
    pa.field("detection_index", pa.int32(), nullable=False),
    pa.field("x1", pa.float32(), nullable=False),
    pa.field("y1", pa.float32(), nullable=False),
    pa.field("x2", pa.float32(), nullable=False),
    pa.field("y2", pa.float32(), nullable=False),
    pa.field("bottom_center_x", pa.float32(), nullable=False),
    pa.field("bottom_center_y", pa.float32(), nullable=False),
    pa.field("confidence", pa.float32(), nullable=False),
    pa.field("class_label", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("tile_id", pa.string(), nullable=True),
    pa.field("config_id", pa.string(), nullable=False),
])


class ValidationError(ValueError):
    """Raised when data violates the detection schema or invariant contracts."""
    pass


def clip_box_to_image(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Clips bounding box coordinates explicitly to image bounds [0, width] x [0, height]."""
    cx1 = max(0.0, min(float(x1), float(width)))
    cy1 = max(0.0, min(float(y1), float(height)))
    cx2 = max(0.0, min(float(x2), float(width)))
    cy2 = max(0.0, min(float(y2), float(height)))
    return cx1, cy1, cx2, cy2


def compute_bottom_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Derives bottom center (bottom_center_x, bottom_center_y) from bounding box."""
    return ((float(x1) + float(x2)) / 2.0, float(y2))


@dataclass(frozen=True)
class DetectionRow:
    schema_version: int
    run_id: str
    chunk_id: int
    frame_id: int
    timestamp_ms: int
    detection_index: int
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
    config_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_detection_row(
    row: DetectionRow | Mapping[str, Any],
    width: float,
    height: float,
    tolerance: float = 1e-3,
) -> None:
    """Validates single row against Detection invariants."""
    data = row.to_dict() if isinstance(row, DetectionRow) else dict(row)

    if data.get("schema_version") != DETECTION_SCHEMA_VERSION:
        raise ValidationError(
            f"Invalid schema_version: expected {DETECTION_SCHEMA_VERSION}, got {data.get('schema_version')}"
        )

    source = data.get("source")
    if source not in ALLOWED_SOURCES:
        raise ValidationError(
            f"Invalid source: '{source}'. Allowed sources: {sorted(ALLOWED_SOURCES)}"
        )

    frame_id = data.get("frame_id")
    if not isinstance(frame_id, int) or frame_id < 0:
        raise ValidationError(f"Invalid frame_id: {frame_id} (must be non-negative integer)")

    timestamp_ms = data.get("timestamp_ms")
    if not isinstance(timestamp_ms, int) or timestamp_ms < 0:
        raise ValidationError(f"Invalid timestamp_ms: {timestamp_ms} (must be non-negative integer)")

    detection_index = data.get("detection_index")
    if not isinstance(detection_index, int) or detection_index < 0:
        raise ValidationError(f"Invalid detection_index: {detection_index} (must be non-negative integer)")

    # Float coordinates and confidence
    for key in ("x1", "y1", "x2", "y2", "bottom_center_x", "bottom_center_y", "confidence"):
        val = data.get(key)
        if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            raise ValidationError(f"Invalid non-finite or missing numeric value for '{key}': {val}")

    x1 = float(data["x1"])
    y1 = float(data["y1"])
    x2 = float(data["x2"])
    y2 = float(data["y2"])
    bc_x = float(data["bottom_center_x"])
    bc_y = float(data["bottom_center_y"])
    conf = float(data["confidence"])

    if not (0.0 <= conf <= 1.0):
        raise ValidationError(f"Confidence {conf} out of range [0.0, 1.0]")

    if not (0.0 <= x1 < x2 <= width):
        raise ValidationError(
            f"Invalid horizontal bounds: 0 <= x1 ({x1}) < x2 ({x2}) <= width ({width})"
        )

    if not (0.0 <= y1 < y2 <= height):
        raise ValidationError(
            f"Invalid vertical bounds: 0 <= y1 ({y1}) < y2 ({y2}) <= height ({height})"
        )

    expected_bc_x = (x1 + x2) / 2.0
    if abs(bc_x - expected_bc_x) > tolerance:
        raise ValidationError(
            f"bottom_center_x mismatch: {bc_x} vs expected {expected_bc_x} (tolerance={tolerance})"
        )

    if abs(bc_y - y2) > tolerance:
        raise ValidationError(
            f"bottom_center_y mismatch: {bc_y} vs expected {y2} (tolerance={tolerance})"
        )


def validate_detection_table(
    table: pa.Table,
    width: float,
    height: float,
    tolerance: float = 1e-3,
) -> None:
    """Vectorized validation for a PyArrow Table of detections."""
    if table is None or not isinstance(table, pa.Table):
        raise ValidationError("Expected a pyarrow.Table instance")

    # 1. Schema check
    expected_names = set(DETECTION_PYARROW_SCHEMA.names)
    actual_names = set(table.column_names)
    if expected_names != actual_names:
        missing = expected_names - actual_names
        extra = actual_names - expected_names
        raise ValidationError(f"Schema column mismatch. Missing: {missing}, Extra: {extra}")

    if table.num_rows == 0:
        return

    # Check nullability on non-nullable fields
    for field in DETECTION_PYARROW_SCHEMA:
        if not field.nullable:
            col = table.column(field.name)
            if col.null_count > 0:
                raise ValidationError(f"Non-nullable column '{field.name}' contains {col.null_count} nulls")

    # 2. Schema version check
    sv_col = table.column("schema_version")
    if not pc.all(pc.equal(sv_col, DETECTION_SCHEMA_VERSION)).as_py():
        raise ValidationError(f"All rows must have schema_version == {DETECTION_SCHEMA_VERSION}")

    # 3. Source check
    sources = set(table.column("source").to_pylist())
    invalid_sources = sources - ALLOWED_SOURCES
    if invalid_sources:
        raise ValidationError(f"Invalid sources found: {invalid_sources}. Allowed: {ALLOWED_SOURCES}")

    # 4. Frame & Timestamp & Detection index non-negative
    frame_ids = table.column("frame_id")
    if pc.any(pc.less(frame_ids, 0)).as_py():
        raise ValidationError("Negative frame_id detected")

    timestamps = table.column("timestamp_ms")
    if pc.any(pc.less(timestamps, 0)).as_py():
        raise ValidationError("Negative timestamp_ms detected")

    det_indices = table.column("detection_index")
    if pc.any(pc.less(det_indices, 0)).as_py():
        raise ValidationError("Negative detection_index detected")

    # 5. Check finite coordinates & confidence
    x1 = table.column("x1")
    y1 = table.column("y1")
    x2 = table.column("x2")
    y2 = table.column("y2")
    bc_x = table.column("bottom_center_x")
    bc_y = table.column("bottom_center_y")
    conf = table.column("confidence")

    for name, col in [("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2),
                      ("bottom_center_x", bc_x), ("bottom_center_y", bc_y), ("confidence", conf)]:
        if not pc.all(pc.is_finite(col)).as_py():
            raise ValidationError(f"Column '{name}' contains non-finite (NaN or Inf) values")

    # 6. Coordinate boundaries
    if pc.any(pc.less(x1, 0.0)).as_py():
        raise ValidationError(f"x1 contains negative coordinates")
    if pc.any(pc.greater(x2, float(width))).as_py():
        raise ValidationError(f"x2 exceeds width boundary ({width})")
    if pc.any(pc.greater_equal(x1, x2)).as_py():
        raise ValidationError("Bounding box invariant x1 < x2 violated")

    if pc.any(pc.less(y1, 0.0)).as_py():
        raise ValidationError(f"y1 contains negative coordinates")
    if pc.any(pc.greater(y2, float(height))).as_py():
        raise ValidationError(f"y2 exceeds height boundary ({height})")
    if pc.any(pc.greater_equal(y1, y2)).as_py():
        raise ValidationError("Bounding box invariant y1 < y2 violated")

    # 7. Confidence in [0, 1]
    if pc.any(pc.less(conf, 0.0)).as_py() or pc.any(pc.greater(conf, 1.0)).as_py():
        raise ValidationError("Confidence contains values outside [0.0, 1.0]")

    # 8. Bottom center derivation
    expected_bc_x = pc.divide(pc.add(x1, x2), 2.0)
    diff_x = pc.abs(pc.subtract(bc_x, expected_bc_x))
    if pc.any(pc.greater(diff_x, tolerance)).as_py():
        raise ValidationError(f"bottom_center_x does not match (x1 + x2)/2 within tolerance {tolerance}")

    diff_y = pc.abs(pc.subtract(bc_y, y2))
    if pc.any(pc.greater(diff_y, tolerance)).as_py():
        raise ValidationError(f"bottom_center_y does not match y2 within tolerance {tolerance}")

    # 9. Key uniqueness: (run_id, source, frame_id, tile_id, detection_index)
    run_ids = table.column("run_id").to_pylist()
    src_list = table.column("source").to_pylist()
    f_list = table.column("frame_id").to_pylist()
    t_list = table.column("tile_id").to_pylist()
    idx_list = table.column("detection_index").to_pylist()

    seen_keys = set()
    for r, s, f, t, idx in zip(run_ids, src_list, f_list, t_list, idx_list):
        k = (r, s, f, t, idx)
        if k in seen_keys:
            raise ValidationError(f"Duplicate detection key found: {k}")
        seen_keys.add(k)

    # 10. Monotonic ordering check by frame_id and timestamp_ms
    # Across consecutive rows, frame_id must be non-decreasing, and timestamp_ms non-decreasing
    for i in range(1, len(f_list)):
        if f_list[i] < f_list[i - 1]:
            raise ValidationError(
                f"Non-monotonic frame_id ordering at row {i}: {f_list[i-1]} -> {f_list[i]}"
            )
        if timestamps[i].as_py() < timestamps[i - 1].as_py():
            raise ValidationError(
                f"Non-monotonic timestamp_ms ordering at row {i}: {timestamps[i-1].as_py()} -> {timestamps[i].as_py()}"
            )
