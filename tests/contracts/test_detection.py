"""Unit tests for Detection schema contracts and validation rules."""

import math
import pytest
import pyarrow as pa

from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    DETECTION_SCHEMA_VERSION,
    ALLOWED_SOURCES,
    DetectionRow,
    ValidationError,
    validate_detection_row,
    validate_detection_table,
    clip_box_to_image,
    compute_bottom_center,
)

SAMPLE_WIDTH = 3840.0
SAMPLE_HEIGHT = 2160.0


def make_valid_row(
    frame_id: int = 0,
    timestamp_ms: int = 0,
    detection_index: int = 0,
    x1: float = 100.0,
    y1: float = 200.0,
    x2: float = 200.0,
    y2: float = 500.0,
    confidence: float = 0.95,
    source: str = "BASE_15FPS",
    tile_id: str | None = None,
) -> DetectionRow:
    bc_x, bc_y = compute_bottom_center(x1, y1, x2, y2)
    return DetectionRow(
        schema_version=DETECTION_SCHEMA_VERSION,
        run_id="run_001",
        chunk_id=0,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        detection_index=detection_index,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        bottom_center_x=bc_x,
        bottom_center_y=bc_y,
        confidence=confidence,
        class_label="player",
        source=source,
        tile_id=tile_id,
        config_id="cfg_abc123",
    )


def test_valid_detection_row_passes():
    row = make_valid_row()
    validate_detection_row(row, SAMPLE_WIDTH, SAMPLE_HEIGHT)


def test_invalid_schema_version_rejected():
    row_dict = make_valid_row().to_dict()
    row_dict["schema_version"] = 2
    with pytest.raises(ValidationError, match="Invalid schema_version"):
        validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)


def test_invalid_source_rejected():
    row_dict = make_valid_row().to_dict()
    row_dict["source"] = "UNKNOWN_SOURCE"
    with pytest.raises(ValidationError, match="Invalid source"):
        validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)


def test_nan_inf_rejected():
    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        row_dict = make_valid_row().to_dict()
        row_dict["x1"] = bad_val
        with pytest.raises(ValidationError, match="non-finite"):
            validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)


def test_inverted_or_negative_box_rejected():
    # Inverted x1 >= x2
    row_dict = make_valid_row(x1=300.0, x2=200.0).to_dict()
    with pytest.raises(ValidationError, match="Invalid horizontal bounds"):
        validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)

    # Inverted y1 >= y2
    row_dict = make_valid_row(y1=600.0, y2=500.0).to_dict()
    with pytest.raises(ValidationError, match="Invalid vertical bounds"):
        validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)

    # Negative x1
    row_dict = make_valid_row(x1=-10.0, x2=100.0).to_dict()
    with pytest.raises(ValidationError, match="Invalid horizontal bounds"):
        validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)


def test_out_of_bounds_and_clipping():
    # Out of bounds x2 > width
    row_dict = make_valid_row(x1=3700.0, x2=3900.0).to_dict()
    with pytest.raises(ValidationError, match="Invalid horizontal bounds"):
        validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)

    # Clipping test
    cx1, cy1, cx2, cy2 = clip_box_to_image(-50.0, -10.0, 3900.0, 2200.0, SAMPLE_WIDTH, SAMPLE_HEIGHT)
    assert cx1 == 0.0
    assert cy1 == 0.0
    assert cx2 == SAMPLE_WIDTH
    assert cy2 == SAMPLE_HEIGHT


def test_bottom_center_tolerance():
    row_dict = make_valid_row(x1=100.0, y1=200.0, x2=200.0, y2=500.0).to_dict()
    # Correct bc_x is 150.0, bc_y is 500.0
    row_dict["bottom_center_x"] = 160.0  # Deviation > 1e-3
    with pytest.raises(ValidationError, match="bottom_center_x mismatch"):
        validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)

    row_dict["bottom_center_x"] = 150.0001  # Deviation within 1e-3
    validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)


def test_confidence_bounds():
    row_dict = make_valid_row(confidence=-0.05).to_dict()
    with pytest.raises(ValidationError, match="Confidence.*out of range"):
        validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)

    row_dict = make_valid_row(confidence=1.05).to_dict()
    with pytest.raises(ValidationError, match="Confidence.*out of range"):
        validate_detection_row(row_dict, SAMPLE_WIDTH, SAMPLE_HEIGHT)


def test_table_validation_valid_and_duplicates():
    r1 = make_valid_row(frame_id=0, detection_index=0)
    r2 = make_valid_row(frame_id=0, detection_index=1, x1=300.0, x2=400.0)
    r3 = make_valid_row(frame_id=1, detection_index=0, timestamp_ms=33)

    rows = [r1.to_dict(), r2.to_dict(), r3.to_dict()]
    table = pa.Table.from_pylist(rows, schema=DETECTION_PYARROW_SCHEMA)
    validate_detection_table(table, SAMPLE_WIDTH, SAMPLE_HEIGHT)

    # Test duplicate key rejection
    duplicate_row = make_valid_row(frame_id=0, detection_index=0).to_dict()
    dup_table = pa.Table.from_pylist([r1.to_dict(), duplicate_row], schema=DETECTION_PYARROW_SCHEMA)
    with pytest.raises(ValidationError, match="Duplicate detection key"):
        validate_detection_table(dup_table, SAMPLE_WIDTH, SAMPLE_HEIGHT)


def test_table_validation_non_monotonic_ordering():
    r1 = make_valid_row(frame_id=10, timestamp_ms=333)
    r2 = make_valid_row(frame_id=5, timestamp_ms=166)  # Frame going backward

    table = pa.Table.from_pylist([r1.to_dict(), r2.to_dict()], schema=DETECTION_PYARROW_SCHEMA)
    with pytest.raises(ValidationError, match="Non-monotonic frame_id"):
        validate_detection_table(table, SAMPLE_WIDTH, SAMPLE_HEIGHT)
