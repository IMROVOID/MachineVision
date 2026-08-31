"""Detection test fixtures package for Football Identity Wave 1."""

from tests.fixtures.detections.synthetic import (
    create_test_detection_row,
    create_test_detection_table,
    create_test_multichunk_partitions,
)

__all__ = [
    "create_test_detection_row",
    "create_test_detection_table",
    "create_test_multichunk_partitions",
]
