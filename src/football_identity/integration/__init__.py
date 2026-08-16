"""Integration package for downstream consumers (PID-2)."""

from football_identity.integration.detection_reader import (
    DetectionItem,
    DetectionStreamReader,
)

__all__ = [
    "DetectionItem",
    "DetectionStreamReader",
]
