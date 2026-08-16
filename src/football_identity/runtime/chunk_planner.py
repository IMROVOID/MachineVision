"""Time-based chunk planning for Detection pipeline."""

from __future__ import annotations

import math
from typing import List

from football_identity.runtime.chunk_state import ChunkRecord


def plan_chunks(
    total_frames: int,
    duration_ms: int,
    fps_num: int,
    fps_den: int,
    chunk_duration_sec: float,
    source: str = "BASE_15FPS",
    config_id: str = "",
) -> list[ChunkRecord]:
    """Generates contiguous, non-overlapping chunk plans covering all source frames.

    Args:
        total_frames: Total number of frames in the source video.
        duration_ms: Total duration in milliseconds.
        fps_num: FPS numerator.
        fps_den: FPS denominator.
        chunk_duration_sec: Configured chunk duration in seconds.
        source: Observation source type (e.g. 'BASE_15FPS').
        config_id: Canonical configuration hash.

    Returns:
        List of planned ChunkRecord instances in 'NOT_STARTED' state.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if chunk_duration_sec <= 0:
        raise ValueError(f"chunk_duration_sec must be positive, got {chunk_duration_sec}")
    if fps_num <= 0 or fps_den <= 0:
        raise ValueError(f"FPS fraction must be positive, got {fps_num}/{fps_den}")

    fps = float(fps_num) / float(fps_den)
    frames_per_chunk = max(1, int(round(chunk_duration_sec * fps)))

    chunks: list[ChunkRecord] = []
    current_frame = 0
    chunk_id = 0

    while current_frame < total_frames:
        end_frame = min(current_frame + frames_per_chunk, total_frames)
        start_time_ms = int(round(current_frame * 1000.0 / fps))
        end_time_ms = int(round(end_frame * 1000.0 / fps))

        chunk = ChunkRecord(
            chunk_id=chunk_id,
            start_frame=current_frame,
            end_frame=end_frame,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            source=source,
            config_id=config_id,
            attempt_count=0,
            state="NOT_STARTED",
        )
        chunks.append(chunk)

        current_frame = end_frame
        chunk_id += 1

    # Invariant verification: no gaps, no overlaps, exact total_frames coverage
    assert chunks[0].start_frame == 0
    assert chunks[-1].end_frame == total_frames
    for i in range(1, len(chunks)):
        assert chunks[i].start_frame == chunks[i - 1].end_frame, (
            f"Gap or overlap detected between chunk {i-1} (end: {chunks[i-1].end_frame}) and chunk {i} (start: {chunks[i].start_frame})"
        )

    return chunks
