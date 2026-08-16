"""Unit tests for time-based Chunk Planner."""

import pytest
from football_identity.runtime.chunk_planner import plan_chunks


def test_chunk_planning_300s_30fps():
    # 10 minutes at 30 fps = 18000 frames
    total_frames = 18000
    duration_ms = 600000
    fps_num = 30
    fps_den = 1
    chunk_dur = 300.0  # 300 seconds -> 9000 frames per chunk

    chunks = plan_chunks(
        total_frames=total_frames,
        duration_ms=duration_ms,
        fps_num=fps_num,
        fps_den=fps_den,
        chunk_duration_sec=chunk_dur,
        source="BASE_15FPS",
        config_id="cfg_123",
    )

    assert len(chunks) == 2
    assert chunks[0].chunk_id == 0
    assert chunks[0].start_frame == 0
    assert chunks[0].end_frame == 9000
    assert chunks[0].start_time_ms == 0
    assert chunks[0].end_time_ms == 300000

    assert chunks[1].chunk_id == 1
    assert chunks[1].start_frame == 9000
    assert chunks[1].end_frame == 18000
    assert chunks[1].start_time_ms == 300000
    assert chunks[1].end_time_ms == 600000


def test_chunk_planning_with_remainder():
    # 700 frames at 30 fps, chunk duration = 10s (300 frames per chunk)
    chunks = plan_chunks(
        total_frames=700,
        duration_ms=23333,
        fps_num=30,
        fps_den=1,
        chunk_duration_sec=10.0,
    )

    assert len(chunks) == 3
    assert (chunks[0].start_frame, chunks[0].end_frame) == (0, 300)
    assert (chunks[1].start_frame, chunks[1].end_frame) == (300, 600)
    assert (chunks[2].start_frame, chunks[2].end_frame) == (600, 700)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        plan_chunks(total_frames=0, duration_ms=1000, fps_num=30, fps_den=1, chunk_duration_sec=10.0)
    with pytest.raises(ValueError):
        plan_chunks(total_frames=100, duration_ms=1000, fps_num=30, fps_den=1, chunk_duration_sec=-5.0)
