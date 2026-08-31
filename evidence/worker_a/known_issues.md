# Known Issues & Limitations — Worker A (Wave 1)

1. **Unresolved Issues:** None. All 46 mandatory and regression tests pass cleanly.
2. **Downstream Invalidation Scope:** As specified by Work Order Section 12.3, Wave 1 invalidation covers Detection artifacts only (PID-1) upon input/config/schema/model identity changes; cascade invalidation for future downstream PIDs (Tracking, Re-ID, Identity Graph) will be handled in subsequent waves.
3. **Hardware Considerations:** Streaming Parquet reader is verified to stream batches iteratively without loading whole 4K match detections into memory, ensuring safe execution within the 16 GB RAM boundary of the RTX 4050 / i7-13700 target system.
