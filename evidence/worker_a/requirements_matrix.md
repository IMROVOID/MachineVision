# Wave 1 Worker A — Requirements-to-Evidence Traceability Matrix

| Req # | Requirement Description | Status | Evidence Location / Test | Note |
|---|---|---|---|---|
| **1** | Logical package boundaries (`contracts/`, `runtime/`, `artifacts/`, `integration/detection_reader.py`) | **PASS** | `src/football_identity/`, `evidence/worker_a/changed_files.txt` | Ownership scope strictly preserved. No out-of-scope files created. |
| **2** | Resolved run config loading and canonical hashing | **PASS** | `src/football_identity/contracts/config.py`<br>`tests/contracts/test_config.py` | Hash invariant to YAML ordering, whitespace, and comments; sensitive to every identity field. |
| **3** | Source-video fingerprint record (byte SHA-256) | **PASS** | `src/football_identity/contracts/video.py`<br>`tests/contracts/test_video.py` | SHA-256 computed on bytes, metadata recorded without fabrication. |
| **4** | Detection schema, column types, and coordinate validation | **PASS** | `src/football_identity/contracts/detection.py`<br>`tests/contracts/test_detection.py` | Schema v1, int16/int32/int64/float32 types, bottom-center derivation, finite/in-bounds checks. |
| **5** | Artifact directory layout (`runs/<run_id>/...`) | **PASS** | `src/football_identity/artifacts/layout.py`<br>`tests/artifacts/test_layout.py` | Exact layout matching Section 7 layout specification. |
| **6** | Parquet Detection writer and streaming reader | **PASS** | `src/football_identity/artifacts/parquet_writer.py`<br>`src/football_identity/artifacts/parquet_reader.py`<br>`tests/artifacts/test_parquet_io.py` | Bounded memory batch iteration, predicate filtering, Snappy compression. |
| **7** | Time-based chunk planning and state records | **PASS** | `src/football_identity/runtime/chunk_planner.py`<br>`src/football_identity/runtime/chunk_state.py`<br>`tests/runtime/test_chunk_planner.py` | Contiguous non-overlapping frame ranges covering all frames. |
| **8** | Atomic artifact publication | **PASS** | `src/football_identity/artifacts/parquet_writer.py`<br>`tests/artifacts/test_atomic_publication.py` | Temp write $\to$ sync $\to$ validate $\to$ atomic rename. |
| **9** | Minimal resume behavior & chunk reuse | **PASS** | `src/football_identity/runtime/state_engine.py`<br>`tests/runtime/test_resume_retry.py` | Matching verified chunks reused; failed/incomplete chunks retried without duplicating data. |
| **10** | Minimal invalidation on incompatible input/config/schema | **PASS** | `src/football_identity/runtime/state_engine.py`<br>`tests/runtime/test_invalidation.py` | Config or video mismatch invalidates incompatible artifacts. |
| **11** | Lightweight structured logging | **PASS** | `src/football_identity/runtime/logger.py` | JSONL logging to `runs/<run_id>/logs/infrastructure.jsonl`. |
| **12** | PID-2 neutral Detection reader interface & boundary fixture | **PASS** | `src/football_identity/integration/detection_reader.py`<br>`tests/integration/test_detection_reader.py` | Detector-neutral reader traverses multi-chunk boundary seamlessly without loading full match. |
| **13** | Unit, integration, negative, corruption, and resume tests | **PASS** | `tests/contracts/`, `tests/runtime/`, `tests/artifacts/`, `tests/integration/` | 44 mandatory and regression tests pass with zero failures and zero skips. |
| **14** | Complete delivery evidence package | **PASS** | `evidence/worker_a/` | All 14 evidence artifacts generated and verified. |
