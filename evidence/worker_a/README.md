# Wave 1 (Worker A) Test Suite, Results & Evidence Master Report

> **Module:** Detection Infrastructure and Integration Foundation  
> **Assigned Role:** Worker A — Infrastructure and Data Plane  
> **Status:** 100% PASS (46 of 46 tests passing, 0 skips, 0 failures)  
> **Target Hardware:** RTX 4050 Laptop GPU, Core i7-13700, 16 GB RAM  
> **Target Input:** One full-match 4K (3840×2160), 30 FPS football video (~180,000 frames)

---

## Quick Navigation
- 📑 **[Handoff Summary](./handoff.md)**
- 📊 **[Requirements Traceability Matrix](./requirements_matrix.md)**
- 💻 **[Test Commands Reference](./test_commands.txt)**
- 📜 **[Full Test Execution Output](./test_output.txt)**
- 🛡️ **[Corruption Test Output](./corruption_test_result.txt)**
- 🔄 **[Resume Test Output](./resume_test_result.txt)**
- 🔍 **[Smoke Integration Test Output](./detection_reader_smoke_result.txt)**
- 📋 **[Sample Detection Manifest](./sample_manifest.json)**
- 📋 **[Sample Chunks Manifest](./sample_chunks.json)**
- 🛠️ **[Preflight Report](./preflight.md)**
- 📝 **[Changed Files Inventory](./changed_files.txt)**
- ⚠️ **[Known Limitations](./known_issues.md)**

---

## 1. Executive Summary & Wave 1 Overview

Wave 1 implements the deterministic **Detection data plane and run-state infrastructure** required to store, validate, resume, and expose PID-1 player detections to downstream stages (PID-2 Local Tracking, Re-ID, Identity Graph).

### Key Architectural Invariants
1. **Deterministic Serialization:** PyArrow Parquet storage with Snappy compression; zero detector-specific Python objects or pickle files.
2. **Strict Invariant Validation:** Zero out-of-bounds bounding boxes, strict float finiteness, derived bottom centers $(x_1+x_2)/2, y_2$ within tolerance, unique composite keys `(run_id, source, frame_id, tile_id, detection_index)`, monotonic frame/time ordering.
3. **Atomic Publication:** Two-phase publish (`.tmp.<uuid>` $\to$ flush/sync $\to$ read-back validate $\to$ `os.replace`) prevents partial or corrupted artifacts on process crash.
4. **Canonical Config Identity:** SHA-256 hash invariant to YAML formatting, comments, or key order, but sensitive to any semantic config change.
5. **Byte-Level Video Fingerprint:** Source video identified by byte SHA-256 rather than file path.
6. **Fault-Tolerant State Engine:** Reuses completed, checksum-verified chunks; retries failed or interrupted chunks; invalidates on config or video mismatch.
7. **Detector-Neutral PID-2 Streaming Reader:** Generator-based streaming reading chunk-by-chunk with bounded memory (< 16 GB RAM) across chunk boundaries without importing PyTorch, Ultralytics, TensorRT, or CUDA.

---

## 2. Directory & Artifact Structure

```text
MachineVision/
├── pyproject.toml
├── README.md
├── dependency_changes.txt
├── docs/
│   └── 01_wave1_detection_infrastructure_work_order.md
├── src/
│   └── football_identity/
│       ├── __init__.py
│       ├── contracts/
│       │   ├── __init__.py
│       │   ├── config.py             # Canonical configuration hashing & loading
│       │   ├── detection.py          # PyArrow detection schema & row/table validation
│       │   └── video.py              # Source video SHA-256 fingerprinting
│       ├── artifacts/
│       │   ├── __init__.py
│       │   ├── layout.py             # Standardized run directory hierarchy
│       │   ├── manifest.py           # Detection manifest & partition inventory
│       │   ├── parquet_writer.py     # Atomic Parquet writer with disk validation
│       │   └── parquet_reader.py     # Streaming Parquet reader with predicate filters
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── chunk_planner.py      # Non-overlapping contiguous chunk scheduler
│       │   ├── chunk_state.py        # Chunk record dataclass & manifest serialization
│       │   ├── logger.py             # Structured JSONL infrastructure logger
│       │   └── state_engine.py       # Execution lifecycle, resume, retry, invalidation
│       └── integration/
│           ├── __init__.py
│           └── detection_reader.py   # Neutral streaming reader for PID-2 consumers
├── tests/
│   ├── README.md                     # Lightweight test quick-start
│   ├── contracts/
│   │   ├── test_config.py            # Canonical hash stability & sensitivity tests
│   │   ├── test_detection.py         # Schema, coordinate, and table invariant tests
│   │   └── test_video.py             # Video fingerprinting & mismatch tests
│   ├── runtime/
│   │   ├── test_chunk_planner.py     # Chunk boundary and remainder calculations
│   │   ├── test_invalidation.py      # Video and config change invalidation tests
│   │   ├── test_regression_repairs.py # Audit repairs & edge cases (9 tests)
│   │   ├── test_resume_retry.py      # Chunk reuse, retry, and corruption recovery
│   │   └── test_state_engine.py      # Complete run execution lifecycle tests
│   ├── artifacts/
│   │   ├── test_atomic_publication.py # Validation failure & disk crash recovery
│   │   ├── test_layout.py            # Directory structure & path resolution tests
│   │   ├── test_manifest.py          # Manifest serialization & checksum integrity
│   │   └── test_parquet_io.py        # Parquet read/write round-trip & filtering
│   └── integration/
│       └── test_detection_reader.py  # Two-chunk boundary traversal & isolation
└── evidence/
    └── worker_a/                     # Formal delivery & audit package
        ├── README.md                 # This master report
        ├── handoff.md
        ├── requirements_matrix.md
        ├── preflight.md
        ├── changed_files.txt
        ├── dependency_versions.txt
        ├── dependency_changes.txt
        ├── artifact_tree.txt
        ├── test_commands.txt
        ├── test_output.txt
        ├── corruption_test_result.txt
        ├── resume_test_result.txt
        ├── detection_reader_smoke_result.txt
        ├── sample_manifest.json
        ├── sample_chunks.json
        └── known_issues.md
```

---

## 3. Complete Test Catalog (All 46 Tests Explained)

### 3.1 `tests/contracts/` — Contracts & Schema Invariants (13 Tests)

| Test Name | File | Description & Verification Goal | Result |
|---|---|---|:---:|
| `test_canonical_hash_stability` | [`../../tests/contracts/test_config.py`](../../tests/contracts/test_config.py) | Proves identical config objects produce the exact same 64-char SHA-256 hash repeatedly. | **PASS** |
| `test_canonical_hash_key_order_and_yaml_insensitivity` | [`../../tests/contracts/test_config.py`](../../tests/contracts/test_config.py) | Verifies hash is invariant to dictionary key order, YAML whitespace, and comments. | **PASS** |
| `test_canonical_hash_sensitivity_to_each_field` | [`../../tests/contracts/test_config.py`](../../tests/contracts/test_config.py) | Proves mutating any single identity-bearing field (model, weights, backend, resolution, FPS policies, batch size, thresholds, tile policy) alters the hash. | **PASS** |
| `test_valid_detection_row_passes` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Validates that a standard compliant `DetectionRow` passes all contract checks. | **PASS** |
| `test_invalid_schema_version_rejected` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Rejects records with `schema_version != 1`. | **PASS** |
| `test_invalid_source_rejected` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Rejects unknown sources outside `{"BASE_15FPS", "AUDIT_TILE_1FPS", "REPAIR_30FPS"}`. | **PASS** |
| `test_nan_inf_rejected` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Ensures `NaN`, `+Inf`, `-Inf` values in coordinates or confidence are rejected. | **PASS** |
| `test_inverted_or_negative_box_rejected` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Rejects inverted boxes ($x_1 \ge x_2$, $y_1 \ge y_2$) or negative pixel coordinates ($x_1 < 0, y_1 < 0$). | **PASS** |
| `test_out_of_bounds_and_clipping` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Tests boundary enforcement against image resolution ($x_2 > W, y_2 > H$) and tests `clip_box_to_image`. | **PASS** |
| `test_bottom_center_tolerance` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Validates bottom center derivation within declared tolerance ($10^{-3}$), rejecting arbitrary coordinates. | **PASS** |
| `test_confidence_bounds` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Rejects confidence values outside $[0.0, 1.0]$. | **PASS** |
| `test_table_validation_valid_and_duplicates` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Vectorized table validation; rejects duplicate keys across `(run_id, source, frame_id, tile_id, detection_index)`. | **PASS** |
| `test_table_validation_non_monotonic_ordering` | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | Ensures frames and timestamps are strictly non-decreasing within each partition. | **PASS** |
| `test_compute_video_sha256` | [`../../tests/contracts/test_video.py`](../../tests/contracts/test_video.py) | Proves streaming byte-level SHA-256 calculation matches file content digest. | **PASS** |
| `test_valid_video_fingerprint` | [`../../tests/contracts/test_video.py`](../../tests/contracts/test_video.py) | Validates serialization and deserialization of `VideoFingerprint`. | **PASS** |
| `test_invalid_video_fingerprint_rejections` | [`../../tests/contracts/test_video.py`](../../tests/contracts/test_video.py) | Rejects invalid video fingerprints (corrupt SHA-256 length, negative dimensions, negative FPS). | **PASS** |

### 3.2 `tests/runtime/` — State Engine, Planner & Resilience (8 Tests)

| Test Name | File | Description & Verification Goal | Result |
|---|---|---|:---:|
| `test_chunk_planning_300s_30fps` | [`../../tests/runtime/test_chunk_planner.py`](../../tests/runtime/test_chunk_planner.py) | Tests planning 18,000 frames (10 min) into 300-second non-overlapping chunks (9,000 frames each). | **PASS** |
| `test_chunk_planning_with_remainder` | [`../../tests/runtime/test_chunk_planner.py`](../../tests/runtime/test_chunk_planner.py) | Tests chunk planning when total frames do not evenly divide, ensuring 100% frame coverage. | **PASS** |
| `test_invalid_parameters` | [`../../tests/runtime/test_chunk_planner.py`](../../tests/runtime/test_chunk_planner.py) | Rejects negative or zero FPS, total frames, or chunk duration. | **PASS** |
| `test_invalidation_on_config_change` | [`../../tests/runtime/test_invalidation.py`](../../tests/runtime/test_invalidation.py) | Proves that attempting to resume a run with a modified model or config transitions run to `INVALIDATED`. | **PASS** |
| `test_invalidation_on_video_mismatch` | [`../../tests/runtime/test_invalidation.py`](../../tests/runtime/test_invalidation.py) | Proves that resuming with a different source video SHA-256 invalidates the existing artifacts. | **PASS** |
| `test_resume_reuses_completed_and_retries_incomplete` | [`../../tests/runtime/test_resume_retry.py`](../../tests/runtime/test_resume_retry.py) | Proves completed chunks with verified checksums are untouched while interrupted/running chunks are retried. | **PASS** |
| `test_corruption_detection_triggers_retry` | [`../../tests/runtime/test_resume_retry.py`](../../tests/runtime/test_resume_retry.py) | Detects on-disk byte corruption via SHA-256 mismatch and automatically marks chunk for re-execution. | **PASS** |
| `test_state_engine_lifecycle` | [`../../tests/runtime/test_state_engine.py`](../../tests/runtime/test_state_engine.py) | Executes full lifecycle: initialize $\to$ running $\to$ completed partitions $\to$ manifest finalization to `COMPLETED`. | **PASS** |

### 3.3 `tests/artifacts/` — Artifacts, Layout & Atomic IO (8 Tests)

| Test Name | File | Description & Verification Goal | Result |
|---|---|---|:---:|
| `test_atomic_publication_on_validation_failure` | [`../../tests/artifacts/test_atomic_publication.py`](../../tests/artifacts/test_atomic_publication.py) | Validates that invalid table data fails before rename, leaving zero corrupt or leftover temp files. | **PASS** |
| `test_atomic_publication_on_disk_error` | [`../../tests/artifacts/test_atomic_publication.py`](../../tests/artifacts/test_atomic_publication.py) | Simulates disk crash during rename; verifies no partial file is published. | **PASS** |
| `test_artifact_layout_structure` | [`../../tests/artifacts/test_layout.py`](../../tests/artifacts/test_layout.py) | Verifies `RunArtifactLayout` creates exact required subdirectories (`base/`, `audit/`, `repair/`, `logs/`, etc.). | **PASS** |
| `test_manifest_save_and_load` | [`../../tests/artifacts/test_manifest.py`](../../tests/artifacts/test_manifest.py) | Tests atomic manifest saving and round-trip loading. | **PASS** |
| `test_manifest_state_transitions` | [`../../tests/artifacts/test_manifest.py`](../../tests/artifacts/test_manifest.py) | Verifies legal states (`IN_PROGRESS`, `COMPLETED`, `FAILED`, `INVALIDATED`) and rejects bogus states. | **PASS** |
| `test_manifest_integrity_verification` | [`../../tests/artifacts/test_manifest.py`](../../tests/artifacts/test_manifest.py) | Tests detection of missing files, size mismatches, and tampered checksums in partition inventory. | **PASS** |
| `test_parquet_write_and_read_roundtrip` | [`../../tests/artifacts/test_parquet_io.py`](../../tests/artifacts/test_parquet_io.py) | Writes valid table to Parquet and reads back full table, verifying numeric and schema fidelity. | **PASS** |
| `test_parquet_streaming_batches_and_filtering` | [`../../tests/artifacts/test_parquet_io.py`](../../tests/artifacts/test_parquet_io.py) | Tests chunked batch streaming with confidence and frame range predicate filtering. | **PASS** |

### 3.4 `tests/integration/` — PID-2 Integration & Smoke (3 Tests)

| Test Name | File | Description & Verification Goal | Result |
|---|---|---|:---:|
| `test_two_chunk_boundary_traversal` | [`../../tests/integration/test_detection_reader.py`](../../tests/integration/test_detection_reader.py) | Streams detections across chunk boundary (frame 290 to 310 across chunk 0 & 1), verifying continuous ordering. | **PASS** |
| `test_source_and_confidence_filtering` | [`../../tests/integration/test_detection_reader.py`](../../tests/integration/test_detection_reader.py) | Verifies multi-source partition filtering (`AUDIT_TILE_1FPS` vs `BASE_15FPS`) and confidence filtering. | **PASS** |
| `test_detector_independence` | [`../../tests/integration/test_detection_reader.py`](../../tests/integration/test_detection_reader.py) | Proves `DetectionStreamReader` does NOT import `torch`, `torchvision`, `ultralytics`, `cv2`, `cuda`, or `tensorrt`. | **PASS** |

### 3.5 `tests/runtime/test_regression_repairs.py` — Audit Repairs & Invariants (9 Tests)

| Test Name | File | Description & Verification Goal | Result |
|---|---|---|:---:|
| `test_multi_source_global_ordering` | [`../../tests/runtime/test_regression_repairs.py`](../../tests/runtime/test_regression_repairs.py) | Verifies `DetectionStreamReader` returns globally monotonic frame ordering across multiple observation sources. | **PASS** |
| `test_reader_expected_identity_rejection` | [`../../tests/runtime/test_regression_repairs.py`](../../tests/runtime/test_regression_repairs.py) | Ensures consumer reader validates expected run ID, video digest, and config identity before streaming. | **PASS** |
| `test_invalid_chunk_state_transitions` | [`../../tests/runtime/test_regression_repairs.py`](../../tests/runtime/test_regression_repairs.py) | Rejects illegal state machine jumps (e.g. `NOT_STARTED -> COMPLETED` or `INVALIDATED -> RUNNING`). | **PASS** |
| `test_pre_commit_row_and_boundary_rejection` | [`../../tests/runtime/test_regression_repairs.py`](../../tests/runtime/test_regression_repairs.py) | Proves atomic writer rejects rows outside chunk bounds `[start_frame, end_frame)` or with mismatched context. | **PASS** |
| `test_resume_incompatibility_invalidates_manifest_and_chunks` | [`../../tests/runtime/test_regression_repairs.py`](../../tests/runtime/test_regression_repairs.py) | Verifies resolution or config changes mark both manifests and chunks as `INVALIDATED` on disk. | **PASS** |
| `test_config_hash_covers_all_fields_without_float_rounding` | [`../../tests/runtime/test_regression_repairs.py`](../../tests/runtime/test_regression_repairs.py) | Proves canonical hash is sensitive to every field without lossy float precision rounding. | **PASS** |
| `test_root_artifacts_created_and_finalized` | [`../../tests/runtime/test_regression_repairs.py`](../../tests/runtime/test_regression_repairs.py) | Validates automated generation of `run_manifest.json`, `config.resolved.yaml`, and `video_fingerprint.json`. | **PASS** |
| `test_run_wide_duplicate_detection_rejection` | [`../../tests/runtime/test_regression_repairs.py`](../../tests/runtime/test_regression_repairs.py) | Detects and rejects duplicate detection keys across chunks throughout the entire run. | **PASS** |
| `test_multi_source_chunk_planning` | [`../../tests/runtime/test_regression_repairs.py`](../../tests/runtime/test_regression_repairs.py) | Verifies multi-source chunk planning independently for base, audit, and repair observation sources. | **PASS** |

---

## 4. Test Execution Guide

### 4.1 Environment Setup

1. **Activate Python Environment:**
   Ensure Python 3.10+ (tested on Python 3.14.3) is active.

2. **Install Package & Dependencies:**
   From the repository root:
   ```bash
   pip install -e .
   ```
   Or install requirements directly:
   ```bash
   pip install pyarrow>=14.0.0 pyyaml>=6.0 pydantic>=2.0 pytest>=8.0
   ```

### 4.2 Running the Test Suite

#### 1. Run Complete Mandatory Test Suite (Single Command)
```bash
python -m pytest tests/contracts tests/runtime tests/artifacts tests/integration/test_detection_reader.py -q
```
*Expected Output:*
```text
..............................................                           [100%]
46 passed in 1.44s
```

#### 2. Run Verbose Output with All Test Names
```bash
python -m pytest tests/contracts tests/runtime tests/artifacts tests/integration/test_detection_reader.py -v
```

#### 3. Run Specific Test Suites
```bash
# Run Contracts Suite
python -m pytest tests/contracts -v

# Run Runtime / State Engine Suite
python -m pytest tests/runtime -v

# Run Artifacts / Parquet IO Suite
python -m pytest tests/artifacts -v

# Run PID-2 Integration Suite
python -m pytest tests/integration -v
```

#### 4. Run Individual Test Scenarios
```bash
# Test Corruption Detection and Automatic Retry
python -m pytest tests/runtime/test_resume_retry.py::test_corruption_detection_triggers_retry -v

# Test Resume and Re-use of Completed Chunks
python -m pytest tests/runtime/test_resume_retry.py::test_resume_reuses_completed_and_retries_incomplete -v

# Test Multi-Chunk Boundary Traversal for PID-2
python -m pytest tests/integration/test_detection_reader.py::test_two_chunk_boundary_traversal -v

# Test Invalidation on Config Change
python -m pytest tests/runtime/test_invalidation.py::test_invalidation_on_config_change -v
```

---

## 5. Manual Testing & Backend API Guide

### 5.1 Writing & Reading Parquet Partitions

```python
import pyarrow as pa
from football_identity.contracts.detection import (
    DETECTION_PYARROW_SCHEMA,
    DETECTION_SCHEMA_VERSION,
    compute_bottom_center,
)
from football_identity.artifacts.parquet_writer import write_detection_partition_atomic
from football_identity.artifacts.parquet_reader import stream_partition_batches

# 1. Create compliant detection rows
x1, y1, x2, y2 = 100.0, 200.0, 200.0, 500.0
bc_x, bc_y = compute_bottom_center(x1, y1, x2, y2)

rows = [{
    "schema_version": DETECTION_SCHEMA_VERSION,
    "run_id": "match_001",
    "chunk_id": 0,
    "frame_id": 0,
    "timestamp_ms": 0,
    "detection_index": 0,
    "x1": x1,
    "y1": y1,
    "x2": x2,
    "y2": y2,
    "bottom_center_x": bc_x,
    "bottom_center_y": bc_y,
    "confidence": 0.95,
    "class_label": "player",
    "source": "BASE_15FPS",
    "tile_id": None,
    "config_id": "cfg_hash_123",
}]

table = pa.Table.from_pylist(rows, schema=DETECTION_PYARROW_SCHEMA)

# 2. Write atomically to disk
out_path = "runs/match_001/pid01_detection/base/chunk-00000.parquet"
row_count, file_size, checksum = write_detection_partition_atomic(
    table=table,
    target_path=out_path,
    source_width=3840.0,
    source_height=2160.0,
    compression="SNAPPY",
)
print(f"Written {row_count} rows, {file_size} bytes, SHA-256: {checksum}")

# 3. Stream batches back with predicate filters
for batch in stream_partition_batches(out_path, min_confidence=0.5, start_frame=0, end_frame=100):
    print(f"Streamed batch with {batch.num_rows} detections")
```

### 5.2 Using the PID-2 Detection Stream Reader

Downstream stages (PID-2 Local Tracking) consume detections without importing ML or detector libraries:

```python
from football_identity.integration.detection_reader import DetectionStreamReader

# Initialize reader for a run directory
reader = DetectionStreamReader(runs_root="runs", run_id="match_001")

# Stream detection items across all chunks in monotonic frame order
for det in reader.stream_detections(
    sources=["BASE_15FPS", "AUDIT_TILE_1FPS"],
    start_frame=0,
    end_frame=1000,
    min_confidence=0.3,
):
    print(
        f"Frame {det.frame_id:05d} | Det #{det.detection_index} | "
        f"Box: ({det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f}) | "
        f"BottomCenter: ({det.bottom_center_x:.1f}, {det.bottom_center_y:.1f}) | "
        f"Conf: {det.confidence:.2f} | Chunk: {det.chunk_id}"
    )
```

### 5.3 Managing Run Lifecycles & Resuming Interrupted Runs

```python
from football_identity.artifacts.layout import RunArtifactLayout
from football_identity.contracts.config import load_resolved_config
from football_identity.contracts.video import VideoFingerprint, validate_video_fingerprint
from football_identity.runtime.state_engine import RunStateEngine

layout = RunArtifactLayout(runs_root="runs", run_id="match_001")
config = load_resolved_config("runs/match_001/config.resolved.yaml")
video_fp = validate_video_fingerprint(...)

engine = RunStateEngine(layout, config, video_fp)

# Automatically plans fresh chunks or resumes uncorrupted completed chunks
all_chunks, pending_chunks = engine.initialize_or_resume(chunk_duration_sec=300.0)

for chunk in pending_chunks:
    engine.mark_chunk_running(chunk.chunk_id, chunk.source)
    # ... Worker B performs inference & writes Parquet ...
    engine.mark_chunk_completed(
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        output_relative_path=f"pid01_detection/base/chunk-{chunk.chunk_id:05d}.parquet",
        row_count=row_count,
        checksum=checksum,
    )

# Finalize run
status = engine.finalize_run()
print(f"Run finalized with status: {status}")
```

---

## 6. Verification Matrix

| Area | Requirement | Implemented In | Test Suite | Result |
|---|---|---|---|:---:|
| **Contracts** | Schema Version 1, Invariants, BBoxes | [`../../src/football_identity/contracts/detection.py`](../../src/football_identity/contracts/detection.py) | [`../../tests/contracts/test_detection.py`](../../tests/contracts/test_detection.py) | **PASS** |
| **Config** | Deterministic Canonical Hash | [`../../src/football_identity/contracts/config.py`](../../src/football_identity/contracts/config.py) | [`../../tests/contracts/test_config.py`](../../tests/contracts/test_config.py) | **PASS** |
| **Video** | Byte SHA-256 Fingerprint | [`../../src/football_identity/contracts/video.py`](../../src/football_identity/contracts/video.py) | [`../../tests/contracts/test_video.py`](../../tests/contracts/test_video.py) | **PASS** |
| **Storage** | Atomic Parquet Write & Stream Read | [`../../src/football_identity/artifacts/parquet_writer.py`](../../src/football_identity/artifacts/parquet_writer.py) | [`../../tests/artifacts/test_parquet_io.py`](../../tests/artifacts/test_parquet_io.py) | **PASS** |
| **Integrity** | Manifest Checksum Verification | [`../../src/football_identity/artifacts/manifest.py`](../../src/football_identity/artifacts/manifest.py) | [`../../tests/artifacts/test_manifest.py`](../../tests/artifacts/test_manifest.py) | **PASS** |
| **Layout** | Exact Hierarchy Layout | [`../../src/football_identity/artifacts/layout.py`](../../src/football_identity/artifacts/layout.py) | [`../../tests/artifacts/test_layout.py`](../../tests/artifacts/test_layout.py) | **PASS** |
| **Runtime** | Time-based Chunk Planning | [`../../src/football_identity/runtime/chunk_planner.py`](../../src/football_identity/runtime/chunk_planner.py) | [`../../tests/runtime/test_chunk_planner.py`](../../tests/runtime/test_chunk_planner.py) | **PASS** |
| **State** | Resume, Retry, Corruption Detection | [`../../src/football_identity/runtime/state_engine.py`](../../src/football_identity/runtime/state_engine.py) | [`../../tests/runtime/test_resume_retry.py`](../../tests/runtime/test_resume_retry.py) | **PASS** |
| **Invalidation** | Incompatible Config / Video Reset | [`../../src/football_identity/runtime/state_engine.py`](../../src/football_identity/runtime/state_engine.py) | [`../../tests/runtime/test_invalidation.py`](../../tests/runtime/test_invalidation.py) | **PASS** |
| **Integration** | Neutral PID-2 Boundary Streamer | [`../../src/football_identity/integration/detection_reader.py`](../../src/football_identity/integration/detection_reader.py) | [`../../tests/integration/test_detection_reader.py`](../../tests/integration/test_detection_reader.py) | **PASS** |
