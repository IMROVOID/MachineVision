# Wave 1 Work Order A

## Detection Infrastructure and Integration Foundation

**Assigned role:** Worker A — Infrastructure and Integration  
**Execution window:** One working week  
**Authority:** Implementation only; no architecture, model, or gate approval authority  
**Project mode:** `IDENTITY_ONLY`  
**Target input:** One full-match 4K, 30 FPS football video  
**Target hardware:** RTX 4050 Laptop GPU, Core i7-13700, 16 GB RAM  
**Coordination assumption:** Worker A must complete this assignment without relying on direct communication with Worker B.  

---

## 1. Mission

Implement the smallest reliable infrastructure required to store, validate, resume, and expose PID-1 Detection results to PID-2.

This assignment is not a request to build a general workflow platform. The required result is a minimal, deterministic Detection data plane and run-state layer for one long 4K video.

The worker must implement the frozen contracts in this document exactly. If an ambiguity prevents exact implementation, the worker must stop the affected item and report `PENDING_MANAGER_DECISION`; the worker must not invent a replacement contract.

---

## 2. Project Context

The complete product reconstructs persistent football-player identities over a full match. Wave 1 creates reusable player/person detections. Tracking, team, role, identity features, jersey recognition, global association, human review, and Pixel-to-Map are later phases.

Detection is expensive and must not be rerun whenever a downstream stage changes. Therefore Wave 1 must produce compact, validated, versioned, chunked artifacts that are independent of detector-specific Python objects.

---

## 3. In Scope

Worker A owns only the following:

1. project package and test skeleton required by this Work Order;
2. resolved run configuration loading and canonical hashing;
3. source-video fingerprint record;
4. Detection schema and validation;
5. artifact directory layout;
6. Parquet Detection writer and reader;
7. chunk planning and chunk-state records;
8. atomic artifact publication;
9. minimal resume behavior;
10. minimal invalidation on incompatible input/config/schema/model identity;
11. lightweight structured logging for infrastructure operations;
12. PID-2-facing Detection reader interface and smoke fixture;
13. unit, integration, negative, corruption, and resume tests;
14. complete delivery evidence.

---

## 4. Explicit Non-Goals

Worker A must not implement or modify:

- video decoding;
- CUDA preprocessing;
- YOLO, RF-DETR, TensorRT, or any detector adapter;
- candidate benchmarking or model selection;
- visual overlay generation;
- local tracking or tracklet creation;
- team, role, Re-ID, memory, jersey, graph, solver, or timeline logic;
- Pixel-to-Map;
- annotation tools or datasets;
- fine-tuning;
- a web UI or dashboard;
- Prefect, Airflow, Ray, Spark, Celery, Kafka, or another distributed framework;
- a database server or vector database;
- a generic plugin system;
- a full dependency graph for all future PIDs;
- production license inventory for all future models;
- precise downstream invalidation beyond the Detection-stage rules in this document.

Adding any non-goal is a scope violation, not an enhancement.

---

## 5. Required Inputs

The Project Manager must provide or confirm:

- repository root;
- supported Python version;
- source-video path for integration tests;
- expected source resolution and nominal FPS;
- run root directory;
- model/config identity values supplied by Worker B or a manager-provided fixture;
- approved base, audit, and repair source labels.

If the actual repository already defines equivalent package paths, Worker A must report the conflict before moving or duplicating code.

---

## 6. Frozen Logical Package Boundaries

Unless the Project Manager supplies an existing equivalent structure, Worker A owns these logical paths:

```text
src/football_identity/contracts/
src/football_identity/runtime/
src/football_identity/artifacts/
src/football_identity/integration/detection_reader.py

tests/contracts/
tests/runtime/
tests/artifacts/
tests/integration/test_detection_reader.py
tests/fixtures/detections/
```

Worker A must not create production files under:

```text
src/football_identity/video/
src/football_identity/detection/
src/football_identity/tracking/
src/football_identity/identity/
```

Test-local fake Detection rows are allowed only under `tests/fixtures/detections/`.

---

## 7. Frozen Artifact Layout

Each run must use the following logical layout:

```text
runs/<run_id>/
  run_manifest.json
  config.resolved.yaml
  video_fingerprint.json
  logs/
    infrastructure.jsonl
  pid01_detection/
    detection_manifest.json
    chunks.json
    base/
      chunk-00000.parquet
      chunk-00001.parquet
    audit/
      chunk-00000.parquet
    repair/
      repair-<interval_id>.parquet
    previews/
    metrics/
```

The worker may add temporary files only with an explicit temporary suffix. Temporary files must never be reported as completed artifacts.

Large JSON detection dumps, full-frame image dumps, and complete crop archives are forbidden.

---

## 8. Source Video Fingerprint Contract

`video_fingerprint.json` must contain at least:

```json
{
  "schema_name": "football_identity.video_fingerprint",
  "schema_version": 1,
  "source_path_recorded": "<string>",
  "file_size_bytes": 0,
  "sha256": "<64 lowercase hex characters>",
  "width": 3840,
  "height": 2160,
  "nominal_fps_num": 30,
  "nominal_fps_den": 1,
  "duration_ms": 0,
  "frame_count_reported": 0,
  "container": "<string>",
  "video_codec": "<string>"
}
```

Rules:

- SHA-256 is computed from the source bytes, not from the path.
- The recorded path is metadata and must not be used as identity.
- Missing or unreliable container metadata must be represented explicitly, not fabricated.
- A different SHA-256 means a different source video even when filenames match.

---

## 9. Detection Row Contract

### 9.1 Storage format

- primary format: Parquet;
- one row per Detection;
- partitioned by source type and chunk or repair interval;
- compression: a standard Parquet compression available in the approved environment;
- no detector-specific object serialization;
- no pickle as a production artifact.

### 9.2 Required columns

| Column | Type | Rule |
|---|---:|---|
| `schema_version` | int16 | Must equal `1`. |
| `run_id` | string | Must match the run manifest. |
| `chunk_id` | int32 | Non-negative base/audit chunk number; repair uses the parent chunk or documented interval mapping. |
| `frame_id` | int64 | Zero-based source-frame identifier. |
| `timestamp_ms` | int64 | Non-negative source timestamp; monotonically non-decreasing by frame. |
| `detection_index` | int32 | Zero-based and unique within `(frame_id, source, tile_id)`. |
| `x1` | float32 | Source-frame pixel coordinate. |
| `y1` | float32 | Source-frame pixel coordinate. |
| `x2` | float32 | Source-frame pixel coordinate and `x2 > x1`. |
| `y2` | float32 | Source-frame pixel coordinate and `y2 > y1`. |
| `bottom_center_x` | float32 | Must equal `(x1 + x2) / 2` within tolerance. |
| `bottom_center_y` | float32 | Must equal `y2` within tolerance. |
| `confidence` | float32 | Finite and within `[0, 1]`. |
| `class_label` | string | Broad person/athlete class defined by the resolved config. |
| `source` | string | One of `BASE_15FPS`, `AUDIT_TILE_1FPS`, `REPAIR_30FPS`. |
| `tile_id` | string/null | Null for full-frame base; populated for tiled audit or repair when applicable. |
| `config_id` | string | Canonical identifier for model and preprocessing settings in the Detection manifest. |

### 9.3 Coordinate rules

- All stored coordinates are in original source-frame coordinates.
- Coordinates must be finite.
- `0 <= x1 < x2 <= source_width`.
- `0 <= y1 < y2 <= source_height`.
- Clipping to image bounds must be explicit and tested.
- Resize, letterbox, and tile offsets must not leak into the stored coordinate system.

### 9.4 Uniqueness

The unique row key is:

```text
(run_id, source, frame_id, tile_id, detection_index)
```

Duplicate keys invalidate the partition.

---

## 10. Detection Manifest Contract

`pid01_detection/detection_manifest.json` must contain:

- schema name and version;
- run ID;
- video SHA-256;
- source resolution;
- model name;
- model weight digest;
- inference backend;
- preprocessing/config canonical digest;
- resolved input resolution;
- base FPS;
- audit FPS;
- batch size;
- confidence thresholds used for storage;
- artifact format and compression;
- code version identifier;
- dependency versions relevant to serialization;
- partition inventory with file size, row count, and checksum;
- creation and completion timestamps;
- final state.

Allowed final states are:

```text
IN_PROGRESS
COMPLETED
FAILED
INVALIDATED
```

`COMPLETED` is legal only after all required files pass validation.

---

## 11. Configuration and Canonical Identity

Worker A must implement deterministic canonical hashing for the resolved Detection configuration.

The same semantic configuration must produce the same `config_id`. At minimum, the hash input must cover:

- model identifier;
- model weight digest;
- backend;
- detector input resolution;
- base/audit/repair FPS policies;
- batch size;
- confidence storage thresholds;
- preprocessing and letterbox policy;
- tile layout and overlap policy;
- schema version;
- implementation version.

Changing any of these values invalidates reuse of incompatible Detection artifacts.

YAML formatting, key ordering, or comments must not change the canonical hash.

---

## 12. Chunking Contract

### 12.1 Initial policy

- chunking is time-based but materialized as exact source-frame ranges;
- initial configured duration: 300 seconds;
- the final accepted duration remains manager-controlled after runtime evidence;
- every chunk records inclusive `start_frame` and exclusive `end_frame`;
- no frame may be silently omitted from the requested observation schedule;
- source-frame ownership between base chunks must not overlap;
- audit and repair observations may overlap base observations because their `source` differs.

### 12.2 Chunk states

```text
NOT_STARTED
RUNNING
COMPLETED
FAILED
INVALIDATED
```

Each chunk record must include:

- chunk ID;
- frame and time boundaries;
- requested source type;
- config ID;
- attempt count;
- state;
- output path when published;
- row count;
- checksum;
- started/completed/failed timestamp;
- concise failure classification.

### 12.3 Resume rules

- a completed chunk with matching video/config/schema identity is reused;
- a temporary or corrupt output is not reused;
- a failed chunk may be retried without rewriting valid completed chunks;
- resume must not create duplicate partitions or duplicate rows;
- a changed video/config/schema/model identity invalidates incompatible PID-1 artifacts;
- Wave 1 does not implement precise downstream invalidation for later PIDs.

---

## 13. Atomic Publication

For every manifest or Parquet partition:

1. write to a temporary path;
2. flush and close;
3. validate schema and content;
4. compute final metadata/checksum;
5. atomically publish to the final path;
6. update chunk state;
7. update the Detection manifest only after the partition is valid.

A crash before publication must leave no artifact that can be mistaken for `COMPLETED`.

---

## 14. PID-2 Reader Interface

Worker A must expose a detector-neutral reader that:

- selects by run ID, source type, frame range, and optional confidence range;
- returns rows ordered by `frame_id`, then `detection_index`;
- validates artifact identity before reading;
- returns source-coordinate boxes and bottom centers;
- does not import YOLO, RF-DETR, TensorRT, CUDA, or video-decoder code;
- can stream batches without loading the entire match into 16 GB RAM;
- exposes low-confidence detections without granting them any identity authority.

The smoke fixture must demonstrate that a future PID-2 consumer can read detections from at least two chunks and across a chunk boundary.

No tracking algorithm is required.

---

## 15. Ordered Implementation Tasks

### Task A1 — Preflight and repository inventory

- record branch, commit, Python version, OS, and dependency state;
- identify existing equivalent modules;
- report path conflicts before implementation;
- confirm that no Tracking or detector logic will be modified.

### Task A2 — Contracts

- implement schema constants and typed validation;
- implement coordinate, type, uniqueness, timestamp, and source validation;
- create small valid and invalid fixtures.

### Task A3 — Configuration identity

- implement resolved config loading;
- implement canonical serialization and digest;
- prove deterministic hashes with tests.

### Task A4 — Video fingerprint metadata

- implement byte-based SHA-256 identity;
- store media metadata without fabricating unknown values;
- validate mismatch behavior.

### Task A5 — Artifact writer and reader

- implement Parquet partition writer;
- implement checksum/inventory generation;
- implement streaming reader;
- validate multi-chunk ordering and filtering.

### Task A6 — Chunk state and resume

- implement chunk planning;
- implement states and atomic transitions;
- implement reuse of matching completed chunks;
- implement retry of failed/incomplete chunks;
- implement rejection of corrupt or incompatible artifacts.

### Task A7 — PID-2 reader smoke integration

- read two fixture chunks;
- cross a chunk boundary;
- expose base, audit, and repair source labels;
- prove detector-library independence.

### Task A8 — Evidence and handoff

- run all required tests;
- produce requirement-to-evidence mapping;
- document unresolved issues without hiding them;
- deliver exact file inventory and run commands.

---

## 16. Required Tests

### 16.1 Unit tests

At minimum:

- valid schema round trip;
- invalid schema version rejection;
- NaN/Inf rejection;
- negative or inverted box rejection;
- out-of-bounds box rejection or explicit clipping test;
- bottom-center derivation validation;
- confidence bounds validation;
- duplicate-key rejection;
- canonical config hash stability;
- config hash change on every identity-bearing field;
- chunk boundary calculation;
- video hash mismatch rejection;
- manifest state-transition validation.

### 16.2 Integration tests

At minimum:

- write and read two valid Parquet chunks;
- ordered streaming across a chunk boundary;
- base/audit/repair source filtering;
- completed-chunk reuse;
- failed-chunk retry;
- interrupted temporary write recovery;
- corruption detection;
- incompatible config invalidation;
- reader memory behavior on a generated multi-partition fixture;
- PID-2 smoke consumer without detector imports.

### 16.3 Negative tests

The suite must prove failure for:

- missing partition referenced by a completed manifest;
- checksum mismatch;
- duplicate unique keys;
- wrong video fingerprint;
- wrong config ID;
- partial temporary file presented as final;
- unsupported schema version;
- non-monotonic frame/timestamp ordering within a partition.

### 16.4 Mandatory test command

The final handoff must provide and execute a single documented command equivalent to:

```bash
python -m pytest tests/contracts tests/runtime tests/artifacts tests/integration/test_detection_reader.py -q
```

The exact repository command may differ only if the Project Manager supplied a different test runner. Exit code must be zero. Skipped mandatory tests count as failures unless the manager approved the skip in writing.

---

## 17. Acceptance Criteria

Worker A's delivery is acceptable only if:

1. every required column and manifest field is implemented;
2. source-coordinate invariants are enforced;
3. Parquet round-trip preserves row values within declared numeric tolerance;
4. completed chunks are reused without byte changes;
5. a simulated interruption resumes without recomputing valid chunks;
6. corruption and identity mismatches are detected;
7. the PID-2 reader streams without loading the full match;
8. no detector, Tracking, database, dashboard, or distributed framework was added;
9. all mandatory tests pass with no unapproved skip;
10. the evidence package is complete.

Worker A does not pass Gate D. Worker A only delivers infrastructure evidence to the Project Manager.

---

## 18. Required Evidence Package

Deliver under a manager-approved evidence directory:

```text
evidence/worker_a/
  preflight.md
  changed_files.txt
  dependency_versions.txt
  test_commands.txt
  test_output.txt
  requirements_matrix.md
  artifact_tree.txt
  sample_manifest.json
  sample_chunks.json
  corruption_test_result.txt
  resume_test_result.txt
  detection_reader_smoke_result.txt
  known_issues.md
  handoff.md
```

The requirements matrix must map every numbered requirement in this Work Order to:

- status: `PASS`, `FAIL`, `BLOCKED`, or `NOT_APPLICABLE`;
- evidence file and location;
- test name or artifact path;
- concise note.

`NOT_APPLICABLE` requires a written explanation and manager approval.

---

## 19. Stop and Escalation Conditions

Worker A must stop the affected work and report the issue if:

- the repository already has a conflicting canonical contract;
- the required Parquet dependency cannot be installed or licensed;
- video metadata is unreliable and would require fabricated values;
- atomic file publication is not supported by the target filesystem;
- an existing user change overlaps owned files;
- a schema change appears necessary;
- Worker B's expected output cannot fit the frozen contract;
- a required test would need Tracking or detector implementation;
- the worker is asked to weaken validation to make a test pass.

The worker must not work around these conditions by silently changing scope.

---

## 20. Forbidden Shortcuts

- storing production detections in pickle;
- storing all detections in one giant JSON file;
- marking a chunk complete before validation;
- using the video path as the video identity;
- ignoring corrupt partitions during normal reads;
- accepting unknown schema versions;
- loading all match detections into memory for ordinary iteration;
- hard-coding one local absolute path;
- implementing fake detector or Tracking success paths in production code;
- weakening tests after a failure;
- removing validation to improve runtime;
- claiming integration based only on unit tests.

---

## 21. Definition of Done

This Work Order is done only when:

- the complete implementation is present;
- all mandatory tests pass;
- the evidence package is complete;
- a valid two-chunk fixture can be written, resumed, read, and consumed by the PID-2 smoke reader;
- corrupted and incompatible artifacts are rejected;
- no out-of-scope subsystem was introduced;
- all blockers are disclosed;
- the Project Manager accepts the handoff.

Code completion without manager acceptance is not completion.

---

## 22. Handoff Summary Template

Worker A must finish with the following summary:

```text
Work Order: Wave 1 / Worker A
Status: PASS | FAIL | BLOCKED | PARTIAL
Repository branch and commit:
Python and dependency versions:
Files created:
Files modified:
Tests executed:
Test result and exit code:
Artifacts produced:
Resume test result:
Corruption test result:
PID-2 reader smoke result:
Scope deviations:
Known limitations:
Manager decisions required:
Requirements matrix path:
Evidence directory:
```

The worker must not use `PASS` if any mandatory item is failed, blocked, skipped without approval, or missing evidence.
