# Baseline Submission Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the baseline import a complete query pack and download only AIC 2026 rule-compliant `submission.zip` files.

**Architecture:** Parse ZIP/multi-TXT input into a query manifest, use that manifest as the task/event-count authority, validate structured selections before writing, then reopen and parse the generated ZIP before returning it. The frontend keeps its current vanilla JavaScript/FastAPI shape and gains pack navigation plus validation reporting.

**Tech Stack:** Python 3.10+, standard-library `zipfile`/`csv`, FastAPI/Pydantic already declared by the project, vanilla JavaScript, pytest, Node built-in test runner.

**Spec:** `docs/superpowers/specs/2026-08-26-baseline-submission-validation-design.md`

## Global Constraints

- Work only in `aic/ui`, `aic/submission`, related tests, and these design/plan docs.
- Do not modify retrieval, fusion, pipeline, or `aic/core`.
- Do not push, open a PR, or upload a submission.
- Query manifest is authoritative; export must not trust the current UI task pill.
- Accept one ZIP or one/many TXT files; never extract ZIP contents to disk.
- KIS is two columns, Q&A is three columns with an exact 1..100-character answer, and TRAKE is `1 + N` columns with strictly increasing frames.
- Preserve exact query IDs, row order, Unicode, and Q&A leading/trailing whitespace.
- Emit UTF-8/no-BOM, comma-delimited, headerless CSV under `submission/` only.
- Every production behavior starts with a failing test and a witnessed RED run.

---

### Task 1: Query-pack parser and manifest

**Files:**
- Create: `aic/submission/query_pack.py`
- Create: `tests/test_query_pack.py`

**Interfaces:**
- Produces: `parse_query_files(files: Sequence[tuple[str, str | bytes]]) -> PackParseResult`
- Produces: `parse_query_zip(data: bytes) -> PackParseResult`
- Produces: `infer_task(query_id: str) -> str | None`
- Produces: `suggest_event_count(query_id: str, text: str) -> int | None`
- Produces dataclasses `QueryDefinition`, `ValidationIssue`, and `PackParseResult`, each with `to_dict()`.

- [ ] **Step 1: Write failing manifest tests**

  Add literal tests proving:

  ```python
  result = parse_query_files([
      ("query-p1-1-kis.txt", "mô tả"),
      ("query-p1-15-qa.txt", "câu hỏi"),
      ("query-p1-16-trake.txt", "3 sự kiện\n1. A\n2. B\n3. C"),
  ])
  assert result.ok
  assert [q.query_id for q in result.manifest] == [
      "query-p1-1-kis", "query-p1-15-qa", "query-p1-16-trake"
  ]
  assert result.manifest[2].n_events == 3
  assert result.manifest[2].events_confirmed is False
  ```

  Add separate tests for strict/terminal suffix matching, exact basename stem,
  preserved order/text, duplicate query IDs, invalid UTF-8, unsafe ZIP paths,
  ignored directories/metadata, unsupported-file warnings, in-memory ZIP,
  per-file/total/count limits, and the exact `query-p1-16-trake -> 3`
  suggestion.

- [ ] **Step 2: Run parser tests and witness RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/test_query_pack.py`

  Expected: collection failure because `aic.submission.query_pack` does not exist.

- [ ] **Step 3: Implement the minimal parser**

  Use standard-library `io.BytesIO`, `zipfile.ZipFile`, `PurePosixPath`, and
  `re`. Never call `extract`/`extractall`. Constants are exactly:

  ```python
  MAX_QUERY_FILES = 500
  MAX_QUERY_FILE_BYTES = 1024 * 1024
  MAX_QUERY_PACK_BYTES = 10 * 1024 * 1024
  EVENT_COUNT_OVERRIDES = {"query-p1-16-trake": 3}
  ```

  Decode bytes using strict UTF-8. Preserve input order. Set
  `events_confirmed=False` for every TRAKE query and true otherwise.

- [ ] **Step 4: Run parser tests and witness GREEN**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/test_query_pack.py`

  Expected: all parser tests pass.

- [ ] **Step 5: Commit parser task**

  ```bash
  git add aic/submission/query_pack.py tests/test_query_pack.py
  git commit -m "feat: parse AIC query packs into manifests"
  ```

### Task 2: Task-aware validator and post-write ZIP validation

**Files:**
- Create: `aic/submission/validator.py`
- Create: `tests/test_submission_validator.py`
- Modify: `aic/submission/writer.py`
- Modify: `tests/test_writer.py`
- Modify: `aic/submission/__init__.py`

**Interfaces:**
- Consumes: `QueryDefinition` and `ValidationIssue` from Task 1.
- Produces: `ValidationReport(ok, errors, warnings)` with `to_dict()`.
- Produces: `normalize_submission_rows(rows) -> list[dict]`.
- Produces: `validate_submission(manifest, rows) -> ValidationReport`.
- Produces: `write_validated_submission(manifest, rows, out_path) -> ValidationReport`.
- Produces: `validate_submission_zip(path_or_bytes, manifest) -> ValidationReport`.

- [ ] **Step 1: Write failing validator tests**

  Use literal manifest/row fixtures and individual tests for:

  ```python
  # valid shapes
  KIS   -> {"frames": [12], "answer": ""}
  QA    -> {"frames": [34], "answer": "  Năm người  "}
  TRAKE -> {"frames": [10, 20, 30], "answer": ""}

  # blocking codes
  missing_query_rows
  unknown_query
  too_many_rows
  kis_frame_count
  kis_unexpected_answer
  qa_frame_count
  qa_missing_answer
  qa_answer_too_long
  trake_events_unconfirmed
  trake_frame_count
  trake_frame_order
  trake_unexpected_answer
  invalid_video_id
  invalid_frame
  ```

  Assert the valid QA answer retains both leading and trailing spaces. Assert
  `.MP4` and `.mp4` are normalized away before validation/output. Assert 100
  rows pass and 101 fail.

- [ ] **Step 2: Write failing writer regression tests**

  Replace the existing mislabeled three-column KIS fixture with a valid Q&A
  fixture. Add a test that `write_validated_submission` rejects a KIS row with
  an answer. Add a valid mixed-task ZIP test that reparses CSV with
  `csv.reader` and asserts exact archive paths and literal rows.

- [ ] **Step 3: Run validator/writer tests and witness RED**

  Run:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
    tests/test_submission_validator.py tests/test_writer.py
  ```

  Expected: missing validator APIs and/or invalid KIS fixture assertion failure.

- [ ] **Step 4: Implement validator and validated writer**

  `validate_submission` looks up every row task from the manifest, preserves
  row order, rejects unknown/missing queries, and returns all errors rather
  than stopping at the first. `write_validated_submission` validates, writes
  through the existing CSV writer, reopens the ZIP, parses every CSV with
  `csv.reader`, validates exact expected paths and task shapes, then returns a
  PASS report. Raise `SubmissionValidationError(report)` on operator input
  errors and `GeneratedArchiveError(report)` on post-write defects.

- [ ] **Step 5: Run validator/writer tests and witness GREEN**

  Run the Step 3 command; expected all pass.

- [ ] **Step 6: Commit validator task**

  ```bash
  git add aic/submission tests/test_submission_validator.py tests/test_writer.py
  git commit -m "feat: validate AIC submissions before writing ZIP"
  ```

### Task 3: FastAPI query-pack and validated-export adapters

**Files:**
- Modify: `aic/ui/app.py`
- Create: `tests/test_ui_submission_api.py`

**Interfaces:**
- Consumes: Task 1 parser and Task 2 validated writer.
- Produces: `POST /api/query-pack/zip`.
- Produces: `POST /api/query-pack/texts`.
- Replaces `/api/export` request with `{manifest, rows}` and PASS-only ZIP output.

- [ ] **Step 1: Write failing API tests**

  With FastAPI `TestClient`, replace retriever loading with dummy mode before
  app import and test real HTTP behavior:

  ```python
  response = client.post(
      "/api/query-pack/texts",
      json={"files": [{"filename": "query-1-kis.txt", "content": "scene"}]},
  )
  assert response.status_code == 200
  assert response.json()["manifest"][0]["query_id"] == "query-1-kis"
  ```

  Add real ZIP-body import, invalid-pack 422, invalid-export 422 with stable
  error codes, valid mixed export returning `application/zip`, header
  `X-Validation-Status: PASS`, and archive paths under `submission/`.

- [ ] **Step 2: Run API tests and witness RED**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/test_ui_submission_api.py`

  Expected: 404 for new endpoints or import/setup failure until the adapter is implemented.

- [ ] **Step 3: Implement Pydantic request models and routes**

  Add `QueryTextFileIn`, `QueryPackTextsRequest`, `QueryManifestIn`, and the
  manifest-bearing `ExportRequest`. Route functions translate dataclass
  results to dictionaries. Parser/validation input failures return HTTP 422
  with the complete report in `detail`. Generated archive validation failures
  return HTTP 500. Successful export includes the PASS header.

- [ ] **Step 4: Run API tests and witness GREEN**

  Run Step 2 command; expected all pass.

- [ ] **Step 5: Commit API task**

  ```bash
  git add aic/ui/app.py tests/test_ui_submission_api.py
  git commit -m "feat: expose query-pack and validated export APIs"
  ```

### Task 4: Baseline UI query manifest and validation report

**Files:**
- Create: `aic/ui/static/submission_helpers.js`
- Create: `tests/js/test_submission_helpers.js`
- Modify: `aic/ui/static/index.html`
- Modify: `aic/ui/static/main.js`
- Modify: `aic/ui/static/style.css`

**Interfaces:**
- Consumes: Task 3 query-pack and export APIs.
- Produces browser global/CommonJS module `AICSubmissionHelpers`.
- Produces persistent `state.manifest`, current-query navigation, per-query
  TRAKE confirmation, and backend validation-report rendering.

- [ ] **Step 1: Write failing pure JavaScript tests**

  Use `node:test` and literal fixtures to test:

  ```javascript
  const qa = helpers.prepareQaAnswer('  Năm người  ');
  assert.equal(qa, '  Năm người  ');
  assert.equal(helpers.canUseIterative('kis'), true);
  assert.equal(helpers.canUseIterative('qa'), false);
  assert.equal(helpers.canUseIterative('trake'), false);
  ```

  Also test strict suffix inference, upserting a manifest item without
  reordering it, updating only the selected TRAKE event count/confirmation,
  and grouping validation issues by query.

- [ ] **Step 2: Run JavaScript tests and witness RED**

  Run: `node --test tests/js/test_submission_helpers.js`

  Expected: module-not-found for `submission_helpers.js`.

- [ ] **Step 3: Implement pure helpers and witness GREEN**

  Implement the minimal UMD/CommonJS-compatible helper file and rerun Step 2.

- [ ] **Step 4: Write a failing UI smoke contract**

  Extend the JavaScript test to load `index.html` text and assert the user-
  visible pack input, query list, TRAKE confirmation control, and validation
  report container exist. Run and witness failure before editing HTML.

- [ ] **Step 5: Implement UI integration**

  Replace the old input with `accept=".zip,.txt" multiple`; upload one ZIP to
  `/api/query-pack/zip` or TXT contents to `/api/query-pack/texts`. Persist and
  render manifest items. Selecting a manifest item sets exact query ID/text,
  task, and its event count. Never silently infer invalid suffixes. Q&A checks
  blankness with `.trim()` but stores the original value. Restrict iterative
  start to KIS. Send `{manifest, rows}` to export, render 422 errors grouped by
  query, and download only a PASS response.

- [ ] **Step 6: Run UI tests and syntax checks**

  Run:

  ```bash
  node --test tests/js/test_submission_helpers.js
  node --check aic/ui/static/submission_helpers.js
  node --check aic/ui/static/main.js
  ```

  Expected: all pass and both syntax checks exit 0.

- [ ] **Step 7: Commit UI task**

  ```bash
  git add aic/ui/static tests/js/test_submission_helpers.js
  git commit -m "feat: manage query packs and validation in baseline UI"
  ```

### Task 5: Full verification and operator documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`

**Interfaces:**
- Consumes all prior tasks.
- Produces documented query-pack workflow and reproducible verification evidence.

- [ ] **Step 1: Document the operator workflow**

  State that official work should import the complete ZIP/multi-TXT pack,
  confirm every TRAKE count, resolve all validation errors, and download only
  when the report says PASS. Document single-TXT mode as development-only.

- [ ] **Step 2: Run focused verification**

  ```bash
  PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider \
    tests/test_query_pack.py tests/test_submission_validator.py \
    tests/test_writer.py tests/test_ui_submission_api.py
  node --test tests/js/test_submission_helpers.js
  node --check aic/ui/static/submission_helpers.js
  node --check aic/ui/static/main.js
  ```

- [ ] **Step 3: Run the repository suite**

  Run: `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider`

  If environmental dependencies block collection, report the exact missing
  package and separately run every dependency-free suite. Do not change
  retrieval/core merely to make the environment green.

- [ ] **Step 4: Run structural checks**

  ```bash
  python -m compileall -q aic
  git diff --check
  git status --short --branch
  ```

- [ ] **Step 5: Commit documentation**

  ```bash
  git add README.md docs/README.md
  git commit -m "docs: explain validated baseline submission workflow"
  ```
