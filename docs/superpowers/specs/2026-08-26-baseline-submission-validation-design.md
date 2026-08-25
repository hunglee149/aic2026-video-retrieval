# Baseline Submission Validation Design

## Goal

Make every `submission.zip` downloadable from the baseline UI conform to the
public AIC 2026 preliminary-round rules for query filenames, KIS/Q&A/TRAKE row
schemas, CSV encoding, and ZIP layout.

This work is limited to the baseline UI, its FastAPI adapter, and
`aic/submission`. It does not change retrieval, fusion, or `aic/core`.

## Organizer rules implemented

- A query comes from a `.txt` filename whose stem becomes the CSV filename.
- Query task is selected by the terminal `kis`, `qa`, or `trake` suffix.
- Each query has its own CSV and at most 100 ranked rows.
- KIS rows have `video_id,frame_id`.
- Q&A rows have `video_id,frame_id,answer`; answer is non-empty, at most 100
  Unicode characters, and is not trimmed or normalized.
- TRAKE rows have one video ID and exactly one frame per event, in temporal
  order.
- Video IDs in output do not include `.mp4`; frames are integers.
- CSV is UTF-8 text, comma-delimited, headerless, and uses LF or CRLF.
- Output is a `.zip` containing only `submission/<query_id>.csv` files.

Frame base, ZIP size, duplicate rows, minimum row count, and BOM rejection are
not defined by the public page. The baseline continues its existing policy of
requiring at least one row per query. Its own writer emits UTF-8 without BOM.

## Architecture

```text
ZIP query pack or one/many TXT files
                 |
                 v
        Query-pack parser
                 |
                 v
           Query manifest
                 |
       search/select/review UI
                 |
                 v
       task-aware validator
                 |
       errors -> block download
                 |
                 v
            ZIP writer
                 |
                 v
      reopen + parse + validate
                 |
                 v
        downloadable ZIP bytes
```

The query manifest is the source of truth. Export never trusts the current UI
task pill or a task supplied on an answer row.

## Query manifest contract

Every imported TXT becomes this JSON-compatible object:

```json
{
  "query_id": "query-p1-16-trake",
  "task": "trake",
  "text": "...",
  "source_name": "query-p1-16-trake.txt",
  "n_events": 3,
  "events_confirmed": true
}
```

Fields:

- `query_id`: exact basename stem; it is never silently renamed.
- `task`: `kis`, `qa`, or `trake`, derived strictly from a terminal suffix
  separated by `-` or `_`, case-insensitively.
- `text`: UTF-8 query contents.
- `source_name`: original basename for operator review.
- `n_events`: positive integer for TRAKE; `null` for KIS/Q&A.
- `events_confirmed`: true for KIS/Q&A. TRAKE import starts false even if an
  event count can be suggested; an operator must confirm it.

The parser accepts either one ZIP or one/many TXT files. ZIP contents are read
in memory and are never extracted to disk. Directories and common metadata
entries are ignored; unsupported regular files produce warnings. Unsafe paths,
duplicate query IDs, invalid UTF-8, or invalid task suffixes are errors.

For safety, an input pack is capped at 500 TXT files, 1 MiB per TXT, and
10 MiB total uncompressed query text.

## TRAKE event count

The parser may suggest an event count from an explicit phrase such as
`3 events`/`3 sự kiện` or from a numbered event list. A suggestion is not
authoritative. The UI displays the number and requires operator confirmation
before export. The published correction for `query-p1-16-trake` suggests 3,
but still requires the same confirmation step.

The validator requires exactly `n_events` frames. Frames must be strictly
increasing because the organizer rule requires temporal event order.

## Answer-row contract

Frontend selections and export requests use:

```json
{
  "query_id": "query-p1-15-qa",
  "video_id": "L01_V028",
  "frames": [3450],
  "answer": "Năm người"
}
```

Task and expected event count are looked up from the manifest. The backend
normalizes a terminal `.mp4` suffix case-insensitively before validation and
serialization so its output always follows the organizer format.

Task-aware validation:

- KIS: exactly one integer frame and empty answer.
- Q&A: exactly one integer frame; answer is a string with length 1..100. The
  exact string is preserved, including leading/trailing whitespace.
- TRAKE: exactly `n_events` integer frames, strictly increasing, and empty
  answer.

Every manifest query must have 1..100 rows. Rows for unknown query IDs are
errors. Row order is preserved and becomes submission rank order.

## Validation report

All parser and submission checks return the same shape:

```json
{
  "ok": false,
  "errors": [
    {
      "code": "trake_frame_count",
      "message": "Expected 3 event frames, got 2",
      "query_id": "query-p1-16-trake",
      "row": 1
    }
  ],
  "warnings": []
}
```

Errors block export. Warnings are shown but do not block. Error codes are
stable identifiers; messages are operator-facing Vietnamese or concise
English.

## API contract

### `POST /api/query-pack/zip`

- Body: raw ZIP bytes (`application/zip`).
- Success: `{ok, manifest, warnings}`.
- Invalid package: HTTP 422 with `{detail: {ok, errors, warnings}}`.

### `POST /api/query-pack/texts`

- Body: `{files: [{filename, content}]}`.
- Same success/error response as ZIP import.

### `POST /api/export`

- Body: `{manifest: [...], rows: [...]}`.
- Runs task-aware validation before writing.
- Creates ZIP, reopens it, decodes every CSV with UTF-8, parses with
  `csv.reader`, and validates exact paths and schemas against the manifest.
- Validation errors return HTTP 422 and no download.
- Success returns `submission.zip` plus response header
  `X-Validation-Status: PASS`.

## UI behavior

- Replace the single TXT picker with a picker accepting one ZIP or one/many
  TXT files.
- Show every imported query in a compact query list with task and readiness.
- Selecting a query restores its text, task, event count, confirmed answers,
  and draft fields.
- TRAKE shows an explicit `Confirm event count` control. Export remains
  blocked until every TRAKE count is confirmed.
- Q&A keeps the exact answer value; blank detection may inspect the value but
  must not rewrite it with `.trim()`.
- Iterative mode is limited to KIS until it can collect task-correct Q&A and
  TRAKE data.
- Export shows the backend validation report. Download starts only after the
  backend returns a PASS ZIP.
- Manifest and selections persist in browser local storage. Each browser is
  still an independent operator session; shared team persistence is outside
  this change.

## Error handling

- Invalid query pack: keep the previous valid manifest and show all new
  errors.
- Duplicate filename/query ID: reject the new pack.
- Missing answers, invalid schemas, or unconfirmed TRAKE counts: show errors
  grouped by query and keep the user on the review screen.
- ZIP post-write validation failure: return HTTP 500 because this indicates a
  baseline defect rather than operator input.

## Tests

- Query parser tests: strict suffixes, multiple TXT, in-memory ZIP, duplicate
  IDs, UTF-8, unsafe paths, file limits, TRAKE suggestion and confirmation.
- Validator tests: valid KIS/Q&A/TRAKE, every task schema error, answer length
  and whitespace preservation, event count/order, unknown/missing queries,
  100-row boundary, case-insensitive `.mp4` normalization.
- Writer tests: exact ZIP paths, UTF-8/no BOM, CSV quoting, preserved order,
  post-write reparse, and rejection of the existing invalid three-column KIS
  fixture.
- UI checks: JavaScript syntax plus pure state/helper tests for strict task
  inference, per-query manifest updates, preserved Q&A whitespace, and KIS-only
  iterative gating.

## Out of scope

- Retrieval quality, translation, fusion, and frame selection.
- Shared server-side team sessions or authentication.
- Uploading the final ZIP to the organizer website.
- Rules not present on the current public organizer pages.
