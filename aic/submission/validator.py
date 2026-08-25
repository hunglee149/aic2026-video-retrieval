"""Validate task-aware AIC submission rows and generated archives."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import io
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence
import zipfile

from .query_pack import QueryDefinition, ValidationIssue


MAX_ROWS_PER_QUERY = 100
MAX_QA_ANSWER_LENGTH = 100
_VIDEO_ID = re.compile(r"L\d{2}_V\d{3}\Z")


@dataclass
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


class SubmissionValidationError(ValueError):
    """Raised when operator-supplied submission rows do not validate."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("Submission validation failed")


class GeneratedArchiveError(RuntimeError):
    """Raised when a ZIP written by the baseline fails post-write validation."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("Generated submission archive validation failed")


def normalize_submission_rows(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    """Copy submission rows and remove only a terminal case-insensitive .mp4."""
    normalized = []
    for row in rows:
        copied = dict(row)
        video_id = copied.get("video_id")
        if isinstance(video_id, str) and video_id.lower().endswith(".mp4"):
            copied["video_id"] = video_id[:-4]
        normalized.append(copied)
    return normalized


def validate_submission(
    manifest: Sequence[QueryDefinition], rows: Iterable[Mapping[str, object]]
) -> ValidationReport:
    """Validate all rows against the tasks and event counts in a manifest."""
    report = ValidationReport()
    queries = {query.query_id: query for query in manifest}
    rows_by_query: dict[str, list[tuple[int, Mapping[str, object]]]] = {
        query.query_id: [] for query in manifest
    }

    for row_number, row in enumerate(rows, start=1):
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or query_id not in queries:
            report.errors.append(
                ValidationIssue(
                    "unknown_query",
                    f"Row refers to unknown query: {query_id!r}",
                    query_id=query_id if isinstance(query_id, str) else None,
                    row=row_number,
                )
            )
            continue
        rows_by_query[query_id].append((row_number, row))

    for query in manifest:
        query_rows = rows_by_query[query.query_id]
        if not query_rows:
            report.errors.append(
                ValidationIssue(
                    "missing_query_rows",
                    f"Query {query.query_id!r} requires at least one row",
                    query_id=query.query_id,
                )
            )
            continue
        if len(query_rows) > MAX_ROWS_PER_QUERY:
            report.errors.append(
                ValidationIssue(
                    "too_many_rows",
                    f"Query {query.query_id!r} has more than {MAX_ROWS_PER_QUERY} rows",
                    query_id=query.query_id,
                )
            )
        for row_number, row in query_rows:
            _validate_row(report, query, row, row_number)

    return report


def validate_submission_zip(
    path_or_bytes: str | Path | bytes, manifest: Sequence[QueryDefinition]
) -> ValidationReport:
    """Reparse an archive and validate its exact paths and task-specific CSVs."""
    report = ValidationReport()
    try:
        archive = _open_zip(path_or_bytes)
    except (OSError, zipfile.BadZipFile):
        report.errors.append(ValidationIssue("invalid_submission_zip", "Invalid ZIP archive"))
        return report

    expected_names = [f"submission/{query.query_id}.csv" for query in manifest]
    parsed_rows: list[dict] = []
    with archive:
        actual_names = archive.namelist()
        if actual_names != expected_names:
            report.errors.append(
                ValidationIssue(
                    "archive_paths_mismatch",
                    "ZIP must contain exactly the expected submission CSV paths",
                )
            )

        for query, expected_name in zip(manifest, expected_names):
            if expected_name not in actual_names:
                continue
            try:
                raw = archive.read(expected_name)
                if raw.startswith(b"\xef\xbb\xbf"):
                    raise UnicodeDecodeError("utf-8", raw, 0, 3, "UTF-8 BOM is not allowed")
                decoded = raw.decode("utf-8", errors="strict")
            except (KeyError, OSError, RuntimeError, UnicodeDecodeError) as error:
                report.errors.append(
                    ValidationIssue(
                        "invalid_csv_encoding",
                        f"Could not decode {expected_name!r} as UTF-8: {error}",
                        query_id=query.query_id,
                    )
                )
                continue

            for row_number, csv_row in enumerate(csv.reader(io.StringIO(decoded)), start=1):
                parsed_rows.append(_csv_to_submission_row(query, csv_row, report, row_number))

    row_report = validate_submission(manifest, parsed_rows)
    report.errors.extend(row_report.errors)
    report.warnings.extend(row_report.warnings)
    return report


def _open_zip(path_or_bytes: str | Path | bytes) -> zipfile.ZipFile:
    if isinstance(path_or_bytes, bytes):
        return zipfile.ZipFile(io.BytesIO(path_or_bytes))
    return zipfile.ZipFile(path_or_bytes)


def _csv_to_submission_row(
    query: QueryDefinition,
    csv_row: list[str],
    report: ValidationReport,
    row_number: int,
) -> dict:
    expected_columns = 2 if query.task == "kis" else 3 if query.task == "qa" else None
    if query.task == "trake" and query.n_events is not None:
        expected_columns = query.n_events + 1
    if expected_columns is not None and len(csv_row) != expected_columns:
        report.errors.append(
            ValidationIssue(
                "archive_row_shape",
                f"CSV row has {len(csv_row)} columns; expected {expected_columns}",
                query_id=query.query_id,
                row=row_number,
            )
        )

    video_id = csv_row[0] if csv_row else ""
    if query.task == "qa":
        frame_values = csv_row[1:2]
        answer = csv_row[2] if len(csv_row) >= 3 else ""
    else:
        frame_values = csv_row[1:]
        answer = ""
    frames = [_parse_frame(value) for value in frame_values]
    return {
        "query_id": query.query_id,
        "video_id": video_id,
        "frames": frames,
        "answer": answer,
    }


def _parse_frame(value: str) -> object:
    try:
        return int(value)
    except ValueError:
        return value


def _validate_row(
    report: ValidationReport,
    query: QueryDefinition,
    row: Mapping[str, object],
    row_number: int,
) -> None:
    video_id = row.get("video_id")
    if not isinstance(video_id, str) or not _VIDEO_ID.fullmatch(video_id):
        _error(report, "invalid_video_id", "Video ID must use the L##_V### format", query, row_number)

    frames = row.get("frames")
    if not isinstance(frames, list):
        frames = []
    if any(not isinstance(frame, int) or isinstance(frame, bool) or frame < 1 for frame in frames):
        _error(report, "invalid_frame", "Frames must be positive integers", query, row_number)

    answer = row.get("answer", "")
    if query.task == "kis":
        if len(frames) != 1:
            _error(report, "kis_frame_count", "KIS requires exactly one frame", query, row_number)
        if answer != "":
            _error(report, "kis_unexpected_answer", "KIS rows must not include an answer", query, row_number)
    elif query.task == "qa":
        if len(frames) != 1:
            _error(report, "qa_frame_count", "Q&A requires exactly one frame", query, row_number)
        if not isinstance(answer, str) or not answer:
            _error(report, "qa_missing_answer", "Q&A requires a non-empty answer", query, row_number)
        elif len(answer) > MAX_QA_ANSWER_LENGTH:
            _error(report, "qa_answer_too_long", "Q&A answer exceeds 100 Unicode characters", query, row_number)
    elif query.task == "trake":
        if not query.events_confirmed or query.n_events is None:
            _error(report, "trake_events_unconfirmed", "TRAKE event count must be confirmed", query, row_number)
        elif len(frames) != query.n_events:
            _error(
                report,
                "trake_frame_count",
                f"Expected {query.n_events} event frames, got {len(frames)}",
                query,
                row_number,
            )
        if len(frames) > 1 and all(isinstance(frame, int) for frame in frames):
            if any(first >= second for first, second in zip(frames, frames[1:])):
                _error(report, "trake_frame_order", "TRAKE frames must be strictly increasing", query, row_number)
        if answer != "":
            _error(report, "trake_unexpected_answer", "TRAKE rows must not include an answer", query, row_number)


def _error(
    report: ValidationReport,
    code: str,
    message: str,
    query: QueryDefinition,
    row_number: int,
) -> None:
    report.errors.append(
        ValidationIssue(code, message, query_id=query.query_id, row=row_number)
    )
