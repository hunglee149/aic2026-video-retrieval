import csv
import io
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .query_pack import QueryDefinition
from .validator import (
    GeneratedArchiveError,
    SubmissionValidationError,
    ValidationReport,
    normalize_submission_rows,
    validate_submission,
    validate_submission_zip,
)

MAX_ROWS_PER_QUERY = 100


def _validate_query_id(query_id: str) -> None:
    """Query ID becomes a filename, so it must not contain a path."""
    if not query_id or Path(query_id).name != query_id or "/" in query_id or "\\" in query_id:
        raise ValueError("query_id must be a non-empty filename without a path")


def write_submission(
    rows_by_query: Mapping[str, Iterable[Sequence[object]]],
    out_path: str | Path,
) -> str:
    """Write BTC-compatible CSV files into ``submission.zip``.

    Each mapping key is the query filename without its extension. Each value is
    the ordered list of CSV rows already converted by ``aic.core.convert``.
    Files are UTF-8 without BOM and live at ``submission/<query_id>.csv``.
    """
    if not rows_by_query:
        raise ValueError("submission must contain at least one query")

    prepared = {}
    for query_id, rows in rows_by_query.items():
        _validate_query_id(query_id)
        materialized = [list(row) for row in rows]
        if not materialized:
            raise ValueError(f"query {query_id!r} must contain at least one row")
        if len(materialized) > MAX_ROWS_PER_QUERY:
            raise ValueError(
                f"query {query_id!r} has {len(materialized)} rows; "
                f"maximum is {MAX_ROWS_PER_QUERY}"
            )
        if any(not row for row in materialized):
            raise ValueError(f"query {query_id!r} contains an empty row")
        prepared[query_id] = materialized

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for query_id, rows in prepared.items():
            buffer = io.StringIO(newline="")
            csv.writer(buffer).writerows(rows)
            archive.writestr(
                "submission/%s.csv" % query_id,
                buffer.getvalue().encode("utf-8"),
            )
    return str(destination)


def write_query_submission(
    query_id: str,
    rows: Iterable[Sequence[object]],
    out_path: str | Path = "submission.zip",
) -> str:
    """Convenience entry point for one query's list of submission rows."""
    return write_submission({query_id: rows}, out_path)


def write_validated_submission(
    manifest: Sequence[QueryDefinition],
    rows: Iterable[Mapping[str, object]],
    out_path: str | Path,
) -> ValidationReport:
    """Validate operator rows, write the archive, then validate it again."""
    normalized = normalize_submission_rows(rows)
    report = validate_submission(manifest, normalized)
    if not report.ok:
        raise SubmissionValidationError(report)

    rows_by_query: dict[str, list[list[object]]] = {
        query.query_id: [] for query in manifest
    }
    for row in normalized:
        query_id = row["query_id"]
        csv_row = [row["video_id"], *[str(frame) for frame in row["frames"]]]
        if next(query for query in manifest if query.query_id == query_id).task == "qa":
            csv_row.append(row["answer"])
        rows_by_query[query_id].append(csv_row)

    write_submission(rows_by_query, out_path)
    archive_report = validate_submission_zip(out_path, manifest)
    if not archive_report.ok:
        raise GeneratedArchiveError(archive_report)
    return archive_report
