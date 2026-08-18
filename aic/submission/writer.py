import csv
import io
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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
