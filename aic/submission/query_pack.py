"""Parse AIC query TXT files into an ordered, task-aware manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import PurePosixPath
import re
from typing import Sequence
import zipfile


MAX_QUERY_FILES = 500
MAX_QUERY_FILE_BYTES = 1024 * 1024
MAX_QUERY_PACK_BYTES = 10 * 1024 * 1024
EVENT_COUNT_OVERRIDES = {"query-p1-16-trake": 3}

_TASK_SUFFIX = re.compile(r"[-_](kis|qa|trake)\Z", re.IGNORECASE)
_EVENT_COUNT = re.compile(
    r"\b(\d+)\s+(?:events?|sự\s+kiện)\b", re.IGNORECASE
)
_NUMBERED_EVENT = re.compile(r"^\s*(\d+)\s*[.)]\s+", re.MULTILINE)


@dataclass
class QueryDefinition:
    query_id: str
    task: str
    text: str
    source_name: str
    n_events: int | None
    events_confirmed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "task": self.task,
            "text": self.text,
            "source_name": self.source_name,
            "n_events": self.n_events,
            "events_confirmed": self.events_confirmed,
        }


@dataclass
class ValidationIssue:
    code: str
    message: str
    query_id: str | None = None
    row: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "query_id": self.query_id,
            "row": self.row,
        }


@dataclass
class PackParseResult:
    manifest: list[QueryDefinition] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "manifest": [query.to_dict() for query in self.manifest],
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def infer_task(query_id: str) -> str | None:
    """Return the task encoded in a strictly terminal query-ID suffix."""
    match = _TASK_SUFFIX.search(query_id)
    return match.group(1).lower() if match else None


def suggest_event_count(query_id: str, text: str) -> int | None:
    """Return a non-authoritative TRAKE event-count suggestion, if available."""
    if infer_task(query_id) != "trake":
        return None

    override = EVENT_COUNT_OVERRIDES.get(query_id.lower())
    if override is not None:
        return override

    explicit = _EVENT_COUNT.search(text)
    if explicit:
        count = int(explicit.group(1))
        if count > 0:
            return count

    numbered = [int(match.group(1)) for match in _NUMBERED_EVENT.finditer(text)]
    if numbered == list(range(1, len(numbered) + 1)):
        return len(numbered) or None
    return None


def parse_query_files(files: Sequence[tuple[str, str | bytes]]) -> PackParseResult:
    """Parse uploaded TXT files, preserving their supplied order and text."""
    result = PackParseResult()
    total_bytes = 0
    query_files = 0

    for filename, content in files:
        source_name = _basename(filename)
        if not _is_text_file(source_name):
            _warn_unsupported(result, source_name)
            continue

        query_files += 1
        if query_files > MAX_QUERY_FILES:
            result.errors.append(
                ValidationIssue(
                    "too_many_query_files",
                    f"Query pack contains more than {MAX_QUERY_FILES} TXT files",
                )
            )
            break

        raw = _encode_content(content, source_name, result)
        if raw is None:
            continue
        total_bytes = _add_query_bytes(
            result, source_name, raw, total_bytes
        )

    return result


def parse_query_zip(data: bytes) -> PackParseResult:
    """Parse a ZIP entirely in memory without extracting any member to disk."""
    result = PackParseResult()
    total_bytes = 0
    query_files = 0

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError):
        result.errors.append(ValidationIssue("invalid_zip", "Invalid ZIP archive"))
        return result

    with archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or name.endswith("/"):
                continue
            if _is_unsafe_zip_path(name):
                result.errors.append(
                    ValidationIssue(
                        "unsafe_zip_path", f"Unsafe ZIP member path: {name!r}"
                    )
                )
                continue
            if _is_metadata_path(name):
                continue

            source_name = PurePosixPath(name).name
            if not _is_text_file(source_name):
                _warn_unsupported(result, source_name)
                continue

            query_files += 1
            if query_files > MAX_QUERY_FILES:
                result.errors.append(
                    ValidationIssue(
                        "too_many_query_files",
                        f"Query pack contains more than {MAX_QUERY_FILES} TXT files",
                    )
                )
                break
            if info.file_size > MAX_QUERY_FILE_BYTES:
                result.errors.append(
                    ValidationIssue(
                        "query_file_too_large",
                        f"Query file {source_name!r} exceeds {MAX_QUERY_FILE_BYTES} bytes",
                    )
                )
                continue
            if total_bytes + info.file_size > MAX_QUERY_PACK_BYTES:
                result.errors.append(
                    ValidationIssue(
                        "query_pack_too_large",
                        f"Query text exceeds {MAX_QUERY_PACK_BYTES} bytes",
                    )
                )
                continue

            try:
                raw = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                result.errors.append(
                    ValidationIssue("invalid_zip", f"Could not read {name!r}: {error}")
                )
                continue

            total_bytes = _add_query_bytes(result, source_name, raw, total_bytes)

    return result


def _add_query_bytes(
    result: PackParseResult, source_name: str, raw: bytes, total_bytes: int
) -> int:
    if len(raw) > MAX_QUERY_FILE_BYTES:
        result.errors.append(
            ValidationIssue(
                "query_file_too_large",
                f"Query file {source_name!r} exceeds {MAX_QUERY_FILE_BYTES} bytes",
            )
        )
        return total_bytes
    if total_bytes + len(raw) > MAX_QUERY_PACK_BYTES:
        result.errors.append(
            ValidationIssue(
                "query_pack_too_large", f"Query text exceeds {MAX_QUERY_PACK_BYTES} bytes"
            )
        )
        return total_bytes

    new_total = total_bytes + len(raw)

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        result.errors.append(
            ValidationIssue("invalid_utf8", f"Query file {source_name!r} is not UTF-8")
        )
        return new_total

    query_id = source_name[:-4]
    task = infer_task(query_id)
    if task is None:
        result.errors.append(
            ValidationIssue(
                "invalid_task_suffix",
                f"Query file {source_name!r} must end with -kis, -qa, or -trake",
                query_id=query_id,
            )
        )
        return new_total
    if any(query.query_id == query_id for query in result.manifest):
        result.errors.append(
            ValidationIssue(
                "duplicate_query_id",
                f"Duplicate query ID: {query_id}",
                query_id=query_id,
            )
        )
        return new_total

    n_events = suggest_event_count(query_id, text) if task == "trake" else None
    result.manifest.append(
        QueryDefinition(
            query_id=query_id,
            task=task,
            text=text,
            source_name=source_name,
            n_events=n_events,
            events_confirmed=task != "trake",
        )
    )
    return new_total


def _encode_content(
    content: str | bytes, source_name: str, result: PackParseResult
) -> bytes | None:
    if isinstance(content, bytes):
        return content
    try:
        return content.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        result.errors.append(
            ValidationIssue("invalid_utf8", f"Query file {source_name!r} is not UTF-8")
        )
        return None


def _basename(filename: str) -> str:
    return PurePosixPath(filename.replace("\\", "/")).name


def _is_text_file(source_name: str) -> bool:
    return source_name.lower().endswith(".txt") and len(source_name) > 4


def _warn_unsupported(result: PackParseResult, source_name: str) -> None:
    result.warnings.append(
        ValidationIssue("unsupported_file", f"Ignored unsupported file: {source_name!r}")
    )


def _is_metadata_path(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        "__MACOSX" in path.parts
        or path.name in {".DS_Store", "Thumbs.db"}
        or path.name.startswith("._")
    )


def _is_unsafe_zip_path(name: str) -> bool:
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name
