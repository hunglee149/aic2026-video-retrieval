import io
import zipfile

from aic.submission.query_pack import (
    MAX_QUERY_FILE_BYTES,
    MAX_QUERY_FILES,
    MAX_QUERY_PACK_BYTES,
    infer_task,
    parse_query_files,
    parse_query_zip,
    suggest_event_count,
)


def _zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return buffer.getvalue()


def test_parses_queries_into_an_ordered_manifest():
    result = parse_query_files(
        [
            ("query-p1-1-kis.txt", "mô tả"),
            ("query-p1-15-qa.txt", "câu hỏi"),
            ("query-p1-16-trake.txt", "3 sự kiện\n1. A\n2. B\n3. C"),
        ]
    )

    assert result.ok
    assert [query.query_id for query in result.manifest] == [
        "query-p1-1-kis",
        "query-p1-15-qa",
        "query-p1-16-trake",
    ]
    assert result.manifest[2].n_events == 3
    assert result.manifest[2].events_confirmed is False


def test_manifest_keeps_exact_basename_stem_source_text_and_input_order():
    result = parse_query_files(
        [
            ("nested/query-p1-2-qa.txt", "  giữ nguyên  \n"),
            ("query_p1_3_KIS.TXT", "thứ hai"),
        ]
    )

    assert result.ok
    assert [query.to_dict() for query in result.manifest] == [
        {
            "query_id": "query-p1-2-qa",
            "task": "qa",
            "text": "  giữ nguyên  \n",
            "source_name": "query-p1-2-qa.txt",
            "n_events": None,
            "events_confirmed": True,
        },
        {
            "query_id": "query_p1_3_KIS",
            "task": "kis",
            "text": "thứ hai",
            "source_name": "query_p1_3_KIS.TXT",
            "n_events": None,
            "events_confirmed": True,
        },
    ]


def test_infer_task_requires_a_terminal_delimited_suffix():
    assert infer_task("query-p1-1-kis") == "kis"
    assert infer_task("query_p1_2_QA") == "qa"
    assert infer_task("query-p1-3-trake-extra") is None
    assert infer_task("query-p1-4-kiss") is None
    assert infer_task("query-p1-5-kis\n") is None
    assert infer_task("trake") is None


def test_rejects_a_txt_file_without_a_supported_task_suffix():
    result = parse_query_files([("query-p1-1-kis-extra.txt", "mô tả")])

    assert not result.ok
    assert result.errors[0].code == "invalid_task_suffix"


def test_rejects_duplicate_query_ids():
    result = parse_query_files(
        [
            ("query-p1-1-kis.txt", "một"),
            ("query-p1-1-kis.TXT", "hai"),
        ]
    )

    assert not result.ok
    assert [issue.code for issue in result.errors] == ["duplicate_query_id"]


def test_rejects_invalid_utf8_bytes():
    result = parse_query_files([("query-p1-1-kis.txt", b"\xff")])

    assert not result.ok
    assert result.errors[0].code == "invalid_utf8"


def test_zip_rejects_unsafe_paths():
    result = parse_query_zip(
        _zip_bytes([("../query-p1-1-kis.txt", "mô tả")])
    )

    assert not result.ok
    assert result.errors[0].code == "unsafe_zip_path"


def test_zip_ignores_directories_and_common_metadata_entries():
    result = parse_query_zip(
        _zip_bytes(
            [
                ("queries/", b""),
                ("__MACOSX/._query-p1-1-kis.txt", b"metadata"),
                (".DS_Store", b"metadata"),
                ("queries/query-p1-1-kis.txt", "mô tả"),
            ]
        )
    )

    assert result.ok
    assert [query.query_id for query in result.manifest] == ["query-p1-1-kis"]
    assert result.warnings == []


def test_warns_about_unsupported_regular_files():
    result = parse_query_files(
        [
            ("readme.pdf", b"not a query"),
            ("query-p1-1-kis.txt", "mô tả"),
        ]
    )

    assert result.ok
    assert [issue.code for issue in result.warnings] == ["unsupported_file"]


def test_parses_an_in_memory_zip_in_archive_order():
    result = parse_query_zip(
        _zip_bytes(
            [
                ("queries/query-p1-2-qa.txt", "câu hỏi"),
                ("query-p1-1-kis.txt", "mô tả"),
            ]
        )
    )

    assert result.ok
    assert [query.query_id for query in result.manifest] == [
        "query-p1-2-qa",
        "query-p1-1-kis",
    ]


def test_rejects_a_query_file_larger_than_the_per_file_limit():
    result = parse_query_files(
        [("query-p1-1-kis.txt", b"x" * (MAX_QUERY_FILE_BYTES + 1))]
    )

    assert not result.ok
    assert result.errors[0].code == "query_file_too_large"


def test_rejects_more_than_the_query_file_limit():
    result = parse_query_files(
        [
            (f"query-p1-{index}-kis.txt", "x")
            for index in range(MAX_QUERY_FILES + 1)
        ]
    )

    assert not result.ok
    assert result.errors[0].code == "too_many_query_files"


def test_rejects_query_text_larger_than_the_total_limit():
    result = parse_query_files(
        [
            ("query-p1-1-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-2-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-3-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-4-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-5-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-6-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-7-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-8-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-9-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-10-kis.txt", "x" * MAX_QUERY_FILE_BYTES),
            ("query-p1-11-kis.txt", "x"),
        ]
    )

    assert MAX_QUERY_PACK_BYTES == 10 * MAX_QUERY_FILE_BYTES
    assert not result.ok
    assert result.errors[0].code == "query_pack_too_large"


def test_suggests_event_count_from_override_phrase_and_numbered_list():
    assert suggest_event_count("query-p1-16-trake", "không có danh sách") == 3
    assert suggest_event_count("query-p1-17-trake", "Có 4 events.") == 4
    assert suggest_event_count("query-p1-18-trake", "1. A\n2. B\n3. C") == 3
    assert suggest_event_count("query-p1-19-kis", "3 events") is None
