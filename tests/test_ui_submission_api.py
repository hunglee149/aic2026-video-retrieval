import csv
import importlib
import io
import os
import sys
import types
import zipfile

from fastapi.testclient import TestClient

from aic.submission.query_pack import ValidationIssue
from aic.submission.validator import GeneratedArchiveError, ValidationReport


os.environ["AIC_USE_DUMMY"] = "1"


class _DummyRetriever:
    NAME = "dummy"

    @staticmethod
    def search(query, k=100, exclude=frozenset()):
        return []


retrieval_stub = types.ModuleType("aic.retrieval")
retrieval_stub.dummy = _DummyRetriever()
sys.modules.setdefault("aic.retrieval", retrieval_stub)

app_module = importlib.import_module("aic.ui.app")
client = TestClient(app_module.app)


def _zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return buffer.getvalue()


def _manifest():
    return [
        {
            "query_id": "query-p1-1-kis",
            "task": "kis",
            "text": "Tìm cảnh",
            "source_name": "query-p1-1-kis.txt",
            "n_events": None,
            "events_confirmed": True,
        },
        {
            "query_id": "query-p1-2-qa",
            "task": "qa",
            "text": "Có bao nhiêu người?",
            "source_name": "query-p1-2-qa.txt",
            "n_events": None,
            "events_confirmed": True,
        },
        {
            "query_id": "query-p1-3-trake",
            "task": "trake",
            "text": "Hai sự kiện",
            "source_name": "query-p1-3-trake.txt",
            "n_events": 2,
            "events_confirmed": True,
        },
    ]


def _valid_export_body():
    return {
        "manifest": _manifest(),
        "rows": [
            {
                "query_id": "query-p1-1-kis",
                "video_id": "L01_V001.MP4",
                "frames": [12],
                "answer": "",
            },
            {
                "query_id": "query-p1-2-qa",
                "video_id": "L01_V002.mp4",
                "frames": [34],
                "answer": "  Năm, người  ",
            },
            {
                "query_id": "query-p1-3-trake",
                "video_id": "L01_V003",
                "frames": [10, 20],
                "answer": "",
            },
        ],
    }


def test_query_pack_zip_imports_raw_archive_body_in_member_order():
    body = _zip_bytes(
        [
            ("queries/query-p1-2-qa.txt", "câu hỏi"),
            ("query-p1-1-kis.txt", "mô tả"),
        ]
    )

    response = client.post(
        "/api/query-pack/zip",
        content=body,
        headers={"content-type": "application/zip"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "manifest": [
            {
                "query_id": "query-p1-2-qa",
                "task": "qa",
                "text": "câu hỏi",
                "source_name": "query-p1-2-qa.txt",
                "n_events": None,
                "events_confirmed": True,
            },
            {
                "query_id": "query-p1-1-kis",
                "task": "kis",
                "text": "mô tả",
                "source_name": "query-p1-1-kis.txt",
                "n_events": None,
                "events_confirmed": True,
            },
        ],
        "errors": [],
        "warnings": [],
    }


def test_query_pack_texts_imports_multiple_json_files_in_request_order():
    response = client.post(
        "/api/query-pack/texts",
        json={
            "files": [
                {"filename": "query-1-kis.txt", "content": "scene"},
                {"filename": "query-2-qa.txt", "content": "question"},
            ]
        },
    )

    assert response.status_code == 200
    assert [query["query_id"] for query in response.json()["manifest"]] == [
        "query-1-kis",
        "query-2-qa",
    ]
    assert response.json()["errors"] == []
    assert response.json()["warnings"] == []


def test_invalid_query_pack_returns_422_with_the_complete_parser_report():
    body = _zip_bytes(
        [
            ("query-p1-1-kis.txt", "valid"),
            ("query-p1-2-unknown.txt", "invalid"),
            ("readme.pdf", "ignored"),
        ]
    )

    response = client.post("/api/query-pack/zip", content=body)

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "ok": False,
        "manifest": [
            {
                "query_id": "query-p1-1-kis",
                "task": "kis",
                "text": "valid",
                "source_name": "query-p1-1-kis.txt",
                "n_events": None,
                "events_confirmed": True,
            }
        ],
        "errors": [
            {
                "code": "invalid_task_suffix",
                "message": (
                    "Query file 'query-p1-2-unknown.txt' must end with -kis, "
                    "-qa, or -trake"
                ),
                "query_id": "query-p1-2-unknown",
                "row": None,
            }
        ],
        "warnings": [
            {
                "code": "unsupported_file",
                "message": "Ignored unsupported file: 'readme.pdf'",
                "query_id": None,
                "row": None,
            }
        ],
    }


def test_invalid_export_returns_422_with_stable_validation_codes():
    body = {
        "manifest": _manifest()[:2],
        "rows": [
            {
                "query_id": "query-p1-1-kis",
                "video_id": "not-a-video",
                "frames": [0, 2],
                "answer": "unexpected",
            }
        ],
    }

    response = client.post("/api/export", json=body)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["ok"] is False
    assert [error["code"] for error in detail["errors"]] == [
        "invalid_video_id",
        "invalid_frame",
        "kis_frame_count",
        "kis_unexpected_answer",
        "missing_query_rows",
    ]
    assert detail["warnings"] == []


def test_valid_mixed_export_returns_a_revalidated_pass_zip():
    response = client.post("/api/export", json=_valid_export_body())

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-validation-status"] == "PASS"
    assert "submission.zip" in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == [
            "submission/query-p1-1-kis.csv",
            "submission/query-p1-2-qa.csv",
            "submission/query-p1-3-trake.csv",
        ]
        assert list(
            csv.reader(
                io.StringIO(
                    archive.read("submission/query-p1-1-kis.csv").decode("utf-8")
                )
            )
        ) == [["L01_V001", "12"]]
        assert list(
            csv.reader(
                io.StringIO(
                    archive.read("submission/query-p1-2-qa.csv").decode("utf-8")
                )
            )
        ) == [["L01_V002", "34", "  Năm, người  "]]
        assert list(
            csv.reader(
                io.StringIO(
                    archive.read("submission/query-p1-3-trake.csv").decode("utf-8")
                )
            )
        ) == [["L01_V003", "10", "20"]]


def test_generated_archive_validation_failure_returns_500_with_full_report(monkeypatch):
    report = ValidationReport(
        errors=[
            ValidationIssue(
                "archive_paths_mismatch",
                "ZIP must contain exactly the expected submission CSV paths",
            )
        ],
        warnings=[ValidationIssue("post_write_warning", "kept for diagnostics")],
    )

    def fail_post_write_validation(manifest, rows, out_path):
        raise GeneratedArchiveError(report)

    monkeypatch.setattr(
        app_module, "write_validated_submission", fail_post_write_validation
    )

    response = client.post("/api/export", json=_valid_export_body())

    assert response.status_code == 500
    assert response.json()["detail"] == report.to_dict()


def test_existing_status_endpoint_remains_available_in_dummy_mode():
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["retriever"] == "dummy"
