import csv
from contextlib import contextmanager
import importlib
import io
import os
import sys
import types
import zipfile

import pytest
from fastapi.testclient import TestClient

import aic as aic_package
import aic.ui as aic_ui_package
from aic.submission.query_pack import ValidationIssue
from aic.submission.validator import GeneratedArchiveError, ValidationReport


_MISSING = object()


class _DummyRetriever:
    NAME = "dummy"

    @staticmethod
    def search(query, k=100, exclude=frozenset()):
        return []


retrieval_stub = types.ModuleType("aic.retrieval")
retrieval_stub.dummy = _DummyRetriever()


def _restore_mapping_item(mapping, key, previous):
    if previous is _MISSING:
        mapping.pop(key, None)
    else:
        mapping[key] = previous


def _restore_attribute(owner, name, previous):
    if previous is _MISSING:
        if hasattr(owner, name):
            delattr(owner, name)
    else:
        setattr(owner, name, previous)


@contextmanager
def _isolated_api_client():
    previous_use_dummy = os.environ.get("AIC_USE_DUMMY", _MISSING)
    previous_app_module = sys.modules.get("aic.ui.app", _MISSING)
    previous_retrieval_module = sys.modules.get("aic.retrieval", _MISSING)
    previous_app_attribute = getattr(aic_ui_package, "app", _MISSING)
    previous_retrieval_attribute = getattr(aic_package, "retrieval", _MISSING)

    try:
        os.environ["AIC_USE_DUMMY"] = "1"
        sys.modules.pop("aic.ui.app", None)
        if hasattr(aic_ui_package, "app"):
            delattr(aic_ui_package, "app")
        sys.modules["aic.retrieval"] = retrieval_stub
        setattr(aic_package, "retrieval", retrieval_stub)

        app_module = importlib.import_module("aic.ui.app")
        test_client = TestClient(
            app_module.app, raise_server_exceptions=False
        )
        try:
            with test_client:
                test_client.app_module = app_module
                yield test_client
        finally:
            test_client.close()
    finally:
        _restore_mapping_item(sys.modules, "aic.ui.app", previous_app_module)
        _restore_mapping_item(
            sys.modules, "aic.retrieval", previous_retrieval_module
        )
        _restore_attribute(aic_ui_package, "app", previous_app_attribute)
        _restore_attribute(
            aic_package, "retrieval", previous_retrieval_attribute
        )
        _restore_mapping_item(os.environ, "AIC_USE_DUMMY", previous_use_dummy)


@pytest.fixture
def client():
    with _isolated_api_client() as test_client:
        yield test_client


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


def test_query_pack_zip_imports_raw_archive_body_in_member_order(client):
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


def test_query_pack_texts_imports_multiple_json_files_in_request_order(client):
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


def test_invalid_query_pack_returns_422_with_the_complete_parser_report(client):
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


def test_invalid_export_returns_422_with_stable_validation_codes(client):
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
    ]
    assert [warning["code"] for warning in detail["warnings"]] == [
        "missing_query_rows",
    ]


def test_empty_export_manifest_returns_the_shared_stable_422_report(client):
    response = client.post("/api/export", json={"manifest": [], "rows": []})

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "ok": False,
        "errors": [
            {
                "code": "empty_manifest",
                "message": "Submission manifest must contain at least one query",
                "query_id": None,
                "row": None,
            }
        ],
        "warnings": [],
    }


@pytest.mark.parametrize(
    ("path", "body", "field"),
    [
        ("/api/export", {"manifest": []}, "rows"),
        (
            "/api/query-pack/texts",
            {"files": [{"filename": "query-p1-1-kis.txt"}]},
            "files.0.content",
        ),
    ],
)
def test_submission_request_schema_failures_use_the_shared_422_report(
    client, path, body, field
):
    response = client.post(path, json=body)

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "ok": False,
        "errors": [
            {
                "code": "invalid_request_schema",
                "message": f"Request field {field!r} is invalid or missing",
                "query_id": None,
                "row": None,
            }
        ],
        "warnings": [],
    }


def test_non_submission_request_schema_failure_keeps_fastapi_default_shape(client):
    response = client.post("/api/translate", json={})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert response.json()["detail"][0]["loc"] == ["body", "text_vi"]


def test_export_rejects_all_manifest_invariant_bypasses_in_one_stable_report(client):
    manifest = [
        {
            "query_id": "query-p1-4-other",
            "task": "kis",
            "text": "unknown suffix",
            "source_name": "query-p1-4-other.txt",
            "n_events": None,
            "events_confirmed": True,
        },
        {
            "query_id": "query-p1-5-kis",
            "task": "other",
            "text": "unknown task",
            "source_name": "query-p1-5-kis.txt",
            "n_events": None,
            "events_confirmed": True,
        },
        {
            "query_id": "query-p1-6-kis",
            "task": "qa",
            "text": "mismatched task",
            "source_name": "query-p1-6-kis.txt",
            "n_events": None,
            "events_confirmed": True,
        },
        {
            "query_id": "../query-p1-7-kis",
            "task": "kis",
            "text": "unsafe ID",
            "source_name": "query-p1-7-kis.txt",
            "n_events": None,
            "events_confirmed": True,
        },
        {
            "query_id": "query-p1-8-qa",
            "task": "qa",
            "text": "first duplicate",
            "source_name": "query-p1-8-qa.txt",
            "n_events": None,
            "events_confirmed": True,
        },
        {
            "query_id": "query-p1-8-qa",
            "task": "qa",
            "text": "second duplicate",
            "source_name": "query-p1-8-qa.txt",
            "n_events": None,
            "events_confirmed": True,
        },
        {
            "query_id": "query-p1-9-trake",
            "task": "trake",
            "text": "zero events",
            "source_name": "query-p1-9-trake.txt",
            "n_events": 0,
            "events_confirmed": True,
        },
        {
            "query_id": "query-p1-10-trake",
            "task": "trake",
            "text": "missing events",
            "source_name": "query-p1-10-trake.txt",
            "n_events": None,
            "events_confirmed": True,
        },
    ]

    response = client.post("/api/export", json={"manifest": manifest, "rows": []})

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "ok": False,
        "errors": [
            {
                "code": "invalid_task_suffix",
                "message": (
                    "Query ID 'query-p1-4-other' must end with -kis, -qa, or -trake"
                ),
                "query_id": "query-p1-4-other",
                "row": None,
            },
            {
                "code": "invalid_manifest_task",
                "message": "Manifest task must be kis, qa, or trake; got 'other'",
                "query_id": "query-p1-5-kis",
                "row": None,
            },
            {
                "code": "manifest_task_mismatch",
                "message": "Query ID suffix implies 'kis', not 'qa'",
                "query_id": "query-p1-6-kis",
                "row": None,
            },
            {
                "code": "unsafe_query_id",
                "message": "Query ID must be a non-empty filename without a path",
                "query_id": "../query-p1-7-kis",
                "row": None,
            },
            {
                "code": "duplicate_query_id",
                "message": "Duplicate query ID: query-p1-8-qa",
                "query_id": "query-p1-8-qa",
                "row": None,
            },
            {
                "code": "invalid_trake_event_count",
                "message": "Confirmed TRAKE query requires a positive event count",
                "query_id": "query-p1-9-trake",
                "row": None,
            },
            {
                "code": "invalid_trake_event_count",
                "message": "Confirmed TRAKE query requires a positive event count",
                "query_id": "query-p1-10-trake",
                "row": None,
            },
        ],
        "warnings": [],
    }


def test_export_preserves_raw_frame_types_for_domain_validation(client):
    response = client.post(
        "/api/export",
        json={
            "manifest": _manifest()[:1],
            "rows": [
                {
                    "query_id": "query-p1-1-kis",
                    "video_id": "L01_V001",
                    "frames": [True, "12"],
                    "answer": "",
                }
            ],
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert [error["code"] for error in detail["errors"]] == [
        "invalid_frame",
        "kis_frame_count",
    ]
    assert detail["warnings"] == []


def test_valid_mixed_export_returns_a_revalidated_pass_zip(client):
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


def test_generated_archive_validation_failure_returns_500_with_full_report(
    monkeypatch, client
):
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
        client.app_module, "write_validated_submission", fail_post_write_validation
    )

    response = client.post("/api/export", json=_valid_export_body())

    assert response.status_code == 500
    assert response.json()["detail"] == report.to_dict()


def test_existing_status_endpoint_remains_available_in_dummy_mode(client):
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["retriever"] == "dummy"


def test_api_client_teardown_restores_preexisting_cache_and_process_state(monkeypatch):
    cached_app = types.ModuleType("aic.ui.app")
    cached_retrieval = types.ModuleType("aic.retrieval")
    monkeypatch.setenv("AIC_USE_DUMMY", "prior-value")
    monkeypatch.setitem(sys.modules, "aic.ui.app", cached_app)
    monkeypatch.setitem(sys.modules, "aic.retrieval", cached_retrieval)
    monkeypatch.setattr(aic_ui_package, "app", cached_app, raising=False)
    monkeypatch.setattr(
        aic_package, "retrieval", cached_retrieval, raising=False
    )

    with _isolated_api_client() as isolated_client:
        fresh_app = isolated_client.app_module
        assert fresh_app is not cached_app
        assert sys.modules["aic.ui.app"] is fresh_app
        assert aic_ui_package.app is fresh_app
        assert os.environ["AIC_USE_DUMMY"] == "1"
        assert sys.modules["aic.retrieval"] is retrieval_stub
        assert aic_package.retrieval is retrieval_stub

    assert isolated_client.is_closed
    assert sys.modules["aic.ui.app"] is cached_app
    assert aic_ui_package.app is cached_app
    assert os.environ["AIC_USE_DUMMY"] == "prior-value"
    assert sys.modules["aic.retrieval"] is cached_retrieval
    assert aic_package.retrieval is cached_retrieval
