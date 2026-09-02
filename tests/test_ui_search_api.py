"""Tests cho /api/search và /api/status sau khi thêm modalities/weights.

Trọng tâm: payload UI cũ (không có modalities/weights) phải chạy y như trước,
và status phải nói rõ nguồn nào ready/error thay vì chỉ một chuỗi tên.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

import aic as aic_package
import aic.ui as aic_ui_package
from aic.core.types import Candidate

_MISSING = object()


class _RecordingDummy:
    """Đứng thay module dummy, ghi lại tham số search nhận được."""

    NAME = "dummy"

    def __init__(self):
        self.calls = []

    def search(self, query, limit=100, exclude=frozenset(), **kwargs):
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "exclude": exclude,
                "kwargs": kwargs,
            }
        )
        return [
            Candidate(
                video_id="L21_V001",
                start_frame=90,
                end_frame=90,
                representative_frames=[90],
                scores={"dummy": 0.9},
                evidence={"caption": "ứng viên giả"},
            ),
            Candidate(
                video_id="L21_V001",
                start_frame=5000,
                end_frame=5000,
                representative_frames=[5000],
                scores={"dummy": 0.5},
                evidence={},
            ),
        ]


def _restore(mapping, key, previous):
    if previous is _MISSING:
        mapping.pop(key, None)
    else:
        mapping[key] = previous


def _restore_attr(owner, name, previous):
    if previous is _MISSING:
        if hasattr(owner, name):
            delattr(owner, name)
    else:
        setattr(owner, name, previous)


@contextmanager
def _dummy_client():
    previous_env = os.environ.get("AIC_USE_DUMMY", _MISSING)
    previous_app = sys.modules.get("aic.ui.app", _MISSING)
    previous_retrieval = sys.modules.get("aic.retrieval", _MISSING)
    previous_app_attr = getattr(aic_ui_package, "app", _MISSING)
    previous_retrieval_attr = getattr(aic_package, "retrieval", _MISSING)

    recorder = _RecordingDummy()
    stub = types.ModuleType("aic.retrieval")
    stub.dummy = recorder

    import aic.core.local_translation
    import aic.core.query_processor
    previous_translate = getattr(aic.core.local_translation, "translate_text", _MISSING)
    previous_qp_translate = getattr(aic.core.query_processor, "_local_translate", _MISSING)
    
    mock_fn = lambda text: text + " (translated)"
    aic.core.local_translation.translate_text = mock_fn
    aic.core.query_processor._local_translate = mock_fn

    try:
        os.environ["AIC_USE_DUMMY"] = "1"
        sys.modules.pop("aic.ui.app", None)
        if hasattr(aic_ui_package, "app"):
            delattr(aic_ui_package, "app")
        sys.modules["aic.retrieval"] = stub
        setattr(aic_package, "retrieval", stub)

        module = importlib.import_module("aic.ui.app")
        client = TestClient(module.app, raise_server_exceptions=False)
        try:
            with client:
                client.app_module = module
                client.recorder = recorder
                yield client
        finally:
            client.close()
    finally:
        if previous_translate is not _MISSING:
            aic.core.local_translation.translate_text = previous_translate
        if previous_qp_translate is not _MISSING:
            aic.core.query_processor._local_translate = previous_qp_translate
        _restore(sys.modules, "aic.ui.app", previous_app)
        _restore(sys.modules, "aic.retrieval", previous_retrieval)
        _restore_attr(aic_ui_package, "app", previous_app_attr)
        _restore_attr(aic_package, "retrieval", previous_retrieval_attr)
        if previous_env is _MISSING:
            os.environ.pop("AIC_USE_DUMMY", None)
        else:
            os.environ["AIC_USE_DUMMY"] = previous_env


@pytest.fixture
def client():
    with _dummy_client() as test_client:
        yield test_client


LEGACY_PAYLOAD = {
    "query_id": "pack1_q1_kis",
    "text_vi": "người đàn ông nấu ăn",
    "text_en": "a man cooking",
    "task": "kis",
    "n_events": 1,
    "k": 10,
    "exclude": [],
}


class TestLegacySearchCompatibility:
    def test_legacy_payload_still_works(self, client):
        response = client.post("/api/search", json=LEGACY_PAYLOAD)

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["candidates"]
        assert body["candidates"][0]["rank"] == 1

    def test_legacy_payload_passes_limit_through(self, client):
        client.post("/api/search", json=LEGACY_PAYLOAD)

        assert client.recorder.calls
        assert client.recorder.calls[0]["limit"] == 10

    def test_missing_optional_fields_defaults_cleanly(self, client):
        response = client.post(
            "/api/search", json={"text_vi": "một chiếc xe máy màu đỏ"}
        )

        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_candidates_keep_two_moments_of_same_video(self, client):
        body = client.post("/api/search", json=LEGACY_PAYLOAD).json()

        frames = [c["start_frame"] for c in body["candidates"]]
        assert frames == [90, 5000]

    def test_no_fake_padding_in_response(self, client):
        payload = dict(LEGACY_PAYLOAD, k=100)

        body = client.post("/api/search", json=payload).json()

        assert body["total"] == 2
        assert all(
            not c["video_id"].startswith("L00_V") for c in body["candidates"]
        )


class TestOptionalRetrievalControls:
    def test_accepts_modalities(self, client):
        payload = dict(LEGACY_PAYLOAD, modalities=["transcript_segment"])

        response = client.post("/api/search", json=payload)

        assert response.status_code == 200
        assert client.recorder.calls[0]["query"].modalities == ["transcript_segment"]

    def test_accepts_weights(self, client):
        payload = dict(LEGACY_PAYLOAD, weights={"clip": 2.0})

        response = client.post("/api/search", json=payload)

        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_exclude_reaches_retriever(self, client):
        payload = dict(LEGACY_PAYLOAD, exclude=["L22_V002"])

        client.post("/api/search", json=payload)

        assert client.recorder.calls[0]["exclude"] == frozenset({"L22_V002"})


class TestStatusEndpoint:
    def test_status_reports_dummy_mode(self, client):
        body = client.get("/api/status").json()

        assert body["ok"] is True
        assert body["use_dummy"] is True

    def test_status_lists_per_retriever_state(self, client):
        body = client.get("/api/status").json()

        assert "retrievers" in body
        assert body["retrievers"]
        entry = body["retrievers"][0]
        assert set(entry) >= {"name", "state", "detail"}
        assert entry["state"] in {"disabled", "idle", "loading", "ready", "error"}

    def test_status_keeps_legacy_retriever_string(self, client):
        body = client.get("/api/status").json()

        assert isinstance(body["retriever"], str)
        assert body["retriever"]

    def test_status_lists_translation_alongside_retrieval(self, client):
        """Model dịch phải có ô trạng thái riêng, không lẫn vào retrieval."""
        body = client.get("/api/status").json()

        by_name = {c["name"]: c for c in body["components"]}
        assert "translation" in by_name
        assert by_name["translation"]["kind"] == "translation"
        assert body["translation"]["name"] == "translation"
        # Và không được lọt vào danh sách nguồn retrieval.
        assert "translation" not in {r["name"] for r in body["retrievers"]}

    def test_status_reports_whether_anything_is_loading(self, client):
        body = client.get("/api/status").json()

        assert body["loading"] is False

    def test_status_paths_describe_source_without_io(self, client):
        """paths phải mô tả được nguồn cấu hình mà không phát sinh tải file."""
        body = client.get("/api/status").json()

        assert body["paths"]["clip_index"]
        assert body["paths"]["text_index"]
