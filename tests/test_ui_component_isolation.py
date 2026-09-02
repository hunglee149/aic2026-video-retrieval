"""Tests độc lập giữa các thành phần ở mức API.

Sự cố cần chặn tái diễn: bấm "dịch query" trên web phải chờ CLIP và BM25 nạp
xong, vì cả ba nạp đồng bộ trong lifespan và dùng chung một lock. Các test ở đây
khoá lại tính độc lập đó ở đúng tầng người dùng chạm vào — HTTP.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import aic.core.local_translation as local_translation
from aic.core.components import ComponentRegistry, LazyComponent
from aic.core.types import Candidate

_MISSING = object()


class FakeRetriever:
    """Retriever tối giản, ghi lại mọi lần search."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def describe(self):
        return f"{self.name}: fake"

    def search(self, query, limit=100, exclude=frozenset(), **kwargs):
        self.calls.append(query)
        return [
            Candidate(
                video_id="L21_V001",
                start_frame=90,
                end_frame=90,
                representative_frames=[90],
                scores={self.name: 0.9},
                evidence={},
            )
        ]


class CountingLoader:
    """Loader đếm số lần được gọi, có thể chặn lại ở một Event."""

    def __init__(self, name, *, gate=None, fail=False):
        self.name = name
        self.calls = 0
        self.entered = threading.Event()
        self.gate = gate
        self.fail = fail

    def __call__(self):
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(timeout=10), "loader không được thả"
        if self.fail:
            raise RuntimeError(f"{self.name}: index hỏng")
        return FakeRetriever(self.name)


class FakeTranslator:
    device = "cpu"

    def translate(self, text_vi):
        return "a man cooking"


@pytest.fixture
def reset_translator(monkeypatch):
    """Dùng translator giả, và trả TRANSLATOR về idle trước/sau mỗi test."""
    monkeypatch.setattr(local_translation, "LocalTranslator", lambda **kw: FakeTranslator())
    local_translation._load_translator.cache_clear()
    local_translation.translate_text.cache_clear()
    local_translation.TRANSLATOR.reset()
    monkeypatch.setenv("AIC_TRANSLATION_DEVICE", "cpu")
    yield
    local_translation._load_translator.cache_clear()
    local_translation.translate_text.cache_clear()
    local_translation.TRANSLATOR.reset()


@contextmanager
def client_with_components(**loaders):
    """TestClient với registry thay bằng các component giả.

    Component ``translation`` luôn là component thật, vì chính nó là thứ phải
    chứng minh được là độc lập.
    """
    import aic.ui.app as ui_app

    previous = ui_app._registry
    registry = ComponentRegistry()
    registry.add(local_translation.TRANSLATOR)
    components = {}
    for name, loader in loaders.items():
        component = LazyComponent(name, loader)
        components[name] = component
        registry.add(component)
    ui_app._registry = registry
    try:
        with TestClient(ui_app.app, raise_server_exceptions=False) as client:
            client.app_module = ui_app
            client.components = components
            yield client
    finally:
        ui_app._registry = previous


SEARCH_PAYLOAD = {
    "query_id": "pack1_q1_kis",
    "text_vi": "người đàn ông nấu ăn",
    "text_en": "a man cooking",
    "task": "kis",
    "k": 10,
}


class TestTranslateIsIndependentOfRetrieval:
    def test_translate_works_while_clip_is_still_loading(self, reset_translator):
        """Yêu cầu chính: dịch không phải chờ CLIP/BM25."""
        gate = threading.Event()
        clip_loader = CountingLoader("clip", gate=gate)
        bm25_loader = CountingLoader("bm25", gate=gate)

        with client_with_components(clip=clip_loader, bm25=bm25_loader) as client:
            # Ép CLIP vào trạng thái loading và giữ nó ở đó.
            worker = threading.Thread(
                target=client.components["clip"].get, daemon=True
            )
            worker.start()
            assert clip_loader.entered.wait(timeout=10)

            try:
                response = client.post(
                    "/api/translate", json={"text_vi": "một người đàn ông nấu ăn"}
                )

                assert response.status_code == 200
                body = response.json()
                assert body["ok"] is True
                assert body["text_en"] == "a man cooking"
                assert client.components["clip"].state == "loading"
            finally:
                gate.set()
                worker.join(timeout=10)

    def test_status_answers_while_clip_is_loading(self, reset_translator):
        gate = threading.Event()
        clip_loader = CountingLoader("clip", gate=gate)

        with client_with_components(clip=clip_loader) as client:
            worker = threading.Thread(
                target=client.components["clip"].get, daemon=True
            )
            worker.start()
            assert clip_loader.entered.wait(timeout=10)

            try:
                body = client.get("/api/status").json()

                assert body["ok"] is True
                assert body["loading"] is True
                by_name = {c["name"]: c["state"] for c in body["components"]}
                assert by_name["clip"] == "loading"
            finally:
                gate.set()
                worker.join(timeout=10)

    def test_translate_never_touches_retrieval_loaders(self, reset_translator):
        clip_loader = CountingLoader("clip")
        bm25_loader = CountingLoader("bm25")

        with client_with_components(clip=clip_loader, bm25=bm25_loader) as client:
            client.post("/api/translate", json={"text_vi": "xe máy màu đỏ"})

            assert clip_loader.calls == 0
            assert bm25_loader.calls == 0
            body = client.get("/api/status").json()
            by_name = {c["name"]: c["state"] for c in body["components"]}
            assert by_name["clip"] == "idle"
            assert by_name["bm25"] == "idle"
            assert by_name["translation"] == "ready"


class TestBrokenSourceDoesNotBreakTheSystem:
    def test_search_still_works_when_bm25_fails(self, reset_translator):
        clip_loader = CountingLoader("clip")
        bm25_loader = CountingLoader("bm25", fail=True)

        with client_with_components(clip=clip_loader, bm25=bm25_loader) as client:
            response = client.post("/api/search", json=SEARCH_PAYLOAD)

            assert response.status_code == 200
            assert response.json()["candidates"]

            body = client.get("/api/status").json()
            by_name = {c["name"]: c for c in body["components"]}
            assert by_name["bm25"]["state"] == "error"
            assert "index hỏng" in by_name["bm25"]["error"]
            assert by_name["clip"]["state"] == "ready"

    def test_search_returns_503_only_when_nothing_loads(self, reset_translator):
        with client_with_components(
            clip=CountingLoader("clip", fail=True),
            bm25=CountingLoader("bm25", fail=True),
        ) as client:
            response = client.post("/api/search", json=SEARCH_PAYLOAD)

            assert response.status_code == 503
            assert response.json()["detail"]["retrievers"]

    def test_translation_failure_does_not_break_search(self):
        """Model dịch hỏng thì search vẫn chạy bằng text tiếng Việt."""
        local_translation.TRANSLATOR.reset()
        try:
            with client_with_components(clip=CountingLoader("clip")) as client:
                with patch.object(
                    local_translation,
                    "_load_translator",
                    side_effect=RuntimeError("thiếu sentencepiece"),
                ):
                    local_translation.translate_text.cache_clear()
                    response = client.post(
                        "/api/search",
                        json={"query_id": "q1", "text_vi": "xe máy màu đỏ", "k": 5},
                    )

                assert response.status_code == 200
                assert response.json()["candidates"]
        finally:
            local_translation.translate_text.cache_clear()
            local_translation.TRANSLATOR.reset()


class TestReloadEndpoint:
    def test_reload_recovers_a_failed_source(self, reset_translator):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("HF timeout")
            return FakeRetriever("bm25")

        with client_with_components(bm25=flaky) as client:
            client.post("/api/search", json=SEARCH_PAYLOAD)
            assert client.components["bm25"].state == "error"

            body = client.post("/api/components/bm25/reload").json()

            assert body["component"]["state"] == "ready"
            assert client.get("/api/status").json()["ready_count"] == 1

    def test_error_is_sticky_until_reload(self, reset_translator):
        loader = CountingLoader("bm25", fail=True)

        with client_with_components(bm25=loader) as client:
            client.post("/api/search", json=SEARCH_PAYLOAD)
            client.post("/api/search", json=SEARCH_PAYLOAD)
            client.post("/api/search", json=SEARCH_PAYLOAD)

            assert loader.calls == 1

    def test_unknown_component_returns_404(self, reset_translator):
        with client_with_components(clip=CountingLoader("clip")) as client:
            assert client.post("/api/components/nope/reload").status_code == 404

    def test_translation_can_be_reloaded(self, reset_translator):
        with client_with_components(clip=CountingLoader("clip")) as client:
            body = client.post("/api/components/translation/reload").json()

            assert body["component"]["name"] == "translation"
            assert body["component"]["state"] == "ready"


class TestNoWorkAtImportTime:
    def test_importing_app_downloads_nothing(self):
        """Import module không được gọi hf_hub_download.

        Uvicorn import module trước khi bind port; phân giải artifact ở đây nghĩa
        là server chưa tồn tại thì đã tải hàng GB.
        """
        previous = sys.modules.get("aic.ui.app", _MISSING)
        sys.modules.pop("aic.ui.app", None)
        try:
            with patch("huggingface_hub.hf_hub_download") as download:
                importlib.import_module("aic.ui.app")

                assert download.call_count == 0
        finally:
            sys.modules.pop("aic.ui.app", None)
            if previous is not _MISSING:
                sys.modules["aic.ui.app"] = previous
            else:
                importlib.import_module("aic.ui.app")

    def test_status_after_startup_is_all_idle(self, reset_translator):
        """AIC_PRELOAD=none ⇒ startup không nạp gì cả."""
        previous = os.environ.get("AIC_PRELOAD", _MISSING)
        os.environ["AIC_PRELOAD"] = "none"
        try:
            with client_with_components(
                clip=CountingLoader("clip"), bm25=CountingLoader("bm25")
            ) as client:
                body = client.get("/api/status").json()

                states = {c["name"]: c["state"] for c in body["components"]}
                assert states == {
                    "translation": "idle",
                    "clip": "idle",
                    "bm25": "idle",
                }
                assert body["loading"] is False
        finally:
            if previous is _MISSING:
                os.environ.pop("AIC_PRELOAD", None)
            else:
                os.environ["AIC_PRELOAD"] = previous


class TestPreloadOrder:
    def test_default_puts_translation_first(self):
        import aic.ui.app as ui_app

        assert ui_app.preload_order()[0] == "translation"

    def test_none_disables_warm_up(self, monkeypatch):
        import aic.ui.app as ui_app

        monkeypatch.setenv("AIC_PRELOAD", "none")

        assert ui_app.preload_order() == ()

    def test_explicit_list_is_respected(self, monkeypatch):
        import aic.ui.app as ui_app

        monkeypatch.setenv("AIC_PRELOAD", "bm25, translation")

        assert ui_app.preload_order() == ("bm25", "translation")


class TestFrameIndexLoadsOnce:
    """Frame index cũng phải là một thành phần có lock và có trạng thái.

    Lỗi cũ: ``_get_frame_mapping`` không có lock. UI render lưới candidate làm
    nhiều request /api/keyframe cùng lúc, mỗi request tự dựng lại mapping 176k
    khoá (đo được 2,03 s và 78 MB một lần) rồi vứt đi tất cả trừ một.
    """

    @contextmanager
    def _client(self, read_fn):
        import aic.ui.app as ui_app
        from aic.core.components import LazyComponent as _Lazy

        previous = ui_app._registry
        registry = ComponentRegistry()
        registry.add(local_translation.TRANSLATOR)
        component = _Lazy("keyframe_map", ui_app._build_frame_index, kind="media")
        registry.add(component)
        ui_app._registry = registry
        with patch.object(ui_app, "clip_meta_path", return_value=Path("meta.json")), \
             patch.object(ui_app, "_read_keyframe_metadata", side_effect=read_fn), \
             patch.object(Path, "exists", return_value=True):
            try:
                with TestClient(ui_app.app, raise_server_exceptions=False) as client:
                    client.app_module = ui_app
                    client.component = component
                    yield client
            finally:
                ui_app._registry = previous

    def test_concurrent_first_reads_parse_metadata_once(self):
        calls = []

        def read(path):
            calls.append(path)
            time.sleep(0.05)
            return {("L21_V001", 100): 5}, {"L21_V001": [100]}

        with self._client(read) as client:
            ui_app = client.app_module
            results = []
            threads = [
                threading.Thread(target=lambda: results.append(ui_app.get_frame_index()))
                for _ in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert len(calls) == 1
            assert len(results) == 20
            assert all(r is results[0] for r in results)

    def test_failure_is_visible_in_status_instead_of_silent_empty(self):
        def read(path):
            raise ValueError("metadata hỏng")

        with self._client(read) as client:
            assert len(client.app_module.get_frame_index()) == 0

            body = client.get("/api/status").json()
            slot = {c["name"]: c for c in body["components"]}["keyframe_map"]
            assert slot["state"] == "error"
            assert "metadata hỏng" in slot["error"]

    def test_keyframe_map_is_not_counted_as_a_retrieval_source(self):
        def read(path):
            return {("L21_V001", 100): 5}, {"L21_V001": [100]}

        with self._client(read) as client:
            client.app_module.get_frame_index()

            body = client.get("/api/status").json()
            assert "keyframe_map" in {c["name"] for c in body["components"]}
            assert "keyframe_map" not in {r["name"] for r in body["retrievers"]}
            assert body["ready_count"] == 0

    def test_keyframe_endpoint_uses_exact_then_nearest_match(self):
        def read(path):
            mapping = {("L21_V001", 100): 5, ("L21_V001", 400): 9}
            return mapping, {"L21_V001": [100, 400]}

        with self._client(read) as client:
            exact = client.get("/api/keyframe/L21_V001/100", follow_redirects=False)
            nearest = client.get("/api/keyframe/L21_V001/390", follow_redirects=False)

            assert exact.headers["location"].endswith("/L21_V001/005.jpg")
            assert nearest.headers["location"].endswith("/L21_V001/009.jpg")

    def test_reload_rebuilds_the_index(self):
        attempts = {"n": 0}

        def read(path):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ValueError("hụt mạng")
            return {("L21_V001", 100): 5}, {"L21_V001": [100]}

        with self._client(read) as client:
            assert len(client.app_module.get_frame_index()) == 0

            body = client.post("/api/components/keyframe_map/reload").json()

            assert body["component"]["state"] == "ready"
            assert len(client.app_module.get_frame_index()) == 1


class TestFrameIndexLookup:
    def test_nearest_lookup_stays_inside_the_same_video(self):
        import aic.ui.app as ui_app

        index = ui_app.FrameIndex(
            {("L21_V001", 100): 5, ("L22_V009", 105): 7},
            {"L21_V001": [100], "L22_V009": [105]},
        )

        assert index.keyframe_num("L21_V001", 104) == 5
        assert index.keyframe_num("L22_V009", 104) == 7
        assert index.keyframe_num("L30_V001", 104) is None

    def test_read_metadata_sorts_frames_per_video(self, tmp_path):
        import json

        import aic.ui.app as ui_app

        path = tmp_path / "meta.json"
        path.write_text(
            json.dumps(
                [
                    {"video_id": "L21_V001", "frame_idx": 400, "keyframe_num": 9},
                    {"video_id": "L21_V001", "frame_idx": 100, "keyframe_num": 5},
                    {"video_id": "L21_V001", "frame_idx": 250},
                ]
            ),
            encoding="utf-8",
        )

        mapping, by_video = ui_app._read_keyframe_metadata(path)

        assert by_video == {"L21_V001": [100, 400]}
        assert mapping == {("L21_V001", 400): 9, ("L21_V001", 100): 5}
