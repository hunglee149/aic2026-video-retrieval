"""Tests cho LazyComponent — khoá lại tính độc lập giữa các thành phần.

Ba tính chất phải giữ bằng mọi giá:

1. ``snapshot()`` trả lời được **trong lúc** loader còn đang chạy. Đây là điều
   khiến ``/api/status`` nói được "đang nạp cái gì" thay vì treo cùng cái lock.
2. Lock theo từng thành phần: nạp CLIP không chặn nạp model dịch.
3. Một thành phần lỗi không kéo theo thành phần khác.
"""

from __future__ import annotations

import threading

import pytest

from aic.core.components import (
    ERROR,
    IDLE,
    LOADING,
    READY,
    ComponentRegistry,
    LazyComponent,
)


class TestStateMachine:
    def test_starts_idle_without_calling_loader(self):
        calls = []
        component = LazyComponent("clip", lambda: calls.append(1) or "value")

        assert component.state == IDLE
        assert calls == []

    def test_idle_to_ready(self):
        component = LazyComponent("clip", lambda: "retriever")

        assert component.get() == "retriever"
        assert component.state == READY

    def test_loader_result_can_carry_detail(self):
        component = LazyComponent("bm25", lambda: ("retriever", "bm25: 354k docs"))

        component.get()

        assert component.snapshot()["detail"] == "bm25: 354k docs"

    def test_detail_falls_back_to_describe(self):
        class Retriever:
            def describe(self):
                return "clip: 300k vectors"

        component = LazyComponent("clip", Retriever)

        component.get()

        assert component.snapshot()["detail"] == "clip: 300k vectors"

    def test_disabled_never_calls_loader(self):
        calls = []
        component = LazyComponent(
            "siglip",
            lambda: calls.append(1),
            disabled_reason="chưa cấu hình",
        )

        assert component.get() is None
        assert component.state == "disabled"
        assert calls == []

    def test_load_seconds_recorded(self):
        component = LazyComponent("clip", lambda: "v")

        component.get()

        assert component.snapshot()["load_seconds"] is not None


class TestErrorIsolation:
    def test_failure_sets_error_and_returns_none(self):
        def boom():
            raise RuntimeError("index hỏng")

        component = LazyComponent("bm25", boom)

        assert component.get() is None
        snapshot = component.snapshot()
        assert snapshot["state"] == ERROR
        assert "index hỏng" in snapshot["error"]

    def test_require_reraises(self):
        component = LazyComponent("bm25", lambda: 1 / 0)

        with pytest.raises(Exception):
            component.require()

    def test_error_is_sticky_no_retry_storm(self):
        """Nguồn hỏng hẳn mà thử lại mỗi request thì mọi lần search gánh thêm timeout."""
        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError("hỏng")

        component = LazyComponent("bm25", boom)
        component.get()
        component.get()
        component.get()

        assert len(calls) == 1

    def test_broken_component_does_not_affect_sibling(self):
        broken = LazyComponent("bm25", lambda: 1 / 0)
        healthy = LazyComponent("clip", lambda: "clip-retriever")
        registry = ComponentRegistry([broken, healthy])

        values = registry.ready_values()

        assert values == ["clip-retriever"]
        by_name = {s["name"]: s["state"] for s in registry.snapshot_all()}
        assert by_name == {"bm25": ERROR, "clip": READY}

    def test_reload_recovers_from_error(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("HF timeout")
            return "retriever"

        component = LazyComponent("clip", flaky)
        assert component.get() is None
        assert component.state == ERROR

        snapshot = component.reload()

        assert snapshot["state"] == READY
        assert component.get() == "retriever"


class TestStatusNeverBlocks:
    def test_snapshot_reports_loading_while_loader_runs(self):
        """Bằng chứng /api/status không bị chặn sau lock nạp."""
        entered = threading.Event()
        release = threading.Event()

        def slow_loader():
            entered.set()
            assert release.wait(timeout=5), "loader không được thả"
            return "retriever"

        component = LazyComponent("clip", slow_loader)
        worker = threading.Thread(target=component.get, daemon=True)
        worker.start()
        assert entered.wait(timeout=5)

        # Đọc trạng thái từ thread chính trong lúc loader còn đang chặn.
        snapshot = component.snapshot()
        assert snapshot["state"] == LOADING

        release.set()
        worker.join(timeout=5)
        assert component.state == READY

    def test_loading_one_component_does_not_block_another(self):
        release = threading.Event()
        entered = threading.Event()

        def slow():
            entered.set()
            assert release.wait(timeout=5)
            return "clip"

        slow_component = LazyComponent("clip", slow)
        fast_component = LazyComponent("translation", lambda: "translator")

        worker = threading.Thread(target=slow_component.get, daemon=True)
        worker.start()
        assert entered.wait(timeout=5)

        # Thành phần thứ hai phải nạp xong ngay, không chờ thành phần thứ nhất.
        assert fast_component.get() == "translator"
        assert fast_component.state == READY
        assert slow_component.state == LOADING

        release.set()
        worker.join(timeout=5)

    def test_concurrent_first_calls_load_exactly_once(self):
        calls = []
        gate = threading.Event()

        def loader():
            calls.append(1)
            gate.wait(timeout=5)
            return "value"

        component = LazyComponent("clip", loader)
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(component.get()))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        gate.set()
        for t in threads:
            t.join(timeout=5)

        assert len(calls) == 1
        assert results == ["value"] * 8


class TestMemoizeOff:
    def test_loader_runs_every_call_but_state_stays_ready(self):
        """Translation dùng chế độ này: cache thật nằm ở lru_cache bên ngoài."""
        calls = []
        component = LazyComponent(
            "translation", lambda: calls.append(1) or "t", memoize=False
        )

        assert component.get() == "t"
        assert component.get() == "t"

        assert len(calls) == 2
        # Không được nhấp nháy về "loading" ở lần gọi thứ hai.
        assert component.state == READY

    def test_second_call_unwraps_value_detail_tuple(self):
        """Lần gọi thứ hai cũng phải tách (value, detail), không trả nguyên tuple.

        Lỗi thật đã gặp: request dịch thứ hai nhận về tuple nên nổ
        "'tuple' object has no attribute 'translate'".
        """
        component = LazyComponent(
            "translation", lambda: ("translator", "detail"), memoize=False
        )

        assert component.get() == "translator"
        assert component.get() == "translator"

    def test_failure_after_ready_is_recorded(self):
        results = ["ok"]

        def loader():
            if not results:
                raise RuntimeError("model biến mất")
            return results.pop()

        component = LazyComponent("translation", loader, memoize=False)
        assert component.get() == "ok"

        assert component.get() is None
        assert component.state == ERROR


class TestRegistry:
    def test_kind_filters_snapshots_and_values(self):
        registry = ComponentRegistry(
            [
                LazyComponent("translation", lambda: "t", kind="translation"),
                LazyComponent("clip", lambda: "c"),
            ]
        )

        assert registry.ready_values(kind="retrieval") == ["c"]
        names = [s["name"] for s in registry.snapshot_all(kind="translation")]
        assert names == ["translation"]

    def test_snapshot_all_does_not_trigger_loading(self):
        calls = []
        registry = ComponentRegistry(
            [LazyComponent("clip", lambda: calls.append(1) or "c")]
        )

        assert [s["state"] for s in registry.snapshot_all()] == [IDLE]
        assert calls == []

    def test_warm_up_runs_in_declared_order(self):
        order = []
        registry = ComponentRegistry(
            [
                LazyComponent("clip", lambda: order.append("clip") or "c"),
                LazyComponent(
                    "translation",
                    lambda: order.append("translation") or "t",
                    kind="translation",
                ),
            ]
        )

        thread = registry.warm_up(["translation", "clip"])
        thread.join(timeout=5)

        assert order == ["translation", "clip"]

    def test_warm_up_survives_a_failing_component(self):
        registry = ComponentRegistry(
            [
                LazyComponent("bm25", lambda: 1 / 0),
                LazyComponent("clip", lambda: "c"),
            ]
        )

        thread = registry.warm_up(["bm25", "clip"])
        thread.join(timeout=5)

        by_name = {s["name"]: s["state"] for s in registry.snapshot_all()}
        assert by_name == {"bm25": ERROR, "clip": READY}

    def test_warm_up_with_empty_order_returns_none(self):
        registry = ComponentRegistry([LazyComponent("clip", lambda: "c")])

        assert registry.warm_up([]) is None

    def test_is_loading_reflects_live_state(self):
        entered = threading.Event()
        release = threading.Event()

        def slow():
            entered.set()
            release.wait(timeout=5)
            return "c"

        registry = ComponentRegistry([LazyComponent("clip", slow)])
        worker = threading.Thread(target=registry.ready_values, daemon=True)
        worker.start()
        assert entered.wait(timeout=5)

        assert registry.is_loading() is True

        release.set()
        worker.join(timeout=5)
        assert registry.is_loading() is False
