"""Khởi tạo lười, theo dõi trạng thái riêng cho từng thành phần nặng.

Vấn đề cần giải: model dịch, CLIP và BM25 đều mất hàng chục giây tới vài phút để
nạp. Nếu gộp chung một lock và một cờ trạng thái thì (a) bấm "dịch" phải chờ cả
CLIP lẫn BM25, (b) ``/api/status`` bị chặn sau đúng cái lock đang nạp nên không
thể hỏi "đang nạp cái gì", (c) một nguồn hỏng kéo sập cả hệ thống.

Cách giải: mỗi thành phần một ``LazyComponent`` với lock riêng, và trạng thái để
trong **một dict bất biến** thay nguyên khối. Reader chỉ đọc tham chiếu dict đó —
không bao giờ chạm vào lock nạp — nên status luôn trả lời được tức thì.

Máy trạng thái::

    disabled                      chưa cấu hình, không bao giờ gọi loader
    idle → loading → ready
                  └→ error        dính lại; chỉ ``reload()`` mới thử lại

``error`` dính là có chủ đích: một nguồn hỏng hẳn mà tự thử lại mỗi request thì
mọi lần search đều phải chịu thêm một lần timeout.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

DISABLED = "disabled"
IDLE = "idle"
LOADING = "loading"
READY = "ready"
ERROR = "error"

VALID_STATES = frozenset({DISABLED, IDLE, LOADING, READY, ERROR})


class LazyComponent:
    """Một thành phần nặng, nạp theo yêu cầu, có ô trạng thái đọc được mọi lúc.

    Parameters
    ----------
    name : tên hiển thị trong ``/api/status`` (``clip``, ``bm25``, ``translation``…)
    loader : callable không tham số, trả về ``(value, detail)`` hoặc chỉ ``value``.
        ``detail`` là chuỗi mô tả cho UI; không có thì lấy ``describe()`` của value.
    kind : nhóm thành phần — ``retrieval`` hoặc ``translation``.
    disabled_reason : có giá trị ⇒ trạng thái ``disabled``, loader không bao giờ chạy.
    memoize : ``False`` thì mỗi ``get()`` vẫn gọi lại loader. Dùng khi loader đã tự
        cache ở nơi khác (``lru_cache``) và ta chỉ muốn theo dõi trạng thái.
    """

    def __init__(
        self,
        name: str,
        loader: Callable[[], Any],
        *,
        kind: str = "retrieval",
        disabled_reason: Optional[str] = None,
        memoize: bool = True,
    ):
        self.name = name
        self.kind = kind
        self._loader = loader
        self._memoize = memoize
        self._load_lock = threading.Lock()
        self._value: Any = None
        if disabled_reason:
            self._state = self._make_state(DISABLED, f"{name}: {disabled_reason}")
        else:
            self._state = self._make_state(IDLE, f"{name}: chưa nạp")

    # -- trạng thái ---------------------------------------------------------

    def _make_state(
        self,
        state: str,
        detail: str,
        error: Optional[str] = None,
        load_seconds: Optional[float] = None,
    ) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "state": state,
            "detail": detail,
            "error": error,
            "load_seconds": load_seconds,
        }

    def snapshot(self) -> dict:
        """Ảnh chụp trạng thái. Không bao giờ chiếm ``_load_lock``.

        Đây chính là tính chất khiến ``/api/status`` trả lời được trong lúc thành
        phần khác đang nạp — đừng thêm lock vào đây.
        """
        return dict(self._state)

    @property
    def state(self) -> str:
        return self._state["state"]

    # -- nạp ----------------------------------------------------------------

    def get(self) -> Any:
        """Trả về value đã nạp, hoặc ``None`` nếu disabled/lỗi.

        Lỗi được nuốt và ghi vào trạng thái: một nguồn hỏng không được phép ném
        exception lên tầng trên và kéo sập các nguồn còn lại.
        """
        try:
            return self.require()
        except Exception:
            return None

    def require(self) -> Any:
        """Như ``get()`` nhưng ném lại lỗi gốc. Dùng cho translation."""
        state = self._state["state"]
        if state == DISABLED:
            return None
        if state == ERROR:
            raise RuntimeError(self._state["error"] or f"{self.name}: nạp thất bại")
        if state == READY:
            if self._memoize:
                return self._value
            # Không memoize ⇒ loader tự cache ở nơi khác. Gọi thẳng, không chiếm
            # lock và không đẩy trạng thái về "loading" — nếu không thì mỗi lần
            # dịch bình thường lại làm /api/status nhấp nháy "đang nạp".
            return self._reload_unmemoized()

        with self._load_lock:
            # Kiểm tra lại sau khi giành được lock: thread khác có thể đã nạp xong.
            state = self._state["state"]
            if state == DISABLED:
                return None
            if state == ERROR:
                raise RuntimeError(
                    self._state["error"] or f"{self.name}: nạp thất bại"
                )
            if state == READY and self._memoize:
                return self._value

            self._state = self._make_state(LOADING, f"{self.name}: đang nạp…")
            started = time.monotonic()
            try:
                result = self._loader()
            except Exception as exc:
                elapsed = time.monotonic() - started
                reason = f"{type(exc).__name__}: {exc}"
                logger.error("[%s] nạp thất bại sau %.1fs — %s", self.name, elapsed, reason)
                self._state = self._make_state(
                    ERROR, f"{self.name}: nạp thất bại", reason, round(elapsed, 2)
                )
                raise

            elapsed = round(time.monotonic() - started, 2)
            value, detail = _split_result(result)

            if value is None:
                # Loader chủ động từ chối (ví dụ ``_load_source`` báo disabled/error).
                self._state = self._make_state(
                    DISABLED if detail is None else ERROR,
                    detail or f"{self.name}: không nạp được",
                    None if detail is None else detail,
                    elapsed,
                )
                return None

            if detail is None:
                describe = getattr(value, "describe", None)
                detail = describe() if callable(describe) else self.name

            self._value = value if self._memoize else None
            self._state = self._make_state(READY, detail, None, elapsed)
            logger.info("[%s] ready sau %.1fs — %s", self.name, elapsed, detail)
            return value

    def _reload_unmemoized(self) -> Any:
        """Đường nhanh cho component không memoize đã ở trạng thái ready.

        Vẫn phải tách ``(value, detail)`` như đường nạp đầy đủ, nếu không lần gọi
        thứ hai sẽ trả về nguyên cái tuple cho caller.
        """
        try:
            value, _detail = _split_result(self._loader())
            return value
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.error("[%s] hỏng sau khi đã ready — %s", self.name, reason)
            self._state = self._make_state(
                ERROR, f"{self.name}: nạp thất bại", reason
            )
            raise

    def reset(self) -> dict:
        """Đưa về ``idle`` mà **không** nạp. ``disabled`` thì giữ nguyên."""
        with self._load_lock:
            if self._state["state"] == DISABLED:
                return self.snapshot()
            self._value = None
            self._state = self._make_state(IDLE, f"{self.name}: chưa nạp")
        return self.snapshot()

    def reload(self) -> dict:
        """Đặt lại về ``idle`` rồi nạp lại. Cách duy nhất thoát khỏi ``error``."""
        self.reset()
        self.get()
        return self.snapshot()


def _split_result(result: Any) -> tuple[Any, Optional[str]]:
    """Loader trả ``(value, detail)`` hoặc chỉ ``value``."""
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


class ComponentRegistry:
    """Tập các ``LazyComponent``, giữ thứ tự khai báo."""

    def __init__(self, components: Iterable[LazyComponent] = ()):
        self._components: dict[str, LazyComponent] = {}
        for component in components:
            self.add(component)

    def add(self, component: LazyComponent) -> LazyComponent:
        self._components[component.name] = component
        return component

    def get(self, name: str) -> Optional[LazyComponent]:
        return self._components.get(name)

    def names(self) -> list[str]:
        return list(self._components)

    def snapshot_all(self, kind: Optional[str] = None) -> list[dict]:
        return [
            c.snapshot()
            for c in self._components.values()
            if kind is None or c.kind == kind
        ]

    def ready_values(self, kind: Optional[str] = None) -> list:
        """Nạp mọi thành phần thuộc ``kind`` và trả về những cái thành công.

        Nguồn nào lỗi thì bị bỏ qua, không làm hỏng các nguồn còn lại.
        """
        values = []
        for component in self._components.values():
            if kind is not None and component.kind != kind:
                continue
            value = component.get()
            if value is not None:
                values.append(value)
        return values

    def is_loading(self) -> bool:
        return any(c.state == LOADING for c in self._components.values())

    def warm_up(self, order: Iterable[str]) -> Optional[threading.Thread]:
        """Nạp trước theo thứ tự, trong **một daemon thread**, không chặn startup.

        Tuần tự chứ không song song: ba thread cùng nạp model sẽ tranh CPU/RAM và
        làm chậm đúng cái thành phần operator cần trước tiên.
        """
        names = [n for n in order if n in self._components]
        if not names:
            return None

        def _run():
            for name in names:
                component = self._components[name]
                if component.state != IDLE:
                    continue
                logger.info("Warm-up: %s", name)
                component.get()  # lỗi đã được nuốt và ghi vào trạng thái

        thread = threading.Thread(target=_run, name="aic-warmup", daemon=True)
        thread.start()
        return thread
