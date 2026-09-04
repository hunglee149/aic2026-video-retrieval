"""AIC 2026 — UI Server.

FastAPI backend + static file server cho giao diện operator.

Chạy:
    uvicorn aic.ui.app:app --reload --port 8000
    hoặc:
    python -m aic.ui

API:
    POST /api/search        — tìm kiếm candidates
    POST /api/translate     — dịch query VI→EN
    POST /api/export        — xuất submission.zip
    GET  /api/status        — trạng thái server
    GET  /api/keyframe/{video_id}/{frame_idx}  — ảnh keyframe
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

# Load .env manually if exists (no dotenv dependency required).
# setdefault chứ không gán đè: biến môi trường truyền tường minh khi chạy lệnh
# phải thắng file .env, đúng ngữ nghĩa dotenv thông thường.
env_path = Path(".env")
if env_path.exists():
    for _line in env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core import local_translation
from ..core.components import ComponentRegistry, LazyComponent
from ..core.convert import to_answer, to_csv_row
from ..core.local_translation import translate_text
from ..core.query_processor import make_query
from ..core.types import Candidate
from ..fusion.rank import fuse
from ..pipeline import retrieve_and_fuse
from ..retrieval import dummy
from ..submission import (
    GeneratedArchiveError,
    QueryDefinition,
    SubmissionValidationError,
    ValidationIssue,
    ValidationReport,
    write_validated_submission,
)
from ..submission.query_pack import infer_task, parse_query_files, parse_query_zip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KEYFRAMES_DIR = Path(os.environ.get("AIC_KEYFRAMES_DIR", "data/keyframes"))

def resolve_path(
    remote_filename: Optional[str], env_var: Optional[str] = None
) -> Optional[Path]:
    """Đường dẫn tới artifact: env var → file sẵn có trong workspace → Hugging Face.

    Thứ tự này quan trọng. Máy dev thường đã có sẵn file (pickle BM25 nặng
    564 MB) nên tải lại từ HF là lãng phí thuần tuý. Chỉ chấp nhận env var khi
    file **thực sự tồn tại**: trỏ sai đường dẫn mà vẫn im lặng chạy tiếp còn tệ
    hơn là tải lại, vì lỗi sẽ nổ ở tận tầng retriever.

    ``env_var`` là đường gọi tường minh của ``artifact_path()``; không truyền thì
    tra bảng ``env_mappings`` bên dưới, để caller cũ chỉ đưa tên file vẫn đúng.
    """
    if not remote_filename:
        return None

    # Mapping remote filenames to env variables
    env_mappings = {
        "local/clip_faiss.index": "AIC_INDEX_PATH",
        "local/clip_metadata.json": "AIC_META_PATH",
        "data/input/input/index/text_search_index.pkl": "AIC_TEXT_INDEX_PATH",
        "data/input/input/index/siglip_faiss.index": "AIC_SIGLIP_INDEX_PATH",
        "data/input/input/index/siglip_metadata.json": "AIC_SIGLIP_META_PATH",
        "local/video_metadata.json": "AIC_VIDEO_METADATA_PATH",
    }
    
    # 1. Kiểm tra biến môi trường tương ứng
    env_name = env_var or env_mappings.get(remote_filename)
    if env_name:
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            p = Path(env_val)
            if p.exists():
                return p
            logger.warning("Đường dẫn cấu hình qua %s = %s không tồn tại locally.", env_name, env_val)

    # 2. Kiểm tra nếu file đã tồn tại ở workspace hiện tại
    p_local = Path(remote_filename)
    if p_local.exists():
        return p_local
        
    p_local_short = Path("local") / p_local.name
    if p_local_short.exists():
        return p_local_short

    # 3. Tải qua Hugging Face Hub
    repo_id = os.environ.get("AIC_HF_REPO_ID", "manhha2502/fullhd")
    revision = os.environ.get("AIC_HF_REVISION", "main")
    cache_dir = os.environ.get("AIC_HF_CACHE_DIR") or None
    try:
        from huggingface_hub import hf_hub_download

        is_mocked = hasattr(hf_hub_download, "mock_add_spec") or hf_hub_download.__class__.__name__ in ("MagicMock", "Mock")
        # Chốt chặn để test không lỡ tải 564 MB thật. Trước đây điều kiện là
        # `"pytest" in sys.modules or "unittest" in sys.modules`, mà `unittest`
        # bị import gián tiếp bởi thư viện khác ngay trong server production —
        # đo được trên chính server đang chạy — nên nhánh này từng chạy ở
        # production và trả về `local/<tên file>` bịa ra. Kết quả là log báo
        # "không tìm thấy local/text_search_index.pkl" trong khi sự thật là
        # không tải được từ HF. Dùng biến môi trường pytest tự đặt thì chính xác.
        in_tests = bool(os.environ.get("PYTEST_CURRENT_TEST")) or (
            os.environ.get("AIC_SKIP_HF_DOWNLOAD", "0") == "1"
        )
        if in_tests and not is_mocked:
            logger.info("Test environment detected. Skipping real HF download for %s", remote_filename)
            # Trả None chứ không bịa đường dẫn: nguồn dùng nó sẽ báo "không phân
            # giải được artifact", đúng nguyên nhân thật.
            return None

        logger.info("Đang kiểm tra/tải file %s từ HF repo %s (revision: %s)...", remote_filename, repo_id, revision)
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_filename,
            revision=revision,
            repo_type="dataset",
            cache_dir=cache_dir
        )
        return Path(cached_path)
    except Exception as e:
        logger.error("Lỗi khi tải file %s từ Hugging Face: %s", remote_filename, e)
        raise e

# Khai báo artifact: tên logic → (env var, đường dẫn trong HF dataset).
# KHÔNG phân giải ở đây. resolve_path() gọi hf_hub_download đồng bộ, mà module này
# được import trước khi uvicorn bind port — phân giải lúc import nghĩa là server
# chưa tồn tại thì đã tải xong hàng GB.
_ARTIFACTS: dict[str, tuple[str, str]] = {
    "clip_index": ("AIC_INDEX_PATH", "local/clip_faiss.index"),
    "clip_meta": ("AIC_META_PATH", "local/clip_metadata.json"),
    "text_index": (
        "AIC_TEXT_INDEX_PATH",
        "data/input/input/index/text_search_index.pkl",
    ),
    "siglip_index": (
        "AIC_SIGLIP_INDEX_PATH",
        "data/input/input/index/siglip_faiss.index",
    ),
    "siglip_meta": (
        "AIC_SIGLIP_META_PATH",
        "data/input/input/index/siglip_metadata.json",
    ),
    "video_metadata": ("AIC_VIDEO_METADATA_PATH", "local/video_metadata.json"),
}


# Chỉ nhớ lần phân giải **thành công**. Nhớ cả thất bại thì một lần HF hụt mạng
# sẽ đóng băng vĩnh viễn, và ngay cả /api/components/<tên>/reload cũng vô dụng.
_artifact_cache: dict[str, Path] = {}


def artifact_path(key: str) -> Optional[Path]:
    """Phân giải artifact theo yêu cầu, không bao giờ ném lúc import.

    Trả ``None`` khi không lấy được — nguồn dùng nó sẽ báo ``error`` riêng chứ
    không kéo sập các nguồn khác.
    """
    cached = _artifact_cache.get(key)
    if cached is not None:
        return cached

    env_var, remote = _ARTIFACTS[key]
    try:
        path = resolve_path(remote, env_var)
    except Exception as exc:
        logger.error("Không phân giải được artifact %s: %s", key, exc)
        return None
    if path is not None:
        _artifact_cache[key] = path
    return path


def reset_artifact_cache(key: Optional[str] = None) -> None:
    """Quên đường dẫn đã nhớ, để lần reload sau phân giải lại từ đầu."""
    if key is None:
        _artifact_cache.clear()
    else:
        _artifact_cache.pop(key, None)


def artifact_source(key: str) -> str:
    """Mô tả nguồn cấu hình, **không** phát sinh I/O — dùng cho /api/status."""
    env_var, remote = _ARTIFACTS[key]
    raw = os.environ.get(env_var, "").strip()
    if raw:
        return raw
    repo_id = os.environ.get("AIC_HF_REPO_ID", "manhha2502/fullhd")
    return f"hf://{repo_id}/{remote}"


def clip_index_path() -> Optional[Path]:
    return artifact_path("clip_index")


def clip_meta_path() -> Optional[Path]:
    return artifact_path("clip_meta")


def text_index_path() -> Optional[Path]:
    return artifact_path("text_index")


def _optional_path(name: str) -> Optional[Path]:
    """Path từ env, hoặc None nếu chưa cấu hình (khác với 'cấu hình sai')."""
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


MAP_KEYFRAMES_DIR = _optional_path("AIC_MAP_KEYFRAMES_DIR")

def _siglip_enabled() -> bool:
    return os.environ.get("AIC_USE_SIGLIP", "0") == "1"


def siglip_index_path() -> Optional[Path]:
    return artifact_path("siglip_index") if _siglip_enabled() else None


def siglip_meta_path() -> Optional[Path]:
    return artifact_path("siglip_meta") if _siglip_enabled() else None

USE_DUMMY = os.environ.get("AIC_USE_DUMMY", "0") == "1"
DISABLE_NEURAL = os.environ.get("AIC_DISABLE_NEURAL", "0").strip() == "1"

AIC_USE_CLOUD_MEDIA = os.environ.get("AIC_USE_CLOUD_MEDIA", "1") == "1"
HF_DATASET_URL = os.environ.get("AIC_HF_DATASET_URL", "https://huggingface.co/datasets/manhha2502/fullhd/resolve/main")
_video_metadata = {}

def load_video_metadata():
    global _video_metadata
    if _video_metadata:
        return _video_metadata
    path = artifact_path("video_metadata")
    if path and path.exists():
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                _video_metadata = json.load(f)
            logger.info("Loaded video metadata with %d mappings", len(_video_metadata))
        except Exception as e:
            logger.warning("Error loading video metadata: %s", e)
    return _video_metadata

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

DEFAULT_PRELOAD_ORDER = ("translation", "clip", "bm25", "siglip")


def preload_order() -> tuple[str, ...]:
    """Thứ tự warm-up từ ``AIC_PRELOAD``.

    Mặc định nạp translation trước: operator bấm "dịch" trước khi bấm "tìm", nên
    đó là thành phần đáng ấm sớm nhất. ``none`` = lười hoàn toàn.
    """
    raw = os.environ.get("AIC_PRELOAD", "").strip()
    if not raw or raw.lower() == "all":
        return DEFAULT_PRELOAD_ORDER
    if raw.lower() == "none":
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@asynccontextmanager
async def lifespan(app: FastAPI):
    import sys

    # Không nạp gì trước `yield`. Trước đây chỗ này nạp CLIP + BM25 + model dịch
    # đồng bộ, nên server không phục vụ nổi một request nào — kể cả /api/status —
    # cho tới khi cả ba xong. Giờ chỉ khởi động một thread warm-up chạy nền.
    registry = get_registry()
    in_tests = "pytest" in sys.modules or "unittest" in sys.modules
    if not in_tests or os.environ.get("AIC_FORCE_PRELOAD") == "1":
        order = preload_order()
        if order:
            logger.info("Warm-up nền theo thứ tự: %s", ", ".join(order))
            registry.warm_up(order)
        else:
            logger.info("AIC_PRELOAD=none — mọi thành phần nạp theo yêu cầu")
    yield


app = FastAPI(title="AIC 2026 Video Retrieval", version="0.1.0", lifespan=lifespan)

from fastapi import WebSocket, WebSocketDisconnect
import json

# Shared state variables
SHARED_STATE_PATH = Path("data/shared_state.json")
shared_manifest: list = []
shared_selections: list = []
shared_query_cache: dict = {}

def load_shared_state():
    global shared_manifest, shared_selections, shared_query_cache
    if SHARED_STATE_PATH.exists():
        try:
            with open(SHARED_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            shared_manifest = data.get("manifest", [])
            shared_selections = data.get("selections", [])
            shared_query_cache = data.get("queryCache", {})
            logger.info("Loaded shared state from %s", SHARED_STATE_PATH)
        except Exception as e:
            logger.warning("Error loading shared state: %s", e)

def save_shared_state():
    try:
        SHARED_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SHARED_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "manifest": shared_manifest,
                "selections": shared_selections,
                "queryCache": shared_query_cache
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Error saving shared state: %s", e)

# Load state on startup
load_shared_state()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.pending_clear: Optional[dict] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if self.pending_clear:
            if self.pending_clear.get("requester") == websocket:
                self.pending_clear = None
                try:
                    asyncio.create_task(self.broadcast({"type": "clear_cache_dismiss"}))
                except Exception:
                    pass
            elif websocket in self.pending_clear.get("pending_others", set()):
                self.pending_clear["pending_others"].discard(websocket)
                remaining = [
                    ws for ws in self.pending_clear["pending_others"]
                    if ws in self.active_connections
                ]
                if not remaining:
                    requester = self.pending_clear.get("requester")
                    self.pending_clear = None
                    if requester in self.active_connections:
                        try:
                            asyncio.create_task(requester.send_json({
                                "type": "clear_cache_rejected",
                                "reason": "Tất cả các máy khác đã ngắt kết nối.",
                            }))
                        except Exception:
                            pass
                    try:
                        asyncio.create_task(self.broadcast({"type": "clear_cache_dismiss"}))
                    except Exception:
                        pass

    async def broadcast(self, message: dict, sender: Optional[WebSocket] = None):
        for connection in list(self.active_connections):
            if sender is None or connection != sender:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()

def validate_ws_update(data: Any) -> tuple[bool, str]:
    """Kiểm tra cấu trúc dữ liệu gửi qua WebSocket."""
    if not isinstance(data, dict):
        return False, "Payload phải là dictionary/JSON object"
    msg_type = data.get("type")
    if msg_type not in (
        "update",
        "delete_query",
        "clear_all",
        "request_clear_cache",
        "clear_cache_response",
        "clear_cache_cancel",
    ):
        return False, f"Loại message không hợp lệ: {msg_type}"

    if msg_type in ("clear_all", "request_clear_cache", "clear_cache_cancel"):
        return True, ""

    if msg_type == "clear_cache_response":
        if not data.get("request_id"):
            return False, "Thiếu request_id trong clear_cache_response"
        return True, ""

    if msg_type == "delete_query":
        query_id = data.get("query_id")
        if not query_id or not isinstance(query_id, str):
            return False, "Thiếu trường 'query_id' hợp lệ trong delete_query"
        return True, ""

    if "manifest" in data:
        if not isinstance(data["manifest"], list):
            return False, "Trường 'manifest' phải là mảng/list"
        for idx, item in enumerate(data["manifest"]):
            if not isinstance(item, dict) or not item.get("query_id"):
                return False, f"Phần tử manifest[{idx}] thiếu trường 'query_id'"

    if "selections" in data:
        if not isinstance(data["selections"], list):
            return False, "Trường 'selections' phải là mảng/list"
        for idx, item in enumerate(data["selections"]):
            if not isinstance(item, dict):
                return False, f"Phần tử selections[{idx}] phải là object"
            if not item.get("video_id") or "frames" not in item:
                return False, f"Phần tử selections[{idx}] thiếu 'video_id' hoặc 'frames'"
            if not isinstance(item["frames"], list):
                return False, f"Trường frames trong selections[{idx}] phải là mảng"

    if "queryCache" in data and not isinstance(data["queryCache"], dict):
        return False, "Trường 'queryCache' phải là dictionary/object"

    return True, ""


def clear_shared_state() -> None:
    """Xóa sạch toàn bộ shared_manifest, shared_selections và shared_query_cache."""
    global shared_manifest, shared_selections, shared_query_cache
    if shared_manifest is not None:
        shared_manifest.clear()
    if shared_selections is not None:
        shared_selections.clear()
    if isinstance(shared_query_cache, dict):
        shared_query_cache.clear()


def delete_shared_query(query_id: str) -> None:
    """Xóa hoàn toàn một query khỏi shared_manifest, shared_selections và shared_query_cache."""
    global shared_manifest, shared_selections, shared_query_cache
    if shared_manifest is not None:
        shared_manifest[:] = [
            m for m in shared_manifest
            if isinstance(m, dict) and m.get("query_id") != query_id
        ]
    if shared_selections is not None:
        shared_selections[:] = [
            s for s in shared_selections
            if isinstance(s, dict) and (s.get("queryId") or s.get("query_id")) != query_id
        ]
    if isinstance(shared_query_cache, dict):
        shared_query_cache.pop(query_id, None)


def merge_shared_state(
    incoming_manifest: Optional[list],
    incoming_selections: Optional[list],
    incoming_query_cache: Optional[dict],
) -> None:
    """Hợp nhất dữ liệu mới, ngăn một người dùng ghi đè dữ liệu của người khác.

    - Manifest: upsert theo query_id, giữ lại các query khác.
    - Selections: hợp nhất theo queryId. Nếu incoming có selections cho query X,
      chỉ cập nhật selections của query X; giữ nguyên selections của các query khác.
    - Query Cache: dict.update theo từng queryId.
    """
    global shared_manifest, shared_selections, shared_query_cache

    # 1. Merge manifest
    if incoming_manifest is not None and isinstance(incoming_manifest, list):
        if not shared_manifest:
            shared_manifest = list(incoming_manifest)
        elif incoming_manifest:
            manifest_by_id = {
                item["query_id"]: item
                for item in shared_manifest
                if isinstance(item, dict) and "query_id" in item
            }
            for item in incoming_manifest:
                if isinstance(item, dict) and "query_id" in item:
                    manifest_by_id[item["query_id"]] = item
            shared_manifest = list(manifest_by_id.values())

    # 2. Merge selections
    if incoming_selections is not None and isinstance(incoming_selections, list):
        if not shared_selections:
            shared_selections = list(incoming_selections)
        elif incoming_selections:
            # Nhận biết các queryId được cập nhật trong payload này
            updated_query_ids = {
                s.get("queryId") or s.get("query_id")
                for s in incoming_selections
                if isinstance(s, dict) and (s.get("queryId") or s.get("query_id"))
            }

            if updated_query_ids:
                # Giữ lại selections thuộc các query khác
                kept_selections = [
                    s for s in shared_selections
                    if isinstance(s, dict) and (s.get("queryId") or s.get("query_id")) not in updated_query_ids
                ]
                shared_selections = kept_selections + list(incoming_selections)
            else:
                def _sel_key(s: dict) -> tuple:
                    return (
                        s.get("queryId") or s.get("query_id", ""),
                        s.get("video_id", ""),
                        tuple(s.get("frames", [])),
                        s.get("answer", ""),
                    )
                seen_keys = {_sel_key(s) for s in shared_selections if isinstance(s, dict)}
                for s in incoming_selections:
                    if isinstance(s, dict) and _sel_key(s) not in seen_keys:
                        shared_selections.append(s)
                        seen_keys.add(_sel_key(s))

    # 3. Merge query cache
    if incoming_query_cache is not None and isinstance(incoming_query_cache, dict):
        shared_query_cache.update(incoming_query_cache)


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    global shared_manifest, shared_selections, shared_query_cache
    await manager.connect(websocket)
    try:
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "manifest": shared_manifest,
            "selections": shared_selections,
            "queryCache": shared_query_cache
        })
        while True:
            data = await websocket.receive_json()
            is_valid, err_msg = validate_ws_update(data)
            if not is_valid:
                logger.warning("Bỏ qua payload WebSocket không hợp lệ: %s", err_msg)
                try:
                    await websocket.send_json({"type": "error", "detail": err_msg})
                except Exception:
                    pass
                continue

            if data.get("type") == "clear_all":
                clear_shared_state()
                save_shared_state()
                manager.pending_clear = None
                await manager.broadcast({
                    "type": "clear_all",
                    "manifest": shared_manifest,
                    "selections": shared_selections,
                    "queryCache": shared_query_cache,
                })
                continue

            if data.get("type") == "request_clear_cache":
                active_others = [ws for ws in list(manager.active_connections) if ws != websocket]
                if not active_others:
                    # Chỉ có 1 máy đang hoạt động -> thực hiện xóa ngay không cần hỏi ai khác
                    clear_shared_state()
                    save_shared_state()
                    manager.pending_clear = None
                    await manager.broadcast({
                        "type": "clear_all",
                        "manifest": shared_manifest,
                        "selections": shared_selections,
                        "queryCache": shared_query_cache,
                    })
                else:
                    # Có nhiều hơn 1 máy -> tạo phiên yêu cầu xóa và hỏi ý kiến các máy còn lại
                    req_id = uuid.uuid4().hex[:8]
                    manager.pending_clear = {
                        "req_id": req_id,
                        "requester": websocket,
                        "pending_others": set(active_others),
                        "rejections": set(),
                    }
                    try:
                        await websocket.send_json({
                            "type": "clear_cache_waiting",
                            "request_id": req_id,
                            "count": len(active_others),
                        })
                    except Exception:
                        pass
                    for other_ws in active_others:
                        try:
                            await other_ws.send_json({
                                "type": "clear_cache_prompt",
                                "request_id": req_id,
                            })
                        except Exception:
                            pass
                continue

            if data.get("type") == "clear_cache_response":
                req_id = data.get("request_id")
                approve = bool(data.get("approve"))
                if not manager.pending_clear or manager.pending_clear.get("req_id") != req_id:
                    continue

                if approve:
                    # Được chấp thuận từ ít nhất 1 máy khác -> có thể xóa ngay!
                    manager.pending_clear = None
                    clear_shared_state()
                    save_shared_state()
                    await manager.broadcast({
                        "type": "clear_all",
                        "manifest": shared_manifest,
                        "selections": shared_selections,
                        "queryCache": shared_query_cache,
                    })
                else:
                    # Máy này từ chối
                    manager.pending_clear["rejections"].add(websocket)
                    manager.pending_clear["pending_others"].discard(websocket)

                    remaining = [
                        ws for ws in manager.pending_clear["pending_others"]
                        if ws in manager.active_connections
                    ]
                    if not remaining:
                        # Toàn bộ người dùng khác không 1 ai chấp nhận -> không được xóa!
                        requester = manager.pending_clear.get("requester")
                        manager.pending_clear = None
                        if requester in manager.active_connections:
                            try:
                                await requester.send_json({
                                    "type": "clear_cache_rejected",
                                    "reason": "Tất cả các thành viên khác đều đã từ chối yêu cầu xóa cache.",
                                })
                            except Exception:
                                pass
                        await manager.broadcast({"type": "clear_cache_dismiss"})
                continue

            if data.get("type") == "clear_cache_cancel":
                req_id = data.get("request_id")
                if manager.pending_clear and (not req_id or manager.pending_clear.get("req_id") == req_id):
                    manager.pending_clear = None
                    await manager.broadcast({"type": "clear_cache_dismiss"})
                continue

            if data.get("type") == "delete_query":
                query_id = data.get("query_id")
                delete_shared_query(query_id)
                save_shared_state()
                await manager.broadcast({
                    "type": "delete_query",
                    "query_id": query_id,
                    "manifest": shared_manifest,
                    "selections": shared_selections,
                    "queryCache": shared_query_cache,
                })
                continue

            merge_shared_state(
                data.get("manifest"),
                data.get("selections"),
                data.get("queryCache"),
            )
            save_shared_state()
            # Broadcast to all other clients
            await manager.broadcast({
                "type": "update",
                "manifest": shared_manifest,
                "selections": shared_selections,
                "queryCache": shared_query_cache
            }, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket handler error: %s", e)
        manager.disconnect(websocket)


@app.post("/api/clear_state")
async def api_clear_state():
    """Xóa sạch toàn bộ shared_manifest, shared_selections và shared_query_cache trên server và đồng bộ tất cả clients."""
    clear_shared_state()
    save_shared_state()
    await manager.broadcast({
        "type": "clear_all",
        "manifest": shared_manifest,
        "selections": shared_selections,
        "queryCache": shared_query_cache,
    })
    return {"ok": True, "message": "Toàn bộ shared state đã được xóa sạch."}

_SUBMISSION_REPORT_PATHS = {"/api/export", "/api/query-pack/texts"}


@app.exception_handler(RequestValidationError)
async def submission_request_validation_handler(
    request: Request, error: RequestValidationError
):
    """Use the submission report contract only on submission-facing APIs."""
    if request.url.path not in _SUBMISSION_REPORT_PATHS:
        return await request_validation_exception_handler(request, error)

    report = ValidationReport()
    seen_fields = set()
    for item in error.errors():
        field = ".".join(
            str(part) for part in item.get("loc", ()) if part != "body"
        ) or "body"
        if field in seen_fields:
            continue
        seen_fields.add(field)
        report.errors.append(
            ValidationIssue(
                "invalid_request_schema",
                f"Request field {field!r} is invalid or missing",
            )
        )
    if not report.errors:
        report.errors.append(
            ValidationIssue(
                "invalid_request_schema",
                "Request body does not match the required schema",
            )
        )
    return JSONResponse(status_code=422, content={"detail": report.to_dict()})

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Retriever registry
#
# Quy tắc: dummy CHỈ được dùng khi AIC_USE_DUMMY=1. Ở chế độ production, nguồn
# nào hỏng thì nguồn đó mang trạng thái "error" và các nguồn còn lại vẫn chạy;
# không bao giờ âm thầm thay bằng dữ liệu giả, vì kết quả giả trông y hệt kết
# quả thật trên UI và sẽ đi thẳng vào bài nộp.
# ---------------------------------------------------------------------------

import threading

_registry: Optional[ComponentRegistry] = None
_registry_lock = threading.Lock()


def _slot(name: str, state: str, detail: str, error: Optional[str] = None) -> dict:
    return {"name": name, "state": state, "detail": detail, "error": error}


def _load_source(name: str, paths, factory, disabled_reason: Optional[str] = None):
    """Nạp một nguồn, trả về ``(retriever hoặc None, slot trạng thái)``."""
    configured = [p for p in paths if p is not None]
    if len(configured) != len(paths):
        return None, _slot(name, "disabled", f"{name}: chưa cấu hình đường dẫn")

    missing = [str(p) for p in configured if not Path(p).exists()]
    if missing:
        reason = "không tìm thấy: " + ", ".join(missing)
        logger.warning("[%s] %s", name, reason)
        return None, _slot(name, "error", f"{name}: {reason}", reason)

    if disabled_reason:
        return None, _slot(name, "disabled", f"{name}: {disabled_reason}")

    try:
        logger.info("Loading %s retriever...", name)
        retriever = factory()
    except Exception as exc:  # nguồn hỏng không được kéo sập cả hệ thống
        reason = f"{type(exc).__name__}: {exc}"
        logger.error("[%s] load thất bại — %s", name, reason)
        detail = f"{name}: lỗi khi nạp {', '.join(str(p) for p in configured)}"
        return None, _slot(name, "error", detail, reason)

    describe = getattr(retriever, "describe", None)
    detail = describe() if callable(describe) else name
    logger.info("[%s] ready — %s", name, detail)
    return retriever, _slot(name, "ready", detail)


def _source_specs(
    *,
    clip_index,
    clip_meta,
    siglip_index,
    siglip_meta,
    text_index,
    disable_neural: bool = False,
) -> list[tuple]:
    """Khai báo các nguồn retrieval: ``(name, paths, factory, disabled_reason)``.

    Một chỗ khai báo duy nhất, hai cách dùng: ``build_retriever_registry()`` nạp
    hết ngay (dùng cho script eval và test contract), còn app dựng mỗi spec thành
    một ``LazyComponent`` nạp riêng.
    """
    neural_off = "AIC_DISABLE_NEURAL=1" if disable_neural else None

    def _clip_factory():
        from ..retrieval.clip import build_clip_retriever

        return build_clip_retriever(clip_index, clip_meta)

    def _siglip_factory():
        from ..retrieval.siglip import build_siglip_retriever

        return build_siglip_retriever(siglip_index, siglip_meta)

    def _text_factory():
        from ..retrieval.text_retriever import TextRetriever

        return TextRetriever(text_index, name="bm25")

    return [
        ("clip", (clip_index, clip_meta), _clip_factory, neural_off),
        ("siglip", (siglip_index, siglip_meta), _siglip_factory, neural_off),
        ("bm25", (text_index,), _text_factory, None),
    ]


def build_retriever_registry(
    *,
    use_dummy: bool,
    clip_index,
    clip_meta,
    siglip_index,
    siglip_meta,
    text_index,
    dummy_module,
    disable_neural: bool = False,
):
    """Dựng danh sách retriever + trạng thái từng nguồn. Hàm thuần, dễ test.

    Nạp hết ngay lập tức. Đường chạy của server **không** dùng hàm này nữa (xem
    ``_build_component_registry``); nó còn ở đây cho ``scripts/eval_retrieval.py``
    và các test contract, nơi nạp đồng bộ mới là hành vi mong muốn.
    """
    if use_dummy:
        logger.info("AIC_USE_DUMMY=1 — dùng dummy retrieval")
        # detail giữ đúng chuỗi "dummy": UI dùng `data.retriever === 'dummy'`
        # để bật badge demo, đổi chuỗi này là làm hỏng chỉ báo đó.
        return [dummy_module], [_slot("dummy", "ready", "dummy")]

    retrievers: list = []
    statuses: list[dict] = []
    specs = _source_specs(
        clip_index=clip_index,
        clip_meta=clip_meta,
        siglip_index=siglip_index,
        siglip_meta=siglip_meta,
        text_index=text_index,
        disable_neural=disable_neural,
    )
    for name, paths, factory, disabled_reason in specs:
        retriever, slot = _load_source(name, paths, factory, disabled_reason)
        if retriever is not None:
            retrievers.append(retriever)
        statuses.append(slot)

    if not retrievers:
        logger.error(
            "Không nạp được nguồn retrieval thật nào; /api/search sẽ trả lỗi."
        )
    return retrievers, statuses


# ---------------------------------------------------------------------------
# Registry lười — mỗi nguồn một ô trạng thái và một lock riêng
# ---------------------------------------------------------------------------


# Thành phần nào dùng artifact nào — để reload biết phải quên cache đường dẫn nào.
_COMPONENT_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "clip": ("clip_index", "clip_meta"),
    "siglip": ("siglip_index", "siglip_meta"),
    "bm25": ("text_index",),
    "keyframe_map": ("clip_meta", "siglip_meta"),
}


class SourceUnavailable(RuntimeError):
    """Nguồn có cấu hình nhưng không nạp được — trạng thái ``error``."""


def _component_loader(name, enabled_fn, paths_fn, factory_fn, disabled_reason):
    """Loader của một LazyComponent: phân giải path rồi mới nạp.

    Path được phân giải **bên trong** loader, nên thời gian tải artifact từ HF
    tính vào đúng nguồn đó, và lỗi tải cũng chỉ làm hỏng đúng nguồn đó.

    Trả ``None`` = disabled (không cấu hình, đừng nạp). Ném ``SourceUnavailable``
    = error (có cấu hình nhưng hỏng). Hai trường hợp này phải phân biệt được:
    SigLIP tắt là bình thường, còn CLIP tải hụt là sự cố cần báo.
    """

    def _load():
        if not enabled_fn():
            return None, None
        if disabled_reason:
            return None, None

        paths = paths_fn()
        if any(path is None for path in paths):
            raise SourceUnavailable(
                f"không phân giải được artifact cho {name} "
                f"(kiểm tra env hoặc kết nối Hugging Face)"
            )

        retriever, slot = _load_source(name, paths, factory_fn(paths), None)
        if retriever is None:
            raise SourceUnavailable(slot["error"] or slot["detail"])
        return retriever, slot["detail"]

    return _load


def _build_component_registry() -> ComponentRegistry:
    """Registry thật của server: translation + từng nguồn retrieval, tất cả đều lười."""
    registry = ComponentRegistry()
    registry.add(local_translation.TRANSLATOR)
    # kind="media": phục vụ hiển thị ảnh, không phải nguồn retrieval — nên không
    # được lọt vào `retrievers` hay `ready_count` của /api/status.
    registry.add(
        LazyComponent("keyframe_map", _build_frame_index, kind="media")
    )

    if USE_DUMMY:
        logger.info("AIC_USE_DUMMY=1 — dùng dummy retrieval")
        # Dummy nạp tức thì (chi phí bằng 0) để detail giữ đúng chuỗi "dummy" mà
        # UI dựa vào để bật badge demo.
        component = LazyComponent("dummy", lambda: (dummy, "dummy"))
        component.get()
        registry.add(component)
        return registry

    neural_off = "AIC_DISABLE_NEURAL=1" if DISABLE_NEURAL else None

    def _clip_factory(paths):
        def _make():
            from ..retrieval.clip import build_clip_retriever

            return build_clip_retriever(*paths)

        return _make

    def _siglip_factory(paths):
        def _make():
            from ..retrieval.siglip import build_siglip_retriever

            return build_siglip_retriever(*paths)

        return _make

    def _text_factory(paths):
        def _make():
            from ..retrieval.text_retriever import TextRetriever

            return TextRetriever(paths[0], name="bm25")

        return _make

    sources = [
        (
            "clip",
            lambda: True,
            lambda: (clip_index_path(), clip_meta_path()),
            _clip_factory,
            neural_off,
        ),
        (
            "siglip",
            _siglip_enabled,
            lambda: (siglip_index_path(), siglip_meta_path()),
            _siglip_factory,
            neural_off,
        ),
        ("bm25", lambda: True, lambda: (text_index_path(),), _text_factory, None),
    ]
    for name, enabled_fn, paths_fn, factory, disabled_reason in sources:
        registry.add(
            LazyComponent(
                name,
                _component_loader(name, enabled_fn, paths_fn, factory, disabled_reason),
                disabled_reason=disabled_reason
                or (None if enabled_fn() else "chưa cấu hình"),
            )
        )
    return registry


def get_registry() -> ComponentRegistry:
    """Registry dùng chung, dựng một lần, không nạp gì cả."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = _build_component_registry()
    return _registry


def get_retrievers():
    """Nạp theo yêu cầu và trả về những nguồn nạp được. Nguồn lỗi bị bỏ qua."""
    return get_registry().ready_values(kind="retrieval")


def get_retriever_status() -> list[dict]:
    return get_registry().snapshot_all(kind="retrieval")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query_id: str = "q1"
    text_vi: str
    text_en: str = ""
    task: str = "kis"   # kis | qa | trake
    n_events: int = 1
    k: int = 100
    exclude: list[str] = []
    # Tuỳ chọn — UI cũ không gửi hai trường này và vẫn phải chạy y như trước.
    modalities: Optional[list[str]] = None
    weights: dict[str, float] = {}


class TranslateRequest(BaseModel):
    text_vi: str


class ExportRow(BaseModel):
    query_id: str
    video_id: str
    frames: list[object]
    answer: str = ""


class QueryTextFileIn(BaseModel):
    filename: str
    content: str


class QueryPackTextsRequest(BaseModel):
    files: list[QueryTextFileIn]


class QueryManifestIn(BaseModel):
    query_id: str
    task: str
    text: str
    source_name: str
    n_events: object | None
    events_confirmed: bool


class ExportRequest(BaseModel):
    manifest: list[QueryManifestIn]
    rows: list[ExportRow]


class CandidateOut(BaseModel):
    video_id: str
    start_frame: int
    end_frame: int
    representative_frames: list[int]
    scores: dict[str, float]
    evidence: dict
    best_score: float
    rank: int


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _candidate_to_out(cand: Candidate, rank: int) -> dict:
    return {
        "video_id": cand.video_id,
        "start_frame": cand.start_frame,
        "end_frame": cand.end_frame,
        "representative_frames": cand.representative_frames,
        "scores": cand.scores,
        "evidence": cand.evidence,
        "best_score": cand.best_score,
        "rank": rank,
    }


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


def _query_pack_response(result):
    report = result.to_dict()
    if not result.ok:
        raise HTTPException(status_code=422, detail=report)
    return report


def _validate_manifest(manifest: list[QueryDefinition]) -> ValidationReport:
    report = ValidationReport()
    seen_query_ids = set()

    for query in manifest:
        inferred_task = infer_task(query.query_id)
        if (
            not query.query_id
            or Path(query.query_id).name != query.query_id
            or "/" in query.query_id
            or "\\" in query.query_id
            or "\x00" in query.query_id
        ):
            report.errors.append(
                ValidationIssue(
                    "unsafe_query_id",
                    "Query ID must be a non-empty filename without a path",
                    query_id=query.query_id,
                )
            )

        if inferred_task is None:
            report.errors.append(
                ValidationIssue(
                    "invalid_task_suffix",
                    f"Query ID {query.query_id!r} must end with -kis, -qa, or -trake",
                    query_id=query.query_id,
                )
            )

        if query.task not in {"kis", "qa", "trake"}:
            report.errors.append(
                ValidationIssue(
                    "invalid_manifest_task",
                    f"Manifest task must be kis, qa, or trake; got {query.task!r}",
                    query_id=query.query_id,
                )
            )
        elif inferred_task is not None and inferred_task != query.task:
            report.errors.append(
                ValidationIssue(
                    "manifest_task_mismatch",
                    f"Query ID suffix implies {inferred_task!r}, not {query.task!r}",
                    query_id=query.query_id,
                )
            )

        if query.query_id in seen_query_ids:
            report.errors.append(
                ValidationIssue(
                    "duplicate_query_id",
                    f"Duplicate query ID: {query.query_id}",
                    query_id=query.query_id,
                )
            )
        else:
            seen_query_ids.add(query.query_id)

        if query.task == "trake" and query.events_confirmed:
            if (
                not isinstance(query.n_events, int)
                or isinstance(query.n_events, bool)
                or query.n_events <= 0
            ):
                report.errors.append(
                    ValidationIssue(
                        "invalid_trake_event_count",
                        "Confirmed TRAKE query requires a positive event count",
                        query_id=query.query_id,
                    )
                )

    return report


@app.post("/api/query-pack/zip")
async def import_query_pack_zip(request: Request):
    """Parse an uploaded query-pack ZIP body into an ordered manifest."""
    return _query_pack_response(parse_query_zip(await request.body()))


@app.post("/api/query-pack/texts")
def import_query_pack_texts(req: QueryPackTextsRequest):
    """Parse JSON-supplied query TXT files into an ordered manifest."""
    files = [(item.filename, item.content) for item in req.files]
    return _query_pack_response(parse_query_files(files))


@app.get("/api/status")
def status():
    """Trạng thái từng thành phần + nguồn cấu hình đang dùng.

    Không bao giờ chặn: mọi số liệu ở đây đọc từ ảnh chụp trạng thái, không đụng
    vào lock nạp, nên hỏi được ngay cả lúc CLIP hay BM25 đang nạp dở.
    """
    registry = get_registry()
    components = registry.snapshot_all()
    statuses = registry.snapshot_all(kind="retrieval")
    ready = [s for s in statuses if s["state"] == "ready"]

    if ready:
        summary = " + ".join(s["detail"] for s in ready)
    elif any(s["state"] == "loading" for s in components):
        summary = "đang nạp…"
    else:
        summary = "không có nguồn retrieval nào sẵn sàng"

    return {
        "ok": True,
        "retriever": summary,
        "retrievers": statuses,
        "components": components,
        "translation": local_translation.translation_status(),
        "loading": registry.is_loading(),
        "ready_count": len(ready),
        "keyframes_dir": str(KEYFRAMES_DIR),
        "use_dummy": USE_DUMMY,
        "paths": {
            "clip_index": artifact_source("clip_index"),
            "clip_meta": artifact_source("clip_meta"),
            "siglip_index": artifact_source("siglip_index") if _siglip_enabled() else None,
            "siglip_meta": artifact_source("siglip_meta") if _siglip_enabled() else None,
            "text_index": artifact_source("text_index"),
        },
    }


@app.post("/api/components/{name}/reload")
def reload_component(name: str):
    """Thử nạp lại một thành phần đang lỗi, khỏi phải restart server.

    Trạng thái ``error`` cố tình dính lại: một nguồn hỏng hẳn mà tự thử lại mỗi
    request thì mọi lần search đều gánh thêm một lần timeout. Đây là lối thoát.
    """
    if name == "translation":
        return {"ok": True, "component": local_translation.reload_translator()}

    component = get_registry().get(name)
    if component is None:
        raise HTTPException(
            status_code=404,
            detail=f"Không có thành phần tên {name!r}",
        )
    # Quên cả đường dẫn đã nhớ: nếu lần trước hụt vì mạng thì phải tải lại,
    # không phải chỉ dựng lại retriever trên cùng một path hỏng.
    for artifact_key in _COMPONENT_ARTIFACTS.get(name, ()):
        reset_artifact_cache(artifact_key)
    return {"ok": True, "component": component.reload()}


@app.post("/api/translate")
def translate(req: TranslateRequest):
    """Dịch câu hỏi tiếng Việt → tiếng Anh bằng Gemini (nếu có) hoặc model local."""
    try:
        from ..core.gemini import is_gemini_available, translate_with_gemini

        if is_gemini_available():
            text_en = translate_with_gemini(req.text_vi)
            if text_en:
                return {"text_en": text_en, "ok": True}
    except Exception as e:
        logger.warning("Gemini translation error: %s. Using local fallback.", e)

    try:
        text_en = translate_text(req.text_vi)
        return {"text_en": text_en, "ok": True}
    except Exception as e:
        logger.warning("Translation error: %s", e)
        return {"text_en": req.text_vi, "ok": False, "error": str(e)}


@app.post("/api/search")
def search(req: SearchRequest):
    """Tìm kiếm candidates bằng các retriever đang sẵn sàng."""
    retrievers = get_retrievers()
    statuses = get_retriever_status()
    if not retrievers:
        # Thà báo lỗi còn hơn trả kết quả giả: ở production không có fallback.
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Không có nguồn retrieval nào sẵn sàng",
                "retrievers": statuses,
            },
        )

    from ..core.query_processor import process_query

    query = make_query(
        query_id=req.query_id,
        text_vi=req.text_vi,
        text_en=req.text_en,
        task=req.task,
        n_events=req.n_events,
    )
    query.modalities = req.modalities
    query.weights = dict(req.weights or {})

    # Dịch local và trích xuất object theo rule nếu UI chưa cung cấp text_en.
    query = process_query(query)

    exclude = frozenset(req.exclude)

    try:
        candidates = retrieve_and_fuse(
            query=query,
            retrievers=retrievers,
            fuse_fn=fuse,
            limit=req.k,
            exclude=exclude,
            fuse_kwargs={"weights": query.weights} if query.weights else None,
        )
    except Exception as e:
        logger.error("Search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "ok": True,
        "total": len(candidates),
        "candidates": [_candidate_to_out(c, i + 1) for i, c in enumerate(candidates)],
    }


class FrameIndex:
    """Tra ``(video_id, frame_idx) → keyframe_num`` cho việc hiển thị ảnh.

    Giữ luôn danh sách frame theo từng video, dựng **một lần** cùng lượt đọc
    metadata. Trước đây danh sách này được dựng lười trong request bằng cách quét
    toàn bộ 176k khoá cho mỗi video — mỗi video một lần quét.
    """

    __slots__ = ("mapping", "by_video")

    def __init__(self, mapping: dict, by_video: dict):
        self.mapping = mapping
        self.by_video = by_video

    def __len__(self) -> int:
        return len(self.mapping)

    def describe(self) -> str:
        return f"keyframe_map: {len(self.mapping):,} keyframe / {len(self.by_video)} video"

    def keyframe_num(self, video_id: str, frame_idx: int):
        """Khớp chính xác, không có thì lấy keyframe gần nhất của cùng video.

        Frame_idx có thể do operator sửa tay nên không rơi đúng keyframe nào;
        hiển thị ảnh gần nhất vẫn hữu ích hơn một ô xám.
        """
        exact = self.mapping.get((video_id, frame_idx))
        if exact is not None:
            return exact
        frames = self.by_video.get(video_id)
        if not frames:
            return None
        closest = min(frames, key=lambda x: abs(x - frame_idx))
        return self.mapping.get((video_id, closest))


EMPTY_FRAME_INDEX = FrameIndex({}, {})


def _read_keyframe_metadata(path: Path) -> tuple[dict, dict]:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    mapping: dict = {}
    by_video: dict[str, list[int]] = {}
    for meta in metadata:
        vid = meta.get("video_id")
        fidx = meta.get("frame_idx")
        n = meta.get("keyframe_num")
        if vid is None or fidx is None or n is None:
            continue
        mapping[(vid, fidx)] = n
        by_video.setdefault(vid, []).append(fidx)
    for frames in by_video.values():
        frames.sort()
    return mapping, by_video


def _build_frame_index() -> tuple[FrameIndex, str]:
    """Loader của component ``keyframe_map``.

    Ném ``SourceUnavailable`` khi không đọc được nguồn nào. Trước đây lỗi bị nuốt
    và hàm trả về dict rỗng, nên operator chỉ thấy ảnh placeholder xám mà không
    có chỗ nào nói vì sao.
    """
    reasons = []
    for label, path in (("clip", clip_meta_path()), ("siglip", siglip_meta_path())):
        if path is None:
            reasons.append(f"{label}: chưa phân giải được metadata")
            continue
        if not path.exists():
            reasons.append(f"{label}: không thấy {path}")
            continue
        try:
            mapping, by_video = _read_keyframe_metadata(path)
        except Exception as exc:
            reasons.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        if mapping:
            index = FrameIndex(mapping, by_video)
            logger.info("Frame index từ %s — %s", path, index.describe())
            return index, index.describe()
        reasons.append(f"{label}: {path} không có keyframe_num nào")

    raise SourceUnavailable("; ".join(reasons) or "không có nguồn metadata keyframe")


def get_frame_index() -> FrameIndex:
    """Frame index dùng chung, nạp một lần duy nhất kể cả khi nhiều request đến cùng lúc."""
    component = get_registry().get("keyframe_map")
    if component is None:
        return EMPTY_FRAME_INDEX
    return component.get() or EMPTY_FRAME_INDEX


def _get_frame_mapping() -> dict:
    """Giữ lại cho các chỗ chỉ cần dict thô (ví dụ ``/api/videos``)."""
    return get_frame_index().mapping


@app.get("/api/keyframe/{video_id}/{frame_idx}")
def get_keyframe(video_id: str, frame_idx: int):
    """Trả về ảnh keyframe (từ local hoặc redirect tới cloud R2/HF)."""
    n = get_frame_index().keyframe_num(video_id, frame_idx)

    if AIC_USE_CLOUD_MEDIA:
        if n is not None:
            return RedirectResponse(f"{HF_DATASET_URL}/keyframes/{video_id}/{n:03d}.jpg")
        else:
            return RedirectResponse(f"{HF_DATASET_URL}/keyframes/{video_id}/{frame_idx:06d}.jpg")

    candidates = []
    
    if n is not None:
        candidates.extend([
            KEYFRAMES_DIR / video_id / f"{n:03d}.jpg",
            KEYFRAMES_DIR / video_id / f"{n:04d}.jpg",
            KEYFRAMES_DIR / video_id / f"{n}.jpg",
            KEYFRAMES_DIR / video_id / f"{n:03d}.png",
            KEYFRAMES_DIR / video_id / f"{n:04d}.png",
        ])

    candidates.extend([
        KEYFRAMES_DIR / video_id / f"{frame_idx:06d}.jpg",
        KEYFRAMES_DIR / video_id / f"{frame_idx}.jpg",
        KEYFRAMES_DIR / video_id / f"{frame_idx:06d}.png",
        KEYFRAMES_DIR / video_id / f"{frame_idx}.png",
    ])
    for p in candidates:
        if p.exists():
            suffix = p.suffix.lower()
            media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
            return FileResponse(str(p), media_type=media_type)

    # Trả về placeholder SVG màu gradient nếu không tìm thấy ảnh
    hue = (hash(f"{video_id}{frame_idx}") % 360)
    hue2 = (hue + 60) % 360
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="hsl({hue},60%,25%)"/>
      <stop offset="100%" stop-color="hsl({hue2},60%,15%)"/>
    </linearGradient>
  </defs>
  <rect width="320" height="180" fill="url(#g)"/>
  <text x="160" y="82" font-family="monospace" font-size="13" fill="rgba(255,255,255,0.6)" text-anchor="middle">{video_id}</text>
  <text x="160" y="102" font-family="monospace" font-size="11" fill="rgba(255,255,255,0.4)" text-anchor="middle">frame #{frame_idx}</text>
  <circle cx="160" cy="55" r="16" fill="rgba(255,255,255,0.12)"/>
  <polygon points="155,47 155,63 171,55" fill="rgba(255,255,255,0.5)"/>
</svg>"""
    return Response(content=svg.encode(), media_type="image/svg+xml")


_video_paths = {}

def _get_video_path(video_id: str):
    if video_id in _video_paths and _video_paths[video_id].exists():
        return _video_paths[video_id]
        
    # Tìm kiếm nhanh theo cấu trúc thư mục thực tế để tránh treo server
    # Cấu trúc phổ biến: data/batch_01/Videos_L27_a/video/L27_V005.mp4
    fast_patterns = [
        f"batch_*/Videos_*/video/{video_id}.mp4",
        f"video/{video_id}.mp4",
        f"{video_id}.mp4"
    ]
    
    for pattern in fast_patterns:
        for p in Path("data").glob(pattern):
            if p.exists():
                _video_paths[video_id] = p
                return p
                
    # Fallback chậm nếu không tìm thấy
    for p in Path("data").rglob(f"{video_id}.mp4"):
        _video_paths[video_id] = p
        return p
        
    return None

_video_fps: Optional[dict] = None
_video_fps_source = "unknown"


def _fps_from_map_keyframes() -> dict:
    """fps chính xác đọc thẳng từ ``map-keyframes/<video_id>.csv`` của BTC."""
    if MAP_KEYFRAMES_DIR is None or not MAP_KEYFRAMES_DIR.exists():
        return {}
    import csv as _csv

    out: dict[str, float] = {}
    for path in MAP_KEYFRAMES_DIR.rglob("*.csv"):
        try:
            with path.open(encoding="utf-8") as handle:
                row = next(_csv.DictReader(handle), None)
            if row and row.get("fps"):
                out[path.stem] = float(row["fps"])
        except Exception as exc:
            logger.warning("Không đọc được fps từ %s: %s", path, exc)
    if out:
        logger.info("Đọc fps chính xác cho %d video từ %s", len(out), MAP_KEYFRAMES_DIR)
    return out


def _fps_from_metadata() -> dict:
    """Suy fps từ metadata keyframe khi không có map-keyframes.

    ``frame_idx = floor(pts_time * fps)`` nên mỗi keyframe cho một khoảng
    ``[frame_idx/pts, (frame_idx+1)/pts)``. Giao tất cả các khoảng lại sẽ kẹp
    fps chặt hơn nhiều so với chia đúng một điểm.

    Đây là đường dự phòng: đo trên batch 1 thì sai số trung bình 5e-5 fps nhưng
    trường hợp xấu nhất vẫn lệch ~0.012 fps, đủ để lệch hơn 20 frame ở phút thứ
    30 — quá rộng cho TRAKE. Vì vậy nên cấu hình ``AIC_MAP_KEYFRAMES_DIR``.
    """
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for retriever in get_retrievers():
        metadata = getattr(retriever, "metadata", None)
        if not metadata:
            continue
        for meta in metadata:
            pts = meta.get("pts_time")
            frame = meta.get("frame_idx")
            video_id = meta.get("video_id")
            if not video_id or pts is None or frame is None:
                continue
            pts = float(pts)
            if pts <= 0:
                continue
            frame = int(frame)
            lower[video_id] = max(lower.get(video_id, 0.0), frame / pts)
            upper[video_id] = min(upper.get(video_id, float("inf")), (frame + 1) / pts)

    out: dict[str, float] = {}
    for video_id, low in lower.items():
        high = upper.get(video_id, float("inf"))
        out[video_id] = round((low + high) / 2, 3) if high > low else round(low, 3)
    if out:
        logger.info("Suy ra fps cho %d video từ metadata keyframe", len(out))
    return out


def _get_video_fps_map() -> dict:
    """fps từng video, ưu tiên nguồn chính xác nhất.

    Vì sao phải đúng: UI tính frame nộp bài bằng ``currentTime * fps``. Dataset
    có cả 25 / 26.44 / 29.97 / 30 fps, nên mặc định 25 cho một video 30 fps sẽ
    lệch hàng trăm giây và nộp sai frame — mà sai frame là R-Score 0.
    """
    global _video_fps, _video_fps_source
    if _video_fps is not None:
        return _video_fps

    _video_fps = _fps_from_map_keyframes()
    _video_fps_source = "map_keyframes"
    if not _video_fps:
        _video_fps = _fps_from_metadata()
        _video_fps_source = "keyframe_metadata"
    return _video_fps


@app.get("/api/video_info/{video_id}")
def get_video_info(video_id: str):
    """Lấy thông tin video (fps), ưu tiên FPS chính xác từ map-keyframes của BTC."""
    fps_map = _get_video_fps_map()
    if video_id in fps_map:
        return {"fps": fps_map[video_id]}

    meta = load_video_metadata()
    video_info = meta.get(video_id)
    if video_info and "fps" in video_info:
        return {"fps": video_info["fps"]}
    return {"fps": 25.0, "source": "fallback", "reliable": False}


@app.get("/api/video/{video_id}")
def get_video(video_id: str, request: Request):
    """Trả về RedirectResponse tới Hugging Face dataset để phát video."""
    meta = load_video_metadata()
    video_info = meta.get(video_id)
    if video_info:
        path_in_repo = video_info["path"]
        return RedirectResponse(f"{HF_DATASET_URL}/{path_in_repo}")
    
    # Fallback dự phòng nếu không tìm thấy video_id trong metadata
    prefix = video_id.split('_')[0]
    return RedirectResponse(f"{HF_DATASET_URL}/videos/Videos_{prefix}_a/video/{video_id}.mp4")


@app.get("/api/videos")
def get_videos():
    """Trả về danh sách tất cả các video_id được phát hiện trong metadata hoặc mapping."""
    meta = load_video_metadata()
    if meta:
        return {"ok": True, "videos": sorted(list(meta.keys()))}
    
    mapping = _get_frame_mapping()
    video_ids = sorted(list({vid for (vid, fidx) in mapping.keys()}))
    return {"ok": True, "videos": video_ids}


@app.post("/api/export")
def export_submission(req: ExportRequest):
    """Validate operator rows and return a reparsed PASS submission ZIP."""
    manifest = [QueryDefinition(**query.model_dump()) for query in req.manifest]
    rows = [row.model_dump() for row in req.rows]

    manifest_report = _validate_manifest(manifest)
    if not manifest_report.ok:
        raise HTTPException(status_code=422, detail=manifest_report.to_dict())

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        try:
            write_validated_submission(manifest, rows, tmp_path)
        except SubmissionValidationError as error:
            raise HTTPException(status_code=422, detail=error.report.to_dict())
        except GeneratedArchiveError as error:
            raise HTTPException(status_code=500, detail=error.report.to_dict())
        zip_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    filename = "submission.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Validation-Status": "PASS",
        },
    )


# ---------------------------------------------------------------------------
# Static files + SPA fallback
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
@app.get("/{full_path:path}")
def serve_spa(full_path: str = ""):
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "AIC 2026 API running. UI static files not found."})


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("aic.ui.app:app", host="0.0.0.0", port=8000, reload=True)
