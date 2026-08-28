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

import logging
import os
import tempfile
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

def resolve_path(remote_filename: Optional[str]) -> Optional[Path]:
    """Tải/đọc trực tiếp file từ Hugging Face Hub cache."""
    if not remote_filename:
        return None

    repo_id = os.environ.get("AIC_HF_REPO_ID", "manhha2502/fullhd")
    revision = os.environ.get("AIC_HF_REVISION", "main")
    cache_dir = os.environ.get("AIC_HF_CACHE_DIR") or None
    try:
        from huggingface_hub import hf_hub_download
        import sys
        is_mocked = hasattr(hf_hub_download, "mock_add_spec") or hf_hub_download.__class__.__name__ in ("MagicMock", "Mock")
        if ("pytest" in sys.modules or "unittest" in sys.modules) and not is_mocked:
            logger.info("Test environment detected. Skipping real HF download for %s", remote_filename)
            return Path("local") / remote_filename.split("/")[-1]

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

INDEX_PATH = resolve_path("local/clip_faiss.index")
META_PATH = resolve_path("local/clip_metadata.json")
TEXT_INDEX_PATH = resolve_path("data/input/input/index/text_search_index.pkl")


def _optional_path(name: str) -> Optional[Path]:
    """Path từ env, hoặc None nếu chưa cấu hình (khác với 'cấu hình sai')."""
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


MAP_KEYFRAMES_DIR = _optional_path("AIC_MAP_KEYFRAMES_DIR")

if os.environ.get("AIC_USE_SIGLIP", "0") == "1":
    SIGLIP_INDEX_PATH = resolve_path("data/input/input/index/siglip_faiss.index")
    SIGLIP_META_PATH = resolve_path("data/input/input/index/siglip_metadata.json")
else:
    SIGLIP_INDEX_PATH = None
    SIGLIP_META_PATH = None

USE_DUMMY = os.environ.get("AIC_USE_DUMMY", "0") == "1"
DISABLE_NEURAL = os.environ.get("AIC_DISABLE_NEURAL", "0").strip() == "1"

AIC_USE_CLOUD_MEDIA = os.environ.get("AIC_USE_CLOUD_MEDIA", "1") == "1"
HF_DATASET_URL = os.environ.get("AIC_HF_DATASET_URL", "https://huggingface.co/datasets/manhha2502/fullhd/resolve/main")
VIDEO_METADATA_PATH = resolve_path("local/video_metadata.json")
_video_metadata = {}

def load_video_metadata():
    global _video_metadata
    if not _video_metadata and VIDEO_METADATA_PATH and VIDEO_METADATA_PATH.exists():
        try:
            import json
            with open(VIDEO_METADATA_PATH, "r", encoding="utf-8") as f:
                _video_metadata = json.load(f)
            logger.info("Loaded video metadata with %d mappings", len(_video_metadata))
        except Exception as e:
            logger.warning("Error loading video metadata: %s", e)
    return _video_metadata

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    import sys
    logger.info("Preloading retrievers...")
    _ensure_retrievers()

    if "pytest" not in sys.modules and "unittest" not in sys.modules:
        logger.info("Preloading translation model...")
        try:
            from ..core.local_translation import get_local_translator
            get_local_translator()
        except Exception as e:
            logger.warning("Failed to preload translation model: %s", e)
    yield


app = FastAPI(title="AIC 2026 Video Retrieval", version="0.1.0", lifespan=lifespan)

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

_retrievers: list = []
_retriever_status: list[dict] = []
_retriever_lock = threading.Lock()


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
    """Dựng danh sách retriever + trạng thái từng nguồn. Hàm thuần, dễ test."""
    if use_dummy:
        logger.info("AIC_USE_DUMMY=1 — dùng dummy retrieval")
        # detail giữ đúng chuỗi "dummy": UI dùng `data.retriever === 'dummy'`
        # để bật badge demo, đổi chuỗi này là làm hỏng chỉ báo đó.
        return [dummy_module], [_slot("dummy", "ready", "dummy")]

    neural_off = "AIC_DISABLE_NEURAL=1" if disable_neural else None
    retrievers: list = []
    statuses: list[dict] = []

    def register(name, paths, factory, disabled_reason=None):
        retriever, slot = _load_source(name, paths, factory, disabled_reason)
        if retriever is not None:
            retrievers.append(retriever)
        statuses.append(slot)

    def _clip_factory():
        from ..retrieval.clip import build_clip_retriever

        return build_clip_retriever(clip_index, clip_meta)

    def _siglip_factory():
        from ..retrieval.siglip import build_siglip_retriever

        return build_siglip_retriever(siglip_index, siglip_meta)

    def _text_factory():
        from ..retrieval.text_retriever import TextRetriever

        return TextRetriever(text_index, name="bm25")

    register("clip", (clip_index, clip_meta), _clip_factory, neural_off)
    register("siglip", (siglip_index, siglip_meta), _siglip_factory, neural_off)
    register("bm25", (text_index,), _text_factory)

    if not retrievers:
        logger.error(
            "Không nạp được nguồn retrieval thật nào; /api/search sẽ trả lỗi."
        )
    return retrievers, statuses


def _ensure_retrievers():
    global _retrievers, _retriever_status
    with _retriever_lock:
        if _retriever_status:
            return _retrievers, _retriever_status
        _retrievers, _retriever_status = build_retriever_registry(
            use_dummy=USE_DUMMY,
            clip_index=INDEX_PATH,
            clip_meta=META_PATH,
            siglip_index=SIGLIP_INDEX_PATH,
            siglip_meta=SIGLIP_META_PATH,
            text_index=TEXT_INDEX_PATH,
            dummy_module=dummy,
            disable_neural=DISABLE_NEURAL,
        )
        return _retrievers, _retriever_status


def get_retrievers():
    return _ensure_retrievers()[0]


def get_retriever_status() -> list[dict]:
    return _ensure_retrievers()[1]


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
    """Trạng thái từng nguồn retrieval + đường dẫn logical đang dùng."""
    statuses = get_retriever_status()
    ready = [s for s in statuses if s["state"] == "ready"]

    if ready:
        summary = " + ".join(s["detail"] for s in ready)
    else:
        summary = "không có nguồn retrieval nào sẵn sàng"

    return {
        "ok": True,
        "retriever": summary,
        "retrievers": statuses,
        "ready_count": len(ready),
        "keyframes_dir": str(KEYFRAMES_DIR),
        "use_dummy": USE_DUMMY,
        "paths": {
            "clip_index": str(INDEX_PATH),
            "clip_meta": str(META_PATH),
            "siglip_index": str(SIGLIP_INDEX_PATH) if SIGLIP_INDEX_PATH else None,
            "siglip_meta": str(SIGLIP_META_PATH) if SIGLIP_META_PATH else None,
            "text_index": str(TEXT_INDEX_PATH),
        },
    }


@app.post("/api/translate")
def translate(req: TranslateRequest):
    """Dịch câu hỏi tiếng Việt → tiếng Anh bằng model local."""
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


_frame_to_n = None

def _get_frame_mapping():
    global _frame_to_n
    if _frame_to_n is not None:
        return _frame_to_n
    
    _frame_to_n = {}
    if META_PATH and META_PATH.exists():
        try:
            import json
            with open(META_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for meta in metadata:
                vid = meta.get("video_id")
                fidx = meta.get("frame_idx")
                n = meta.get("keyframe_num")
                if vid is not None and fidx is not None and n is not None:
                    _frame_to_n[(vid, fidx)] = n
            logger.info("Loaded frame mapping with %d items from %s", len(_frame_to_n), META_PATH)
        except Exception as e:
            logger.warning("Error loading metadata for frame mapping: %s", e)

    if not _frame_to_n and SIGLIP_META_PATH and SIGLIP_META_PATH.exists():
        try:
            import json
            with open(SIGLIP_META_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for meta in metadata:
                vid = meta.get("video_id")
                fidx = meta.get("frame_idx")
                n = meta.get("keyframe_num")
                if vid is not None and fidx is not None and n is not None:
                    _frame_to_n[(vid, fidx)] = n
            logger.info("Loaded frame mapping with %d items from %s (SigLIP)", len(_frame_to_n), SIGLIP_META_PATH)
        except Exception as e:
            logger.warning("Error loading SigLIP metadata for frame mapping: %s", e)

    return _frame_to_n


_video_fidx_cache = {}


@app.get("/api/keyframe/{video_id}/{frame_idx}")
def get_keyframe(video_id: str, frame_idx: int):
    """Trả về ảnh keyframe (từ local hoặc redirect tới cloud R2/HF)."""
    mapping = _get_frame_mapping()
    n = mapping.get((video_id, frame_idx))
    
    # Nếu không có keyframe khớp chính xác (do frame_idx bị chỉnh sửa thủ công),
    # tìm keyframe gần nhất của video này để tránh hiển thị ảnh lỗi.
    if n is None and mapping:
        global _video_fidx_cache
        if video_id not in _video_fidx_cache:
            _video_fidx_cache[video_id] = [fidx for (vid, fidx) in mapping.keys() if vid == video_id]
        
        v_keys = _video_fidx_cache[video_id]
        if v_keys:
            closest_fidx = min(v_keys, key=lambda x: abs(x - frame_idx))
            n = mapping.get((video_id, closest_fidx))

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
    """Lấy thông tin video (fps) từ video_metadata.json."""
    meta = load_video_metadata()
    video_info = meta.get(video_id)
    if video_info:
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