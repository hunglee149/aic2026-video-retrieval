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

# Load .env manually if exists (no dotenv dependency required)
env_path = Path(".env")
if env_path.exists():
    for _line in env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip().strip("'\"")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.convert import to_answer, to_csv_row
from ..core.query_processor import make_query, translate_query
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
INDEX_PATH = Path(os.environ.get("AIC_INDEX_PATH", "local/clip_faiss.index"))
META_PATH = Path(os.environ.get("AIC_META_PATH", "local/clip_metadata.json"))
TEXT_INDEX_PATH = Path(os.environ.get("AIC_TEXT_INDEX_PATH", "data/input/input/index/text_search_index.pkl"))
USE_DUMMY = os.environ.get("AIC_USE_DUMMY", "0") == "1"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AIC 2026 Video Retrieval", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Lazy retriever loader
# ---------------------------------------------------------------------------

import threading

_retrievers = []
_retriever_lock = threading.Lock()

def get_retrievers():
    global _retrievers
    with _retriever_lock:
        if _retrievers:
            return _retrievers
        if USE_DUMMY:
            logger.info("Using dummy retriever (set AIC_USE_DUMMY=0 for CLIP)")
            _retrievers = [dummy]
            return _retrievers
            
        try:
            from ..retrieval.clip import build_clip_retriever
            logger.info("Loading CLIP retriever...")
            clip_retriever = build_clip_retriever(INDEX_PATH, META_PATH)
            _retrievers.append(clip_retriever)
            logger.info("CLIP retriever ready.")
        except Exception as e:
            logger.warning("CLIP load failed (%s), falling back to dummy", e)
            _retrievers.append(dummy)
            
        try:
            from ..retrieval.text_retriever import TextRetriever
            if TEXT_INDEX_PATH.exists():
                logger.info("Loading Text retriever (BM25)...")
                text_retriever = TextRetriever(TEXT_INDEX_PATH, name="bm25")
                _retrievers.append(text_retriever)
                logger.info("Text retriever ready.")
            else:
                logger.warning("Text index not found at %s", TEXT_INDEX_PATH)
        except Exception as e:
            logger.warning("Text retriever load failed: %s", e)
            
        return _retrievers


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
    retrievers = get_retrievers()
    retriever_names = []
    for r in retrievers:
        name = getattr(r, "NAME", getattr(r, "name", "unknown"))
        if hasattr(r, "num_vectors"):
            name = f"CLIP ({r.num_vectors:,} vectors)"
        elif hasattr(r, "num_documents"):
            name = f"BM25 ({r.num_documents:,} docs)"
        elif r is dummy:
            name = "dummy"
        retriever_names.append(name)
        
    return {
        "ok": True,
        "retriever": " + ".join(retriever_names),
        "keyframes_dir": str(KEYFRAMES_DIR),
        "use_dummy": USE_DUMMY,
    }


@app.post("/api/translate")
def translate(req: TranslateRequest):
    """Dịch câu hỏi tiếng Việt → tiếng Anh qua Gemini."""
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        return {"text_en": req.text_vi, "ok": False, "error": "Chưa cài đặt biến môi trường GEMINI_API_KEY"}
        
    q = make_query("_tmp", text_vi=req.text_vi)
    from ..core.query_processor import _gemini_translate
    try:
        text_en = _gemini_translate(req.text_vi)
        return {"text_en": text_en, "ok": True}
    except Exception as e:
        logger.warning("Translation error: %s", e)
        return {"text_en": req.text_vi, "ok": False, "error": str(e)}


@app.post("/api/search")
def search(req: SearchRequest):
    """Tìm kiếm candidates bằng retriever hiện tại."""
    retrievers = get_retrievers()
    from ..core.query_processor import process_query
    
    query = make_query(
        query_id=req.query_id,
        text_vi=req.text_vi,
        text_en=req.text_en,
        task=req.task,
        n_events=req.n_events,
    )
    
    # Bật lại LLM Gemini để kết quả tốt hơn (dịch + synonyms)
    query = process_query(query)
    
    exclude = frozenset(req.exclude)

    try:
        candidates = retrieve_and_fuse(
            query=query,
            retrievers=retrievers,
            fuse_fn=fuse,
            limit=req.k,
            exclude=exclude,
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
    
    retrievers = get_retrievers()
    _frame_to_n = {}
    for retriever in retrievers:
        if hasattr(retriever, "metadata") and retriever.metadata:
            for meta in retriever.metadata:
                vid = meta.get("video_id")
                fidx = meta.get("frame_idx")
                n = meta.get("keyframe_num")
                if vid is not None and fidx is not None and n is not None:
                    _frame_to_n[(vid, fidx)] = n
    return _frame_to_n


@app.get("/api/keyframe/{video_id}/{frame_idx}")
def get_keyframe(video_id: str, frame_idx: int):
    """Trả về ảnh keyframe từ thư mục data/keyframes."""
    candidates = []
    
    mapping = _get_frame_mapping()
    n = mapping.get((video_id, frame_idx))
    
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

@app.get("/api/video_info/{video_id}")
def get_video_info(video_id: str):
    """Lấy thông tin video (fps)."""
    video_path = _get_video_path(video_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")
        
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return {"fps": fps if fps > 0 else 25.0}

@app.get("/api/video/{video_id}")
def get_video(video_id: str, request: Request):
    """Trả về video với hỗ trợ Range để tua."""
    video_path = _get_video_path(video_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")
        
    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range")
    
    if range_header:
        byte1, byte2 = 0, None
        try:
            match = range_header.replace("bytes=", "").split("-")
            if match[0]:
                byte1 = int(match[0])
            if len(match) > 1 and match[1]:
                byte2 = int(match[1])
        except ValueError:
            pass
            
        if byte2 is None:
            byte2 = file_size - 1
            
        length = byte2 - byte1 + 1
        
        def video_generator(path, start, length, chunk_size=1024*1024):
            with open(path, "rb") as f:
                f.seek(start)
                while length > 0:
                    chunk = f.read(min(length, chunk_size))
                    if not chunk:
                        break
                    length -= len(chunk)
                    yield chunk
                    
        headers = {
            "Content-Range": f"bytes {byte1}-{byte2}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        }
        return StreamingResponse(
            video_generator(video_path, byte1, length), 
            status_code=206, 
            headers=headers, 
            media_type="video/mp4"
        )
        
    return FileResponse(str(video_path), media_type="video/mp4")


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
