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

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.convert import to_answer, to_csv_row
from ..core.query_processor import make_query, translate_query
from ..core.types import Candidate
from ..fusion.rank import fuse
from ..pipeline import retrieve_and_fuse
from ..retrieval import dummy
from ..submission.writer import write_submission

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KEYFRAMES_DIR = Path(os.environ.get("AIC_KEYFRAMES_DIR", "data/keyframes"))
INDEX_PATH = Path(os.environ.get("AIC_INDEX_PATH", "local/clip_faiss.index"))
META_PATH = Path(os.environ.get("AIC_META_PATH", "local/clip_metadata.json"))
USE_DUMMY = os.environ.get("AIC_USE_DUMMY", "1") == "1"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AIC 2026 Video Retrieval", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Lazy retriever loader
# ---------------------------------------------------------------------------

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    if USE_DUMMY:
        logger.info("Using dummy retriever (set AIC_USE_DUMMY=0 for CLIP)")
        _retriever = dummy
        return _retriever
    try:
        from ..retrieval.clip import build_clip_retriever

        logger.info("Loading CLIP retriever...")
        _retriever = build_clip_retriever(INDEX_PATH, META_PATH)
        logger.info("CLIP retriever ready.")
    except Exception as e:
        logger.warning("CLIP load failed (%s), falling back to dummy", e)
        _retriever = dummy
    return _retriever


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
    video_id: str
    frames: list[int]
    answer: str = ""


class ExportRequest(BaseModel):
    query_id: str
    task: str = "kis"
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


@app.get("/api/status")
def status():
    retriever = get_retriever()
    retriever_name = getattr(retriever, "NAME", getattr(retriever, "name", "unknown"))
    if hasattr(retriever, "num_vectors"):
        retriever_name = f"CLIP ({retriever.num_vectors:,} vectors)"
    elif retriever is dummy:
        retriever_name = "dummy"
    return {
        "ok": True,
        "retriever": retriever_name,
        "keyframes_dir": str(KEYFRAMES_DIR),
        "use_dummy": USE_DUMMY,
    }


@app.post("/api/translate")
def translate(req: TranslateRequest):
    """Dịch câu hỏi tiếng Việt → tiếng Anh qua Gemini."""
    q = make_query("_tmp", text_vi=req.text_vi)
    try:
        q = translate_query(q)
        return {"text_en": q.text_en, "ok": True}
    except Exception as e:
        logger.warning("Translation error: %s", e)
        return {"text_en": req.text_vi, "ok": False, "error": str(e)}


@app.post("/api/search")
def search(req: SearchRequest):
    """Tìm kiếm candidates bằng retriever hiện tại."""
    retriever = get_retriever()
    query = make_query(
        query_id=req.query_id,
        text_vi=req.text_vi,
        text_en=req.text_en,
        task=req.task,
        n_events=req.n_events,
    )
    exclude = frozenset(req.exclude)

    try:
        candidates = retrieve_and_fuse(
            query=query,
            retrievers=[retriever],
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


@app.get("/api/keyframe/{video_id}/{frame_idx}")
def get_keyframe(video_id: str, frame_idx: int):
    """Trả về ảnh keyframe từ thư mục data/keyframes.

    Cấu trúc thư mục kỳ vọng:
        data/keyframes/{video_id}/{frame_idx:06d}.jpg
    hoặc:
        data/keyframes/{video_id}/{frame_idx}.jpg
    """
    candidates = [
        KEYFRAMES_DIR / video_id / f"{frame_idx:06d}.jpg",
        KEYFRAMES_DIR / video_id / f"{frame_idx}.jpg",
        KEYFRAMES_DIR / video_id / f"{frame_idx:06d}.png",
        KEYFRAMES_DIR / video_id / f"{frame_idx}.png",
    ]
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


@app.post("/api/export")
def export_submission(req: ExportRequest):
    """Tạo submission.zip và trả về để tải xuống."""
    rows = []
    for row in req.rows:
        vid = row.video_id.removesuffix(".mp4")
        r = [vid] + [str(f) for f in row.frames]
        if row.answer:
            r.append(row.answer)
        rows.append(r)

    if not rows:
        raise HTTPException(status_code=400, detail="No rows to export")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_submission({req.query_id: rows}, tmp_path)
        zip_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    filename = f"{req.query_id}.zip"
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
