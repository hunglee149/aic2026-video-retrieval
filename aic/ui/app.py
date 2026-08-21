"""AIC 2026 — UI Server.

FastAPI backend + static file server cho giao diện operator.
Hỗ trợ Multi-Modal Hybrid Search (CLIP + SigLIP + BM25 + Objects)
và đọc ảnh trực tiếp từ dữ liệu gốc trong D:/AIC 2026/batch_01/

Chạy:
    python -m uvicorn aic.ui.app:app --reload --port 8000
    hoặc:
    python -m aic.ui

API:
    POST /api/search        — tìm kiếm candidates đa kênh
    POST /api/translate     — dịch query VI→EN
    POST /api/export        — xuất submission.zip
    GET  /api/status        — trạng thái server & model
    GET  /api/keyframe/{video_id}/{frame_idx}  — ảnh keyframe thật từ zip/disk
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.convert import to_answer, to_csv_row
from ..core.query_processor import make_query, process_query, translate_query
from ..core.types import Candidate, Query
from ..fusion.rank import fuse
from ..pipeline import retrieve_and_fuse
from ..retrieval import dummy
from ..submission.writer import write_submission

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config & Paths
# ---------------------------------------------------------------------------

KEYFRAMES_DIR = Path(os.environ.get("AIC_KEYFRAMES_DIR", "data/keyframes"))
LOCAL_DIR = Path(os.environ.get("AIC_LOCAL_DIR", "local"))
USE_DUMMY = os.environ.get("AIC_USE_DUMMY", "0") == "1"

# ---------------------------------------------------------------------------
# Keyframe Fast Zip Loader (Đọc ảnh trực tiếp từ Zip trong data/keyframes/)
# ---------------------------------------------------------------------------

_video_zip_map: dict[str, str] = {}
_video_first_frame: dict[str, str] = {}
_open_zips: dict[str, zipfile.ZipFile] = {}


def _init_keyframe_map():
    global _video_zip_map, _video_first_frame
    if _video_zip_map:
        return
    map_file = LOCAL_DIR / "video_keyframes_map.json"
    if map_file.exists():
        try:
            import json
            with open(map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            _video_zip_map = data.get("video_to_zip", {})
            _video_first_frame = data.get("video_first_frame", {})
        except Exception as e:
            logger.warning("Failed loading video_keyframes_map.json: %s", e)


def _get_keyframe_from_zip(video_id: str, frame_idx: int) -> bytes | None:
    global _video_zip_map, _open_zips
    if not KEYFRAMES_DIR.exists():
        return None

    _init_keyframe_map()

    zip_name = _video_zip_map.get(video_id)
    if not zip_name:
        return None

    zip_path = str(KEYFRAMES_DIR / zip_name)
    z = _open_zips.get(zip_path)
    if not z:
        try:
            z = zipfile.ZipFile(zip_path, "r")
            _open_zips[zip_path] = z
        except Exception as e:
            logger.warning("Cannot open zip %s: %s", zip_path, e)
            return None

    # Mẫu tên ảnh trong zip (thử cả 0-based và 1-based)
    patterns = [
        f"keyframes/{video_id}/{frame_idx:03d}.jpg",
        f"keyframes/{video_id}/{frame_idx + 1:03d}.jpg",
        f"keyframes/{video_id}/{frame_idx:06d}.jpg",
        f"keyframes/{video_id}/{frame_idx + 1:06d}.jpg",
        f"keyframes/{video_id}/{frame_idx}.jpg",
        f"keyframes/{video_id}/{frame_idx + 1}.jpg",
        f"keyframes/{video_id}/{frame_idx:03d}.png",
        f"keyframes/{video_id}/{frame_idx + 1:03d}.png",
    ]

    first_fname = _video_first_frame.get(video_id)
    if first_fname:
        patterns.append(f"keyframes/{video_id}/{first_fname}")

    for p in patterns:
        try:
            return z.read(p)
        except KeyError:
            continue
    return None


# ---------------------------------------------------------------------------
# App & Static
# ---------------------------------------------------------------------------

app = FastAPI(title="AIC 2026 Video Retrieval", version="0.2.0")
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Multi-Modal Retrievers Loader
# ---------------------------------------------------------------------------

_retrievers: list = []
_object_filter = None
_loaded: bool = False


def get_retrievers():
    """Nạp tất cả các bộ tìm kiếm có sẵn từ thư mục local/."""
    global _retrievers, _object_filter, _loaded
    if _loaded:
        return _retrievers, _object_filter

    if USE_DUMMY:
        logger.info("AIC_USE_DUMMY=1: Using dummy retriever")
        _retrievers = [dummy]
        _loaded = True
        return _retrievers, None

    active_retrievers = []

    # 1. BM25 Bilingual Text Retriever (ASR + OCR + Captions)
    text_index = LOCAL_DIR / "text_search_index.pkl"
    if text_index.exists():
        try:
            import gc
            from ..retrieval.text_retriever import build_text_retriever

            logger.info("Loading BM25 Text Retriever from %s ...", text_index)
            tr = build_text_retriever(text_index)
            active_retrievers.append(tr)
            gc.collect()  # Giải phóng RAM sau khi load xong
            logger.info("  ✓ BM25 Text Retriever ready (%d documents)", tr.num_documents)
        except Exception as e:
            import traceback
            logger.warning("BM25 load failed: %s\n%s", e, traceback.format_exc())

    # 2. CLIP FAISS Vector Search (Visual)
    # Tự động nạp CLIP nếu có file index và metadata
    clip_index = LOCAL_DIR / "clip_faiss.index"
    clip_meta = LOCAL_DIR / "clip_metadata.json"
    if clip_index.exists() and clip_meta.exists() and os.environ.get("AIC_DISABLE_NEURAL", "0") != "1":
        try:
            from ..retrieval.clip import build_clip_retriever
            logger.info("Loading CLIP retriever from %s ...", clip_index)
            cr = build_clip_retriever(clip_index, clip_meta)
            active_retrievers.append(cr)
            logger.info("  ✓ CLIP Visual Retriever ready (%d vectors)", cr.num_vectors)
        except Exception as e:
            logger.info("CLIP skipped: %s", e)

    # 4. Object Detection Filter (Tắt theo yêu cầu — chỉ dùng điểm BM25 text match)
    _object_filter = None

    # Fallback to dummy if no retrievers could be loaded
    if not active_retrievers:
        logger.warning("No models loaded. Falling back to dummy retriever.")
        active_retrievers = [dummy]

    _retrievers = active_retrievers
    _loaded = True
    return _retrievers, _object_filter


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query_id: str = "q1"
    text_vi: str
    text_en: str = ""
    task: str = "kis"  # kis | qa | trake
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
# Helpers
# ---------------------------------------------------------------------------


def _candidate_to_out(cand: Candidate, rank: int) -> dict:
    evidence = {}
    for ev_k in ("transcript_match", "ocr_match", "caption_match"):
        if ev_k in cand.evidence and cand.evidence[ev_k]:
            evidence[ev_k] = cand.evidence[ev_k]
    if not evidence and cand.evidence:
        evidence = cand.evidence

    scores = {}
    for k, v in cand.scores.items():
        if k != "object_match":
            scores[k] = round(v, 4)

    # Điểm hiển thị trên huy hiệu card
    display_score = cand.best_score
    if "bm25" in cand.scores and len(scores) == 1:
        display_score = cand.scores["bm25"]

    return {
        "video_id": cand.video_id,
        "start_frame": cand.start_frame,
        "end_frame": cand.end_frame,
        "representative_frames": cand.representative_frames,
        "scores": scores,
        "evidence": evidence,
        "best_score": round(display_score, 4),
        "rank": rank,
    }


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get("/api/status")
def status():
    """Trả về trạng thái hệ thống — chỉ kiểm tra file, KHÔNG load model."""
    names = []

    if _loaded:
        for r in _retrievers:
            name = getattr(r, "NAME", getattr(r, "name", "unknown"))
            if hasattr(r, "num_vectors"):
                name = f"{name} ({r.num_vectors:,} vectors)"
            elif hasattr(r, "num_documents"):
                name = f"BM25 ({r.num_documents:,} docs)"
            names.append(name)
    else:
        if (LOCAL_DIR / "text_search_index.pkl").exists():
            names.append("BM25 (629,404 docs)")

    return {
        "ok": True,
        "retriever": " + ".join(names) if names else "dummy",
        "keyframes_dir": str(KEYFRAMES_DIR),
        "use_dummy": USE_DUMMY,
    }


@app.post("/api/translate")
def translate(req: TranslateRequest):
    """Dịch câu hỏi tiếng Việt → tiếng Anh qua Gemini / Fallback."""
    q = make_query("_tmp", text_vi=req.text_vi)
    try:
        q = translate_query(q)
        return {"text_en": q.text_en, "ok": True}
    except Exception as e:
        logger.warning("Translation error: %s", e)
        return {"text_en": req.text_vi, "ok": False, "error": str(e)}


@app.post("/api/search")
def search(req: SearchRequest):
    """Tìm kiếm candidates qua BM25 text match (ASR / OCR / Caption)."""
    retrievers, _ = get_retrievers()

    # 1. Khởi tạo & xử lý mở rộng query
    query = make_query(
        query_id=req.query_id,
        text_vi=req.text_vi,
        text_en=req.text_en,
        task=req.task,
        n_events=req.n_events,
    )
    query = process_query(query)
    exclude = frozenset(req.exclude)

    try:
        # 2. Tìm kiếm song song qua các nguồn & gộp điểm RRF
        candidates = retrieve_and_fuse(
            query=query,
            retrievers=retrievers,
            fuse_fn=fuse,
            limit=req.k,
            exclude=exclude,
        )

    except Exception as e:
        import traceback
        logger.error("Search error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e) or repr(e))

    return {
        "ok": True,
        "total": len(candidates),
        "candidates": [_candidate_to_out(c, i + 1) for i, c in enumerate(candidates)],
    }


@app.get("/api/keyframe/{video_id}/{frame_idx}")
def get_keyframe(video_id: str, frame_idx: int):
    """Trả về ảnh keyframe từ disk hoặc trực tiếp từ file Zip gốc trong D:/AIC 2026/batch_01."""
    # 1. Thử đọc từ local disk nếu đã giải nén
    candidates = [
        KEYFRAMES_DIR / video_id / f"{frame_idx:03d}.jpg",
        KEYFRAMES_DIR / video_id / f"{frame_idx:06d}.jpg",
        KEYFRAMES_DIR / video_id / f"{frame_idx}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return FileResponse(str(p), media_type="image/jpeg")

    # 2. Thử đọc trực tiếp từ Keyframes_*.zip trong D:/AIC 2026/batch_01
    zip_bytes = _get_keyframe_from_zip(video_id, frame_idx)
    if zip_bytes:
        return Response(content=zip_bytes, media_type="image/jpeg")

    # 3. Trả về placeholder SVG màu gradient nếu không tìm thấy ảnh
    hue = hash(f"{video_id}{frame_idx}") % 360
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
  <polygon points="155,47 155,63 171,55" fill="rgba(255,255,255,0.5)"/>
</svg>"""
    return Response(content=svg.encode(), media_type="image/svg+xml")


@app.get("/api/video_keyframes/{video_id}")
def get_video_keyframes(video_id: str):
    """Trả về danh sách tất cả keyframes có trong video để duyệt timeline."""
    retrievers, _ = get_retrievers()
    for r in retrievers:
        if hasattr(r, "keyframe_map") and video_id in r.keyframe_map:
            return {"ok": True, "video_id": video_id, "keyframes": r.keyframe_map[video_id]}

    _init_keyframe_map()
    zip_name = _video_zip_map.get(video_id)
    if zip_name:
        zip_path = str(KEYFRAMES_DIR / zip_name)
        z = _open_zips.get(zip_path)
        if not z:
            try:
                z = zipfile.ZipFile(zip_path, "r")
                _open_zips[zip_path] = z
            except Exception:
                z = None
        if z:
            prefix = f"keyframes/{video_id}/"
            kf_list = []
            for name in sorted(z.namelist()):
                if name.startswith(prefix) and (name.endswith(".jpg") or name.endswith(".png")):
                    fname = name.removeprefix(prefix)
                    stem = Path(fname).stem
                    try:
                        num = int(stem)
                        kf_list.append({"kf_num": num, "frame_idx": num, "pts_time": None})
                    except ValueError:
                        pass
            return {"ok": True, "video_id": video_id, "keyframes": kf_list}

    return {"ok": False, "video_id": video_id, "keyframes": []}


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
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("aic.ui.app:app", host="0.0.0.0", port=8000, reload=True)
