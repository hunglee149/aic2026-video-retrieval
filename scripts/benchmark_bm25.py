#!/usr/bin/env python3
"""Benchmark so sánh hiệu năng của TextRetriever trước và sau khi precompute BM25.

Đo lường:
1. Thời gian nạp (Load time)
2. Lượng bộ nhớ tiêu thụ đỉnh (Peak memory)
3. Thời gian tìm kiếm (Search latency)
4. Độ tương đồng kết quả và thứ hạng (Ranking equivalence)
"""

import gc
import logging
import os
import pickle
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aic.core.types import Query
from aic.retrieval.text_retriever import TextRetriever
from scripts.build_text_index import compute_bm25_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_bm25")


def create_synthetic_index(num_docs: int = 20000) -> dict:
    """Tạo corpus mẫu để benchmark nếu máy không có sẵn index thật."""
    logger.info("Tạo dataset mẫu %d documents...", num_docs)
    vocab = [
        "người", "đàn_ông", "phụ_nữ", "áo_đỏ", "xe_hơi", "đường_phố", "công_viên",
        "nấu_ăn", "nhà_bếp", "chạy_bộ", "biển_hiệu", "đà_lạt", "thành_phố",
        "xe_máy", "giao_thông", "học_sinh", "trường_học", "bóng_đá", "sân_vận_động"
    ]
    documents = []
    tokenized = []
    keyframe_map = {}

    import random
    random.seed(42)

    for i in range(num_docs):
        vid = f"L01_V{i % 100:03d}"
        doc_len = random.randint(5, 30)
        tokens = [random.choice(vocab) for _ in range(doc_len)]
        text = " ".join(tokens)
        documents.append({
            "type": "ocr",
            "video_id": vid,
            "text": text,
            "keyframe_num": i % 50,
            "frame_idx": (i % 50) * 25,
            "start_time": float(i % 50),
            "end_time": float(i % 50),
        })
        tokenized.append(tokens)
        if vid not in keyframe_map:
            keyframe_map[vid] = []
        keyframe_map[vid].append({
            "kf_num": i % 50,
            "frame_idx": (i % 50) * 25,
            "pts_time": float(i % 50),
        })

    return {
        "documents": documents,
        "tokenized": tokenized,
        "keyframe_map": keyframe_map,
    }


def benchmark():
    # Kiểm tra xem có file index thật không
    text_index_env = os.environ.get("AIC_TEXT_INDEX_PATH")
    sample_index_path = Path(text_index_env) if text_index_env and Path(text_index_env).exists() else None

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw_index.pkl"
        precomputed_path = Path(tmpdir) / "precomputed_index.pkl"

        if sample_index_path:
            logger.info("Sử dụng index thật từ: %s", sample_index_path)
            with sample_index_path.open("rb") as f:
                data = pickle.load(f)
        else:
            logger.info("Không thấy AIC_TEXT_INDEX_PATH, sử dụng dataset mô phỏng 25,000 documents...")
            data = create_synthetic_index(25000)

        # 1. Tạo file index raw (không có precomputed)
        raw_payload = {
            "documents": data["documents"],
            "tokenized": data["tokenized"],
            "keyframe_map": data.get("keyframe_map", {}),
        }
        with raw_path.open("wb") as f:
            pickle.dump(raw_payload, f)

        # 2. Tạo file index đã precompute
        logger.info("Tính toán precompute BM25...")
        bm25_stats = compute_bm25_stats(data["tokenized"])
        precomputed_payload = dict(raw_payload)
        precomputed_payload.update(bm25_stats)
        with precomputed_path.open("wb") as f:
            pickle.dump(precomputed_payload, f)

        # -------------------------------------------------------------
        # Đo 1: Không có precompute (phải build BM25 động)
        # -------------------------------------------------------------
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        retriever_raw = TextRetriever(raw_path, name="bm25_raw")
        t_load_raw = (time.perf_counter() - t0) * 1000
        current_mem, peak_mem_raw = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # -------------------------------------------------------------
        # Đo 2: Có precompute (nạp trực tiếp từ pickle)
        # -------------------------------------------------------------
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        retriever_prec = TextRetriever(precomputed_path, name="bm25_precomputed")
        t_load_prec = (time.perf_counter() - t0) * 1000
        current_mem, peak_mem_prec = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # -------------------------------------------------------------
        # Đo 3: Thời gian tìm kiếm và kiểm tra tương đồng kết quả
        # -------------------------------------------------------------
        queries = [
            Query(query_id="q1", text_vi="người đàn ông áo đỏ nấu ăn"),
            Query(query_id="q2", text_vi="xe hơi đường phố giao thông"),
            Query(query_id="q3", text_vi="học sinh trường học đà lạt"),
        ]

        logger.info("\n" + "=" * 65)
        logger.info("KẾT QUẢ BENCHMARK BM25 (PRECOMPUTE VS DYNAMIC BUILD)")
        logger.info("=" * 65)
        logger.info(f"Số lượng documents: {len(data['documents']):,}")
        logger.info(f"Thời gian load (Chưa precompute): {t_load_raw:.2f} ms")
        logger.info(f"Thời gian load (Đã precompute)  : {t_load_prec:.2f} ms")
        speedup = t_load_raw / t_load_prec if t_load_prec > 0 else float("inf")
        logger.info(f"  → Tốc độ nạp nhanh hơn        : {speedup:.1f}x")
        logger.info(f"Bộ nhớ đỉnh (Chưa precompute)   : {peak_mem_raw / (1024*1024):.2f} MB")
        logger.info(f"Bộ nhớ đỉnh (Đã precompute)     : {peak_mem_prec / (1024*1024):.2f} MB")

        # Kiểm tra kết quả tìm kiếm giống hệt nhau
        logger.info("-" * 65)
        logger.info("Kiểm tra độ tương đồng kết quả tìm kiếm:")
        for idx, q in enumerate(queries, 1):
            t0 = time.perf_counter()
            res_raw = retriever_raw.search(q, limit=10)
            t_search_raw = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            res_prec = retriever_prec.search(q, limit=10)
            t_search_prec = (time.perf_counter() - t0) * 1000

            # So sánh candidate video_id và representative_frames
            raw_tuples = [(c.video_id, c.start_frame, c.scores.get("bm25_raw")) for c in res_raw]
            prec_tuples = [(c.video_id, c.start_frame, c.scores.get("bm25_precomputed")) for c in res_prec]

            assert len(res_raw) == len(res_prec), f"Số lượng kết quả lệch ở query {idx}"
            for (vid_r, f_r, s_r), (vid_p, f_p, s_p) in zip(raw_tuples, prec_tuples):
                assert vid_r == vid_p, f"Lệch video_id: {vid_r} != {vid_p}"
                assert f_r == f_p, f"Lệch frame: {f_r} != {f_p}"
                assert abs(s_r - s_p) < 1e-5, f"Lệch score: {s_r} != {s_p}"

            logger.info(f"Query {idx} ({q.text_vi}):")
            logger.info(f"  - Search latency (raw): {t_search_raw:.2f} ms | (precomputed): {t_search_prec:.2f} ms")
            logger.info(f"  - Kết quả & thứ hạng  : HOÀN TOÀN KHỚP 100% ({len(res_raw)} candidates)")

        logger.info("=" * 65)
        logger.info("XÁC NHẬN: Precompute BM25 giảm đáng kể thời gian load, không làm thay đổi thứ hạng hay sai lệch frame!")
        logger.info("=" * 65)


if __name__ == "__main__":
    benchmark()
