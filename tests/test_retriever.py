"""Tests cho FaissRetriever — dùng mock FAISS index nhỏ."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from aic.core.types import Candidate, Query

# Chỉ chạy nếu faiss đã cài
faiss = pytest.importorskip("faiss")

from aic.retrieval.faiss_retriever import FaissRetriever


@pytest.fixture
def tiny_index(tmp_path):
    """Tạo FAISS index nhỏ (10 vectors, dim=8) và metadata tương ứng."""
    dim = 8
    n = 10

    # Tạo random vectors, normalize
    rng = np.random.RandomState(42)
    vectors = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Tạo FAISS index (inner product cho cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = tmp_path / "test.index"
    faiss.write_index(index, str(index_path))

    # Tạo metadata
    metadata = []
    for i in range(n):
        metadata.append({
            "video_id": f"L{21 + i // 3}_V{1 + i % 3:03d}",
            "keyframe_num": i + 1,
            "frame_idx": i * 100,
            "pts_time": i * 4.0,
        })

    meta_path = tmp_path / "test_metadata.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    return index_path, meta_path, vectors, metadata


def dummy_encode(text: str) -> np.ndarray:
    """Fake text encoder: deterministic vector from text hash."""
    rng = np.random.RandomState(hash(text) % (2**31))
    vec = rng.randn(8).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestFaissRetriever:

    def test_load(self, tiny_index):
        idx_path, meta_path, vectors, metadata = tiny_index
        retriever = FaissRetriever(idx_path, meta_path, dummy_encode, name="test")

        assert retriever.num_vectors == 10
        assert retriever.dim == 8

    def test_search_returns_candidates(self, tiny_index):
        idx_path, meta_path, vectors, metadata = tiny_index
        retriever = FaissRetriever(idx_path, meta_path, dummy_encode, name="test")
        query = Query(query_id="q1", text_vi="test", text_en="test query")

        results = retriever.search(query, k=5)

        assert len(results) == 5
        assert all(isinstance(c, Candidate) for c in results)
        assert all("test" in c.scores for c in results)

    def test_search_respects_k(self, tiny_index):
        idx_path, meta_path, vectors, metadata = tiny_index
        retriever = FaissRetriever(idx_path, meta_path, dummy_encode, name="test")
        query = Query(query_id="q1", text_vi="test", text_en="test query")

        results = retriever.search(query, k=3)
        assert len(results) == 3

    def test_search_exclude_filters(self, tiny_index):
        idx_path, meta_path, vectors, metadata = tiny_index
        retriever = FaissRetriever(idx_path, meta_path, dummy_encode, name="test")
        query = Query(query_id="q1", text_vi="test", text_en="test query")

        results = retriever.search(query, k=10, exclude=frozenset({"L21_V001"}))
        video_ids = [c.video_id for c in results]
        assert "L21_V001" not in video_ids

    def test_search_sorted_descending(self, tiny_index):
        idx_path, meta_path, vectors, metadata = tiny_index
        retriever = FaissRetriever(idx_path, meta_path, dummy_encode, name="test")
        query = Query(query_id="q1", text_vi="test", text_en="test query")

        results = retriever.search(query, k=5)
        scores = [c.scores["test"] for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_get_video_ids(self, tiny_index):
        idx_path, meta_path, vectors, metadata = tiny_index
        retriever = FaissRetriever(idx_path, meta_path, dummy_encode, name="test")

        video_ids = retriever.get_video_ids()
        assert len(video_ids) > 0
        assert all(isinstance(v, str) for v in video_ids)

    def test_search_by_embedding(self, tiny_index):
        idx_path, meta_path, vectors, metadata = tiny_index
        retriever = FaissRetriever(idx_path, meta_path, dummy_encode, name="test")

        embedding = vectors[0]  # search bằng vector đầu tiên
        results = retriever.search_by_embedding(embedding, k=3)

        assert len(results) == 3
        # Vector gần nhất với chính nó phải có score cao nhất
        assert results[0].scores["test"] >= results[1].scores["test"]
