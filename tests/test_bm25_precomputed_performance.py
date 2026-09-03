"""Tests for BM25 precomputed statistics, array-compression, and lazy loading performance."""

import pickle
import math
from array import array
from pathlib import Path
from unittest.mock import patch

import pytest

from aic.core.types import Query
from aic.retrieval.text_retriever import TextRetriever, build_text_retriever


@pytest.fixture
def sample_corpus():
    docs = [
        {"type": "caption", "video_id": "V001", "text": "người đi xe đạp áo đỏ", "start_time": 5.0, "end_time": 10.0},
        {"type": "ocr", "video_id": "V001", "text": "biển hiệu đà lạt 2026", "keyframe_num": 1, "start_time": 5.0, "end_time": 5.0},
        {"type": "transcript_segment", "video_id": "V002", "text": "chúc mừng năm mới áo đỏ", "start_time": 12.0, "end_time": 15.0},
        {"type": "caption", "video_id": "V003", "text": "xe cấp cứu bệnh viện", "start_time": 20.0, "end_time": 25.0},
    ]
    tokens = [
        ["người", "đi", "xe_đạp", "áo", "đỏ"],
        ["biển_hiệu", "đà_lạt", "2026"],
        ["chúc_mừng", "năm_mới", "áo", "đỏ"],
        ["xe_cấp_cứu", "bệnh_viện"],
    ]
    keyframe_map = {
        "V001": [{"kf_num": 1, "frame_idx": 125, "pts_time": 5.0}],
        "V002": [{"kf_num": 1, "frame_idx": 300, "pts_time": 12.0}],
        "V003": [{"kf_num": 1, "frame_idx": 500, "pts_time": 20.0}],
    }
    return docs, tokens, keyframe_map


def test_legacy_index_backward_compatibility(tmp_path, sample_corpus):
    """Verifies that legacy index format with tokenized is still 100% supported."""
    docs, tokens, keyframe_map = sample_corpus
    legacy_payload = {
        "documents": docs,
        "tokenized": tokens,
        "keyframe_map": keyframe_map,
    }
    idx_file = tmp_path / "legacy_index.pkl"
    with idx_file.open("wb") as f:
        pickle.dump(legacy_payload, f)

    retriever = TextRetriever(idx_file)
    assert retriever.N == 4
    assert len(retriever.doc_lengths) == 4
    assert retriever.avgdl == (5 + 3 + 4 + 2) / 4.0
    assert "áo" in retriever.inverted

    # Test search
    res = retriever.search(Query(query_id="q1", text_vi="áo đỏ"), limit=5)
    assert len(res) >= 2
    video_ids = [c.video_id for c in res]
    assert "V001" in video_ids
    assert "V002" in video_ids


def test_precomputed_array_index_loading(tmp_path, sample_corpus):
    """Verifies that index with precomputed inverted_arrays, idf, avgdl, doc_lengths loads with tokenized=[]"""
    docs, tokens, keyframe_map = sample_corpus
    N = len(docs)
    doc_lengths = array('I', [len(t) for t in tokens])
    avgdl = sum(doc_lengths) / float(N)

    # Compute idf & inverted
    df = {"người": 1, "đi": 1, "xe_đạp": 1, "áo": 2, "đỏ": 2, "biển_hiệu": 1, "đà_lạt": 1, "2026": 1}
    idf = {k: math.log((N - v + 0.5) / (v + 0.5) + 1.0) for k, v in df.items()}
    inverted_arrays = {
        "áo": (array('I', [0, 2]), array('H', [1, 1])),
        "đỏ": (array('I', [0, 2]), array('H', [1, 1])),
        "đà_lạt": (array('I', [1]), array('H', [1])),
    }
    accent_index = {"da_lat": ["đà_lạt"], "ao": ["áo"], "do": ["đỏ"]}

    precomputed_payload = {
        "documents": docs,
        "tokenized": [],  # Empty tokenized list to save RAM
        "keyframe_map": keyframe_map,
        "N": N,
        "avgdl": avgdl,
        "doc_lengths": doc_lengths,
        "idf": idf,
        "inverted": inverted_arrays,
        "accent_index": accent_index,
    }

    idx_file = tmp_path / "precomputed_index.pkl"
    with idx_file.open("wb") as f:
        pickle.dump(precomputed_payload, f)

    retriever = TextRetriever(idx_file)
    assert retriever.N == 4
    assert len(retriever.doc_lengths) == 4
    assert retriever.avgdl == avgdl
    assert retriever._accent_index == accent_index

    # Search with unaccented query to verify accent_index mapping
    res = retriever.search(Query(query_id="q1", text_vi="da lat"), limit=5)
    assert len(res) == 1
    assert res[0].video_id == "V001"
    assert res[0].start_frame == 125


def test_ranking_equivalence_between_legacy_and_precomputed(tmp_path, sample_corpus):
    """Ensures dynamic BM25 build and precomputed array BM25 produce identical candidate rankings and scores."""
    docs, tokens, keyframe_map = sample_corpus
    
    # 1. Build legacy
    legacy_file = tmp_path / "legacy.pkl"
    with legacy_file.open("wb") as f:
        pickle.dump({"documents": docs, "tokenized": tokens, "keyframe_map": keyframe_map}, f)
    r_legacy = TextRetriever(legacy_file)

    # 2. Build precomputed
    N = len(docs)
    doc_lengths = array('I', [len(t) for t in tokens])
    avgdl = sum(doc_lengths) / float(N)
    inverted_arrays = {}
    for term, postings in r_legacy.inverted.items():
        doc_ids = array('I', [p[0] for p in postings])
        counts = array('H', [p[1] for p in postings])
        inverted_arrays[term] = (doc_ids, counts)

    precomputed_file = tmp_path / "precomputed.pkl"
    with precomputed_file.open("wb") as f:
        pickle.dump({
            "documents": docs,
            "tokenized": [],
            "keyframe_map": keyframe_map,
            "N": N,
            "avgdl": avgdl,
            "doc_lengths": doc_lengths,
            "idf": r_legacy.idf,
            "inverted": inverted_arrays,
            "accent_index": r_legacy._accent_index,
        }, f)
    r_precomputed = TextRetriever(precomputed_file)

    # Compare search outputs for several queries
    for query_str in ["áo đỏ", "xe cấp cứu", "đà lạt", "người đi xe"]:
        q = Query(query_id="test", text_vi=query_str)
        res1 = r_legacy.search(q, limit=10)
        res2 = r_precomputed.search(q, limit=10)
        assert len(res1) == len(res2)
        for c1, c2 in zip(res1, res2):
            assert c1.video_id == c2.video_id
            assert c1.start_frame == c2.start_frame
            assert pytest.approx(c1.scores["bm25"], 1e-4) == c2.scores["bm25"]


def test_translation_isolated_from_bm25():
    """Verifies that translation functions independently even when BM25 is uninitialized or missing."""
    from aic.core.local_translation import translate_text

    with patch("aic.retrieval.text_retriever.TextRetriever") as mock_retriever:
        with patch("aic.core.local_translation._load_translator") as mock_load:
            mock_trans = mock_load.return_value
            mock_trans.translate.return_value = "A person riding a bicycle"
            result = translate_text("Một người đi xe đạp")
            assert result == "A person riding a bicycle"
            assert not mock_retriever.called
