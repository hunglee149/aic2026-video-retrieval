"""Contract tests cho tầng retrieval sau khi tích hợp core của Hiếu.

Các test ở đây khoá lại những quy ước mà brief tích hợp yêu cầu:

- retriever nhận ``limit``/``exclude`` thống nhất (có shim ``k`` được test);
- CLIP ưu tiên ``query.text_en``;
- Candidate mang actual video frame, không phải ordinal keyframe;
- BM25 đọc được index thật và map timestamp sang frame thật;
- fusion là weighted RRF, giữ nhiều moment trong cùng video, không padding giả;
- production (``AIC_USE_DUMMY=0``) không âm thầm rơi về dummy.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from aic.core.types import Candidate, Query

faiss = pytest.importorskip("faiss")

from aic.fusion.rank import DEFAULT_RRF_K, fuse  # noqa: E402
from aic.retrieval.faiss_retriever import FaissRetriever  # noqa: E402
from aic.retrieval.text_retriever import TextRetriever  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


@pytest.fixture
def tiny_faiss(tmp_path):
    """Index 6 vector, 2 video, mỗi video 3 keyframe với frame_idx thật."""
    dim = 4
    vectors = np.eye(6, dim, dtype=np.float32)[:, :dim]
    vectors[4] = [0.5, 0.5, 0.5, 0.5]
    vectors[5] = [0.9, 0.1, 0.0, 0.0]
    vectors = np.array([_unit(v) for v in vectors], dtype=np.float32)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    index_path = tmp_path / "tiny.index"
    faiss.write_index(index, str(index_path))

    # keyframe_num là ordinal 1-based; frame_idx là frame thật, khác hẳn ordinal
    metadata = [
        {"video_id": "L21_V001", "keyframe_num": 1, "frame_idx": 0},
        {"video_id": "L21_V001", "keyframe_num": 2, "frame_idx": 90},
        {"video_id": "L21_V001", "keyframe_num": 3, "frame_idx": 261},
        {"video_id": "L22_V002", "keyframe_num": 1, "frame_idx": 0},
        {"video_id": "L22_V002", "keyframe_num": 2, "frame_idx": 125},
        {"video_id": "L22_V002", "keyframe_num": 3, "frame_idx": 400},
    ]
    meta_path = tmp_path / "tiny_meta.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    return index_path, meta_path, vectors, metadata


@pytest.fixture
def recording_encoder():
    """Encoder ghi lại text được truyền vào, trả vector cố định."""
    seen: list[str] = []

    def encode(text: str) -> np.ndarray:
        seen.append(text)
        return _unit(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    encode.seen = seen  # type: ignore[attr-defined]
    return encode


@pytest.fixture
def tiny_text_index(tmp_path):
    """Pickle BM25 nhỏ, cùng schema với text_search_index.pkl thật."""
    documents = [
        {
            "type": "transcript_segment",
            "video_id": "L21_V001",
            "text": "một người đàn ông nấu ăn trong bếp",
            "start_time": 3.0,
            "end_time": 6.0,
            "language": "vi",
        },
        {
            "type": "transcript_segment",
            "video_id": "L21_V001",
            "text": "cảnh sát giao thông điều tiết trên đường",
            "start_time": 60.0,
            "end_time": 63.0,
            "language": "vi",
        },
        {
            "type": "transcript_segment",
            "video_id": "L22_V002",
            "text": "đà lạt vào mùa hoa nở rất đẹp",
            "start_time": 10.0,
            "end_time": 14.0,
            "language": "vi",
        },
        {
            "type": "media_info",
            "video_id": "L22_V002",
            "text": "bản tin du lịch đà lạt",
            "start_time": 0,
            "end_time": 100,
            "language": None,
        },
        {
            # OCR biết chính xác keyframe nào, nên map frame là tra bảng chứ
            # không phải suy từ mốc thời gian.
            "type": "ocr",
            "video_id": "L21_V001",
            "text": "chào mừng quý khách đến ga hà nội",
            "keyframe_num": 3,
            "frame_idx": 1800,
            "start_time": 60.0,
            "end_time": 60.0,
            "language": "vi",
        },
    ]
    tokenized = [
        ["một", "người", "đàn_ông", "nấu_ăn", "trong", "bếp"],
        ["cảnh_sát", "giao_thông", "điều_tiết", "trên", "đường"],
        ["đà_lạt", "vào", "mùa", "hoa", "nở", "rất", "đẹp"],
        ["bản_tin", "du_lịch", "đà_lạt"],
        ["chào_mừng", "quý_khách", "đến", "ga", "hà_nội"],
    ]
    keyframe_map = {
        "L21_V001": [
            {"kf_num": 1, "frame_idx": 0, "pts_time": 0.0},
            {"kf_num": 2, "frame_idx": 90, "pts_time": 3.0},
            {"kf_num": 3, "frame_idx": 1800, "pts_time": 60.0},
        ],
        "L22_V002": [
            {"kf_num": 1, "frame_idx": 0, "pts_time": 0.0},
            {"kf_num": 2, "frame_idx": 250, "pts_time": 10.0},
        ],
    }
    path = tmp_path / "text_search_index.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "documents": documents,
                "tokenized": tokenized,
                "keyframe_map": keyframe_map,
            },
            handle,
        )
    return path


def make_candidate(video_id: str, frame: int, source: str, score: float) -> Candidate:
    return Candidate(
        video_id=video_id,
        start_frame=frame,
        end_frame=frame,
        representative_frames=[frame],
        scores={source: score},
    )


# ---------------------------------------------------------------------------
# 1. Retriever nhận limit và exclude nhất quán
# ---------------------------------------------------------------------------


class TestRetrieverInterface:
    def test_faiss_search_accepts_limit_keyword(self, tiny_faiss, recording_encoder):
        index_path, meta_path, _, _ = tiny_faiss
        retriever = FaissRetriever(index_path, meta_path, recording_encoder, name="t")

        results = retriever.search(Query("q1", "xin chào"), limit=2)

        assert len(results) == 2

    def test_faiss_search_k_shim_still_supported(self, tiny_faiss, recording_encoder):
        """Shim tương thích ngược: ``k`` vẫn dùng được và tương đương ``limit``."""
        index_path, meta_path, _, _ = tiny_faiss
        retriever = FaissRetriever(index_path, meta_path, recording_encoder, name="t")

        by_k = retriever.search(Query("q1", "xin chào"), k=3)
        by_limit = retriever.search(Query("q1", "xin chào"), limit=3)

        assert len(by_k) == 3
        assert [c.start_frame for c in by_k] == [c.start_frame for c in by_limit]

    def test_faiss_search_positional_limit(self, tiny_faiss, recording_encoder):
        index_path, meta_path, _, _ = tiny_faiss
        retriever = FaissRetriever(index_path, meta_path, recording_encoder, name="t")

        assert len(retriever.search(Query("q1", "xin chào"), 2)) == 2

    def test_faiss_exclude_removes_video(self, tiny_faiss, recording_encoder):
        index_path, meta_path, _, _ = tiny_faiss
        retriever = FaissRetriever(index_path, meta_path, recording_encoder, name="t")

        results = retriever.search(
            Query("q1", "xin chào"), limit=10, exclude=frozenset({"L21_V001"})
        )

        assert results
        assert all(c.video_id != "L21_V001" for c in results)

    def test_text_search_accepts_limit_and_exclude(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index)
        query = Query("q1", "đà lạt")

        results = retriever.search(query, limit=5, exclude=frozenset({"L22_V002"}))

        assert all(c.video_id != "L22_V002" for c in results)

    def test_text_search_k_shim_still_supported(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index)
        query = Query("q1", "người đàn ông nấu ăn")

        by_k = retriever.search(query, k=2)
        by_limit = retriever.search(query, limit=2)

        assert len(by_k) <= 2
        assert [c.video_id for c in by_k] == [c.video_id for c in by_limit]


# ---------------------------------------------------------------------------
# 2. CLIP dùng query.text_en khi có
# ---------------------------------------------------------------------------


class TestQueryTextSelection:
    def test_faiss_prefers_text_en(self, tiny_faiss, recording_encoder):
        index_path, meta_path, _, _ = tiny_faiss
        retriever = FaissRetriever(index_path, meta_path, recording_encoder, name="t")
        query = Query("q1", text_vi="người đàn ông nấu ăn", text_en="a man cooking")

        retriever.search(query, limit=1)

        assert recording_encoder.seen == ["a man cooking"]

    def test_faiss_falls_back_to_text_vi(self, tiny_faiss, recording_encoder):
        index_path, meta_path, _, _ = tiny_faiss
        retriever = FaissRetriever(index_path, meta_path, recording_encoder, name="t")

        retriever.search(Query("q1", text_vi="người đàn ông nấu ăn"), limit=1)

        assert recording_encoder.seen == ["người đàn ông nấu ăn"]


# ---------------------------------------------------------------------------
# 3. Candidate giữ actual frame_idx, không lấy keyframe_num
# ---------------------------------------------------------------------------


class TestActualFrameMapping:
    def test_faiss_candidate_uses_frame_idx_not_ordinal(
        self, tiny_faiss, recording_encoder
    ):
        index_path, meta_path, _, metadata = tiny_faiss
        retriever = FaissRetriever(index_path, meta_path, recording_encoder, name="t")

        results = retriever.search(Query("q1", "xin chào"), limit=6)

        real_frames = {m["frame_idx"] for m in metadata}
        for cand in results:
            assert cand.start_frame in real_frames
            assert cand.representative_frames == [cand.start_frame]
        # 261 và 400 chỉ tồn tại nếu dùng frame_idx; ordinal cao nhất chỉ là 3
        assert {c.start_frame for c in results} == real_frames

    def test_faiss_rejects_metadata_without_frame_idx(self, tmp_path, recording_encoder):
        index = faiss.IndexFlatIP(4)
        index.add(np.eye(2, 4, dtype=np.float32))
        index_path = tmp_path / "bad.index"
        faiss.write_index(index, str(index_path))
        meta_path = tmp_path / "bad_meta.json"
        meta_path.write_text(
            json.dumps([{"video_id": "L21_V001", "keyframe_num": 1}] * 2),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="frame_idx"):
            FaissRetriever(index_path, meta_path, recording_encoder, name="t")

    def test_bm25_maps_timestamp_to_actual_frame(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index)

        results = retriever.search(Query("q1", "cảnh sát giao thông"), limit=5)

        assert results
        top = results[0]
        assert top.video_id == "L21_V001"
        # start_time 60.0 → keyframe pts_time 60.0 → frame_idx 1800 (không phải kf_num 3)
        assert top.start_frame == 1800

    def test_bm25_never_uses_keyframe_ordinal_as_frame(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index)

        results = retriever.search(Query("q1", "đà lạt hoa nở"), limit=10)

        assert results
        assert all(c.start_frame in {0, 250, 90, 1800} for c in results)
        assert any(c.start_frame == 250 for c in results)

    def test_bm25_skips_video_without_keyframe_map(self, tmp_path):
        path = tmp_path / "no_map.pkl"
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "documents": [
                        {
                            "type": "transcript_segment",
                            "video_id": "L99_V999",
                            "text": "không có keyframe map",
                            "start_time": 5.0,
                            "end_time": 7.0,
                        }
                    ],
                    "tokenized": [["không", "có", "keyframe", "map"]],
                    "keyframe_map": {},
                },
                handle,
            )
        retriever = TextRetriever(path)

        results = retriever.search(Query("q1", "keyframe map"), limit=5)

        # Không map được frame thật → không được bịa ra candidate
        assert results == []
        assert retriever.unmapped_documents > 0


# ---------------------------------------------------------------------------
# 4. BM25 load được index path hiện tại
# ---------------------------------------------------------------------------


class TestBm25Loading:
    def test_constructor_stores_index_path_before_use(self, tiny_text_index):
        """Bug cũ: __init__ dùng self.index_path trước khi gán → AttributeError."""
        retriever = TextRetriever(tiny_text_index)

        assert retriever.index_path == Path(tiny_text_index)
        assert retriever.num_documents == 5

    def test_modality_filter_selects_documents(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index, modalities=["media_info"])

        assert retriever.num_documents == 1
        results = retriever.search(Query("q1", "du lịch đà lạt"), limit=5)
        assert results
        assert all("media_info_match" in c.evidence for c in results)

    def test_evidence_is_modality_specific(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index)

        results = retriever.search(Query("q1", "nấu ăn trong bếp"), limit=5)

        assert results
        assert "transcript_match" in results[0].evidence
        assert "start_time" in results[0].evidence

    def test_stopwords_keep_meaningful_words(self, tiny_text_index):
        """Stopword chỉ được ăn hư từ; từ mang nghĩa phải sống sót."""
        from aic.retrieval.text_retriever import STOPWORDS

        meaningful = [
            "người", "đàn", "ông", "xe", "máy", "đỏ", "cháy", "cảnh", "sát",
            "giao", "thông", "hoa", "bếp", "nấu", "hai", "ba", "trắng", "chó",
        ]
        assert [w for w in meaningful if w in STOPWORDS] == []

    def test_query_expansion_matches_segmented_compounds(self, tiny_text_index):
        """Index đã word-segment nên query rời phải sinh n-gram nối bằng '_'."""
        retriever = TextRetriever(tiny_text_index)

        terms = retriever.expand_query_terms("cảnh sát giao thông")

        assert "giao_thông" in terms
        assert "cảnh_sát" in terms

    def test_query_expansion_maps_unaccented_to_accented(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index)

        terms = retriever.expand_query_terms("da lat")

        assert "đà_lạt" in terms

    def test_unaccented_query_finds_same_video(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index)

        accented = retriever.search(Query("q1", "đà lạt"), limit=5)
        unaccented = retriever.search(Query("q2", "da lat"), limit=5)

        assert accented and unaccented
        assert {c.video_id for c in unaccented} == {c.video_id for c in accented}

    def test_bm25_uses_real_document_length(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index)

        assert retriever.doc_lengths == [6, 5, 7, 3, 5]
        assert retriever.avgdl == pytest.approx(26 / 5)

    def test_short_exact_document_outranks_long_diluted_one(self, tmp_path):
        """Kiểm tra thứ hạng BM25 trên corpus nhỏ biết trước đáp án."""
        documents = [
            {"type": "transcript_segment", "video_id": "L21_V001",
             "text": "hỏa hoạn", "start_time": 0.0, "end_time": 1.0},
            {"type": "transcript_segment", "video_id": "L22_V002",
             "text": "hỏa hoạn kèm rất nhiều từ khác", "start_time": 0.0,
             "end_time": 1.0},
        ]
        tokenized = [
            ["hỏa_hoạn"],
            ["hỏa_hoạn"] + [f"chữ{i}" for i in range(40)],
        ]
        keyframe_map = {
            "L21_V001": [{"kf_num": 1, "frame_idx": 0, "pts_time": 0.0}],
            "L22_V002": [{"kf_num": 1, "frame_idx": 0, "pts_time": 0.0}],
        }
        path = tmp_path / "rank.pkl"
        with path.open("wb") as handle:
            pickle.dump(
                {"documents": documents, "tokenized": tokenized,
                 "keyframe_map": keyframe_map},
                handle,
            )
        retriever = TextRetriever(path)

        results = retriever.search(Query("q1", "hỏa hoạn"), limit=5)

        assert [c.video_id for c in results] == ["L21_V001", "L22_V002"]

    def test_ocr_document_maps_via_keyframe_num(self, tiny_text_index):
        """OCR gắn với đúng một keyframe → map frame bằng tra bảng, chính xác."""
        retriever = TextRetriever(tiny_text_index)

        results = retriever.search(Query("q1", "ga hà nội"), limit=5)

        assert results
        top = results[0]
        assert top.video_id == "L21_V001"
        assert top.start_frame == 1800
        assert "ocr_match" in top.evidence

    def test_ocr_modality_filter(self, tiny_text_index):
        retriever = TextRetriever(tiny_text_index, modalities=["ocr"])

        assert retriever.num_documents == 1
        results = retriever.search(Query("q1", "quý khách ga hà nội"), limit=5)
        assert results
        assert all(c.evidence.get("doc_type") == "ocr" for c in results)

    def test_ocr_ignores_stale_frame_idx_in_document(self, tmp_path):
        """``keyframe_num`` là nguồn đúng, thắng cả ``frame_idx`` ghi sẵn lẫn
        mốc thời gian trong document."""
        path = tmp_path / "stale.pkl"
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "documents": [
                        {
                            "type": "ocr",
                            "video_id": "L21_V001",
                            "text": "biển hiệu đà lạt",
                            "keyframe_num": 2,
                            "frame_idx": 999999,  # sai, cố tình
                            # mốc thời gian cố tình trỏ sang keyframe 1: chỉ
                            # đường keyframe_num mới cho ra frame đúng
                            "start_time": 0.0,
                            "end_time": 0.0,
                        }
                    ],
                    "tokenized": [["biển_hiệu", "đà_lạt"]],
                    "keyframe_map": {
                        "L21_V001": [
                            {"kf_num": 1, "frame_idx": 0, "pts_time": 0.0},
                            {"kf_num": 2, "frame_idx": 90, "pts_time": 3.0},
                        ]
                    },
                },
                handle,
            )
        retriever = TextRetriever(path)

        results = retriever.search(Query("q1", "đà lạt"), limit=5)

        assert results
        assert results[0].start_frame == 90

    def test_real_index_loads_when_present(self):
        raw = os.environ.get("AIC_TEXT_INDEX_PATH")
        if not raw:
            pytest.skip("AIC_TEXT_INDEX_PATH chưa cấu hình")
        path = Path(raw)
        if not path.exists():
            pytest.skip(f"index thật không tồn tại: {path}")

        retriever = TextRetriever(path)

        assert retriever.num_documents > 0
        assert retriever.keyframe_map
        results = retriever.search(Query("q1", "giao thông"), limit=5)
        assert results
        for cand in results:
            frames = {
                entry["frame_idx"] for entry in retriever.keyframe_map[cand.video_id]
            }
            assert cand.start_frame in frames


class TestRealVectorIndexes:
    """Chạy trên index thật khi có cấu hình; bỏ qua nếu máy không có artifacts."""

    @staticmethod
    def _stub_encoder(dim: int):
        def encode(text: str) -> np.ndarray:
            rng = np.random.RandomState(abs(hash(text)) % (2**31))
            return _unit(rng.randn(dim).astype(np.float32))

        return encode

    def _paths(self, index_env: str, meta_env: str):
        index_raw = os.environ.get(index_env)
        meta_raw = os.environ.get(meta_env)
        if not index_raw or not meta_raw:
            pytest.skip(f"{index_env}/{meta_env} chưa cấu hình")
        index_path, meta_path = Path(index_raw), Path(meta_raw)
        if not index_path.exists() or not meta_path.exists():
            pytest.skip("index thật không tồn tại")
        return index_path, meta_path

    def test_clip_index_matches_encoder_dimension(self):
        index_path, meta_path = self._paths("AIC_INDEX_PATH", "AIC_META_PATH")
        retriever = FaissRetriever(
            index_path, meta_path, self._stub_encoder(512), name="clip",
            expected_dim=512,
        )
        assert retriever.dim == 512
        assert retriever.num_vectors == len(retriever.metadata)

    def test_siglip_index_matches_encoder_dimension(self):
        index_path, meta_path = self._paths(
            "AIC_SIGLIP_INDEX_PATH", "AIC_SIGLIP_META_PATH"
        )
        retriever = FaissRetriever(
            index_path, meta_path, self._stub_encoder(1152), name="siglip",
            expected_dim=1152,
        )
        assert retriever.dim == 1152
        results = retriever.search(Query("q1", text_vi="thử", text_en="a test"), limit=5)
        assert len(results) == 5
        real_frames = {m["frame_idx"] for m in retriever.metadata}
        assert all(c.start_frame in real_frames for c in results)

    def test_wrong_dimension_encoder_is_rejected(self):
        """Bẫy chính: model 768 chiều (google/siglip2-base) trên index 1152."""
        index_path, meta_path = self._paths(
            "AIC_SIGLIP_INDEX_PATH", "AIC_SIGLIP_META_PATH"
        )
        with pytest.raises(ValueError, match="embedding space"):
            FaissRetriever(
                index_path, meta_path, self._stub_encoder(768), name="siglip",
                expected_dim=768,
            )


# ---------------------------------------------------------------------------
# 5–7. Fusion: RRF, đa moment, không padding giả
# ---------------------------------------------------------------------------


class TestFusion:
    def test_keeps_two_moments_from_same_video(self):
        early = make_candidate("L21_V001", 100, "clip", 0.8)
        late = make_candidate("L21_V001", 5000, "clip", 0.7)

        result = fuse([[early, late]], limit=10)

        assert [c.start_frame for c in result] == [100, 5000]

    def test_merges_identical_moment_across_sources(self):
        from_clip = make_candidate("L21_V001", 100, "clip", 0.8)
        from_siglip = make_candidate("L21_V001", 100, "siglip", 0.6)

        result = fuse([[from_clip], [from_siglip]], limit=10)

        assert len(result) == 1
        assert result[0].scores["clip"] == 0.8
        assert result[0].scores["siglip"] == 0.6

    def test_merge_radius_groups_nearby_frames(self):
        first = make_candidate("L21_V001", 100, "clip", 0.8)
        near = make_candidate("L21_V001", 105, "siglip", 0.6)

        assert len(fuse([[first], [near]], limit=10, merge_radius=0)) == 2
        assert len(fuse([[first], [near]], limit=10, merge_radius=10)) == 1

    def test_uses_reciprocal_rank_fusion_formula(self):
        top = make_candidate("L21_V001", 10, "clip", 0.9)
        second = make_candidate("L22_V002", 20, "clip", 0.1)

        result = fuse([[top, second]], limit=10)

        assert result[0].scores["fused"] == pytest.approx(1.0 / (DEFAULT_RRF_K + 1))
        assert result[1].scores["fused"] == pytest.approx(1.0 / (DEFAULT_RRF_K + 2))

    def test_rrf_beats_raw_score_scale_mismatch(self):
        """Nguồn điểm thô lớn không được tự động thắng nếu thứ hạng thấp."""
        big_scale = [
            make_candidate("L21_V001", 10, "bm25", 900.0),
            make_candidate("L22_V002", 20, "bm25", 800.0),
        ]
        small_scale = [
            make_candidate("L22_V002", 20, "clip", 0.31),
            make_candidate("L21_V001", 10, "clip", 0.30),
        ]

        result = fuse([big_scale, small_scale], limit=10)

        # Cả hai đều rank 1 ở một nguồn và rank 2 ở nguồn kia → hoà điểm RRF
        assert result[0].scores["fused"] == pytest.approx(result[1].scores["fused"])

    def test_weights_are_applied_per_source(self):
        runs = [
            [make_candidate("L21_V001", 10, "clip", 0.9)],
            [make_candidate("L22_V002", 20, "bm25", 0.9)],
        ]

        result = fuse(runs, limit=10, weights={"clip": 2.0, "bm25": 1.0})

        assert result[0].video_id == "L21_V001"
        assert result[0].scores["fused"] == pytest.approx(2.0 / (DEFAULT_RRF_K + 1))
        assert result[1].scores["fused"] == pytest.approx(1.0 / (DEFAULT_RRF_K + 1))

    def test_is_deterministic(self):
        runs = [
            [make_candidate("L21_V001", 10, "clip", 0.9)],
            [make_candidate("L22_V002", 20, "siglip", 0.9)],
        ]

        first = fuse([list(r) for r in runs], limit=10)
        second = fuse([list(r) for r in runs], limit=10)

        assert [(c.video_id, c.start_frame, c.scores["fused"]) for c in first] == [
            (c.video_id, c.start_frame, c.scores["fused"]) for c in second
        ]

    def test_never_pads_with_fake_candidates(self):
        only = make_candidate("L21_V001", 10, "clip", 0.9)

        result = fuse([[only]], limit=100)

        assert len(result) == 1
        assert all(not c.video_id.startswith("L00_V") for c in result)

    def test_honours_limit(self):
        run = [make_candidate("L21_V%03d" % i, i * 10, "clip", 1.0) for i in range(1, 8)]

        assert len(fuse([run], limit=3)) == 3

    def test_empty_runs_return_empty(self):
        assert fuse([], limit=100) == []
        assert fuse([[], []], limit=100) == []


# ---------------------------------------------------------------------------
# 8. Không silent fallback sang dummy khi AIC_USE_DUMMY=0
# ---------------------------------------------------------------------------


class TestNoSilentDummyFallback:
    def _build(self, **kwargs):
        from aic.ui.app import build_retriever_registry

        defaults = dict(
            use_dummy=False,
            clip_index=Path("/nonexistent/clip.index"),
            clip_meta=Path("/nonexistent/clip.json"),
            siglip_index=None,
            siglip_meta=None,
            text_index=Path("/nonexistent/text.pkl"),
            dummy_module=object(),
            disable_neural=True,
        )
        defaults.update(kwargs)
        return build_retriever_registry(**defaults)

    def test_missing_indexes_report_error_not_dummy(self):
        retrievers, statuses = self._build()

        assert retrievers == []
        by_name = {s["name"]: s for s in statuses}
        assert by_name["clip"]["state"] == "error"
        assert by_name["bm25"]["state"] == "error"
        assert all(s["state"] != "ready" for s in statuses)
        assert "dummy" not in by_name

    def test_error_status_explains_reason(self):
        _, statuses = self._build()

        clip = next(s for s in statuses if s["name"] == "clip")
        assert clip["error"]
        assert "clip.index" in clip["detail"]

    def test_dummy_only_when_explicitly_enabled(self):
        sentinel = object()
        retrievers, statuses = self._build(use_dummy=True, dummy_module=sentinel)

        assert retrievers == [sentinel]
        assert statuses[0]["name"] == "dummy"
        assert statuses[0]["state"] == "ready"

    def test_siglip_disabled_when_not_configured(self):
        _, statuses = self._build()

        siglip = next(s for s in statuses if s["name"] == "siglip")
        assert siglip["state"] == "disabled"

    def test_working_source_survives_broken_sibling(self, tiny_text_index):
        retrievers, statuses = self._build(text_index=tiny_text_index)

        by_name = {s["name"]: s for s in statuses}
        assert by_name["bm25"]["state"] == "ready"
        assert by_name["clip"]["state"] == "error"
        assert len(retrievers) == 1
