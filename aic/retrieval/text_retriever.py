"""BM25-based text retriever — search trên ASR + OCR + Caption gộp.

Dùng file text_search_index.pkl đã rebuild (451K documents).
Trả về list[Candidate] cùng interface với FaissRetriever.
"""

from __future__ import annotations

import logging
import pickle
import math
from collections import Counter, defaultdict
from pathlib import Path

from ..core.types import Candidate, Query

logger = logging.getLogger(__name__)


def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt (ví dụ: 'Đà Lạt' -> 'Da Lat')."""
    import unicodedata
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    res = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return res.replace("đ", "d").replace("Đ", "d")


def tokenize_bilingual(text: str) -> list[str]:
    """Tokenize sinh cả token có dấu và token không dấu."""
    import re
    text_lower = text.lower().strip()
    tokens = re.findall(r'[\w]+', text_lower, re.UNICODE)
    unaccented_text = remove_accents(text_lower)
    if unaccented_text != text_lower:
        unaccented_tokens = re.findall(r'[\w]+', unaccented_text, re.UNICODE)
        tokens.extend(unaccented_tokens)
    return tokens


class TextRetriever:
    """BM25 retriever trên unified text search index.

    Parameters
    ----------
    index_path : đường dẫn tới text_search_index.pkl
    name : tên retriever, ghi vào Candidate.scores
    doc_types : lọc chỉ search trên các loại document cụ thể
                (None = search tất cả). Ví dụ: ["ocr", "caption"]
    """

    def __init__(
        self,
        index_path: str | Path,
        name: str = "bm25",
        doc_types: list[str] | None = None,
    ):
        self.name = name

        logger.info("Loading text search index from %s", index_path)
        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.keyframe_map = data.get("keyframe_map", {})
        all_documents = data["documents"]

        # 1. Fast path: Đã được tính toán sẵn inverted index và idf
        if "inverted" in data and "idf" in data and not doc_types:
            self.documents = all_documents
            self.inverted = data["inverted"]
            self.idf = data["idf"]
            self.N = data.get("N", len(all_documents))
            self.avgdl = data.get("avgdl", 10.0)
            self.tokenized = []
            logger.info("  ✓ Instant loaded precomputed BM25 index (%d documents, %d unique terms)",
                        self.N, len(self.idf))
            return

        all_tokenized = data.get("tokenized", [])

        # Filter by doc_types if specified
        if doc_types is not None:
            self.documents = []
            self.tokenized = []
            for doc, tok in zip(all_documents, all_tokenized):
                if doc.get("type") in doc_types:
                    self.documents.append(doc)
                    self.tokenized.append(tok)
        else:
            self.documents = all_documents
            self.tokenized = all_tokenized

        logger.info("  → %d documents loaded (types: %s)",
                     len(self.documents),
                     doc_types or "all")

        # Pre-compute BM25 statistics
        self._build_bm25()

    def _build_bm25(self):
        """Pre-compute IDF and document lengths for BM25."""
        self.N = len(self.documents)
        self.avgdl = sum(len(t) for t in self.tokenized) / max(self.N, 1)

        # Document frequency: how many docs contain each term
        self.df = Counter()
        for tokens in self.tokenized:
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] += 1

        # Pre-compute IDF
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

        # Inverted index for fast lookup
        self.inverted = defaultdict(list)
        for i, tokens in enumerate(self.tokenized):
            tf = Counter(tokens)
            for term, count in tf.items():
                self.inverted[term].append((i, count))

        logger.info("  → BM25 index built: %d unique terms", len(self.df))

    def _bm25_score(self, query_tokens: list[str], doc_idx: int, k1=1.5, b=0.75) -> float:
        """Compute BM25 score for a single document."""
        doc_tokens = self.tokenized[doc_idx]
        dl = len(doc_tokens)
        tf = Counter(doc_tokens)

        score = 0.0
        for qt in query_tokens:
            if qt not in self.idf:
                continue
            term_freq = tf.get(qt, 0)
            if term_freq == 0:
                continue
            idf = self.idf[qt]
            tf_norm = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * dl / self.avgdl))
            score += idf * tf_norm

        return score

    def search(
        self,
        query: Query,
        k: int = 100,
        exclude: frozenset = frozenset(),
    ) -> list[Candidate]:
        """BM25 search siêu tốc qua Inverted Index, trả về top-k Candidate."""
        # 1. Tokenize query
        if hasattr(query, "for_bm25"):
            query_text = query.for_bm25().lower()
        else:
            text_vi = query.for_text().lower()
            text_en = query.for_clip().lower()
            query_text = f"{text_vi} {text_en}".strip()

        query_tokens = tokenize_bilingual(query_text)
        if not query_tokens:
            return []

        # 2. Tích lũy điểm BM25 trực tiếp từ Inverted Index
        k1 = 1.5
        b = 0.75
        doc_scores = defaultdict(float)

        for qt in query_tokens:
            if qt not in self.inverted or qt not in self.idf:
                continue
            idf = self.idf[qt]
            for doc_idx, term_freq in self.inverted[qt]:
                tf_norm = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b))
                doc_scores[doc_idx] += idf * tf_norm

        if not doc_scores:
            return []

        # 3. Gom nhóm theo video_id, chọn document có điểm cao nhất mỗi video
        video_best: dict[str, tuple[float, dict]] = {}
        for doc_idx, score in doc_scores.items():
            if score <= 0 or doc_idx >= len(self.documents):
                continue
            doc = self.documents[doc_idx]
            vid = doc.get("video_id", "")
            if not vid or vid in exclude:
                continue
            if vid not in video_best or score > video_best[vid][0]:
                video_best[vid] = (score, doc)

        if not video_best:
            return []

        # 4. Chuẩn hóa điểm [0, 1] và sắp xếp giảm dần
        max_score = max(s for s, _ in video_best.values()) if video_best else 1.0
        sorted_videos = sorted(video_best.items(), key=lambda x: x[1][0], reverse=True)[:k]

        candidates = []
        for vid, (score, doc) in sorted_videos:
            norm_score = score / max_score if max_score > 0 else 0.0

            kf_num = doc.get("keyframe_num", 0)
            # Nếu keyframe_num = 0 nhưng có start_time (transcript), tự động ánh xạ sang keyframe gần nhất
            if kf_num == 0 and ("start_time" in doc or "pts_time" in doc):
                st = doc.get("start_time") or doc.get("pts_time", 0)
                sec = 0.0
                if isinstance(st, (int, float)):
                    sec = float(st)
                elif isinstance(st, str) and st:
                    parts = st.split(":")
                    try:
                        if len(parts) == 3:
                            sec = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                        elif len(parts) == 2:
                            sec = float(parts[0]) * 60 + float(parts[1])
                        else:
                            sec = float(st)
                    except Exception:
                        sec = 0.0

                kfs = self.keyframe_map.get(vid, [])
                if kfs:
                    closest = min(kfs, key=lambda x: abs(x.get("pts_time", 0.0) - sec))
                    kf_num = closest.get("kf_num", closest.get("frame_idx", 1))

            if kf_num == 0:
                kf_num = 1  # Fallback to frame 1

            frame_idx = kf_num

            # Build evidence
            evidence = {}
            doc_type = doc.get("type", "")
            text_snippet = doc.get("text", "")[:200]
            if doc_type == "ocr":
                evidence["ocr_match"] = text_snippet
            elif doc_type == "caption":
                evidence["caption_match"] = text_snippet
            elif doc_type in ("transcript_segment", "transcript_full"):
                evidence["transcript_match"] = text_snippet

            candidates.append(
                Candidate(
                    video_id=vid,
                    start_frame=frame_idx,
                    end_frame=frame_idx,
                    representative_frames=[frame_idx],
                    scores={self.name: round(norm_score, 6)},
                    evidence=evidence,
                )
            )

        return candidates

    @property
    def num_documents(self) -> int:
        return len(self.documents)


def build_text_retriever(
    index_path: str | Path = "local/text_search_index.pkl",
    name: str = "bm25",
    doc_types: list[str] | None = None,
) -> TextRetriever:
    """Factory function — tạo TextRetriever từ file hoặc thư mục."""
    p = Path(index_path)
    if p.is_dir():
        p = p / "text_search_index.pkl"
    return TextRetriever(
        index_path=p,
        name=name,
        doc_types=doc_types,
    )
