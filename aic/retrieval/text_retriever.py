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

        all_documents = data["documents"]
        all_tokenized = data["tokenized"]
        self.keyframe_map = data.get("keyframe_map", {})

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
        
        logger.info("  → %d text documents loaded", len(self.documents))
        
        bm25_cache_path = self.index_path.with_name("bm25_index.pkl")
        if bm25_cache_path.exists():
            logger.info("Loading pre-built BM25 index from %s", bm25_cache_path)
            with open(bm25_cache_path, "rb") as f:
                bm25_data = pickle.load(f)
                self.doc_lengths = bm25_data["doc_lengths"]
                self.avgdl = bm25_data["avgdl"]
                self.idf = bm25_data["idf"]
                self.inverted = bm25_data["inverted"]
                self.N = len(self.documents)
            logger.info("  → BM25 index loaded instantly.")
        else:
            self._build_bm25()
            
            logger.info("Saving BM25 index to %s", bm25_cache_path)
            try:
                with open(bm25_cache_path, "wb") as f:
                    pickle.dump({
                        "doc_lengths": self.doc_lengths,
                        "avgdl": self.avgdl,
                        "idf": self.idf,
                        "inverted": dict(self.inverted)
                    }, f)
            except Exception as e:
                logger.warning("Failed to save BM25 cache: %s", e)

    def _build_bm25(self):
        """Pre-compute IDF and document lengths for BM25."""
        logger.info("Building BM25 index from scratch (this may take ~50s)...")
        self.N = len(self.documents)
        self.doc_lengths = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)

        # Document frequency: how many docs contain each term
        self.df = Counter()
        for tokens in self.tokenized:
            self.df.update(set(tokens))

        # Compute IDF
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

        # Build inverted index
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, tokens in enumerate(self.tokenized):
            tf = Counter(tokens)
            for term, count in tf.items():
                self.inverted[term].append((doc_idx, count))

        logger.info("  → BM25 index built: %d unique terms", len(self.df))

    def search(
        self,
        query: Query,
        k: int = 100,
        exclude: frozenset = frozenset(),
    ) -> list[Candidate]:
        """BM25 search, trả về top-k Candidate theo score giảm dần."""
        # Tokenize query (song ngữ + từ mở rộng + không dấu)
        if hasattr(query, "for_bm25"):
            query_text = query.for_bm25().lower()
        else:
            text_vi = query.for_text().lower()
            text_en = query.for_clip().lower()
            query_text = f"{text_vi} {text_en}".strip()
        
        query_tokens = tokenize_bilingual(query_text)

        if not query_tokens:
            return []

        # Accumulate scores for each document efficiently
        scores = defaultdict(float)
        k1 = 1.5
        b = 0.75
        avgdl = self.avgdl
        
        for qt in query_tokens:
            if qt not in self.inverted:
                continue
            idf = self.idf.get(qt, 0.0)
            if idf <= 0.0:
                continue
                
            for doc_idx, count in self.inverted[qt]:
                dl = self.doc_lengths[doc_idx]
                tf_norm = (count * (k1 + 1)) / (count + k1 * (1 - b + b * dl / avgdl))
                scores[doc_idx] += idf * tf_norm

        # Group by video, keep best score per video
        video_best: dict[str, tuple[float, dict]] = {}
        for doc_idx, score in scores.items():
            doc = self.documents[doc_idx]
            vid = doc.get("video_id", "")
            if not vid or vid in exclude:
                continue
            if vid not in video_best or score > video_best[vid][0]:
                video_best[vid] = (score, doc)

        if not video_best:
            return []

        # Normalize scores to [0, 1]
        max_score = max(s for s, _ in video_best.values())
        
        candidates = []
        for vid, (score, doc) in sorted(video_best.items(), key=lambda x: x[1][0], reverse=True):
            norm_score = score / max_score if max_score > 0 else 0.0

            kf_num = doc.get("keyframe_num", 0)
            frame_idx = kf_num

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

        return candidates[:k]



    @property
    def num_documents(self) -> int:
        return len(self.documents)


def build_text_retriever(
    index_dir: str | Path = "local/index",
    name: str = "bm25",
    doc_types: list[str] | None = None,
) -> TextRetriever:
    """Factory function — tạo TextRetriever từ thư mục index."""
    index_dir = Path(index_dir)
    return TextRetriever(
        index_path=index_dir / "text_search_index.pkl",
        name=name,
        doc_types=doc_types,
    )
