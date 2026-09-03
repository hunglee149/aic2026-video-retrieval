"""BM25 retriever trên unified text search index (ASR / OCR / caption / metadata).

Schema pickle được hỗ trợ::

    {
      "documents":  [{"type", "video_id", "text", "start_time", "end_time", ...}],
      "tokenized":  [[token, ...], ...],          # khớp 1:1 với documents
      "keyframe_map": {video_id: [{"kf_num", "frame_idx", "pts_time"}, ...]},
      # tuỳ chọn — nếu có thì dùng luôn, khỏi build lại:
      "inverted": {...}, "idf": {...}, "avgdl": float, "N": int
    }

Hai điểm cần biết về frame:

- Document text **không có frame index**; nó có mốc thời gian (giây). Frame thật
  chỉ ra được qua ``keyframe_map``. Không map được thì bỏ document đó, tuyệt đối
  không lấy ``keyframe_num``/ordinal làm frame nộp bài.
- Frame trả ra luôn là actual video frame 0-based, và luôn rơi đúng vào một
  keyframe có thật nên UI mở được ảnh tương ứng.

Về tokenize: index tiếng Việt đã qua word-segmentation nên chứa token ghép kiểu
``giao_thông``. Query người dùng gõ rời nên phải sinh thêm n-gram nối bằng ``_``
thì mới khớp được, đồng thời map token không dấu về đúng token có dấu trong vocab.
"""

from __future__ import annotations

import logging
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from ..core.types import Candidate, Query

logger = logging.getLogger(__name__)

# Document cấp video: mốc thời gian trải cả video nên không định vị được moment.
# Neo về keyframe đầu tiên thay vì bịa ra một điểm giữa vô nghĩa.
VIDEO_LEVEL_TYPES = frozenset({"media_info", "transcript_full", "caption_summary"})

# Document neo vào đúng một keyframe: frame lấy bằng tra bảng ``kf_num``.
KEYFRAME_LEVEL_TYPES = frozenset({"ocr", "ocr_vi", "ocr_en", "caption",
                                  "caption_vi", "caption_en"})

# Tên modality thân thiện → doc type thật trong index.
MODALITY_ALIASES: dict[str, tuple[str, ...]] = {
    "asr": ("transcript_segment", "transcript_full"),
    "transcript": ("transcript_segment", "transcript_full"),
    "speech": ("transcript_segment", "transcript_full"),
    "ocr": ("ocr", "ocr_vi", "ocr_en"),
    "caption": ("caption", "caption_vi", "caption_en"),
    "summary": ("caption_summary",),
    "meta": ("media_info",),
    "metadata": ("media_info",),
}

# Khoá evidence theo modality, để UI hiển thị đúng loại bằng chứng.
_EVIDENCE_KEYS: dict[str, str] = {
    "transcript_segment": "transcript_match",
    "transcript_full": "transcript_match",
    "ocr": "ocr_match",
    "ocr_vi": "ocr_match",
    "ocr_en": "ocr_match",
    "caption": "caption_match",
    "caption_vi": "caption_match",
    "caption_en": "caption_match",
    "caption_summary": "summary_match",
    "media_info": "media_info_match",
}

# Stopword giữ ở mức vừa phải: chỉ hư từ thuần chức năng. Không đụng vào từ mang
# nghĩa (màu sắc, số đếm, danh từ, động từ) vì query thi đấu sống nhờ chúng.
VI_STOPWORDS = frozenset(
    """
    và của là các một những cho với từ trong ngoài trên dưới khi thì mà nên
    được bị đã đang sẽ rằng ở về theo cùng như hay hoặc nếu vì do bởi tại
    này đó kia ấy nào đây cái sự việc rất quá lắm cũng vẫn còn chỉ đến tới
    """.split()
)

EN_STOPWORDS = frozenset(
    """
    a an the of in on at to for with from by and or but is are was were be
    been being this that these those it its as into about over under then
    there here which who whom whose what when where how
    """.split()
)

STOPWORDS = VI_STOPWORDS | EN_STOPWORDS

# Trần số biến thể có dấu sinh ra từ một token không dấu, tránh query nổ ra
# hàng trăm term nhiễu.
_MAX_ACCENT_VARIANTS = 8

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def remove_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt: 'Đà Lạt' → 'da lat'."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.replace("đ", "d").replace("Đ", "D")


def tokenize(text: str) -> list[str]:
    """Tách từ đơn, lowercase, giữ nguyên dấu."""
    return _TOKEN_RE.findall(text.lower().strip())


def tokenize_bilingual(text: str) -> list[str]:
    """Token có dấu + token không dấu (giữ lại cho code gọi kiểu cũ)."""
    tokens = tokenize(text)
    unaccented = tokenize(remove_accents(text))
    if unaccented != tokens:
        tokens = tokens + unaccented
    return tokens


def _ngrams(tokens: list[str], sizes=(2, 3)) -> list[str]:
    """Nối n-gram bằng '_' để khớp token ghép của index đã segment."""
    out: list[str] = []
    for size in sizes:
        for i in range(len(tokens) - size + 1):
            out.append("_".join(tokens[i : i + size]))
    return out


class TextRetriever:
    """BM25 retriever trên unified text search index.

    Parameters
    ----------
    index_path : đường dẫn tới ``text_search_index.pkl``
    name : tên retriever, ghi vào ``Candidate.scores``
    modalities : lọc theo modality/doc type (None = tất cả)
    per_video_limit : số moment tối đa giữ lại cho mỗi video
    """

    def __init__(
        self,
        index_path: str | Path,
        name: str = "bm25",
        modalities: list[str] | None = None,
        doc_types: list[str] | None = None,
        per_video_limit: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.name = name
        # Gán TRƯỚC mọi chỗ dùng: bản cũ đọc self.index_path trong __init__
        # trước khi gán nên vỡ ngay khi có index thật.
        self.index_path = Path(index_path)
        self.per_video_limit = max(1, per_video_limit)
        self.k1 = k1
        self.b = b
        self.unmapped_documents = 0

        if not self.index_path.exists():
            raise FileNotFoundError(f"Text index không tồn tại: {self.index_path}")

        logger.info("Loading text search index from %s", self.index_path)
        with self.index_path.open("rb") as handle:
            data = pickle.load(handle)

        if "documents" not in data:
            raise ValueError(f"Text index thiếu khoá 'documents': {self.index_path}")

        all_documents = data["documents"]
        all_tokenized = data.get("tokenized")

        # Kiểm tra xem index đã có sẵn precomputed stats (inverted, idf, avgdl, N) chưa
        has_precomputed = (
            ("inverted" in data or "inverted_arrays" in data)
            and "idf" in data
            and "avgdl" in data
        )

        if not has_precomputed:
            if all_tokenized is None:
                raise ValueError(f"Text index thiếu khoá 'tokenized': {self.index_path}")
            if len(all_documents) != len(all_tokenized):
                raise ValueError(
                    f"documents ({len(all_documents)}) != tokenized "
                    f"({len(all_tokenized)}) trong {self.index_path}"
                )

        self.keyframe_map: dict[str, list[dict]] = data.get("keyframe_map", {})
        self.available_types = sorted(
            {doc.get("type", "") for doc in all_documents if doc.get("type")}
        )

        selected = self._resolve_modalities(modalities or doc_types)
        self.modalities = selected
        if selected is None:
            self.documents = all_documents
            self.tokenized = all_tokenized or []
        else:
            self.documents = []
            self.tokenized = []
            if all_tokenized and len(all_tokenized) == len(all_documents):
                for doc, tokens in zip(all_documents, all_tokenized):
                    if doc.get("type") in selected:
                        self.documents.append(doc)
                        self.tokenized.append(tokens)
            else:
                for doc in all_documents:
                    if doc.get("type") in selected:
                        self.documents.append(doc)

        logger.info("  → %d text documents loaded", len(self.documents))

        self._prepare_keyframe_lookup()
        self._load_or_build_bm25(data, used_subset=selected is not None)
        if "accent_index" in data and not (selected is not None):
            self._accent_index = data["accent_index"]
        else:
            self._build_accent_index()

    # ------------------------------------------------------------------
    # Khởi tạo
    # ------------------------------------------------------------------

    def _resolve_modalities(self, requested) -> set[str] | None:
        if not requested:
            return None
        resolved: set[str] = set()
        for item in requested:
            key = str(item).strip().lower()
            if not key:
                continue
            resolved.update(MODALITY_ALIASES.get(key, (key,)))
        return resolved or None

    def _prepare_keyframe_lookup(self) -> None:
        """Sắp keyframe theo pts_time để tra cứu bằng nhị phân."""
        self._kf_times: dict[str, list[float]] = {}
        self._kf_frames: dict[str, list[int]] = {}
        # kf_num → frame thật, cho document biết chính xác nó thuộc keyframe nào
        # (OCR, caption). Tra bảng thế này chính xác tuyệt đối, không như suy từ
        # mốc thời gian.
        self._kf_by_num: dict[str, dict[int, int]] = {}
        for video_id, entries in self.keyframe_map.items():
            ordered = sorted(entries, key=lambda e: float(e.get("pts_time", 0.0)))
            self._kf_times[video_id] = [float(e.get("pts_time", 0.0)) for e in ordered]
            self._kf_frames[video_id] = [int(e["frame_idx"]) for e in ordered]
            self._kf_by_num[video_id] = {
                int(e["kf_num"]): int(e["frame_idx"])
                for e in entries
                if e.get("kf_num") is not None
            }

    def _load_or_build_bm25(self, data: dict, used_subset: bool) -> None:
        """Dùng thống kê precompute nếu có và còn hợp lệ, không thì build."""
        precomputed = (
            not used_subset
            and (("inverted" in data or "inverted_arrays" in data) and "idf" in data and "avgdl" in data)
        )
        if precomputed:
            logger.info("  → dùng inverted index/IDF có sẵn trong pickle")
            self.inverted = data.get("inverted_arrays") or data["inverted"]
            self.idf = data["idf"]
            self.avgdl = float(data["avgdl"])
            self.N = int(data.get("N", len(self.documents)))
            if "doc_lengths" in data and len(data["doc_lengths"]) == self.N:
                self.doc_lengths = data["doc_lengths"]
            elif self.tokenized:
                self.doc_lengths = [len(t) for t in self.tokenized]
            else:
                # Tính doc_lengths từ inverted index nếu thiếu
                doc_lengths = [0] * self.N
                for postings in self.inverted.values():
                    if isinstance(postings, tuple) and len(postings) == 2:
                        doc_ids, counts = postings
                        for doc_idx, count in zip(doc_ids, counts):
                            doc_lengths[doc_idx] += count
                    else:
                        for doc_idx, count in postings:
                            doc_lengths[doc_idx] += count
                self.doc_lengths = doc_lengths
            return
        self._build_bm25()

    def _build_bm25(self) -> None:
        self.N = len(self.documents)
        self.doc_lengths = [len(tokens) for tokens in self.tokenized]
        total_length = sum(self.doc_lengths)
        # avgdl là mẫu số thật của BM25; thay bằng hằng số là sai công thức.
        self.avgdl = total_length / self.N if self.N else 0.0

        document_freq: Counter[str] = Counter()
        for tokens in self.tokenized:
            document_freq.update(set(tokens))

        self.idf = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in document_freq.items()
        }

        inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, tokens in enumerate(self.tokenized):
            for term, count in Counter(tokens).items():
                inverted[term].append((doc_idx, count))
        self.inverted = dict(inverted)

        logger.info("  → BM25 built: %d documents, %d terms", self.N, len(self.idf))

    def _build_accent_index(self) -> None:
        """unaccented(term) → các term có dấu thật trong vocab."""
        accent_index: dict[str, list[str]] = defaultdict(list)
        for term in self.inverted:
            folded = remove_accents(term)
            if folded != term:
                accent_index[folded].append(term)
        self._accent_index = {
            folded: sorted(terms)[:_MAX_ACCENT_VARIANTS]
            for folded, terms in accent_index.items()
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def expand_query_terms(self, text: str) -> list[str]:
        """Sinh danh sách term tra cứu từ câu query thô.

        Gồm: từ đơn (bỏ stopword) + n-gram nối ``_`` + biến thể có dấu suy ra
        từ token không dấu. Chỉ giữ term thật sự tồn tại trong vocab.
        """
        tokens = tokenize(text)
        if not tokens:
            return []

        # n-gram dựng từ chuỗi token gốc (chưa bỏ stopword) để không phá vỡ
        # các từ ghép kiểu "giao thông", "đà lạt".
        raw_candidates = _ngrams(tokens) + [t for t in tokens if t not in STOPWORDS]

        terms: list[str] = []
        seen: set[str] = set()
        for candidate in raw_candidates:
            if candidate in self.inverted:
                if candidate not in seen:
                    seen.add(candidate)
                    terms.append(candidate)
                continue
            for variant in self._accent_index.get(remove_accents(candidate), ()):
                if variant not in seen:
                    seen.add(variant)
                    terms.append(variant)
        return terms

    def _score_documents(self, terms: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl
        if avgdl <= 0:
            return {}
        for term in terms:
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            if isinstance(postings, tuple) and len(postings) == 2:
                doc_ids, counts = postings
                for doc_idx, count in zip(doc_ids, counts):
                    doc_len = self.doc_lengths[doc_idx]
                    denominator = count + k1 * (1 - b + b * doc_len / avgdl)
                    scores[doc_idx] += idf * (count * (k1 + 1)) / denominator
            else:
                for doc_idx, count in postings:
                    doc_len = self.doc_lengths[doc_idx]
                    # Mẫu số dùng độ dài document thật, không phải hằng số.
                    denominator = count + k1 * (1 - b + b * doc_len / avgdl)
                    scores[doc_idx] += idf * (count * (k1 + 1)) / denominator
        return scores

    def _document_frame(self, doc: dict) -> int | None:
        """Đổi mốc thời gian của document sang actual video frame gần nhất.

        Trả ``None`` nếu video không có keyframe map — khi đó không có cách nào
        biết frame thật, và bịa ra một con số là cách chắc chắn nhất để nộp sai.
        """
        video_id = doc.get("video_id", "")
        times = self._kf_times.get(video_id)
        frames = self._kf_frames.get(video_id)
        if not times or not frames:
            return None

        # Document gắn với đúng một keyframe (OCR, caption) thì tra thẳng bảng.
        # Ưu tiên hơn cả ``frame_idx`` ghi sẵn trong document lẫn mốc thời gian:
        # keyframe_map là nguồn duy nhất được đối chiếu với map-keyframes của BTC.
        kf_num = doc.get("keyframe_num")
        if kf_num is not None:
            mapped = self._kf_by_num.get(video_id, {}).get(int(kf_num))
            if mapped is not None:
                return mapped
            # Có ordinal nhưng không tra được frame thật → không đoán bừa.
            return None

        if doc.get("type") in VIDEO_LEVEL_TYPES:
            return frames[0]

        start = doc.get("start_time")
        end = doc.get("end_time")
        if start is None:
            return frames[0]
        start = float(start)
        end = float(end) if end is not None else start

        import bisect

        # Ưu tiên keyframe nằm trong [start, end]; lấy cái gần giữa nhất.
        left = bisect.bisect_left(times, start)
        right = bisect.bisect_right(times, end)
        if left < right:
            middle = (start + end) / 2.0
            best = min(
                range(left, right), key=lambda i: (abs(times[i] - middle), i)
            )
            return frames[best]

        # Không keyframe nào rơi vào cửa sổ → lấy keyframe gần nhất theo thời gian.
        nearest = min(
            (i for i in (left - 1, left) if 0 <= i < len(times)),
            key=lambda i: (abs(times[i] - start), i),
            default=None,
        )
        return frames[nearest] if nearest is not None else None

    def _evidence_for(self, doc: dict) -> dict:
        doc_type = doc.get("type", "")
        key = _EVIDENCE_KEYS.get(doc_type, "text_match")
        evidence: dict = {key: (doc.get("text") or "")[:200], "doc_type": doc_type}
        if doc.get("start_time") is not None:
            evidence["start_time"] = doc.get("start_time")
        if doc.get("end_time") is not None:
            evidence["end_time"] = doc.get("end_time")
        if doc.get("language"):
            evidence["language"] = doc["language"]
        return evidence

    def search(
        self,
        query: Query,
        limit: int = 100,
        exclude: frozenset = frozenset(),
        k: int | None = None,
    ) -> list[Candidate]:
        """BM25 search, trả về tối đa ``limit`` Candidate theo score giảm dần.

        ``k`` là shim tương thích ngược; nếu truyền thì nó thắng ``limit``.
        """
        if k is not None:
            limit = k
        if limit <= 0:
            return []

        if hasattr(query, "for_bm25"):
            query_text = query.for_bm25()
        else:
            query_text = f"{query.for_text()} {query.for_clip()}".strip()

        terms = self.expand_query_terms(query_text)
        if not terms:
            return []

        # Modality có thể siết thêm theo từng request; None = giữ nguyên cấu
        # hình lúc khởi tạo, nên payload UI cũ chạy y như trước.
        request_types = self._resolve_modalities(getattr(query, "modalities", None))

        scores = self._score_documents(terms)
        if not scores:
            return []

        self.unmapped_documents = 0
        # (video_id, frame) → (score, doc). Nhiều document có thể cùng trỏ về một
        # keyframe; giữ document điểm cao nhất cho moment đó.
        moments: dict[tuple[str, int], tuple[float, dict]] = {}
        for doc_idx, score in scores.items():
            doc = self.documents[doc_idx]
            if request_types is not None and doc.get("type") not in request_types:
                continue
            video_id = doc.get("video_id", "")
            if not video_id or video_id in exclude:
                continue
            frame = self._document_frame(doc)
            if frame is None:
                self.unmapped_documents += 1
                continue
            key = (video_id, frame)
            if key not in moments or score > moments[key][0]:
                moments[key] = (score, doc)

        if not moments:
            return []

        ordered = sorted(
            moments.items(), key=lambda item: (-item[1][0], item[0][0], item[0][1])
        )

        max_score = ordered[0][1][0]
        per_video: Counter[str] = Counter()
        candidates: list[Candidate] = []
        for (video_id, frame), (score, doc) in ordered:
            if per_video[video_id] >= self.per_video_limit:
                continue
            per_video[video_id] += 1
            normalized = score / max_score if max_score > 0 else 0.0
            candidates.append(
                Candidate(
                    video_id=video_id,
                    start_frame=frame,
                    end_frame=frame,
                    representative_frames=[frame],
                    scores={self.name: round(normalized, 6)},
                    evidence=self._evidence_for(doc),
                )
            )
            if len(candidates) >= limit:
                break

        return candidates

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    def describe(self) -> str:
        return f"{self.name} ({len(self.documents):,} docs)"


def build_text_retriever(
    index_path: str | Path = "local/index",
    name: str = "bm25",
    modalities: list[str] | None = None,
    doc_types: list[str] | None = None,
    **kwargs,
) -> TextRetriever:
    """Factory — nhận thẳng file ``.pkl`` hoặc thư mục chứa nó."""
    path = Path(index_path)
    if path.is_dir():
        path = path / "text_search_index.pkl"
    return TextRetriever(
        index_path=path,
        name=name,
        modalities=modalities,
        doc_types=doc_types,
        **kwargs,
    )
