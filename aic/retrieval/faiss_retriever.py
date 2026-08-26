"""FAISS retriever dùng chung cho CLIP và SigLIP.

Metadata format (JSON array), khớp 1:1 theo thứ tự với vector trong index::

    [{"video_id": "L21_V001", "keyframe_num": 1, "frame_idx": 0, "pts_time": 0.0}, ...]

Quy ước frame quan trọng:

- ``keyframe_num`` là **ordinal của ảnh keyframe** (1-based, khớp tên file
  ``001.jpg``). Nó KHÔNG phải frame của video.
- ``frame_idx`` là **actual video frame index, 0-based** (= ``int(pts_time * fps)``).
  Đây là giá trị duy nhất được phép đi vào ``Candidate``.
- Việc +1 sang hệ 1-based của BTC chỉ xảy ra ở boundary nộp bài
  (``aic/core/convert.py``), không phải ở đây.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from ..core.types import Candidate, Query

logger = logging.getLogger(__name__)


def read_faiss_index(index_path: str | Path):
    """Đọc FAISS index, ưu tiên mmap để không nạp cả file vào RAM.

    mmap không dùng được với mọi loại index/filesystem, nên fallback sang đọc
    thường thay vì để cả retriever chết.
    """
    path = str(index_path)
    try:
        return faiss.read_index(path, faiss.IO_FLAG_MMAP)
    except (RuntimeError, OSError, AttributeError) as exc:
        logger.info("mmap không dùng được cho %s (%s); đọc thường.", path, exc)
        return faiss.read_index(path)


class FaissRetriever:
    """Tìm keyframe bằng cosine similarity trên FAISS index.

    Parameters
    ----------
    index_path : đường dẫn tới file ``*.index`` (FAISS)
    metadata_path : đường dẫn tới file ``*.json`` chứa metadata
    encode_fn : hàm nhận ``str`` trả về ``np.ndarray`` shape ``(dim,)``
    name : tên retriever, ghi vào ``Candidate.scores``
    expected_dim : nếu truyền, index phải đúng số chiều này, sai thì raise
    """

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        encode_fn: Callable[[str], np.ndarray],
        name: str = "faiss",
        expected_dim: int | None = None,
    ):
        self.name = name
        self.encode_fn = encode_fn
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index không tồn tại: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata không tồn tại: {self.metadata_path}")

        logger.info("Loading FAISS index from %s", self.index_path)
        self.index = read_faiss_index(self.index_path)
        logger.info("  → %d vectors, dim=%d", self.index.ntotal, self.index.d)

        if expected_dim is not None and self.index.d != expected_dim:
            raise ValueError(
                f"[{name}] index dim {self.index.d} != encoder dim {expected_dim}; "
                f"index và text encoder không cùng embedding space"
            )

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata: list[dict] = json.load(handle)

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"[{name}] metadata {len(self.metadata)} record != index "
                f"{self.index.ntotal} vector; index và metadata không khớp"
            )
        self._validate_metadata()
        logger.info("  → %d entries loaded", len(self.metadata))

        self._video_index: dict[str, list[int]] = defaultdict(list)
        for position, meta in enumerate(self.metadata):
            self._video_index[meta["video_id"]].append(position)

    def _validate_metadata(self) -> None:
        """Bắt lỗi thiếu ``frame_idx`` ngay lúc load, không phải lúc nộp bài."""
        if not self.metadata:
            raise ValueError(f"[{self.name}] metadata rỗng")
        for position in (0, len(self.metadata) // 2, len(self.metadata) - 1):
            meta = self.metadata[position]
            if "video_id" not in meta:
                raise ValueError(
                    f"[{self.name}] metadata[{position}] thiếu 'video_id'"
                )
            if "frame_idx" not in meta:
                raise ValueError(
                    f"[{self.name}] metadata[{position}] thiếu 'frame_idx' — "
                    f"keyframe_num là ordinal, không dùng thay frame thật được"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: Query,
        limit: int = 100,
        exclude: frozenset = frozenset(),
        k: int | None = None,
    ) -> list[Candidate]:
        """Tìm top-``limit`` keyframe gần nhất với query text.

        ``k`` là shim tương thích ngược cho code gọi kiểu cũ; nếu truyền thì
        nó thắng ``limit``.
        """
        if k is not None:
            limit = k
        text = query.for_clip()  # ưu tiên bản tiếng Anh
        embedding = self.encode_fn(text)
        return self._search_vector(embedding, limit, exclude, self.name)

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        limit: int = 100,
        exclude: frozenset = frozenset(),
        name_override: str | None = None,
        k: int | None = None,
    ) -> list[Candidate]:
        """Search bằng embedding vector trực tiếp (cho SQR-style refinement)."""
        if k is not None:
            limit = k
        return self._search_vector(
            embedding, limit, exclude, name_override or self.name
        )

    def _search_vector(
        self,
        embedding,
        limit: int,
        exclude: frozenset,
        tag: str,
    ) -> list[Candidate]:
        if limit <= 0:
            return []

        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.index.d:
            raise ValueError(
                f"[{self.name}] embedding dim {vector.shape[1]} != index dim "
                f"{self.index.d}"
            )
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Over-fetch để bù cho exclude; index dùng inner product trên vector
        # đã normalize nên distance chính là cosine similarity.
        fetch_k = min(max(limit * 3, limit), self.index.ntotal)
        distances, indices = self.index.search(vector, fetch_k)

        candidates: list[Candidate] = []
        for distance, position in zip(distances[0], indices[0]):
            if position < 0:  # FAISS trả -1 khi không đủ kết quả
                continue
            meta = self.metadata[position]
            if meta["video_id"] in exclude:
                continue

            frame_idx = int(meta["frame_idx"])
            candidates.append(
                Candidate(
                    video_id=meta["video_id"],
                    start_frame=frame_idx,
                    end_frame=frame_idx,
                    representative_frames=[frame_idx],
                    scores={tag: round(float(distance), 6)},
                    evidence={"keyframe_num": meta.get("keyframe_num")},
                )
            )
            if len(candidates) >= limit:
                break

        candidates.sort(key=lambda c: c.scores.get(tag, 0.0), reverse=True)
        return candidates

    @property
    def num_vectors(self) -> int:
        return self.index.ntotal

    @property
    def dim(self) -> int:
        return self.index.d

    def get_video_ids(self) -> list[str]:
        """Danh sách unique video_id trong index."""
        return sorted(self._video_index.keys())

    def describe(self) -> str:
        return f"{self.name} ({self.index.ntotal:,} vectors, dim={self.index.d})"
