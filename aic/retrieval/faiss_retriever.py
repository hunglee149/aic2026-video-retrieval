"""Generic FAISS-based retriever.

Hỗ trợ cả CLIP lẫn SigLIP — chỉ cần truyền đúng index file,
metadata file, và hàm encode text.

Metadata format (JSON array):
    [{"video_id": "L21_V001", "keyframe_num": 1, "frame_idx": 0, "pts_time": 0.0}, ...]

Mỗi phần tử khớp 1:1 với vector tương ứng trong FAISS index.
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


class FaissRetriever:
    """Tìm keyframe bằng cosine similarity trên FAISS index.

    Parameters
    ----------
    index_path : đường dẫn tới file ``*.index`` (FAISS)
    metadata_path : đường dẫn tới file ``*.json`` chứa metadata
    encode_fn : hàm nhận ``str`` trả về ``np.ndarray`` shape ``(dim,)``
    name : tên retriever, ghi vào ``Candidate.scores``
    """

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        encode_fn: Callable[[str], np.ndarray],
        name: str = "faiss",
    ):
        self.name = name
        self.encode_fn = encode_fn

        logger.info("Loading FAISS index from %s", index_path)
        try:
            self.index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
        except Exception:
            self.index = faiss.read_index(str(index_path))
        logger.info(
            "  → %d vectors, dim=%d", self.index.ntotal, self.index.d
        )

        logger.info("Loading metadata from %s", metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata: list[dict] = json.load(f)
        assert len(self.metadata) == self.index.ntotal, (
            f"metadata length {len(self.metadata)} != index size {self.index.ntotal}"
        )
        logger.info("  → %d entries loaded", len(self.metadata))

        # Tạo reverse lookup: video_id → list[int] (index positions)
        self._video_index: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(self.metadata):
            self._video_index[m["video_id"]].append(i)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: Query,
        limit: int = 100,
        exclude: frozenset = frozenset(),
        k: int = None,
    ) -> list[Candidate]:
        """Tìm top-k keyframes gần nhất với query text.

        Trả về list[Candidate] đã sắp xếp giảm dần theo score.
        """
        if k is not None:
            limit = k
        k = limit
        text = query.for_clip()  # ưu tiên bản tiếng Anh
        embedding = self.encode_fn(text)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        # Normalize cho cosine similarity (FAISS inner product)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Over-fetch để bù cho exclude
        fetch_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(embedding, fetch_k)

        candidates: list[Candidate] = []
        seen_videos: set[str] = set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS trả -1 nếu không đủ kết quả
                continue
            meta = self.metadata[idx]
            vid = meta["video_id"]

            if vid in exclude:
                continue

            score = float(dist)
            # Nếu cosine similarity (inner product sau normalize) thì score
            # nằm trong [-1, 1]. Clip về [0, 1] cho dễ đọc.
            score = max(0.0, min(1.0, score))

            candidates.append(
                Candidate(
                    video_id=vid,
                    start_frame=meta["frame_idx"],
                    end_frame=meta["frame_idx"],
                    representative_frames=[meta["frame_idx"]],
                    scores={self.name: round(score, 6)},
                    evidence={},
                )
            )

            if len(candidates) >= k:
                break

        # Sắp xếp giảm dần theo score
        candidates.sort(key=lambda c: c.scores.get(self.name, 0), reverse=True)
        return candidates

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        k: int = 100,
        exclude: frozenset = frozenset(),
        name_override: str | None = None,
    ) -> list[Candidate]:
        """Search bằng embedding vector trực tiếp (cho SQR-style refinement)."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        fetch_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(embedding, fetch_k)
        tag = name_override or self.name

        candidates: list[Candidate] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            if meta["video_id"] in exclude:
                continue
            score = max(0.0, min(1.0, float(dist)))
            candidates.append(
                Candidate(
                    video_id=meta["video_id"],
                    start_frame=meta["frame_idx"],
                    end_frame=meta["frame_idx"],
                    representative_frames=[meta["frame_idx"]],
                    scores={tag: round(score, 6)},
                    evidence={},
                )
            )
            if len(candidates) >= k:
                break

        candidates.sort(key=lambda c: c.scores.get(tag, 0), reverse=True)
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
