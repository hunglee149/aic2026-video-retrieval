"""Fusion — trộn kết quả nhiều retriever bằng weighted Reciprocal Rank Fusion.

Vì sao là RRF chứ không phải cộng điểm thô: CLIP/SigLIP trả cosine trong
``[0, 1]`` còn BM25 trả điểm không chặn trên. Cộng thẳng thì nguồn nào có thang
đo lớn hơn sẽ luôn thắng, bất kể thứ hạng. RRF chỉ dùng *thứ hạng* nên không
phụ thuộc thang đo::

    RRF(candidate) = Σ_source  weight(source) / (rrf_k + rank_source)

Hai quy ước quan trọng:

- **Danh tính candidate là một moment, không phải một video.** Khoá là
  ``(video_id, anchor_frame)``. Gộp mọi moment của cùng một video thành một
  dòng sẽ giết TRAKE và làm mất các cảnh khác nhau trong cùng video.
- **Không bao giờ độn candidate giả.** Trả tối đa ``limit`` moment thật; ít hơn
  thì trả ít hơn. Dòng ``L00_V000`` độn cho đủ 100 chỉ làm bẩn bài nộp.
"""

from __future__ import annotations

from collections import Counter

from aic.core.types import Candidate

# Hằng số RRF kinh điển; đủ lớn để top-1 không áp đảo phần còn lại.
DEFAULT_RRF_K = 60

# Trọng số mặc định cho từng nguồn. Giữ bằng nhau cho tới khi có đánh giá
# định lượng trên ground truth thật — đoán trọng số là tự bịa.
DEFAULT_WEIGHTS: dict[str, float] = {
    "clip": 1.0,
    "siglip": 1.0,
    "bm25": 1.0,
}

# 0 = chỉ gộp khi trùng đúng frame. Nới ra chỉ khi có rule temporal được đo đạc.
DEFAULT_MERGE_RADIUS = 0

_FUSED_KEY = "fused"


def _anchor_frame(candidate: Candidate) -> int:
    """Frame đại diện của candidate — luôn là actual video frame (0-based)."""
    if candidate.representative_frames:
        return int(candidate.representative_frames[0])
    return int(candidate.start_frame)


def _run_source(run: list[Candidate], fallback: str) -> str:
    """Tên nguồn của một run, suy ra từ khoá điểm mà retriever đã gắn."""
    keys: Counter[str] = Counter()
    for candidate in run:
        for key in candidate.scores:
            if key != _FUSED_KEY:
                keys[key] += 1
    if not keys:
        return fallback
    # Deterministic: nhiều nhất trước, hoà thì lấy tên nhỏ nhất theo alphabet.
    return min(keys.items(), key=lambda item: (-item[1], item[0]))[0]


class _Moment:
    """Một moment đã gộp từ nhiều nguồn."""

    __slots__ = ("video_id", "anchor", "start", "end", "scores", "evidence", "rrf")

    def __init__(self, candidate: Candidate, anchor: int):
        self.video_id = candidate.video_id
        self.anchor = anchor
        self.start = int(candidate.start_frame)
        self.end = int(candidate.end_frame)
        self.scores: dict[str, float] = {}
        self.evidence: dict = {}
        self.rrf = 0.0
        self.absorb(candidate)

    def absorb(self, candidate: Candidate) -> None:
        self.start = min(self.start, int(candidate.start_frame))
        self.end = max(self.end, int(candidate.end_frame))
        for key, value in candidate.scores.items():
            if key == _FUSED_KEY:
                continue
            if key not in self.scores or value > self.scores[key]:
                self.scores[key] = value
        for key, value in candidate.evidence.items():
            if key == "objects":
                merged = set(self.evidence.get("objects", []))
                merged.update(value or [])
                self.evidence["objects"] = sorted(merged)
            elif key not in self.evidence:
                self.evidence[key] = value

    def to_candidate(self) -> Candidate:
        scores = dict(self.scores)
        scores[_FUSED_KEY] = round(self.rrf, 9)
        return Candidate(
            video_id=self.video_id,
            start_frame=self.start,
            end_frame=self.end,
            # Giữ đúng MỘT frame đại diện: to_answer() dùng list này làm frame
            # nộp bài khi operator chưa chọn tay, nên nhiều phần tử sẽ đẻ ra
            # dòng KIS/Q&A sai định dạng.
            representative_frames=[self.anchor],
            scores=scores,
            evidence=dict(self.evidence),
        )


def fuse(
    runs,
    limit: int = 100,
    weights: dict[str, float] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    merge_radius: int = DEFAULT_MERGE_RADIUS,
) -> list[Candidate]:
    """Trộn nhiều run thành một danh sách moment đã xếp hạng.

    Parameters
    ----------
    runs : list[list[Candidate]]
        Mỗi phần tử là kết quả *đã xếp hạng* của một retriever.
    limit : số moment tối đa trả về. Không độn cho đủ.
    weights : trọng số theo tên nguồn; thiếu thì lấy ``DEFAULT_WEIGHTS`` rồi 1.0.
    rrf_k : hằng số RRF.
    merge_radius : gộp hai moment cùng video nếu anchor cách nhau <= giá trị này.
    """
    effective_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        effective_weights.update(weights)

    moments: list[_Moment] = []
    # video_id → list[(anchor, vị trí trong `moments`)], giữ thứ tự xuất hiện.
    by_video: dict[str, list[tuple[int, int]]] = {}

    for run_idx, run in enumerate(runs or []):
        if not run:
            continue
        source = _run_source(run, fallback=f"source{run_idx}")
        weight = effective_weights.get(source, 1.0)

        for rank, candidate in enumerate(run, start=1):
            anchor = _anchor_frame(candidate)
            slot = None
            for existing_anchor, position in by_video.get(candidate.video_id, []):
                if abs(existing_anchor - anchor) <= merge_radius:
                    slot = position
                    break

            if slot is None:
                moment = _Moment(candidate, anchor)
                moments.append(moment)
                by_video.setdefault(candidate.video_id, []).append(
                    (anchor, len(moments) - 1)
                )
            else:
                moment = moments[slot]
                moment.absorb(candidate)

            moment.rrf += weight / (rrf_k + rank)

    moments.sort(key=lambda m: (-m.rrf, m.video_id, m.anchor))
    return [moment.to_candidate() for moment in moments[:limit]]


def rrf_fuse(runs, limit: int = 100, **kwargs) -> list[Candidate]:
    """Alias tường minh cho `fuse` — hữu ích khi đọc code pipeline."""
    return fuse(runs, limit=limit, **kwargs)
