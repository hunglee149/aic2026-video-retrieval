"""Tests cho fusion.

Spec đã đổi so với bản đầu (xem docs/fusion.md):

- xếp hạng bằng weighted RRF thay vì cộng điểm thô;
- danh tính là moment ``(video_id, anchor_frame)`` chứ không phải ``video_id``,
  nên hai khoảnh khắc khác nhau trong cùng một video cùng tồn tại;
- không độn candidate giả ``L00_Vxxx`` cho đủ ``limit``.

Hai test cũ (`keeps_first_duplicate_video`, `pads_to_limit`) khoá đúng hành vi
vừa bị thay nên được viết lại ở đây thành test cho hành vi mới.
"""

from aic.core.types import Candidate
from aic.fusion import rank


def make_candidate(video_id: str, start_frame: int, score: float, source="source"):
    return Candidate(
        video_id=video_id,
        start_frame=start_frame,
        end_frame=start_frame + 60,
        representative_frames=[start_frame],
        scores={source: score},
    )


def test_fuse_keeps_distinct_moments_of_same_video():
    first = make_candidate("L21_V001", 100, 0.8)
    second = make_candidate("L22_V002", 200, 0.7)
    later_moment = make_candidate("L21_V001", 300, 0.99)

    result = rank.fuse([[first, second], [later_moment]], limit=10)

    keys = {(c.video_id, c.start_frame) for c in result}
    assert keys == {("L21_V001", 100), ("L22_V002", 200), ("L21_V001", 300)}


def test_fuse_merges_same_moment_from_two_runs():
    from_clip = make_candidate("L21_V001", 100, 0.8, source="clip")
    from_bm25 = make_candidate("L21_V001", 100, 12.5, source="bm25")

    result = rank.fuse([[from_clip], [from_bm25]], limit=10)

    assert len(result) == 1
    assert result[0].scores["clip"] == 0.8
    assert result[0].scores["bm25"] == 12.5
    assert result[0].scores["fused"] > 0


def test_fuse_honours_limit():
    first = make_candidate("L21_V001", 100, 0.8)
    second = make_candidate("L22_V002", 200, 0.7)
    third = make_candidate("L23_V003", 300, 0.6)

    result = rank.fuse([[first, second], [third]], limit=2)

    assert len(result) == 2
    assert [c.video_id for c in result] == ["L21_V001", "L23_V003"]


def test_fuse_ranks_by_rrf_not_raw_score():
    """Điểm thô lớn ở thứ hạng thấp không được thắng thứ hạng cao."""
    bm25_run = [
        make_candidate("L21_V001", 10, 0.1, source="bm25"),
        make_candidate("L22_V002", 20, 999.0, source="bm25"),
    ]

    result = rank.fuse([bm25_run], limit=10)

    assert result[0].video_id == "L21_V001"
    assert result[0].scores["fused"] > result[1].scores["fused"]


def test_fuse_does_not_pad_to_limit():
    first = make_candidate("L21_V001", 100, 0.8)

    result = rank.fuse([[first]], limit=5)

    assert len(result) == 1
    assert result[0].video_id == "L21_V001"
    assert all(not c.video_id.startswith("L00_V") for c in result)


def test_fuse_does_not_mutate_input_candidates():
    original = make_candidate("L21_V001", 100, 0.8, source="clip")

    rank.fuse([[original]], limit=10)

    assert original.scores == {"clip": 0.8}
