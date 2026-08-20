from aic.core.types import Candidate
from aic.fusion import rank


def make_candidate(video_id: str, start_frame: int, score: float) -> Candidate:
    return Candidate(
        video_id=video_id,
        start_frame=start_frame,
        end_frame=start_frame + 60,
        scores={"source": score},
    )


def test_fuse_preserves_order_and_keeps_first_duplicate_video():
    first = make_candidate("L21_V001", 100, 0.8)
    second = make_candidate("L22_V002", 200, 0.7)
    duplicate = make_candidate("L21_V001", 300, 0.99)

    result = rank.fuse([[first, second], [duplicate]], limit=2)

    assert result == [first, second]
    assert result[0] is first


def test_fuse_honours_limit():
    first = make_candidate("L21_V001", 100, 0.8)
    second = make_candidate("L22_V002", 200, 0.7)
    third = make_candidate("L23_V003", 300, 0.6)

    result = rank.fuse([[first, second], [third]], limit=2)

    assert result == [first, second]


def test_fuse_pads_to_limit():
    first = make_candidate("L21_V001", 100, 0.8)
    result = rank.fuse([[first]], limit=5)
    
    assert len(result) == 5
    assert result[0] == first
    assert result[1].video_id == "L00_V000"
    assert result[2].video_id == "L00_V001"
    assert result[3].video_id == "L00_V002"
    assert result[4].video_id == "L00_V003"
