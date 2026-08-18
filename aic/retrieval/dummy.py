import random

from ..core.types import Candidate, Query

NAME = "dummy"


def search(query: Query, k: int = 100, exclude: frozenset = frozenset()) -> list:
    """Trả về k ứng viên bịa, đã xếp hạng giảm dần theo điểm."""
    rng = random.Random(query.query_id)
    out = []
    seen = set()
    while len(out) < k and len(seen) < 3000:
        video_id = "L%02d_V%03d" % (rng.randint(21, 30), rng.randint(1, 80))
        start = rng.randrange(0, 30000, 30)
        key = (video_id, start)
        if key in seen:
            continue
        seen.add(key)
        if video_id in exclude:
            continue
        end = start + rng.randrange(60, 600, 30)
        out.append(
            Candidate(
                video_id=video_id,
                start_frame=start,
                end_frame=end,
                representative_frames=[(start + end) // 2],
                scores={"dummy": round(rng.uniform(0.2, 0.95), 4)},
                evidence={"caption": "ứng viên giả cho %s" % query.query_id},
            )
        )
    out.sort(key=lambda c: c.best_score, reverse=True)
    return out[:k]
