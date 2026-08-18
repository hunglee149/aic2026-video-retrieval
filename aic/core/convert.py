from .types import Answer, Candidate

FRAME_OFFSET = 1

def to_answer(
    cand: Candidate,
    query_id: str,
    rank: int,
    frames: list | None = None,
    answer: str = "",
) -> Answer:
    
    chosen = list(frames) if frames else list(cand.representative_frames)
    if not chosen:
        chosen = [cand.middle_frame()]
    return Answer(
        query_id=query_id,
        rank=rank,
        video_id=cand.video_id,
        frames=[int(f) + FRAME_OFFSET for f in chosen],
        answer=answer,
    )

def to_csv_row(ans: Answer) -> list:
    
    video_id = ans.video_id.removesuffix(".mp4")
    row = [video_id] + [str(f) for f in ans.frames]
    if ans.answer:
        row.append(ans.answer)
    return row