from .core.convert import to_answer, to_csv_row


def retrieve_and_fuse(query, retrievers, fuse_fn, limit=100, exclude=frozenset()):
    """Gọi mọi nguồn tìm kiếm rồi trộn lại thành một danh sách ứng viên."""
    runs = [r.search(query, limit, exclude) for r in retrievers]
    return fuse_fn(runs, limit=limit)


def to_rows(query, candidates, chosen_frames=None, answers=None):
    chosen_frames = chosen_frames or {}
    answers = answers or {}
    rows = []
    for i, cand in enumerate(candidates, start=1):
        ans = to_answer(
            cand,
            query_id=query.query_id,
            rank=i,
            frames=chosen_frames.get(i),
            answer=answers.get(i, ""),
        )
        rows.append(to_csv_row(ans))
    return rows


def run(query, retrievers, fuse_fn, write_fn, out_path, limit=100, review_fn=None):
    candidates = retrieve_and_fuse(query, retrievers, fuse_fn, limit=limit)
    chosen_frames = answers = None
    if review_fn is not None:
        candidates, chosen_frames, answers = review_fn(candidates)
    rows = to_rows(query, candidates, chosen_frames, answers)
    return write_fn({query.query_id: rows}, out_path)
