"""Pipeline — nối retrieval → fusion → review → submission.

Có hai chế độ:
1. `run()` — single-pass: retrieve → fuse → review → write (đã có)
2. `iterative_run()` — multi-round: retrieve → verify → exclude not_matched → re-retrieve

Quy ước:
    - frame_idx trong hệ thống là 0-based
    - Chỉ đổi sang 1-based ở bước cuối (to_csv_row trong convert.py)
    - "not_matched" → loại khỏi exclude set, không bao giờ trả lại
    - "unsure" → giữ trong danh sách, xếp sau matched
"""

from .core.convert import to_answer, to_csv_row


def retrieve_and_fuse(
    query, retrievers, fuse_fn, limit=100, exclude=frozenset(), fuse_kwargs=None
):
    """Gọi mọi nguồn tìm kiếm rồi trộn lại thành một danh sách ứng viên.

    ``fuse_kwargs`` chuyển tiếp tuỳ chọn cho fusion (ví dụ ``weights``); để None
    thì fusion dùng mặc định của nó.
    """
    runs = [r.search(query, limit, exclude) for r in retrievers]
    return fuse_fn(runs, limit=limit, **(fuse_kwargs or {}))


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
    """Single-pass pipeline: retrieve → fuse → review → write."""
    candidates = retrieve_and_fuse(query, retrievers, fuse_fn, limit=limit)
    chosen_frames = answers = None
    if review_fn is not None:
        candidates, chosen_frames, answers = review_fn(candidates)
    rows = to_rows(query, candidates, chosen_frames, answers)
    return write_fn({query.query_id: rows}, out_path)


# -------------------------------------------------------------------------
# Iterative retrieve — inspired by VideoSearch-R1
# -------------------------------------------------------------------------


def iterative_retrieve(
    query,
    retrievers,
    fuse_fn,
    verify_fn=None,
    max_rounds=3,
    limit=100,
):
    """Multi-round retrieve → verify → exclude → re-retrieve.

    Parameters
    ----------
    query : Query object
    retrievers : list of retriever modules (each has .search())
    fuse_fn : fusion function (list[list[Candidate]] → list[Candidate])
    verify_fn : callable(Candidate) → str ("matched" | "not_matched" | "unsure")
                Nếu None, tất cả đều "matched" (no filtering)
    max_rounds : số lượt tối đa
    limit : top-K trả về cuối cùng

    Returns
    -------
    list[Candidate] đã xếp hạng, tối đa `limit` phần tử.
    Thứ tự: matched → unsure (theo score giảm dần trong mỗi nhóm).
    """
    exclude = set()
    matched = []
    unsure = []

    for round_num in range(1, max_rounds + 1):
        # Retrieve với exclude set hiện tại
        candidates = retrieve_and_fuse(
            query, retrievers, fuse_fn,
            limit=limit, exclude=frozenset(exclude),
        )

        if not candidates:
            break

        if verify_fn is None:
            # Không verify → tất cả matched
            matched.extend(candidates)
            break

        # Verify từng candidate
        new_excludes = 0
        for cand in candidates:
            # Bỏ qua video đã xếp loại
            if cand.video_id in exclude:
                continue

            verdict = verify_fn(cand)

            if verdict == "matched":
                matched.append(cand)
            elif verdict == "not_matched":
                exclude.add(cand.video_id)
                new_excludes += 1
            else:  # "unsure"
                unsure.append(cand)

        # Nếu không loại thêm ai → dừng sớm
        if new_excludes == 0:
            break

        # Đã đủ kết quả → dừng
        if len(matched) + len(unsure) >= limit:
            break

    # Kết hợp: matched trước, unsure sau
    # Trong mỗi nhóm, giữ nguyên thứ tự score
    result = matched + unsure

    # Dedup theo (video_id, start_frame)
    seen = set()
    deduped = []
    for cand in result:
        key = (cand.video_id, cand.start_frame)
        if key not in seen:
            seen.add(key)
            deduped.append(cand)

    return deduped[:limit]


def iterative_run(
    query,
    retrievers,
    fuse_fn,
    write_fn,
    out_path,
    verify_fn=None,
    review_fn=None,
    max_rounds=3,
    limit=100,
):
    """Full iterative pipeline: multi-round retrieve → review → write.

    Parameters
    ----------
    verify_fn : auto verify per-candidate ("matched"/"not_matched"/"unsure")
    review_fn : manual review by operator (UI callback)
    """
    candidates = iterative_retrieve(
        query, retrievers, fuse_fn,
        verify_fn=verify_fn, max_rounds=max_rounds, limit=limit,
    )

    chosen_frames = answers = None
    if review_fn is not None:
        candidates, chosen_frames, answers = review_fn(candidates)

    rows = to_rows(query, candidates, chosen_frames, answers)
    return write_fn({query.query_id: rows}, out_path)
