from aic.submission.query_pack import QueryDefinition
from aic.submission.validator import normalize_submission_rows, validate_submission


MANIFEST = [
    QueryDefinition(
        query_id="query-p1-1-kis",
        task="kis",
        text="Tìm cảnh",
        source_name="query-p1-1-kis.txt",
        n_events=None,
        events_confirmed=True,
    ),
    QueryDefinition(
        query_id="query-p1-2-qa",
        task="qa",
        text="Có bao nhiêu người?",
        source_name="query-p1-2-qa.txt",
        n_events=None,
        events_confirmed=True,
    ),
    QueryDefinition(
        query_id="query-p1-3-trake",
        task="trake",
        text="Ba sự kiện",
        source_name="query-p1-3-trake.txt",
        n_events=3,
        events_confirmed=True,
    ),
]


def _row(query_id, *, video_id="L01_V001", frames=None, answer=""):
    return {
        "query_id": query_id,
        "video_id": video_id,
        "frames": [12] if frames is None else frames,
        "answer": answer,
    }


def _valid_rows():
    return [
        _row("query-p1-1-kis", frames=[12]),
        _row("query-p1-2-qa", frames=[34], answer="  Năm người  "),
        _row("query-p1-3-trake", frames=[10, 20, 30]),
    ]


def _codes(report):
    return [issue.code for issue in report.errors]


def test_valid_rows_keep_qa_whitespace_and_normalize_terminal_mp4_suffix():
    rows = _valid_rows()
    rows[0]["video_id"] = "L01_V001.MP4"
    rows[1]["video_id"] = "L01_V002.mp4"

    normalized = normalize_submission_rows(rows)
    report = validate_submission(MANIFEST, normalized)

    assert report.ok
    assert normalized[0]["video_id"] == "L01_V001"
    assert normalized[1]["video_id"] == "L01_V002"
    assert normalized[1]["answer"] == "  Năm người  "


def test_requires_rows_for_every_manifest_query():
    report = validate_submission(MANIFEST, _valid_rows()[:2])

    assert "missing_query_rows" in _codes(report)


def test_rejects_rows_for_an_unknown_query():
    report = validate_submission(MANIFEST, _valid_rows() + [_row("query-p1-99-kis")])

    assert "unknown_query" in _codes(report)


def test_allows_exactly_100_ranked_rows_for_one_query():
    rows = [
        _row("query-p1-1-kis", frames=[index + 1]) for index in range(100)
    ]
    rows.extend(_valid_rows()[1:])

    report = validate_submission(MANIFEST, rows)

    assert report.ok


def test_rejects_more_than_100_ranked_rows_for_one_query():
    rows = [
        _row("query-p1-1-kis", frames=[index + 1]) for index in range(101)
    ]
    rows.extend(_valid_rows()[1:])

    report = validate_submission(MANIFEST, rows)

    assert "too_many_rows" in _codes(report)


def test_rejects_kis_with_more_than_one_frame():
    rows = _valid_rows()
    rows[0]["frames"] = [12, 13]

    assert "kis_frame_count" in _codes(validate_submission(MANIFEST, rows))


def test_rejects_kis_with_an_answer():
    rows = _valid_rows()
    rows[0]["answer"] = "không được phép"

    assert "kis_unexpected_answer" in _codes(validate_submission(MANIFEST, rows))


def test_rejects_qa_with_more_than_one_frame():
    rows = _valid_rows()
    rows[1]["frames"] = [34, 35]

    assert "qa_frame_count" in _codes(validate_submission(MANIFEST, rows))


def test_rejects_qa_without_an_answer():
    rows = _valid_rows()
    rows[1]["answer"] = ""

    assert "qa_missing_answer" in _codes(validate_submission(MANIFEST, rows))


def test_rejects_qa_answer_longer_than_100_unicode_characters():
    rows = _valid_rows()
    rows[1]["answer"] = "á" * 101

    assert "qa_answer_too_long" in _codes(validate_submission(MANIFEST, rows))


def test_rejects_trake_until_its_event_count_is_confirmed():
    unconfirmed = [*MANIFEST]
    unconfirmed[2] = QueryDefinition(
        **{**MANIFEST[2].to_dict(), "events_confirmed": False}
    )

    assert "trake_events_unconfirmed" in _codes(
        validate_submission(unconfirmed, _valid_rows())
    )


def test_rejects_trake_with_the_wrong_number_of_event_frames():
    rows = _valid_rows()
    rows[2]["frames"] = [10, 20]

    assert "trake_frame_count" in _codes(validate_submission(MANIFEST, rows))


def test_rejects_trake_frames_that_are_not_strictly_increasing():
    rows = _valid_rows()
    rows[2]["frames"] = [10, 10, 30]

    assert "trake_frame_order" in _codes(validate_submission(MANIFEST, rows))


def test_rejects_trake_with_an_answer():
    rows = _valid_rows()
    rows[2]["answer"] = "không được phép"

    assert "trake_unexpected_answer" in _codes(validate_submission(MANIFEST, rows))


def test_rejects_an_invalid_video_id():
    rows = _valid_rows()
    rows[0]["video_id"] = "not a video"

    assert "invalid_video_id" in _codes(validate_submission(MANIFEST, rows))


def test_rejects_a_non_positive_or_non_integer_frame():
    rows = _valid_rows()
    rows[0]["frames"] = [0]
    rows[1]["frames"] = ["34"]

    assert "invalid_frame" in _codes(validate_submission(MANIFEST, rows))
