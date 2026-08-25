import csv
import io
import zipfile

import pytest

from aic.submission import writer
from aic.submission.query_pack import QueryDefinition
from aic.submission.validator import SubmissionValidationError


MANIFEST = [
    QueryDefinition(
        query_id="pack1_q1_kis",
        task="kis",
        text="Tìm cảnh",
        source_name="pack1_q1_kis.txt",
        n_events=None,
        events_confirmed=True,
    ),
    QueryDefinition(
        query_id="pack1_q2_qa",
        task="qa",
        text="Có bao nhiêu người?",
        source_name="pack1_q2_qa.txt",
        n_events=None,
        events_confirmed=True,
    ),
    QueryDefinition(
        query_id="pack1_q3_trake",
        task="trake",
        text="Ba sự kiện",
        source_name="pack1_q3_trake.txt",
        n_events=3,
        events_confirmed=True,
    ),
]


def test_write_submission_creates_utf8_zip_without_bom(tmp_path):
    out_path = tmp_path / "submission.zip"
    rows = {"pack1_q3_qa": [["L21_V001", "712", "Hy Lạp"]]}

    result = writer.write_submission(rows, str(out_path))

    assert result == str(out_path)
    with zipfile.ZipFile(out_path) as archive:
        assert archive.namelist() == ["submission/pack1_q3_qa.csv"]
        raw = archive.read("submission/pack1_q3_qa.csv")

    assert not raw.startswith(b"\xef\xbb\xbf")
    decoded_rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    assert decoded_rows == rows["pack1_q3_qa"]


def test_write_validated_submission_rejects_kis_with_an_answer(tmp_path):
    rows = [
        {
            "query_id": "pack1_q1_kis",
            "video_id": "L21_V001",
            "frames": [712],
            "answer": "Hy Lạp",
        },
        {
            "query_id": "pack1_q2_qa",
            "video_id": "L21_V002",
            "frames": [713],
            "answer": "Hai người",
        },
        {
            "query_id": "pack1_q3_trake",
            "video_id": "L21_V003",
            "frames": [10, 20, 30],
            "answer": "",
        },
    ]

    with pytest.raises(SubmissionValidationError) as error:
        writer.write_validated_submission(MANIFEST, rows, tmp_path / "submission.zip")

    assert [issue.code for issue in error.value.report.errors] == [
        "kis_unexpected_answer"
    ]


def test_write_validated_submission_reparses_a_valid_mixed_task_zip(tmp_path):
    out_path = tmp_path / "submission.zip"
    rows = [
        {
            "query_id": "pack1_q1_kis",
            "video_id": "L21_V001.MP4",
            "frames": [712],
            "answer": "",
        },
        {
            "query_id": "pack1_q2_qa",
            "video_id": "L21_V002.mp4",
            "frames": [713],
            "answer": "  Hai người  ",
        },
        {
            "query_id": "pack1_q3_trake",
            "video_id": "L21_V003",
            "frames": [10, 20, 30],
            "answer": "",
        },
    ]

    report = writer.write_validated_submission(MANIFEST, rows, out_path)

    assert report.ok
    with zipfile.ZipFile(out_path) as archive:
        assert archive.namelist() == [
            "submission/pack1_q1_kis.csv",
            "submission/pack1_q2_qa.csv",
            "submission/pack1_q3_trake.csv",
        ]
        assert list(csv.reader(io.StringIO(
            archive.read("submission/pack1_q1_kis.csv").decode("utf-8")
        ))) == [["L21_V001", "712"]]
        assert list(csv.reader(io.StringIO(
            archive.read("submission/pack1_q2_qa.csv").decode("utf-8")
        ))) == [["L21_V002", "713", "  Hai người  "]]
        assert list(csv.reader(io.StringIO(
            archive.read("submission/pack1_q3_trake.csv").decode("utf-8")
        ))) == [["L21_V003", "10", "20", "30"]]


def test_write_submission_supports_multiple_queries_and_creates_parent(tmp_path):
    out_path = tmp_path / "output" / "submission.zip"
    rows = {
        "pack1_q1_kis": [["L21_V001", "712"]],
        "pack1_q2_trake": [["L22_V002", "10", "20", "30"]],
    }

    writer.write_submission(rows, out_path)

    with zipfile.ZipFile(out_path) as archive:
        assert archive.namelist() == [
            "submission/pack1_q1_kis.csv",
            "submission/pack1_q2_trake.csv",
        ]


def test_write_query_submission_accepts_a_list_for_one_query(tmp_path):
    out_path = tmp_path / "submission.zip"

    writer.write_query_submission(
        "pack1_q3_kis", [["L21_V001", "712"]], out_path
    )

    with zipfile.ZipFile(out_path) as archive:
        assert archive.namelist() == ["submission/pack1_q3_kis.csv"]


def test_write_submission_rejects_more_than_100_rows(tmp_path):
    rows = {"pack1_q1_kis": [["L21_V001", str(i + 1)] for i in range(101)]}

    with pytest.raises(ValueError, match="maximum is 100"):
        writer.write_submission(rows, tmp_path / "submission.zip")


@pytest.mark.parametrize("query_id", ["", "../query", "folder/query"])
def test_write_submission_rejects_unsafe_query_id(tmp_path, query_id):
    with pytest.raises(ValueError, match="query_id"):
        writer.write_submission(
            {query_id: [["L21_V001", "712"]]}, tmp_path / "submission.zip"
        )
