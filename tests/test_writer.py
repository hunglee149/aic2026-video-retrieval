import csv
import io
import zipfile

import pytest

from aic.submission import writer


def test_write_submission_creates_utf8_zip_without_bom(tmp_path):
    out_path = tmp_path / "submission.zip"
    rows = {"pack1_q3_kis": [["L21_V001", "712", "Hy Lạp"]]}

    result = writer.write_submission(rows, str(out_path))

    assert result == str(out_path)
    with zipfile.ZipFile(out_path) as archive:
        assert archive.namelist() == ["submission/pack1_q3_kis.csv"]
        raw = archive.read("submission/pack1_q3_kis.csv")

    assert not raw.startswith(b"\xef\xbb\xbf")
    decoded_rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    assert decoded_rows == rows["pack1_q3_kis"]


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
