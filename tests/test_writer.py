import csv
import io
import zipfile

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
