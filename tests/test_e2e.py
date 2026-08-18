"""Kiểm cả đường ống, từ câu hỏi tới file nộp.

| Mã | Kiểm gì                          | Trạng thái mong đợi                        |
|----|----------------------------------|--------------------------------------------|
| 01 | đường ống nối thông              | XANH ngay từ đầu, không phụ thuộc vào ai   |
| 02 | đường ống trên module thật       | BỎ QUA kèm tên người còn thiếu, tự xanh dần|

Bài 01 dùng bản trộn và bản ghi file tối giản viết ngay trong file này, nên nó
chỉ kiểm phần nối và phần đổi số khung hình. Bài 02 gọi module thật của cả
team; ai chưa xong thì bài này bỏ qua và in ra tên người đó, rồi tự chuyển
thành chạy thật ngay khi người đó đẩy code lên.

Cả hai bài đều đi qua cùng một bộ kiểm ở `assert_valid_submission`:

    1. trong file nén có thư mục submission/
    2. tên file CSV khớp tên file câu hỏi
    3. mã hoá UTF-8 và không có dấu BOM
    4. số dòng nằm trong giới hạn
    5. tên video không còn đuôi .mp4
    6. khung hình đánh số từ 1 trở lên
"""

import csv
import io
import zipfile

import pytest

from aic import pipeline
from aic.core.types import Query
from aic.retrieval import dummy

QUERY = Query(query_id="pack1_q3_kis", text_vi="cảnh cháy rừng ở châu Âu", task="kis")


# --------------------------------------------------------------------------
# Bản tối giản, chỉ phục vụ bài 01. Không phải code thật của Hà và Hạ.
# --------------------------------------------------------------------------


def _fuse_minimal(runs, limit=100):
    seen, out = set(), []
    for run in runs:
        for candidate in run:
            key = (candidate.video_id, candidate.start_frame)
            if key not in seen:
                seen.add(key)
                out.append(candidate)
    return out[:limit]


def _write_minimal(rows_by_query, out_path):
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, rows in rows_by_query.items():
            buffer = io.StringIO(newline="")
            csv.writer(buffer).writerows(rows)
            archive.writestr("submission/%s.csv" % name, buffer.getvalue())
    return str(out_path)


# --------------------------------------------------------------------------
# Bộ kiểm dùng chung
# --------------------------------------------------------------------------


def assert_valid_submission(zip_path, query_name, max_rows=100):
    """Sáu điều kiện một file nộp hợp lệ phải thoả."""
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()

        assert any(n.startswith("submission/") for n in names), (
            "thiếu thư mục submission/ trong file nén, đang có: %s" % names
        )

        csv_name = "submission/%s.csv" % query_name
        assert csv_name in names, "thiếu %s, đang có: %s" % (csv_name, names)

        raw = archive.read(csv_name)

    assert not raw.startswith(b"\xef\xbb\xbf"), (
        "file CSV đang có dấu BOM — BTC sẽ loại toàn bộ bài nộp"
    )
    text = raw.decode("utf-8")  # hỏng ở đây nghĩa là không phải UTF-8

    rows = list(csv.reader(io.StringIO(text)))
    assert 0 < len(rows) <= max_rows, "có %d dòng, tối đa %d" % (len(rows), max_rows)

    for row in rows:
        assert not row[0].endswith(".mp4"), "tên video còn đuôi .mp4: %s" % row[0]
        for value in row[1:]:
            if value.lstrip("-").isdigit():
                assert int(value) >= 1, "khung hình phải từ 1 trở lên, đang là %s" % value


def _review_minimal(candidates):
    """Giả lập người vận hành: giữ 5 ứng viên đầu, đổi khung hình của hạng 1,
    và nhập một câu trả lời cho hạng 1."""
    kept = candidates[:5]
    chosen_frames = {1: [kept[0].start_frame]}
    answers = {1: "Hy Lạp"}
    return kept, chosen_frames, answers


# --------------------------------------------------------------------------
# Ba bài kiểm tra
# --------------------------------------------------------------------------


def test_01_pipeline_is_wired(tmp_path):
    """Câu hỏi giả đi hết đường ống và ra file nộp đúng cấu trúc."""
    out_path = tmp_path / "submission.zip"
    pipeline.run(QUERY, [dummy], _fuse_minimal, _write_minimal, str(out_path), limit=100)
    assert_valid_submission(out_path, QUERY.query_id)


def test_02_pipeline_with_real_modules(tmp_path):
    """Như bài 01 nhưng dùng module thật. Bỏ qua khi còn người chưa xong."""
    from aic.fusion import rank
    from aic.submission import writer

    out_path = tmp_path / "submission.zip"
    try:
        pipeline.run(QUERY, [dummy], rank.fuse, writer.write_submission, str(out_path))
    except NotImplementedError as exc:
        pytest.skip("còn thiếu: %s" % exc)
    assert_valid_submission(out_path, QUERY.query_id)


def test_03_review_step_is_honoured(tmp_path):
    """Bước người duyệt phải ảnh hưởng được tới file nộp.

    Đây là chỗ nối của Dương. Không có bài này thì phần giao diện làm xong vẫn
    không ai gọi tới và không ai biết.
    """
    out_path = tmp_path / "submission.zip"
    pipeline.run(
        QUERY,
        [dummy],
        _fuse_minimal,
        _write_minimal,
        str(out_path),
        review_fn=_review_minimal,
    )
    assert_valid_submission(out_path, QUERY.query_id, max_rows=5)

    with zipfile.ZipFile(out_path) as archive:
        text = archive.read("submission/%s.csv" % QUERY.query_id).decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))

    assert len(rows) == 5, "người duyệt giữ 5 ứng viên, file nộp phải có 5 dòng"
    assert rows[0][-1] == "Hy Lạp", "câu trả lời người dùng nhập bị mất"
