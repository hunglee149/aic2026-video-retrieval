"""Kiểm ranh giới tầng 1 sang tầng 2 — chỗ duy nhất được cộng thêm 1.

Toàn bộ file này phải XANH ngay từ đầu, không phụ thuộc vào phần của ai.

| Mã | Kiểm gì                                    | Đỏ nghĩa là                        |
|----|--------------------------------------------|------------------------------------|
| 01 | frame_idx 711 thành 712 khi nộp            | mọi câu 0 điểm, không có báo lỗi   |
| 02 | không có khung hình đề xuất thì lấy giữa   | nộp nhầm khung hình ở mép cảnh     |
| 03 | dòng KIS: 2 cột                            | sai định dạng, BTC không đọc được  |
| 04 | dòng Q&A: 3 cột, câu trả lời ở cuối        | mất câu trả lời                    |
| 05 | dòng TRAKE: n+1 cột, giữ đúng thứ tự       | sai thứ tự sự kiện                 |
| 06 | tên video bỏ đuôi .mp4                     | BTC loại dòng đó                   |

Dữ liệu dùng trong file này lấy thật từ L21_V001: khung hình số 7 nằm ở
frame_idx 711, pts_time 23.70.
"""

from aic.core.convert import to_answer, to_csv_row
from aic.core.types import Candidate


def make_candidate() -> Candidate:
    """Ứng viên mẫu, lấy số liệu thật của L21_V001."""
    return Candidate(
        video_id="L21_V001",
        start_frame=711,
        end_frame=858,
        representative_frames=[711],
        scores={"clip": 0.76},
    )


def test_01_frame_offset_applied_once():
    """Bên trong hệ thống đánh số từ 0, BTC đánh số từ 1."""
    answer = to_answer(make_candidate(), query_id="pack1_q3", rank=1)
    assert answer.frames == [712]


def test_02_falls_back_to_middle_frame():
    """Không có khung hình đề xuất thì lấy giữa khoảng, không lấy mép."""
    candidate = make_candidate()
    candidate.representative_frames = []
    answer = to_answer(candidate, query_id="pack1_q3", rank=1)
    assert answer.frames == [(711 + 858) // 2 + 1]


def test_03_kis_row_format():
    """KIS: tên video và một khung hình."""
    answer = to_answer(make_candidate(), query_id="pack1_q3", rank=1)
    assert to_csv_row(answer) == ["L21_V001", "712"]


def test_04_qa_row_format():
    """Q&A: thêm câu trả lời ở cuối dòng."""
    answer = to_answer(make_candidate(), query_id="pack1_q3", rank=1, answer="Hy Lạp")
    assert to_csv_row(answer) == ["L21_V001", "712", "Hy Lạp"]


def test_05_trake_row_format_keeps_order():
    """TRAKE: nhiều khung hình, thứ tự sự kiện phải giữ nguyên."""
    answer = to_answer(
        make_candidate(), query_id="pack1_q3", rank=1, frames=[711, 744, 802, 857]
    )
    assert to_csv_row(answer) == ["L21_V001", "712", "745", "803", "858"]


def test_06_strips_mp4_extension():
    """BTC yêu cầu tên video không kèm đuôi .mp4."""
    candidate = make_candidate()
    candidate.video_id = "L21_V001.mp4"
    answer = to_answer(candidate, query_id="pack1_q3", rank=1)
    assert to_csv_row(answer)[0] == "L21_V001"
