# AIC 2026

Hệ thống tìm sự kiện trong video cho cuộc thi AI Challenge HCMC 2026. Nhận một
câu mô tả, tìm đúng khoảnh khắc trong kho video, xuất ra file nộp cho BTC.

Ba loại câu hỏi: KIS tìm khoảnh khắc, Q&A tìm khoảnh khắc kèm câu trả lời,
TRAKE tìm nhiều mốc sự kiện trong cùng một video. Đợt 1 tập trung KIS; Q&A và
TRAKE vận hành thủ công qua giao diện.

## Pipeline 

```
câu hỏi -> tìm kiếm -> trộn và lọc →  -> xuất file nộp
```

Mỗi mũi tên là một thư mục, mỗi thư mục một người giữ. Không ai cần đọc code
của người khác, chỉ cần biết mình nhận vào cái gì và trả ra cái gì.

## Hai tầng dữ liệu

**Tầng 1 — `Candidate`.** Tầng tìm kiếm trả về: frame nào trong video nào, điểm số.

```
video_id, start_frame, end_frame, representative_frames, scores, evidence
```

**Tầng 2 — `Answer`.** Sau khi operator chọn xong.

```
query_id, rank, video_id, frames, answer
```

Ba loại câu hỏi khác nhau **chỉ ở số phần tử trong `frames` và có `answer` hay
không**:

```
KIS     1 khung hình, không answer   →  L21_V001, 712
Q&A     1 khung hình, có answer      →  L21_V001, 712, Hy Lạp
TRAKE   n khung hình, không answer   →  L21_V001, 712, 745, 803, 858
```

## Ai làm phần gì

| Thư mục | Người l |
|---|---|
| `aic/core/` | chung, muốn thay đổi thì nhắn cho cả nh |
| `aic/retrieval/clip.py` | Hiếu |
| `aic/fusion/` | Hà |
| `aic/ui/` | Dương |
| `aic/submission/` | Hạ |
| `aic/retrieval/dummy.py`, `aic/pipeline.py`, `tests/` | H |
| `docs/` | mỗi người một file |


## Làm việc trên nhánh

Mỗi người một nhánh, đặt tên `tên-mình/phần-mình-làm`. Ví dụ `Hung/test`.

Tạo PR nếu cần, merge vào `main` nếu chạy được và có recheck.

## Chạy thử

```bash
pip install -r requirements.txt
pytest
```

Chạy UI
```bash
python -m uvicorn aic.ui.app:app --reload --port 8000
```

### Dịch query tiếng Việt bằng model local

Mặc định UI dùng `Helsinki-NLP/opus-mt-vi-en` để dịch query sang tiếng Anh
cho CLIP. Không cần `GEMINI_API_KEY`. Lần dịch đầu tiên Hugging Face sẽ tải
model về cache của máy; các lần chạy sau dùng lại cache đó.

Thiết bị được chọn tự động: CUDA nếu PyTorch nhận GPU, ngược lại dùng CPU.
Có thể ép thiết bị khi chạy:

```bash
AIC_TRANSLATION_DEVICE=cuda python -m uvicorn aic.ui.app:app --reload --port 8000
```

Các biến cấu hình tùy chọn:

- `AIC_TRANSLATION_DEVICE=auto|cuda|cpu`
- `AIC_TRANSLATION_MODEL=Helsinki-NLP/opus-mt-vi-en`

## Xuất submission đã kiểm tra

Dùng **query pack đầy đủ** BTC cung cấp: nạp file ZIP, hoặc nạp toàn bộ các
file TXT của một pack. Chế độ một TXT chỉ dành cho phát triển/kiểm thử, không
dùng cho bài nộp chính thức. Với từng query TRAKE, kiểm tra lại số event, nhập
số đó và bấm xác nhận trước khi chọn frame.

Trong màn hình xuất, xử lý mọi lỗi hiện trong validation report (mỗi query cần
có dòng hợp lệ). Chỉ tải `submission.zip` khi export trả về trạng thái **PASS**;
hệ thống kiểm tra lại ZIP vừa tạo trước khi trả file. Kiểm tra này hỗ trợ bắt
lỗi định dạng và không thay thế submission validator chính thức của BTC — luôn
chạy validator đó trước khi nộp.

## Viết tài liệu

Mỗi người một file trong `docs/`: phần này làm gì, nhận
vào gì trả ra gì, vài dòng ví dụ cách gọi, và cái gì chưa làm.
