# Đo chất lượng retrieval

Không có bộ đánh giá thì mọi thay đổi đều là đoán. Có nó rồi mới nói được "đổi X
làm top-5 tăng 6%" thay vì "chắc là tốt hơn", và mới chỉnh được trọng số RRF có
căn cứ thay vì bịa.

## Vì sao phải người gán nhãn, không tự sinh được

Đây là giới hạn thật, không phải lười:

- Câu lấy từ **ASR** không dùng để chấm kênh hình được. Lời đang nói thường
  không phải thứ đang hiện trên màn hình — người dẫn nói về Nhật Bản trong khi
  hình chiếu cảnh khác hẳn. CLIP không sai khi không tìm ra.
- Câu lấy từ **OCR** thì quay vòng với chính BM25, vì OCR đã nằm trong index.
  BM25 sẽ tìm ra ngay lập tức và con số đẹp một cách vô nghĩa.

Nên chỉ có người mở video, xem, rồi viết mô tả mới cho ra ground truth dùng được.
Máy làm phần còn lại: chọn sẵn khoảnh khắc kiểm chứng được và chấm điểm.

## Quy trình

### 1. Sinh phiếu nháp

```bash
python scripts/make_sanity_drafts.py --count 30 --out eval/sanity_set.jsonl
```

Script chỉ chọn video **có file `.mp4`** để bạn tua được, tránh đoạn đầu/cuối
video (hình hiệu, quảng cáo), mỗi video một khoảnh khắc, và in kèm ngữ cảnh
ASR/OCR quanh đó làm gợi ý.

### 2. Điền mô tả

Mỗi dòng mở ảnh ở `_keyframe_image` (hoặc tua video tới `_pts_time`), rồi:

- điền `text_vi`: **tả cảnh nhìn thấy**;
- đổi `verified` thành `true`.

Bốn quy tắc, nếu phá thì bộ này hết đo được cái cần đo:

1. Tả **cái nhìn thấy**, đừng chép lại chữ trong `_ocr_hint` — chép vào là biến
   nó thành bài kiểm tra trí nhớ của BM25.
2. Đừng nhắc tên video, tên kênh, ngày phát sóng.
3. Viết như đề thi BTC: một câu tiếng Việt, đủ chi tiết để phân biệt với cảnh
   khác trong kho.
4. Cảnh nào nhìn không rõ thì bỏ dòng đó, đừng đoán.

Trường bắt đầu bằng `_` chỉ là gợi ý cho người gán nhãn; script chấm điểm bỏ qua
hết. Chúng chứa đường dẫn tuyệt đối của máy bạn nên sinh lại trên máy khác là được.

30 câu, mỗi câu khoảng 2 phút, tổng chừng một tiếng.

### 3. Chấm

```bash
python scripts/eval_retrieval.py --set eval/sanity_set.jsonl
```

Chỉ dòng `verified: true` được tính vào con số chính; phiếu chưa kiểm báo riêng.
Chưa có dòng nào `verified` thì script in cảnh báo to và bảo đừng trích dẫn.

## Đọc kết quả

| Chỉ số | Nghĩa |
|---|---|
| **Video Recall@K** | video đúng có nằm trong K ứng viên đầu không |
| **Moment hit@K** | có ứng viên vừa đúng video **vừa đúng cửa sổ frame** không |
| **Điểm dạng BTC** | `mean(R@1, R@5, R@20, R@50, R@100)` theo công thức đề thi |
| **Nguồn lập công** | ứng viên đúng do CLIP, SigLIP hay BM25 đưa vào |

Nhìn **Moment hit** chứ đừng nhìn Video Recall. Đúng video mà sai frame thì
R-Score vẫn bằng 0 — Video Recall chỉ là chặn trên.

Cột "nguồn lập công" là thứ dùng để chỉnh trọng số. Nếu BM25 lập công ở hầu hết
các câu thì tăng trọng số BM25 là có căn cứ; nếu nó chỉ thắng ở những câu bạn
viết lẫn chữ trên màn hình vào thì đó là bộ nhãn hỏng, không phải BM25 giỏi.

## Frame ở hệ nào

`frame` và `window` trong sanity set là **actual video frame, 0-based**, cùng hệ
với `Candidate.start_frame`. **Không** phải frame 1-based lúc nộp bài. Việc `+1`
chỉ xảy ra ở boundary submission.

Thiếu `window` thì lấy `frame ± --frame-tolerance` (mặc định 150 ≈ 5 giây ở
30fps — chặt hơn cửa sổ KIS thật, nên con số ra sẽ hơi bi quan).

## So sánh cấu hình

```bash
# thử trọng số khác
python scripts/eval_retrieval.py --set eval/sanity_set.jsonl \
    --weights clip=1.0,bm25=2.0

# tắt một nguồn để xem nó đóng góp bao nhiêu
python scripts/eval_retrieval.py --set eval/sanity_set.jsonl --disable bm25
```

Đổi một thứ mỗi lần, ghi lại số. Trọng số hiện tại để bằng nhau (`clip = siglip
= bm25 = 1.0`) chính là vì chưa có bộ này để đo.

## Cỡ mẫu

30 câu thì sai số quanh mỗi con số khoảng ±9 điểm phần trăm. Đủ để thấy khác
biệt lớn (top-5 từ 40% lên 70%), **không** đủ để phân biệt 62% với 65%. Muốn
chốt những khác biệt nhỏ thì cần 100+ câu.
