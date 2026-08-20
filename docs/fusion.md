# Fusion

Module này fuse danh sách các ứng viên được trả về từ nhiều bộ máy tìm kiếm khác nhau, lọc các video trùng lặp và đảm bảo đầu ra luôn đủ số lượng kết quả yêu cầu (mặc định 100). Module được thiết kế để tối ưu hóa thứ hạng của các kết quả tốt nhất, đáp ứng mục tiêu đạt điểm cao ở các mốc R@1, R@5, R@20.

## Input / Output

- **Input:** 
  - `runs`: Danh sách kết quả từ nhiều retriever. Định dạng: `list[list[Candidate]]`.
  - `limit`: Số lượng kết quả đầu ra tối đa cần lấy (mặc định 100).
- **Output:** Trả về một `list[Candidate]` với độ dài chính xác bằng `limit`, đã trải qua các bước xử lý:
  1. **Lọc trùng:** Đảm bảo mỗi `video_id` chỉ xuất hiện 1 lần duy nhất trong danh sách.
  2. **Gộp điểm (Score Fusion):** Nếu một video được trả về nhiều lần từ nhiều nguồn, hệ thống tự động gộp các loại điểm (ví dụ: `clip`, `vlm_rerank`) và giữ lại điểm số cao nhất cho mỗi loại.
  3. **Xếp hạng (Reranking bằng RRF):** Sử dụng thuật toán **Reciprocal Rank Fusion (RRF)**. Khi gộp nhiều nguồn, điểm số thô thường khác thang đo (scale) nên không thể cộng trực tiếp. RRF quy đổi thứ hạng (`rank`) của mỗi video trong từng nguồn thành điểm số theo công thức `1 / (k + rank)` (với `k=60`). Sau đó cộng dồn điểm RRF từ các nguồn lại để xếp hạng chung, đảm bảo sự cân bằng và ưu tiên các kết quả xuất hiện nhiều lần ở thứ hạng cao.
  4. **Padding:** Nếu số lượng video độc lập ít hơn `limit`, tự động sinh ra các kết quả đệm (như `L00_V000`, `L00_V001`) với `start_frame=0` để đủ 100 dòng theo yêu cầu nộp bài.

## Chạy thế nào

Ví dụ cách gọi hàm `fuse` trong mã nguồn thực tế:

```python
from aic.core.types import Candidate
from aic.fusion.rank import fuse

# Kết quả giả định từ CLIP retriever
run_clip = [
    Candidate(video_id="L01_V010", start_frame=100, end_frame=150, scores={"clip": 0.8}),
    Candidate(video_id="L02_V020", start_frame=200, end_frame=250, scores={"clip": 0.9}),
]

# Kết quả giả định từ VLM reranker
run_vlm = [
    Candidate(video_id="L01_V010", start_frame=110, end_frame=160, scores={"vlm": 0.95}),
]

# Fuse kết quả và giới hạn 100 kết quả
fused_results = fuse([run_clip, run_vlm], limit=100)

for cand in fused_results:
    print(f"Video: {cand.video_id}, Điểm RRF: {cand.scores.get('fused')}")
```

## Chưa làm / Blockers

- Đối với truy vấn loại TRAKE (yêu cầu nhiều frame cho nhiều khoảnh khắc trong cùng một video), logic "giữ 1 frame đại diện duy nhất cho mỗi video" có thể sẽ cần điều chỉnh lại ở giai đoạn sau. Hiện tại hệ thống đang ưu tiên phục vụ truy vấn KIS.

