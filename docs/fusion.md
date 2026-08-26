# Fusion

Trộn kết quả từ nhiều retriever thành **một danh sách moment đã xếp hạng**, để
tối ưu các mốc R@1, R@5, R@20, R@50, R@100 của BTC.

## Input / Output

- **Input**
  - `runs`: `list[list[Candidate]]` — mỗi phần tử là kết quả *đã xếp hạng* của
    một retriever.
  - `limit`: số moment tối đa trả về (mặc định 100).
  - `weights`: trọng số theo tên nguồn (tuỳ chọn).
  - `rrf_k`: hằng số RRF (mặc định 60).
  - `merge_radius`: bán kính gộp moment theo frame (mặc định 0).
- **Output**: `list[Candidate]` dài **tối đa** `limit` — có thể ngắn hơn.

## Xếp hạng bằng weighted RRF

Điểm thô của các nguồn khác thang đo: CLIP/SigLIP cho cosine trong `[0, 1]`, BM25
cho điểm không chặn trên. Cộng thẳng thì nguồn có thang đo lớn hơn luôn thắng bất
kể thứ hạng. RRF chỉ dùng **thứ hạng** nên miễn nhiễm với chuyện đó:

```text
RRF(candidate) = Σ_source  weight(source) / (rrf_k + rank_source)
```

`rank_source` là vị trí 1-based trong run của nguồn đó. Trọng số mặc định để
bằng nhau (`clip = siglip = bm25 = 1.0`) và nằm ở `DEFAULT_WEIGHTS` trong
`aic/fusion/rank.py`. Chưa tinh chỉnh vì chưa có ground truth để đo — đoán trọng
số lúc này là tự bịa.

Thứ tự cuối được sắp theo `(-rrf, video_id, anchor_frame)` nên **hoàn toàn tất
định**, chạy lại cho ra đúng một kết quả.

## Danh tính là moment, không phải video

Khoá gộp là `(video_id, anchor_frame)`, trong đó `anchor_frame` là
`representative_frames[0]` (một actual video frame 0-based).

Gộp mọi kết quả của cùng một video thành một dòng sẽ:

- giết TRAKE, vì TRAKE cần nhiều mốc sự kiện **trong cùng một video**;
- vứt mất các cảnh khác nhau của một video dài, trong khi mỗi cảnh là một ứng
  viên độc lập.

`merge_radius=0` nghĩa là chỉ gộp khi **trùng đúng frame** — tức khi hai nguồn
cùng trỏ về đúng một keyframe. Đây là mức an toàn: không nới rộng bằng heuristic
chưa được đo. Nới ra thì hai moment cách nhau `<= merge_radius` frame sẽ nhập
làm một.

Khi gộp: điểm lấy `max` theo từng khoá nguồn, evidence hợp nhất, `start_frame` /
`end_frame` mở rộng thành bao lồi, nhưng `representative_frames` vẫn giữ **đúng
một** anchor — vì `to_answer()` dùng list đó làm frame nộp bài khi operator chưa
chọn tay, nhiều phần tử sẽ đẻ ra dòng KIS/Q&A sai định dạng.

`fuse` **không sửa** các `Candidate` đầu vào; nó dựng object mới.

## Không độn candidate giả

Bản trước độn `L00_V000`, `L00_V001`… cho đủ `limit`. Đã bỏ hẳn. Dòng độn không
bao giờ ghi điểm mà lại có nguy cơ đi thẳng vào `submission.zip`. Ít kết quả thật
thì trả ít, đúng bằng số thật.

## Chạy thế nào

```python
from aic.core.types import Candidate
from aic.fusion.rank import fuse

run_clip = [
    Candidate(video_id="L21_V001", start_frame=100, end_frame=100,
              representative_frames=[100], scores={"clip": 0.31}),
    Candidate(video_id="L21_V001", start_frame=5000, end_frame=5000,
              representative_frames=[5000], scores={"clip": 0.29}),
]
run_bm25 = [
    Candidate(video_id="L21_V001", start_frame=100, end_frame=100,
              representative_frames=[100], scores={"bm25": 12.5}),
]

fused = fuse([run_clip, run_bm25], limit=100, weights={"clip": 1.0, "bm25": 1.0})
for c in fused:
    print(c.video_id, c.start_frame, c.scores["fused"])
```

Hai moment `100` và `5000` của cùng `L21_V001` vẫn tồn tại song song; moment
`100` được cả hai nguồn trả nên xếp trên.

## Chưa làm

- Trọng số theo loại query (KIS / Q&A / TRAKE) chưa có.
- `merge_radius` theo thời gian thực (giây × fps) thay vì theo frame cố định.
- Chưa có rerank bằng VLM.
