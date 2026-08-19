# Module Retrieval (Bùi Trung Hiếu)

Module chịu trách nhiệm tiếp nhận câu truy vấn (query), tìm kiếm trong cơ sở dữ liệu keyframes (FAISS index) hoặc sinh ứng viên giả lập (Dummy), đồng thời hỗ trợ cơ chế loại trừ (exclude) và phân tầng kết quả theo vòng lặp suy luận Iterative Retrieval (lấy cảm hứng từ VideoSearch-R1).

---

## 1. Phần này làm gì?

1. **Quản lý & Tìm kiếm FAISS (`aic/retrieval/faiss_retriever.py`)**:
   - Tải `clip_faiss.index` (hoặc `siglip_faiss.index`) và metadata JSON tương ứng (177,321 keyframes).
   - Mã hoá câu truy vấn thành embedding vector, thực hiện tìm kiếm tương đồng cosine (Inner Product).
   - Hỗ trợ tham số `exclude` để loại bỏ hoàn toàn các video đã được xác định là không khớp từ các vòng trước.

2. **CLIP & SigLIP Retrievers (`aic/retrieval/clip.py`, `aic/retrieval/siglip.py`)**:
   - Kết nối với mô hình text encoder của OpenCLIP (ViT-B-32) và SigLIP2.
   - Trả về danh sách đối tượng `Candidate` chuẩn cho pipeline.

3. **Iterative Retrieval (`aic/pipeline.py -> iterative_retrieve`)**:
   - Chạy vòng lặp nhiều lượt (Retrieve ➡️ Verify ➡️ Exclude & Refine ➡️ Re-retrieve).
   - Phân loại trạng thái:
     - `not_matched`: đưa vào danh sách `exclude` cho lượt sau.
     - `unsure`: giữ lại trong danh sách kết quả nhưng xếp hạng sau nhóm `matched`.

4. **Query Processor (`aic/core/query_processor.py`)**:
   - Đọc file `.txt` từ BTC, tự động nhận diện loại bài toán (`kis`, `qa`, `trake`).
   - Hỗ trợ dịch tự động câu truy vấn từ Tiếng Việt sang Tiếng Anh.

---

## 2. Input / Output

### Input
- **`query: Query`**: Đối tượng truy vấn chứa `query_id`, `text_vi`, `text_en`, `task` (`"kis"` | `"qa"` | `"trake"`).
- **`k: int` / `limit: int`**: Số lượng ứng viên tối đa cần trả về (mặc định = 100).
- **`exclude: frozenset[str]`**: Tập hợp các `video_id` cần bỏ qua, không được trả về trong kết quả.
- **`verify_fn: Callable`** *(trong iterative_retrieve)*: Hàm nhận vào một `Candidate` và trả về một trong ba trạng thái: `"matched"`, `"not_matched"`, `"unsure"`.

### Output
- **`list[Candidate]`**: Danh sách ứng viên đã xếp hạng giảm dần theo độ phù hợp.
  Mỗi `Candidate` gồm:
  - `video_id: str`: Tên video (đã loại bỏ đuôi `.mp4`).
  - `start_frame: int`, `end_frame: int`: Mốc khung hình bắt đầu / kết thúc (0-based).
  - `representative_frames: list[int]`: Danh sách khung hình tiêu biểu (0-based).
  - `scores: dict[str, float]`: Điểm số tương đồng theo từng model (ví dụ `{"clip": 0.854}`).
  - `evidence: dict`: Chứa caption, metadata, hoặc reasoning từ VLM.

---

## 3. Chạy thế nào (Ví dụ gọi code)

### a. Tìm kiếm đơn giản với Dummy Retriever (Testing)
```python
from aic.core.types import Query
from aic.retrieval import dummy

query = Query(query_id="pack1_q3_kis", text_vi="cảnh cháy rừng ở châu Âu", task="kis")
candidates = dummy.search(query, k=100, exclude=frozenset({"L21_V001"}))

print(f"Số lượng kết quả: {len(candidates)}")
print(f"Top-1 video: {candidates[0].video_id}, score: {candidates[0].best_score}")
```

### b. Tìm kiếm với CLIP FAISS Index thật
```python
from aic.core.query_processor import make_query, translate_query
from aic.retrieval.clip import build_clip_retriever

# 1. Khởi tạo retriever
clip_retriever = build_clip_retriever(
    index_path="local/clip_faiss.index",
    metadata_path="local/clip_metadata.json"
)

# 2. Tạo query và dịch sang tiếng Anh cho CLIP
query = make_query("q1", text_vi="Người phụ nữ mặc áo đỏ đang đi dạo trong công viên")
query = translate_query(query)  # Sinh query.text_en

# 3. Tìm kiếm top 100
candidates = clip_retriever.search(query, k=100)
```

### c. Chạy vòng lặp Iterative Retrieval (Retrieve + Exclude + Unsure)
```python
from aic.pipeline import iterative_retrieve
from aic.retrieval import dummy
from aic.fusion import rank

# Hàm verify giả lập (hoặc kết nối Gemini/UI)
def mock_verify(cand):
    if cand.video_id == "L21_V001":
        return "not_matched"  # Loại bỏ video này ở các vòng sau
    if cand.video_id == "L22_V002":
        return "unsure"       # Giữ lại nhưng xếp sau nhóm matched
    return "matched"

candidates = iterative_retrieve(
    query=query,
    retrievers=[dummy],
    fuse_fn=rank.fuse,
    verify_fn=mock_verify,
    max_rounds=3,
    limit=100
)
```

---

## 4. Chưa làm / Blockers (Kế hoạch các ngày tới)

- [ ] **Tích hợp VLM Agentic Verify**: Thay hàm `verify_fn` giả lập bằng prompt Gemini 2.0 Flash / Qwen3-VL để tự động sinh suy luận `<think>...</think>` và chấm điểm match/not_matched/unsure trên ảnh keyframe thật.
- [ ] **Soft Query Refinement (SQR / Text Refinement)**: Khi nhận phản hồi `not_matched`, tự động viết lại câu truy vấn tiếng Anh (reformulate query) trước khi gọi round tiếp theo thay vì chỉ dùng lại query cũ.
- [ ] **Two-Stage Temporal Grounding**: Giải mã FPS gốc quanh khoảng 5–10s của keyframe tìm được để định vị chính xác khung hình theo yêu cầu của từng task (đặc biệt là TRAKE).
