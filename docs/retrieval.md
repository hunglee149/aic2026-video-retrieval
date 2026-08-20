# Module Retrieval (Bùi Trung Hiếu)

Module chịu trách nhiệm tiếp nhận câu truy vấn từ đề thi của BTC, phân tích mở rộng query (Query Expansion), tìm kiếm đa kênh song song (**Visual Vector Search + BM25 Bilingual Text Search + Object Detection Soft Filter**), hỗ trợ cơ chế loại trừ (`exclude`) và phân tầng kết quả theo vòng lặp suy luận Iterative Retrieval (lấy cảm hứng từ VideoSearch-R1).

---

## 1. Kiến trúc Đa Kênh (Multi-Modal Hybrid Retrieval)

Hệ thống kết hợp **4 nguồn dữ liệu độc lập** nhằm khắc phục triệt để điểm yếu của từng kênh riêng lẻ:

```
                            FILE QUERY BTC
                       (pack1_q3_kis.txt)
                               │
                               ▼
                   [Query Processor (LLM / Fallback)]
             Sinh song ngữ + Từ đồng nghĩa + Nhãn Object
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
 [Kênh 1 & 2: Visual]    [Kênh 3: BM25 Text]     [Kênh 4: Object Filter]
  CLIP & SigLIP2          ASR + OCR + Captions    OpenImages Soft Filter
  (177K Keyframes)        (629K Docs Song Ngữ)    (114K Keyframes có điểm)
       │                       │                       │
       ▼                       ▼                       ▼
  Scores: clip/siglip     Score: bm25             Score: object_match
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               ▼
                    [Score Fusion & Ranking]
                     (Khử trùng & xếp hạng)
                               ▼
                   Top-100 Candidates Final
```

---

## 2. Các Thành Phần Chính Trong Module

### 1. Visual Vector Search (`aic/retrieval/faiss_retriever.py`, `clip.py`, `siglip.py`)
- **CLIP ViT-B/32 (`clip_faiss.index`)**: 177,321 vectors 512-dim. Tốc độ tìm kiếm ~0.005s.
- **SigLIP2 (`siglip_faiss.index`)**: 177,321 vectors 1152-dim. Nắm bắt ngữ nghĩa hình ảnh chuyên sâu.
- **Tính năng**: Hỗ trợ lọc `exclude` (loại bỏ video `not_matched` từ các vòng trước).

### 2. BM25 Text Retriever (`aic/retrieval/text_retriever.py`)
- Quét trên **629,404 tài liệu văn bản** trong `text_search_index.pkl`:
  - **177,321 Caption Tiếng Việt** (VLM sinh mô tả chi tiết từng keyframe).
  - **177,321 Caption Tiếng Anh** (VLM sinh mô tả tiếng Anh chuẩn hóa).
  - **136,928 OCR Tiếng Việt** (Chữ trên màn hình: tít tin, bảng hiệu, logo đài).
  - **134,371 ASR Tiếng Việt** (Lời thoại bóc tách từ Faster-Whisper kèm timestamp).
  - **1,746 Tóm tắt Video** (Chủ đề chương trình, tên kênh).
- **Hỗ trợ tìm kiếm không dấu / có dấu / chữ hoa:** Tự động chuẩn hóa song song Unicode (`tokenize_bilingual`), đảm bảo gõ có dấu, gõ không dấu (`da lat`), hay chữ hoa bảng hiệu (`DA LAT`) đều khớp 100%.

### 3. Object Detection Soft Filter (`aic/retrieval/object_filter.py`)
- Nạp trực tiếp từ `objects_index.pkl` (13.6 MB) trong **0.05 giây**.
- Chứa toàn bộ 873 video, 114,885 keyframe có vật thể (`Person`, `Car`, `Food`, `Tree`, `Boat`...) đi kèm **Confidence Score thật**.
- **Cơ chế Soft Scoring**: Không loại bỏ ứng viên (tránh False Negative) mà cộng điểm thưởng (`object_match`) dựa trên tỷ lệ khớp và độ tin cậy của vật thể.

### 4. Query Processor Toàn Diện (`aic/core/query_processor.py`)
- **`process_query(query)`**:
  - Dịch sang tiếng Anh chuẩn (`query.text_en`) cho CLIP/SigLIP.
  - Tự động sinh từ đồng nghĩa tiếng Việt & tiếng Anh (`query.expanded_vi`, `query.expanded_en`).
  - Tự động rút trích danh sách vật thể (`query.objects`) cho `ObjectFilter`.
  - **Fallback Rule-based**: Hoạt động trơn tru ngay cả khi offline / mất kết nối API LLM.
- **`query.for_bm25()`**: Kết hợp tất cả từ khóa mở rộng thành chuỗi tìm kiếm phong phú cho BM25.

---

## 3. Cấu Trúc Dữ Liệu Đã Tối Ưu (`local/index/`)

Toàn bộ dữ liệu của **Batch 1 (10 batches L21 $\rightarrow$ L30, 873 video, 177,321 keyframe)** đã được nén gọn thành **6 file duy nhất** trong `local/index/`:

| Tên File | Dung lượng | Mô tả nội dung |
| :--- | :---: | :--- |
| `clip_faiss.index` | 346 MB | 177,321 vector CLIP ViT-B/32 (512 chiều) |
| `clip_metadata.json` | 14.4 MB | Metadata 1:1 cho vector CLIP |
| `siglip_faiss.index` | 779 MB | 177,321 vector SigLIP2 (1152 chiều) |
| `siglip_metadata.json` | 11.1 MB | Metadata 1:1 cho vector SigLIP2 |
| `text_search_index.pkl` | 537 MB | 629,404 documents (ASR + OCR + Captions song ngữ + BM25 index) |
| `objects_index.pkl` | 13.6 MB | 177,321 keyframe objects kèm detection scores của 873 video |

> **Ghi chú đối soát:** Đã kiểm tra đối chiếu trực tiếp với 14 file zip gốc trong `D:\AIC 2026\batch_01`, **trùng khớp 100%** không thiếu một video hay keyframe nào.

---

## 4. Hướng Dẫn Sử Dụng Code

### a. Tìm kiếm Text BM25 (Song ngữ + Không dấu)
```python
from aic.core.types import Query
from aic.retrieval.text_retriever import build_text_retriever

# 1. Khởi tạo retriever (load trong vài giây)
text_retriever = build_text_retriever("local/index/text_search_index.pkl")

# 2. Tìm kiếm (chấp nhận cả tiếng Việt có dấu, không dấu hoặc tiếng Anh)
query = Query(query_id="q1", text_vi="nguoi dan ong nau an trong bep")
candidates = text_retriever.search(query, k=100)

print(f"Top-1: {candidates[0].video_id}, Evidence: {candidates[0].evidence}")
```

### b. Chấm điểm bổ trợ bằng Object Filter
```python
from aic.retrieval.object_filter import build_object_filter

obj_filter = build_object_filter("local/index")

# Cộng điểm bonus cho các candidate nếu có chứa "Person" và "Food"
candidates = obj_filter.apply_scores(
    candidates,
    query_objects=["Person", "Food"],
    score_key="object_match",
    weight=0.3
)
```

### c. Chạy Pipeline Đa Kênh Tự Động
```python
from aic.core.query_processor import parse_query_file, process_query
from aic.retrieval import build_clip_retriever, build_siglip_retriever, build_text_retriever, build_object_filter
from aic.fusion.rank import fuse
from aic.pipeline import run

# 1. Xử lý câu hỏi đề thi
query = parse_query_file("queries/pack1_q3_kis.txt")
query = process_query(query)  # Tự động dịch, sinh từ đồng nghĩa, trích object

# 2. Nạp các nguồn tìm kiếm
retrievers = [
    build_clip_retriever("local/index/clip_faiss.index", "local/index/clip_metadata.json"),
    build_siglip_retriever("local/index/siglip_faiss.index", "local/index/siglip_metadata.json"),
    build_text_retriever("local/index/text_search_index.pkl"),
]

# 3. Tìm kiếm & gộp kết quả
candidates = run(query, retrievers, fuse_fn=fuse, write_fn=None, out_path=None)
```

---

## 5. Trạng Thái Hoàn Thành & Việc Tiếp Theo

- [x] **FAISS Vector Search**: CLIP & SigLIP2 hoạt động 100%.
- [x] **BM25 Text Search**: Quét đồng thời ASR, OCR, Captions song ngữ, hỗ trợ không dấu / viết hoa.
- [x] **Object Filter**: Gộp 178K frame objects thành `objects_index.pkl` (13.6MB) với điểm score chi tiết.
- [x] **Query Expansion**: Tự động sinh từ đồng nghĩa song ngữ & trích xuất nhãn vật thể.
- [x] **Đối soát dữ liệu**: 873/873 video, 177,321/177,321 keyframe khớp hoàn hảo 100%.
- [ ] **Kế hoạch tiếp theo**:
  - [ ] **VLM Agentic Verify**: Tích hợp Gemini 2.0 Flash / Qwen-VL duyệt ảnh keyframe thật để phân loại `matched`/`not_matched`/`unsure`.
  - [ ] **RRF Score Fusion**: Nâng cấp công thức gộp điểm của Hà sang Reciprocal Rank Fusion.
  - [ ] **Two-Stage Temporal Grounding**: Mở rộng khung hình lân cận cho bài toán TRAKE.
