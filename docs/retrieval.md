# Module Retrieval

Nhận query của BTC, tìm kiếm song song trên nhiều kênh, chuẩn hoá về `Candidate`
với **actual video frame**, rồi để `aic/fusion` trộn lại.

> Tài liệu này mô tả **những gì code đang làm**, đã đối chiếu với test và với
> artifacts thật trên máy. Phần chưa làm được ghi rõ ở mục 6.

---

## 1. Kiến trúc

```text
                  query .txt của BTC (tiếng Việt)
                               │
                               ▼
              aic/core/query_processor.process_query
        Helsinki-NLP dịch VI→EN  +  trích nhãn object theo rule
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   CLIP ViT-B/32          SigLIP2 SO400M          BM25 văn bản
   512-dim, FAISS         1152-dim, FAISS         ASR + OCR + media info
   scores["clip"]         scores["siglip"]        scores["bm25"]
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                 aic/fusion/rank.fuse — weighted RRF
                 danh tính = (video_id, anchor_frame)
                               ▼
                    top-K Candidate thật (không độn)
```

Mọi retriever có cùng interface:

```python
search(query: Query, limit: int = 100, exclude: frozenset = frozenset())
    -> list[Candidate]
```

`k=` vẫn dùng được như shim tương thích ngược và có test riêng.

---

## 2. Quy ước frame — sai chỗ này là mất trắng điểm

| Khái niệm | Ý nghĩa | Ví dụ |
|---|---|---|
| `keyframe_num` | ordinal ảnh keyframe, **1-based**, khớp tên file | `18` → `018.jpg` |
| `frame_idx` | **actual video frame, 0-based** = `int(pts_time × fps)` | `1500` |
| frame nộp bài | 1-based, cộng `+1` **chỉ** ở boundary submission | `1501` |

- `Candidate.start_frame` / `representative_frames` **luôn** là `frame_idx`.
- Việc `+1` chỉ xảy ra ở `aic/core/convert.py` và `candidateToSubmissionFrame()`.
- Document BM25 không có frame; nó có mốc thời gian. Frame được suy ra qua
  `keyframe_map`. **Video nào không có keyframe map thì bỏ hẳn document đó** —
  không bao giờ lấy ordinal thay frame thật.

Đã kiểm chứng trên toàn bộ 177.321 keyframe của batch 1: `frame_idx` trong
metadata khớp 100% với `map-keyframes/*.csv`, và `frame_idx == int(pts_time × fps)`
đúng cho cả 177.321 dòng.

---

## 3. Các kênh

### 3.1 CLIP (`clip.py` + `faiss_retriever.py`)

- Index: `clip_faiss.index`, `IndexFlatIP`, 177.321 vector × 512 chiều, đã
  L2-normalize.
- Nguồn: chính bộ `clip-features-32` BTC phát (`clip-ViT-B-32`, trọng số OpenAI).
  Đã đối chiếu trực tiếp: cosine giữa vector trong index và file `.npy` gốc = **1.0000**.
- Text encoder: `open_clip` **`ViT-B-32-quickgelu`** + `openai`.
  Dùng `ViT-B-32` thường sẽ nạp được nhưng lệch không gian embedding — đo bằng
  image tower trên chính feature BTC: quickgelu khớp `0.9994`, bản thường `0.9548`.
- Số chiều encoder được đo lúc khởi tạo và đối chiếu với `index.d`; lệch thì
  raise ngay chứ không âm thầm đổi encoder khác.

### 3.2 SigLIP2 (`siglip.py`)

- Index: `siglip_faiss.index`, `IndexFlatIP`, 177.321 vector × 1152 chiều.
- **Model phải là `open_clip` `hf-hub:timm/ViT-SO400M-14-SigLIP2`** — đúng thứ
  đã dùng để dựng index trong `drive/notebooks/trake_indexing.ipynb`.
  `transformers.SiglipTextModel` với `google/siglip2-*` cũng cho ra 1152 chiều
  và vẫn search được, nhưng khác không gian embedding nên kết quả là rác im lặng.
  Trùng số chiều **không** đủ để kết luận cùng model.
- Không cấu hình `AIC_SIGLIP_INDEX_PATH`/`AIC_SIGLIP_META_PATH` thì kênh này báo
  `disabled`, không phải `error`.

### 3.3 BM25 (`text_retriever.py`)

Index thật trên máy hiện tại (`text_search_index.pkl`, 82,1 MB, 181.793 document):

| Loại document | Số lượng | Phủ |
|---|---:|---|
| `transcript_segment` (ASR theo đoạn, có timestamp) | 134.371 | 873 video |
| `ocr` (chữ trên màn hình, mỗi keyframe một document) | 45.705 | **206/873 video** |
| `media_info` (tiêu đề, mô tả, tag kênh) | 873 | 873 video |
| `transcript_full` (ASR cả video) | 844 | 844 video |

OCR được bổ sung bằng `scripts/build_text_index.py` từ `drive/ocr/`. Còn thiếu
667 video chưa chạy OCR — chạy xong thì gọi lại đúng script đó, nó thay thế
phần `ocr` cũ nên chạy bao nhiêu lần cũng ra một kết quả.

Script dựng index cần `underthesea` (đúng tokenizer đã dựng index gốc — đã đối
chiếu 300/300 document mẫu). Nó **chỉ cần lúc dựng**, không cần lúc chạy app,
nên không nằm trong `requirements.txt`: `pip install underthesea`.

**Caption thì chưa có gì** (cần VLM sinh mô tả cho 177K keyframe).

Document OCR ghi kèm `keyframe_num`, nên frame lấy bằng **tra bảng
`keyframe_map`** — chính xác tuyệt đối, khác với document ASR phải suy từ mốc
thời gian. Đã đối chiếu mẫu với `map-keyframes`: 0 sai lệch.

Xử lý query:

- Index tiếng Việt đã word-segment nên chứa token ghép (`giao_thông`, `đà_lạt`);
  chiếm 22,2% tổng số token. Query người dùng gõ rời, nên phải sinh thêm n-gram
  nối bằng `_` (2-gram và 3-gram) thì mới khớp được.
- Tra không dấu: dựng sẵn map `unaccented(term) → term có dấu` từ vocab, nên
  `da lat` tìm ra `đà_lạt`. Giới hạn 8 biến thể/token để query không nổ ra nhiễu.
- Stopword giữ ở mức vừa phải, chỉ hư từ thuần chức năng; có test khoá lại rằng
  từ mang nghĩa (màu sắc, số đếm, danh từ, động từ) không bị xoá.
- BM25 chuẩn (`k1=1.5`, `b=0.75`), mẫu số dùng **độ dài document thật**.
- Dùng luôn `inverted`/`idf`/`avgdl` nếu pickle đã precompute; không thì build
  trong bộ nhớ (~4,6s cho 182K document) và **không ghi file cache** vào thư mục
  artifacts dùng chung.
- Lọc theo modality được cả ở lúc khởi tạo lẫn theo từng request
  (`Query.modalities`).
- Mỗi video giữ tối đa `per_video_limit=3` moment khác nhau.

### 3.4 ObjectFilter (`object_filter.py`)

Module có sẵn nhưng **chưa nối vào pipeline/UI**, và trên máy hiện tại **không có
`objects_index.pkl`**. Xem mục 6.

---

## 4. Cấu hình

Xem `.env.example`. Các biến của module này:

| Biến | Bắt buộc | Ý nghĩa |
|---|---|---|
| `AIC_INDEX_PATH` / `AIC_META_PATH` | có | CLIP index + metadata |
| `AIC_TEXT_INDEX_PATH` | có | pickle BM25 |
| `AIC_SIGLIP_INDEX_PATH` / `AIC_SIGLIP_META_PATH` | không | bật kênh SigLIP |
| `AIC_CLIP_MODEL` / `AIC_CLIP_PRETRAINED` | không | mặc định `ViT-B-32-quickgelu` / `openai` |
| `AIC_SIGLIP_MODEL` | không | mặc định `hf-hub:timm/ViT-SO400M-14-SigLIP2` |
| `AIC_CLIP_DEVICE` / `AIC_SIGLIP_DEVICE` | không | `auto` \| `cuda` \| `cpu` |
| `AIC_HF_CACHE_DIR` | không | thư mục cache HuggingFace |
| `AIC_DISABLE_NEURAL` | không | `1` để chỉ chạy BM25 |

Không hardcode đường dẫn tuyệt đối của máy nào trong code.

---

## 5. Dùng thế nào

```python
from aic.core.types import Query
from aic.retrieval.clip import build_clip_retriever
from aic.retrieval.text_retriever import TextRetriever
from aic.fusion.rank import fuse
from aic.pipeline import retrieve_and_fuse

clip = build_clip_retriever("<...>/clip_faiss.index", "<...>/clip_metadata.json")
bm25 = TextRetriever("<...>/text_search_index.pkl", name="bm25")

query = Query(
    query_id="pack1_q1_kis",
    text_vi="cảnh sát giao thông điều tiết trên đường phố đông đúc",
    text_en="traffic police directing traffic on a crowded street",
)

candidates = retrieve_and_fuse(query, [clip, bm25], fuse, limit=100)
for c in candidates[:5]:
    print(c.video_id, c.start_frame, c.scores)
```

Trạng thái từng nguồn xem ở `GET /api/status`; mỗi nguồn có `state` là một trong
`disabled` / `loading` / `ready` / `error`.

---

## 6. Chưa làm

- **OCR mới phủ 206/873 video**; 667 video còn lại chưa chạy OCR. Caption do VLM
  sinh thì chưa có gì.
- **ObjectFilter chưa nối** vào pipeline và thiếu `objects_index.pkl`.
- **Query expansion ngữ nghĩa chưa có.** Helsinki-NLP chỉ dịch, không sinh từ
  đồng nghĩa, nên `expanded_vi` / `expanded_en` mặc định vẫn rỗng.
- **Iterative multi-round chưa nối UI.** `exclude` đã hoạt động đúng ở mọi
  retriever nên phần backend đã sẵn sàng, nhưng UI Iterative hiện chỉ phân loại
  candidate đã có, chưa gọi lại `/api/search`.
- **Trọng số RRF chưa được tinh chỉnh** — đang để bằng nhau. Bộ đo đã có
  (`docs/evaluation.md`), còn thiếu 30 câu gán nhãn tay để chạy.
- **TRAKE chưa có truy hồi đa sự kiện tự động**; Q&A chưa có VLM sinh câu trả lời.
