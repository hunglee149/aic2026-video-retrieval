# Module Retrieval & Web UI (Hệ Thống Tìm Kiếm Video AIC 2026)

Tài liệu chi tiết về kiến trúc tìm kiếm đa phương thức (Multi-Modal Retrieval), các tác vụ thi đấu (Tasks), định dạng Input/Output, cấu trúc thư mục dữ liệu, và giao diện tương tác người dùng.

---

## 1. Các Tác Vụ Thi Đấu (Competition Tasks)

Hệ thống hỗ trợ đầy đủ 3 bài toán chính của cuộc thi **Ho Chi Minh City AI Challenge (AIC 2026)**:

| Tác vụ | Mô tả đề bài | Input | Output |
| :--- | :--- | :--- | :--- |
| **1. KIS (Known-Item Search)** | Tìm video và **chính xác khung hình (frame index)** nơi diễn ra sự kiện được mô tả | 1 đoạn văn bản mô tả sự kiện (tiếng Việt) | `video_id`, `frame_idx` (1 frame duy nhất) |
| **2. QA (Video Question Answering)** | Tìm video chứa ngữ cảnh và trả lời câu hỏi ngắn về sự kiện/đối tượng | Dòng 1: Mô tả bối cảnh<br>Dòng 2: Câu hỏi cụ thể (`?`) | `video_id`, `frame_idx`, `answer` (câu trả lời) |
| **3. TRAKE (Temporal Retrieval)** | Định vị chuỗi sự kiện diễn ra tuần tự theo dòng thời gian của video | Danh sách $N$ sự kiện liên tiếp ($E_1, E_2, \dots, E_N$) | `video_id`, chuỗi frame $[f_1, f_2, \dots, f_N]$ |

---

## 2. Cấu Trúc Thư Mục Dữ Liệu (Data Structure)

Dữ liệu được tổ chức chuẩn hóa, gọn gàng và phân tách rõ ràng giữa mã nguồn, file chỉ mục (`local/`), và dữ liệu gốc (`data/`):

```
aic2026-video-retrieval/
├── aic/                            # Mã nguồn hệ thống
│   ├── core/                       # types, query_processor, convert
│   ├── retrieval/                  # text_retriever, faiss_retriever, clip, siglip, object_filter
│   ├── fusion/                     # rank (RRF, Borda, Weighted)
│   ├── pipeline.py                 # Pipeline tìm kiếm & lặp (single-pass, iterative)
│   └── ui/                         # FastAPI Web App + Giao diện HTML/CSS/JS
│       ├── app.py                  # API endpoints backend
│       └── static/                 # index.html, style.css, main.js
│
├── local/                          # Thư mục chứa toàn bộ file chỉ mục (Đã làm phẳng)
│   ├── text_search_index.pkl       # 629,404 văn bản (ASR + OCR + Captions + Inverted Index + IDF)
│   ├── clip_faiss.index            # 177,321 vector CLIP ViT-B/32 (346 MB)
│   ├── clip_metadata.json          # Metadata vector CLIP
│   ├── siglip_faiss.index          # 177,321 vector SigLIP2 (779 MB)
│   ├── siglip_metadata.json        # Metadata vector SigLIP2
│   ├── objects_index.pkl           # 114K keyframes nhận diện vật thể OpenImages
│   └── video_keyframes_map.json    # Bảng ánh xạ 873 video vào 14 file zip (Đọc ảnh < 1ms)
│
├── data/                           # Thư mục dữ liệu media gốc
│   └── keyframes/                  # 14 file zip chứa toàn bộ keyframe của Batch 1 (L21 -> L30)
│       ├── Keyframes_L21.zip
│       ├── Keyframes_L22.zip
│       ├── ...
│       └── Keyframes_L30.zip
│
└── docs/                           # Tài liệu kỹ thuật, slide tập huấn và hướng dẫn
```

> [!NOTE]
> **Không cần giải nén file Zip:** Server đọc ảnh trực tiếp từ 14 file zip trong `data/keyframes/` thông qua `video_keyframes_map.json` với tốc độ `< 1ms/ảnh`, tiết kiệm hơn **60 GB** dung lượng ổ cứng.

---

## 3. Kiến Trúc Tìm Kiếm & Luồng Xử Lý (Pipeline Architecture)

```
                            CÂU TRUY VẤN (QUERY)
                                     │
                                     ▼
                   [Query Processor: Mở rộng Caption]
          Tiếng Việt + Dịch Caption Tiếng Anh + Từ đồng nghĩa
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       [BM25 Text Retriever]                   [Visual Retrievers]*
     (ASR + OCR + Captions 629K)                 (CLIP & SigLIP2)
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                       [Score Fusion & Normalization]
                      Chuẩn hóa điểm số về [0.0000 - 1.0000]
                                     │
                                     ▼
                     [Candidate Selection & Ranking]
                     Top-K Video Candidates + Evidence
                                     │
                                     ▼
                     [Interactive Keyframe Scrubber]
                 Duyệt Keyframe Timeline & Xuất kết quả CSV
```
*\*Lưu ý: CLIP/SigLIP có thể bật khi có card đồ họa bằng biến môi trường `set AIC_ENABLE_NEURAL=1`.*

---

## 4. Chi Tiết Thuật Toán & Cơ Chế Tính Điểm

### 4.1. BM25 Text Search & Trọng Số IDF
File `text_search_index.pkl` gom **629,404 tài liệu văn bản** thuộc 6 nhóm:
- **354,642 Captions:** Mô tả chi tiết hình ảnh từng keyframe.
- **136,928 OCR:** Chữ trích xuất từ màn hình video (bảng hiệu, logo, tít tin).
- **134,371 ASR Transcript Segments:** Lời thoại bóc tách theo timestamp.
- **1,746 Caption Summaries & 844 Full Transcripts.**

Điểm số được tính toán siêu tốc thông qua Inverted Index:
$$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
- **IDF cao:** Từ khóa hiếm, danh từ riêng, chữ OCR đặc trưng ($\text{IDF} \approx 8.0 \sim 12.0$) tạo điểm bứt phá cho ứng viên chính xác.
- **Hỗ trợ không dấu/có dấu song song:** Tự động tokenize cả 2 dạng (`da lat` $\leftrightarrow$ `Đà Lạt`).

### 4.2. Khớp Chữ OCR & Định Vị Khung Hình Cho KIS
- Mỗi bản ghi OCR gắn liền với **đúng 1 `keyframe_num`**. Khi câu truy vấn chứa chữ trên màn hình (ví dụ `HTV9 60 giây`, `Thanh Niên`), hệ thống tự động trỏ đến đúng keyframe có chứa dòng chữ đó.
- Đối với lời thoại ASR: Hệ thống tự động chuyển đổi timestamp (`start_time`) sang keyframe gần nhất trên timeline.

### 4.3. Mở Rộng Query Theo Chuẩn Caption Dataset
Khi bấm **"Dịch"**, `query_processor.py` tự động chuyển đổi câu hỏi tự nhiên sang phong cách Caption của mô hình thị giác:
- **Shot Type:** `A medium shot of...`, `A close-up shot of...`, `An aerial drone shot of...`, `A wide angle shot of...`
- **Subject & Action:** Bổ sung các từ khóa trực quan liên quan đến bối cảnh và hành động.

---

## 5. Các Tính Năng Giao Diện Web Mới (Web UI)

1. **Thanh kéo mở rộng màn hình (Resizable Splitter):** Kéo thả chuột giữa 2 khung để mở rộng panel xem ảnh to tùy ý (tự động lưu kích thước vào `localStorage`).
2. **Bộ duyệt Keyframe Timeline (Keyframe Scrubber):**
   - Thanh trượt scrubber tua qua lại toàn bộ video.
   - Nút `◀` / `▶` và phím tắt bàn phím **◄ / ►** để bước từng frame chính xác.
   - Nút chuyển chế độ **`⊞ Lưới`** để xem đồng thời toàn bộ dải ảnh của video.
3. **Hiển thị ảnh 100% không cắt góc (`object-fit: contain`):** Không bị xẹp hay cắt nửa khung hình.
4. **Lightbox Phóng to Toàn màn hình:** Click ảnh preview hoặc nhấp đúp thẻ bất kỳ để phóng to ảnh tối đa (bấm `ESC` để đóng).
5. **Hộp Evidence Trực Quan:** Hiển thị rõ đoạn trích văn bản khớp (ASR, OCR hoặc Caption).

---

## 6. Hướng Dẫn Khởi Chạy Cho Thành Viên Khác

```powershell
# 1. Cài đặt các dependencies
pip install -r requirements.txt

# 2. Chạy Web Server
py -m uvicorn aic.ui.app:app --port 8000
```
Truy cập: **`http://localhost:8000`** để sử dụng.
