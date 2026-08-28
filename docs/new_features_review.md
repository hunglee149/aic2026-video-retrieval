# Báo cáo Tính năng Mới và Sửa đổi Giao diện (So với Commit 86e3944)

Tài liệu này tóm tắt toàn bộ các tính năng mới, cải tiến kiến trúc hệ thống và các lỗi giao diện đã được thay đổi/sửa đổi kể từ commit [86e3944](https://github.com/hunglee149/aic2026-video-retrieval/commit/86e394416f6fd07788b7e0b3e2656894ccebd82a).

---

## 1. Tích hợp Đầy đủ Dữ liệu từ Hugging Face (Cloud Hub Integration)
* **Thay đổi**: Toàn bộ hệ thống dữ liệu tìm kiếm và video media đã được liên kết với Hugging Face dataset `manhha2502/fullhd`.
* **Cơ chế tải tự động**: 
  * Hàm `resolve_path` được viết lại để luôn kiểm tra và tải các file index/metadata (CLIP, BM25, SigLIP) trực tiếp từ Hugging Face qua `hf_hub_download` và lưu vào thư mục cache của máy. 
  * Khi khởi chạy lần thứ 2, hệ thống tự động đọc từ cache ổ cứng, không cần internet và không tải lại.
* **Stream video trực tuyến**:
  * Đơn giản hóa endpoint `/api/video/{video_id}` để luôn trả về `RedirectResponse` dẫn trực tiếp tới URL của video trên Hugging Face. Không còn bất kỳ dòng code nào kiểm tra file video local trên máy, giúp tiết kiệm hàng trăm GB dung lượng ổ đĩa.
  * Endpoint `/api/video_info/{video_id}` đọc chỉ số FPS chính xác của 873 video từ file `video_metadata.json` đã tải về từ Hugging Face.
* **Cấu hình & Dependency**:
  * Thêm package `huggingface_hub` vào file `requirements.txt`.
  * Bổ sung các biến cấu hình (`AIC_HF_REPO_ID`, `AIC_HF_REVISION`, `AIC_HF_CACHE_DIR`, `AIC_HF_DATASET_URL`, `AIC_USE_SIGLIP`) vào file `.env.example`.

---

## 2. Nạp Sẵn mô hình ở Startup bằng Lifespan Handler
* **Thay đổi**: Thay thế các hàm `@app.on_event` cũ bằng bộ quản lý vòng đời `lifespan` context manager hiện đại của FastAPI.
* **Nạp trước mô hình dịch**:
  * Khi server vừa khởi động, hệ thống tự động tải và nạp sẵn mô hình dịch thuật local `opus-mt-vi-en` cùng các index tìm kiếm (CLIP, BM25) vào RAM.
  * Giúp thao tác dịch câu hỏi đầu tiên trên UI phản hồi ngay lập tức, không còn độ khựng/chậm để load mô hình như trước đây.
  * Tự động bỏ qua việc nạp dịch thuật khi chạy test (`pytest`) để tránh lỗi mạng.

---

## 3. Tính năng "Xóa Cache" trên Giao diện UI
* **Thay đổi**: Thêm nút **Xóa Cache** (màu đỏ) ở vị trí bên trái thanh menu bar trên cùng.
* **Chức năng**: Khi operator click vào nút này, hệ thống sẽ hiển thị bảng xác nhận. Nếu đồng ý, toàn bộ dữ liệu lưu trữ tạm thời trong trình duyệt (`LocalStorage`) bao gồm:
  * Tất cả các câu hỏi đã up lên trước đó (manifest queries).
  * Tất cả các lựa chọn và câu trả lời đã xác nhận (selections).
  * Lịch sử kết quả tìm kiếm đã cache (query cache).
  * Sẽ bị xóa sạch hoàn toàn và tải lại trang, giúp hệ thống quay về trạng thái ban đầu để làm lượt thi mới.

---

## 4. Làm Sạch Giao diện (Clean UI Status Labels)
* **Thay đổi**: Loại bỏ toàn bộ các biểu tượng cảm xúc (emojis) và ký tự tiền tố rườm rà ở đầu các badge trạng thái, các nút và bảng kết quả để giao diện nhìn hiện đại, gọn gàng và chuyên nghiệp hơn.
* **Các nhãn được làm sạch**:
  * Trạng thái câu hỏi sẵn sàng: đổi từ `✓ Ready` thành `Ready`.
  * Trạng thái thiếu thông tin: đổi từ `⚠ Thiếu event` hoặc `⚠️ Chưa có dòng` thành `Thiếu event` hoặc `Chưa có dòng`.
  * Các nút hành động trong bảng Export: đổi từ `👁️ Xem` thành `Xem`, và từ `✕ Xoá` thành `Xoá`.
  * Nút tải xuống bài nộp: đổi từ `📥 Tải xuống submission.zip` thành `Tải xuống submission.zip`.

---

## 5. Sửa lỗi Giao diện Co Sập & Tối ưu số cột Candidates Grid
* **Thay đổi**: Sửa đổi toàn bộ CSS của khu vực hiển thị kết quả tìm kiếm.
* **Sửa lỗi co sập**: Sửa thuộc tính CSS Grid của `#candidates-grid` và dùng padding aspect-ratio hack (`padding-bottom: 56.25%`, `height: 0`) cho khung ảnh `.card-thumb`. Các ô kết quả tìm kiếm đã khôi phục về dạng hình chữ nhật to đẹp chuẩn tỷ lệ 16:9, không bị co dẹt thành các đường thanh ngang (18px) như trước.
* **Tối ưu hiển thị 3 cột/hàng**: Đổi thuộc tính độ rộng tối thiểu của mỗi ô về `240px` (`minmax(240px, 1fr)`). Khi hiển thị trên màn hình laptop tiêu chuẩn của bạn, giao diện tự động co giãn đều đặn và hiển thị **vừa khít đúng 3 ô kết quả trên một hàng**, mang lại trải nghiệm tối ưu nhất cho operator.

---

## 6. Cơ chế lưu Cache Dữ liệu Phiên làm việc (Session & Query Cache)
* **Thay đổi**: Bổ sung cơ chế tự động lưu trữ và khôi phục trạng thái làm bài của operator thông qua LocalStorage trình duyệt (`aic_query_cache`).
* **Lưu cache kết quả tìm kiếm theo câu hỏi**:
  * Khi thực hiện tìm kiếm cho một câu hỏi (`query_id`), toàn bộ danh sách kết quả (candidates), các frame nháp đã lưu (draft frames) và trạng thái đánh giá (verdict marks) của câu hỏi đó sẽ được tự động lưu lại.
  * Khi chuyển qua lại giữa các câu hỏi khác nhau hoặc khi tải lại trang (F5), kết quả tìm kiếm và thiết lập nháp của câu hỏi đó sẽ được khôi phục ngay lập tức mà không cần phải thực hiện lại lệnh tìm kiếm tốn thời gian.
* **Tự động lưu khi tắt/tải lại trang**: Đăng ký sự kiện lắng nghe `beforeunload` để tự động lưu trạng thái nháp hiện tại ngay trước khi operator tắt tab trình duyệt hoặc F5 trang.
* **Xóa cache thông minh khi đổi gói thi**: Khi nạp (upload) một Query Pack ZIP mới, hệ thống tự động xóa sạch các cache kết quả tìm kiếm cũ để tránh làm lẫn lộn dữ liệu kết quả của lượt thi mới.

---

## 7. Tối ưu hóa Phát lại Video (Video Player Reuse Optimization)
* **Thay đổi**: Tự động giữ nguyên và tái sử dụng trình phát video nếu candidate tiếp theo thuộc cùng một video file với candidate đang mở.
* **Chức năng**: 
  * Khi bạn duyệt qua các candidate card khác nhau nhưng của **cùng một video** (chỉ khác khoảng frame hoặc góc máy), thẻ `<video>` trên UI sẽ không bị reset và tải lại từ đầu (không làm mất kết nối HTTP).
  * Trình phát video chỉ thực hiện nhảy (seek) trực tiếp thời gian phát (`vid.currentTime`) tới vị trí frame mới.
  * **Hiệu quả**: Giúp việc chuyển đổi, đối chiếu các khung hình trong cùng một video diễn ra **tức thì trong mili giây**, hoàn toàn không bị khựng, giật hình hay phải chờ load bộ đệm (buffering) mạng, tăng tốc độ làm bài thi cực kỳ lớn cho operator.

---

## 8. Tránh lệch hiển thị khi chuyển đổi câu hỏi (Query Mismatch Protection)
* **Thay đổi**: Bổ sung cơ chế tự động dọn dẹp danh sách candidates cũ trước khi tải dữ liệu câu hỏi mới.
* **Chức năng**: 
  * Khi operator chọn một câu hỏi mới từ danh sách bên trái, hệ thống sẽ lập tức gán `state.selected = null` và làm rỗng mảng kết quả `state.candidates = []`.
  * **Hiệu quả**: Ngăn ngừa hiện tượng giao diện bị lag và hiển thị nhầm lẫn các candidate card của câu hỏi trước đó trong lúc chờ gọi API tải dữ liệu của câu hỏi mới.

---

## 9. Cải tiến lưu trữ Frame Nháp và Khôi phục Thông minh
* **Thay đổi**: Tối ưu hóa cách hoạt động của các frame nháp (draft frames) của các candidate kết quả.
* **Thông minh hóa khôi phục frame đã xác nhận**:
  * Khi chọn một candidate đã từng được lưu/xác nhận (confirm), hệ thống sẽ ưu tiên đọc lại frame đã confirm hoặc draft frame trước đó để nhảy video tới đúng vị trí đó, thay vì luôn ép về frame đại diện mặc định (representative frame).
* **Tự động dọn nháp khi đổi candidate**: Khi operator chuyển chọn từ candidate A sang candidate B, danh sách nháp đang chỉnh dở sẽ được dọn sạch để tránh việc lưu đè nhầm frame.
* **Hủy nhanh nháp bằng cách click lại card**: Khi click lại chính card candidate đang được chọn, hệ thống tự động xóa draft frame tạm thời và đưa video quay trở lại mốc frame gốc đại diện của video.
