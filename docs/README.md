# Tài liệu

## Quy trình nộp bài cho operator

1. Nạp ZIP query pack đầy đủ của BTC, hoặc toàn bộ các file TXT trong pack.
   Một TXT chỉ dành cho phát triển/kiểm thử, không phải quy trình nộp chính thức.
2. Với mọi query TRAKE, xem nội dung query, xác nhận số event đúng, rồi chọn
   đủ các frame event theo thứ tự tăng dần.
3. Trước khi xuất, sửa toàn bộ lỗi trong validation report. Mỗi query phải có
   ít nhất một dòng hợp lệ.
4. Chỉ tải `submission.zip` nếu màn hình export trả trạng thái **PASS**. Hệ
   thống tự kiểm tra lại archive vừa tạo, nhưng không thay thế validator chính
   thức của BTC; chạy validator BTC trước khi nộp.

Mỗi người một file trong thư mục này, đặt tên theo phần mình làm:

## T

```markdown
# Tên task

...

## Input / Output

...

## Chạy thế nào

Ví dụ

## Chưa làm / Blockers

- ...
```
