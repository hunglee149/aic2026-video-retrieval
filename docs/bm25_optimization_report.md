# Báo Cáo Đo Đạc Tối Ưu Hoá BM25 (Load Time & Memory Footprint)

**Thời gian thực hiện:** 2026-09-03 13:54:35
**Môi trường:** Python 3.12.7 on win32
**Tệp dữ liệu:** `local/text_search_index.pkl` (383.31 MB, 629,404 documents)

---

## 1. Bảng So Sánh Chỉ Số Hiệu Năng Trước & Sau Khi Tối Ưu

| Chỉ số đo đạc | Trước khi tối ưu (Baseline) | Sau khi tối ưu (Optimized) | Mức cải thiện |
|:---|:---:|:---:|:---:|
| **Thời gian nạp index (`t_load`)** | **~7.38 giây** | **1.310 giây** | ⚡ **Nhanh hơn 5.6x** |
| **Lượng bộ nhớ RAM tiêu thụ** | **~3,677 MB** | **742.36 MB** | 📉 **Tiết kiệm 5.0x RAM** |
| **Kích thước file lưu trữ** | **468.29 MB** | **383.31 MB** | 💾 **Nhẹ hơn 85.0 MB** |
| **Độ trễ tìm kiếm trung bình** | **~350 - 650 ms** | **556.94 ms** | 🚀 **Mượt mà, ổn định** |
| **Độ toàn vẹn khung hình & thứ hạng** | 100% | **100% khớp tuyệt đối** | 🎯 **Không lệch kết quả** |
| **Tính độc lập của module Dịch** | Độc lập | **100% Độc lập (Lazy Component)** | ✅ **Không phụ thuộc BM25** |

---

## 2. Chi Tiết Kết Quả Truy Vấn Kiểm Thử (Top Candidates & Frame Mapping)

### Query `q1`: *"người đi xe đạp áo đỏ đội nón trắng rưới nước"* (Độ trễ: `709.34 ms`)
| Thứ hạng | Video ID | Frame Index | Điểm BM25 |
|:---:|:---:|:---:|:---:|
| #1 | **`L23_V017`** | `7775` | `1.0000` |
| #2 | **`L23_V004`** | `9625` | `0.9915` |
| #3 | **`L23_V010`** | `625` | `0.9859` |
| #4 | **`L23_V006`** | `6321` | `0.9768` |
| #5 | **`L23_V004`** | `8125` | `0.9721` |

### Query `q2`: *"bảng nguyên liệu nấu ăn bột cà ri nước cốt dừa nấm mèo"* (Độ trễ: `426.02 ms`)
| Thứ hạng | Video ID | Frame Index | Điểm BM25 |
|:---:|:---:|:---:|:---:|
| #1 | **`L26_V034`** | `0` | `1.0000` |
| #2 | **`L26_V012`** | `8640` | `0.8918` |
| #3 | **`L26_V012`** | `8594` | `0.8852` |
| #4 | **`L26_V034`** | `8194` | `0.8276` |
| #5 | **`L26_V034`** | `8235` | `0.8276` |

### Query `q3`: *"giáo viên nam đeo kính giảng bài slide bài học"* (Độ trễ: `575.28 ms`)
| Thứ hạng | Video ID | Frame Index | Điểm BM25 |
|:---:|:---:|:---:|:---:|
| #1 | **`L25_V080`** | `24000` | `1.0000` |
| #2 | **`L25_V054`** | `27000` | `0.9730` |
| #3 | **`L25_V020`** | `18985` | `0.9702` |
| #4 | **`L25_V054`** | `49500` | `0.9663` |
| #5 | **`L25_V081`** | `25350` | `0.9588` |

### Query `q4`: *"xe cấp cứu chạy trên đường bệnh viện"* (Độ trễ: `180.28 ms`)
| Thứ hạng | Video ID | Frame Index | Điểm BM25 |
|:---:|:---:|:---:|:---:|
| #1 | **`L21_V005`** | `11913` | `1.0000` |
| #2 | **`L21_V014`** | `16167` | `0.9937` |
| #3 | **`L22_V003`** | `25191` | `0.9224` |
| #4 | **`L22_V020`** | `33240` | `0.9224` |
| #5 | **`L21_V006`** | `20637` | `0.9032` |

### Query `q5`: *"cô gái đeo tạp dề trắng lọ hoa riềng tía dĩa trắng"* (Độ trễ: `893.81 ms`)
| Thứ hạng | Video ID | Frame Index | Điểm BM25 |
|:---:|:---:|:---:|:---:|
| #1 | **`L26_V378`** | `6656` | `1.0000` |
| #2 | **`L26_V109`** | `4736` | `0.9981` |
| #3 | **`L26_V071`** | `1613` | `0.9647` |
| #4 | **`L26_V410`** | `512` | `0.9587` |
| #5 | **`L26_V448`** | `411` | `0.9371` |

---

## 3. Đánh Giá & Kết Luận
1. **Thời gian load và bộ nhớ:** Việc chuyển đổi postings list sang định dạng mảng nhị phân nén `(array('I'), array('H'))` và lưu trữ sẵn `doc_lengths`, `idf`, `avgdl`, `N`, `accent_index` đã loại bỏ hoàn toàn chi phí khởi tạo lặp lại và giảm 80% RAM.
2. **Khung hình và thứ hạng:** Tất cả các truy vấn đều giữ nguyên thứ hạng và map frame chính xác 100%.
3. **Lazy Initialization:** Module dịch khởi tạo tức thì mà không cần chờ nạp BM25.
