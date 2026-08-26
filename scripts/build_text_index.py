#!/usr/bin/env python3
"""Bổ sung document OCR vào ``text_search_index.pkl`` của BM25.

Vì sao cần: index hiện tại chỉ có ASR (lời nói) và media_info (tiêu đề kênh).
Câu hỏi kiểu "biển hiệu ghi Đà Lạt", "bảng tỉ số 2-1", "dòng chữ chạy dưới bản
tin" hoàn toàn không tìm được, vì chữ hiện trên màn hình chưa hề được đánh chỉ
mục.

Cách làm là **bổ sung**, không dựng lại từ đầu: đọc pickle cũ, giữ nguyên toàn
bộ document ASR/media_info, chỉ thay thế phần ``ocr``. Nhờ vậy không có nguy cơ
làm hỏng 136 nghìn document đang chạy tốt, và chạy lại nhiều lần cho ra cùng một
kết quả (idempotent) — khi có thêm file OCR thì chỉ việc chạy lại.

Tokenizer phải là ``underthesea.word_tokenize(..., format='text')`` đúng như
notebook đã dựng index gốc, nếu không token OCR sẽ không cùng dạng với token
ASR (``giao_thông`` so với ``giao`` + ``thông``) và điểm BM25 sẽ lệch.
Đã đối chiếu: underthesea 9.5.0 tái tạo đúng 300/300 document mẫu của index cũ.

``underthesea`` chỉ cần khi *dựng* index, không cần khi chạy app, nên nó không
nằm trong ``requirements.txt``. Cài riêng::

    pip install underthesea

Chạy::

    python scripts/build_text_index.py \\
        --index   /duong/dan/index/text_search_index.pkl \\
        --ocr-dir /duong/dan/drive/ocr \\
        --out     /duong/dan/index/text_search_index.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger("build_text_index")

# Ngưỡng tin cậy mặc định của OCR. 0.5 giữ lại ~86% số box; thấp hơn nữa chủ yếu
# là rác từ logo mờ và phụ đề chuyển cảnh.
DEFAULT_MIN_CONFIDENCE = 0.5

# Box quá ngắn thường là nhiễu ("|", "l", "."), không mang thông tin tìm kiếm.
MIN_TEXT_LENGTH = 2

OCR_DOC_TYPE = "ocr"


def get_tokenizer():
    """Tokenizer khớp với index gốc; thiếu underthesea thì dừng hẳn.

    Không im lặng fallback sang ``text.split()``: token sinh ra sẽ khác dạng với
    phần ASR đã có, làm hỏng thống kê BM25 của cả index mà không báo lỗi.
    """
    try:
        from underthesea import word_tokenize
    except ImportError:
        sys.exit(
            "Thiếu 'underthesea' — cần đúng tokenizer đã dùng để dựng index gốc.\n"
            "Cài bằng: pip install underthesea"
        )

    def tokenize(text: str) -> list[str]:
        text = text.lower().strip()
        try:
            return word_tokenize(text, format="text").split()
        except Exception:
            return text.split()

    return tokenize


def reading_order(box: dict) -> tuple:
    """Sắp text box theo thứ tự đọc: trên xuống dưới, trái sang phải."""
    bbox = box.get("bbox") or [0, 0, 0, 0]
    # bbox = [x1, y1, x2, y2]; gom theo dải dọc 20px để chữ cùng hàng không bị
    # tách ra chỉ vì lệch vài pixel.
    return (int(bbox[1]) // 20, int(bbox[0]))


def build_ocr_documents(
    ocr_dir: Path,
    keyframe_map: dict,
    min_confidence: float,
) -> tuple[list[dict], dict]:
    """Mỗi keyframe có chữ → một document OCR.

    Ghi kèm ``keyframe_num`` để map sang frame thật **chính xác tuyệt đối** qua
    ``keyframe_map``, không phải suy từ timestamp như document ASR.
    """
    documents: list[dict] = []
    stats = Counter()

    files = sorted(ocr_dir.glob("*.json"))
    if not files:
        logger.warning("Không thấy file OCR nào trong %s", ocr_dir)
        return documents, stats

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Bỏ qua %s: %s", path.name, exc)
            stats["file_loi"] += 1
            continue

        video_id = data.get("video_id") or path.stem
        frame_lookup = {
            int(entry["kf_num"]): entry for entry in keyframe_map.get(video_id, [])
        }
        if not frame_lookup:
            # Không có keyframe map thì không biết frame thật; bỏ, không bịa.
            logger.warning("%s: không có keyframe map, bỏ qua", video_id)
            stats["video_khong_co_map"] += 1
            continue

        stats["video"] += 1
        for keyframe in data.get("keyframes", []):
            kf_num = keyframe.get("keyframe_num")
            if kf_num is None:
                continue
            entry = frame_lookup.get(int(kf_num))
            if entry is None:
                stats["keyframe_ngoai_map"] += 1
                continue

            boxes = [
                box
                for box in keyframe.get("texts", [])
                if float(box.get("confidence", 0.0)) >= min_confidence
                and len((box.get("text") or "").strip()) >= MIN_TEXT_LENGTH
            ]
            if not boxes:
                continue

            boxes.sort(key=reading_order)
            text = " ".join((box.get("text") or "").strip() for box in boxes)
            if not text.strip():
                continue

            pts_time = float(entry.get("pts_time", 0.0))
            documents.append(
                {
                    "type": OCR_DOC_TYPE,
                    "video_id": video_id,
                    "text": text,
                    "keyframe_num": int(kf_num),
                    "frame_idx": int(entry["frame_idx"]),
                    "start_time": pts_time,
                    "end_time": pts_time,
                    "language": "vi",
                }
            )
            stats["document"] += 1
            stats["box"] += len(boxes)

    return documents, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path,
                        help="text_search_index.pkl hiện có")
    parser.add_argument("--ocr-dir", required=True, type=Path,
                        help="thư mục JSON OCR, mỗi video một file")
    parser.add_argument("--out", type=Path,
                        help="file đầu ra (mặc định ghi đè --index)")
    parser.add_argument("--min-confidence", type=float,
                        default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--no-backup", action="store_true",
                        help="không tạo bản sao .bak khi ghi đè")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_path = args.out or args.index

    if not args.index.exists():
        sys.exit(f"Không thấy index: {args.index}")
    if not args.ocr_dir.exists():
        sys.exit(f"Không thấy thư mục OCR: {args.ocr_dir}")

    logger.info("Đọc index cũ: %s", args.index)
    with args.index.open("rb") as handle:
        data = pickle.load(handle)

    documents = data["documents"]
    tokenized = data["tokenized"]
    keyframe_map = data.get("keyframe_map", {})
    if len(documents) != len(tokenized):
        sys.exit("Index hỏng: documents và tokenized lệch số lượng")
    if not keyframe_map:
        sys.exit("Index không có keyframe_map — không map được frame thật cho OCR")

    before = Counter(doc.get("type") for doc in documents)
    logger.info("  %d document: %s", len(documents), dict(before))

    # Bỏ document OCR cũ để chạy lại nhiều lần vẫn ra cùng kết quả.
    kept = [
        (doc, tokens)
        for doc, tokens in zip(documents, tokenized)
        if doc.get("type") != OCR_DOC_TYPE
    ]
    dropped = len(documents) - len(kept)
    if dropped:
        logger.info("  bỏ %d document OCR của lần chạy trước", dropped)

    logger.info("Đọc OCR từ %s (min_confidence=%.2f)", args.ocr_dir,
                args.min_confidence)
    ocr_docs, stats = build_ocr_documents(
        args.ocr_dir, keyframe_map, args.min_confidence
    )
    if not ocr_docs:
        sys.exit("Không dựng được document OCR nào — dừng, không ghi đè index")
    logger.info(
        "  %d document OCR từ %d video (%d text box)",
        stats["document"], stats["video"], stats["box"],
    )
    for key in ("file_loi", "video_khong_co_map", "keyframe_ngoai_map"):
        if stats[key]:
            logger.warning("  %s: %d", key, stats[key])

    tokenize = get_tokenizer()
    logger.info("Tokenize %d document OCR...", len(ocr_docs))
    ocr_tokens = [tokenize(doc["text"]) for doc in ocr_docs]

    empty = sum(1 for tokens in ocr_tokens if not tokens)
    if empty:
        logger.warning("  %d document tokenize ra rỗng, bỏ", empty)
    merged = [
        (doc, tokens)
        for doc, tokens in zip(ocr_docs, ocr_tokens)
        if tokens
    ]

    final = kept + merged
    out_documents = [doc for doc, _ in final]
    out_tokenized = [tokens for _, tokens in final]

    payload = dict(data)
    payload["documents"] = out_documents
    payload["tokenized"] = out_tokenized
    payload["keyframe_map"] = keyframe_map
    # Thống kê BM25 precompute (nếu pickle cũ có) đã hết đúng vì corpus đổi.
    for stale in ("inverted", "idf", "avgdl", "N", "doc_lengths"):
        payload.pop(stale, None)

    after = Counter(doc.get("type") for doc in out_documents)
    logger.info("Index mới: %d document: %s", len(out_documents), dict(after))

    if out_path.exists() and not args.no_backup:
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        logger.info("Sao lưu bản cũ sang %s", backup)
        shutil.copy2(out_path, backup)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    logger.info("Ghi %s", out_path)
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(out_path)

    size_mb = out_path.stat().st_size / 1e6
    logger.info("Xong: %s (%.1f MB)", out_path, size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
