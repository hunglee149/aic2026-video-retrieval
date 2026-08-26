#!/usr/bin/env python3
"""Sinh phiếu nháp cho bộ sanity set — phần mô tả vẫn do người viết.

Vì sao không tự sinh luôn cả câu hỏi: không có cách nào tạo ground truth hợp lệ
bằng máy ở đây.

- Lấy câu từ **ASR** thì không dùng chấm kênh hình được: lời đang nói thường
  không phải thứ đang hiện trên màn hình.
- Lấy câu từ **OCR** thì quay vòng với chính BM25, vì OCR giờ đã nằm trong index.

Nên script này chỉ làm phần máy làm được: chọn sẵn những khoảnh khắc **kiểm chứng
được** (có file video để tua, có keyframe để xem), rồi in kèm ngữ cảnh ASR/OCR
quanh đó. Việc còn lại của người gán nhãn là mở ảnh keyframe, xem, và viết một
câu mô tả **cảnh nhìn thấy** — đúng như cách BTC ra đề.

Quy tắc viết mô tả, để bộ này còn đo được cái cần đo:

1. Tả **cái nhìn thấy**, không chép lại chữ trong ô ``ocr_hint``.
2. Không nhắc tên video, tên kênh, ngày phát sóng.
3. Viết như đề thi: một câu tiếng Việt, đủ chi tiết để phân biệt với cảnh khác.
4. Xem xong thì đổi ``verified`` thành ``true``.

Chạy::

    python scripts/make_sanity_drafts.py --count 30 --out eval/sanity_set.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("make_sanity_drafts")


def load_env_file(path: Path) -> None:
    """Đọc .env giống app (setdefault, biến truyền tay thắng file)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def videos_with_local_file(keyframes_dir: Path) -> set[str]:
    """Chỉ chọn video **xem được**: người gán nhãn phải tua được video thật.

    Keyframe rời không đủ để viết mô tả tin cậy, và TRAKE thì bắt buộc phải xem
    video vì cửa sổ mỗi sự kiện dưới 10 frame.
    """
    video_dir = keyframes_dir.parent / "video"
    if not video_dir.exists():
        logger.warning("Không thấy thư mục video cạnh %s", keyframes_dir)
        return set()
    return {p.stem for p in video_dir.glob("*.mp4")}


def build_context(index_path: Path):
    """Gom ASR và OCR theo (video, keyframe) để in kèm làm gợi ý."""
    with index_path.open("rb") as handle:
        data = pickle.load(handle)

    keyframe_map = data.get("keyframe_map", {})
    asr = defaultdict(list)
    ocr = {}
    for doc in data["documents"]:
        doc_type = doc.get("type")
        video_id = doc.get("video_id")
        if doc_type == "transcript_segment":
            asr[video_id].append(
                (float(doc.get("start_time", 0.0)),
                 float(doc.get("end_time", 0.0)),
                 doc.get("text", ""))
            )
        elif doc_type == "ocr" and doc.get("keyframe_num") is not None:
            ocr[(video_id, int(doc["keyframe_num"]))] = doc.get("text", "")
    for segments in asr.values():
        segments.sort()
    return keyframe_map, asr, ocr


def asr_around(segments, pts_time: float, span: float = 6.0) -> str:
    hits = [
        text for start, end, text in segments
        if start <= pts_time + span and end >= pts_time - span
    ]
    return " ".join(hits)[:200]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("eval/sanity_set.jsonl"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per-video", type=int, default=1,
                        help="số khoảnh khắc tối đa lấy từ một video")
    parser.add_argument("--force", action="store_true",
                        help="ghi đè file đã có")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_env_file(Path(".env"))

    keyframes_dir = Path(os.environ.get("AIC_KEYFRAMES_DIR", "data/keyframes"))
    index_path = Path(os.environ.get("AIC_TEXT_INDEX_PATH", ""))
    map_dir = os.environ.get("AIC_MAP_KEYFRAMES_DIR", "")

    if not index_path.exists():
        sys.exit(f"Không thấy AIC_TEXT_INDEX_PATH: {index_path}")
    if not keyframes_dir.exists():
        sys.exit(f"Không thấy AIC_KEYFRAMES_DIR: {keyframes_dir}")
    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} đã tồn tại — dùng --force nếu thực sự muốn ghi đè")

    playable = videos_with_local_file(keyframes_dir)
    logger.info("Video có file .mp4 để tua: %d", len(playable))
    if not playable:
        sys.exit("Không có video nào xem được — không gán nhãn tin cậy được")

    keyframe_map, asr, ocr = build_context(index_path)

    # Chỉ giữ video vừa xem được vừa có keyframe map.
    usable = sorted(v for v in playable if keyframe_map.get(v))
    logger.info("Trong đó có keyframe map: %d", len(usable))

    rng = random.Random(args.seed)
    rng.shuffle(usable)

    drafts = []
    for video_id in usable:
        if len(drafts) >= args.count:
            break
        entries = keyframe_map[video_id]
        if len(entries) < 8:
            continue
        # Tránh đầu/cuối video: hay là hình hiệu, quảng cáo, danh sách credit.
        middle = entries[len(entries) // 8: -len(entries) // 8 or None]
        if not middle:
            continue
        picks = rng.sample(middle, min(args.per_video, len(middle)))
        for entry in picks:
            if len(drafts) >= args.count:
                break
            kf_num = int(entry["kf_num"])
            frame_idx = int(entry["frame_idx"])
            pts = float(entry.get("pts_time", 0.0))
            drafts.append({
                "query_id": f"s{len(drafts) + 1:02d}",
                "text_vi": "",
                "text_en": "",
                "video_id": video_id,
                "frame": frame_idx,
                "task": "kis",
                "verified": False,
                "_keyframe_image": str(keyframes_dir / video_id / f"{kf_num:03d}.jpg"),
                "_keyframe_num": kf_num,
                "_pts_time": round(pts, 2),
                "_asr_hint": asr_around(asr.get(video_id, []), pts),
                "_ocr_hint": ocr.get((video_id, kf_num), ""),
            })

    if not drafts:
        sys.exit("Không chọn được khoảnh khắc nào")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for draft in drafts:
            handle.write(json.dumps(draft, ensure_ascii=False) + "\n")

    missing_image = sum(
        1 for d in drafts if not Path(d["_keyframe_image"]).exists()
    )
    logger.info("\nĐã ghi %d phiếu nháp vào %s", len(drafts), args.out)
    logger.info("  video khác nhau : %d", len({d["video_id"] for d in drafts}))
    logger.info("  có gợi ý OCR    : %d", sum(1 for d in drafts if d["_ocr_hint"]))
    logger.info("  có gợi ý ASR    : %d", sum(1 for d in drafts if d["_asr_hint"]))
    if missing_image:
        logger.warning("  thiếu ảnh keyframe: %d", missing_image)

    logger.info("\nViệc tiếp theo — với mỗi dòng:")
    logger.info("  1. mở ảnh ở _keyframe_image (hoặc tua video tới _pts_time)")
    logger.info("  2. điền text_vi: tả CẢNH NHÌN THẤY, đừng chép _ocr_hint")
    logger.info("  3. đổi verified thành true")
    logger.info("\nXong thì chấm bằng:")
    logger.info("  python scripts/eval_retrieval.py --set %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
