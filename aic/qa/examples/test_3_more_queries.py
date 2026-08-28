"""Benchmark 3 Additional Real Grounded VQA Queries with Gemini 3.6 Flash."""

import io
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from aic.qa.vlm_engine import GroundedQAEngine, QAResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_frame(video_id: str, frame_num: int) -> Image.Image:
    """Extract keyframe from zip file or local directory."""
    local_paths = [
        Path(f"keyframes/{video_id}/{frame_num:03d}.jpg"),
        Path(f"keyframes/{video_id}/{frame_num:04d}.jpg"),
        Path(f"data/keyframes/{video_id}/{frame_num:03d}.jpg"),
    ]
    for lp in local_paths:
        if lp.exists():
            return Image.open(lp)

    prefix = video_id.split("_")[0]  # e.g. L21, L22, L26, L28, L30
    zip_candidates = [
        Path(f"data/keyframes/Keyframes_{prefix}.zip"),
        Path(f"data/keyframes/Keyframes_{prefix}_a.zip"),
        Path(f"data/keyframes/Keyframes_{prefix}_b.zip"),
        Path(f"data/keyframes/Keyframes_{prefix}_c.zip"),
        Path(f"data/keyframes/Keyframes_{prefix}_d.zip"),
        Path(f"data/keyframes/Keyframes_{prefix}_e.zip"),
        Path(f"data/Keyframes_{prefix}.zip"),
    ]

    target_suffixes = (
        f"{video_id}/{frame_num:03d}.jpg",
        f"{video_id}/{frame_num:04d}.jpg",
        f"{video_id}/{frame_num:05d}.jpg",
        f"{video_id}/{frame_num}.jpg",
        f"{video_id}_{frame_num:03d}.jpg",
    )

    for zp in zip_candidates:
        if not zp.exists():
            continue
        try:
            with zipfile.ZipFile(zp, "r") as z:
                for name in z.namelist():
                    if name.endswith(target_suffixes):
                        with z.open(name) as f:
                            return Image.open(io.BytesIO(f.read()))
        except Exception as e:
            logger.debug("Error checking %s: %s", zp, e)

    logger.warning("Frame %d of %s not found in local/zip. Using blank placeholder.", frame_num, video_id)
    return Image.new("RGB", (320, 240), color="gray")


def run_3_more_queries(api_key: str):
    print("=" * 85)
    print("       AIC 2026 TASK 2: GROUNDED VQA BENCHMARK (3 MORE QUERIES)")
    print("=" * 85)

    engine = GroundedQAEngine(api_key=api_key, model_name="gemini-3.6-flash")

    test_queries = [
        {
            "num": 1,
            "id": "query-p1-1-exercise",
            "video_id": "L30_V046",
            "question": "Nhóm người đang xếp hàng trong sân thể thao đang cùng thực hiện động tác thể dục gì?",
            "frames": [92, 94, 96, 97],
            "expected_ans": "Hai tay chạm mũi chân / Gập người chạm ngón chân",
        },
        {
            "num": 2,
            "id": "query-p1-5-tofu",
            "video_id": "L26_V035",
            "question": "Món ăn gì màu vàng trắng được cắt thành các khối vuông đang được chiên/nấu trong chảo ở phần đầu đoạn clip?",
            "frames": [1, 2, 3, 4, 5],
            "expected_ans": "Đậu hũ / Đậu phụ (Tofu)",
        },
        {
            "num": 3,
            "id": "query-p1-2-dam",
            "video_id": "L28_V018",
            "question": "Công trình thủy lợi xây dựng quy mô lớn chắn ngang dòng nước được quay từ trên cao là công trình gì?",
            "frames": [1, 2, 3, 4, 5, 6],
            "expected_ans": "Con đập / Đập thủy lợi (Dam)",
        },
    ]


    results = []

    for item in test_queries:
        print(f"\n[CÂU {item['num']}] ({item['id']}) - Video: {item['video_id']}")
        print(f"  Câu hỏi:    \"{item['question']}\"")
        print(f"  Kỳ vọng:    Đáp án: {item['expected_ans']}")

        candidate_frames = []
        for f_id in item["frames"]:
            img = extract_frame(item["video_id"], f_id)
            candidate_frames.append((f_id, img))

        # Run inference
        res: QAResult = engine.answer_query(
            question=item["question"],
            video_id=item["video_id"],
            candidate_frames=candidate_frames,
        )
        results.append((item, res))

        print(f"  >>> KẾT QUẢ VLM:")
        print(f"      • Frame bằng chứng:  {res.frame_id}")
        print(f"      • Câu trả lời:       '{res.answer}'")
        print(f"      • Độ tin cậy (Conf): {res.confidence:.2f}")
        print(f"      • Xác thực Grounded: {'ĐẠT (TRUE)' if res.is_grounded else 'KHÔNG ĐẠT (UNKNOWN/LOW CONF)'}")
        print(f"      • Thời gian xử lý:   {res.latency_ms:.1f} ms")
        print(f"      • Bằng chứng thị giác: {res.evidence}")
        print(f"      • Dòng CSV nộp bài:  {res.to_submission_row()}")

    # Summary table
    print("\n" + "=" * 85)
    print("                             TỔNG HỢP KẾT QUẢ 3 CÂU BỔ SUNG")
    print("=" * 85)
    print(f"{'STT':<4} | {'Query ID':<20} | {'Video':<9} | {'Frame':<6} | {'Đáp án VLM':<24} | {'Conf':<5} | {'Latency':<9} | {'Grounding'}")
    print("-" * 85)
    for item, res in results:
        ground_status = "PASSED" if res.is_grounded else "FAILED"
        print(f"{item['num']:<4} | {item['id']:<20} | {res.video_id:<9} | {res.frame_id:<6} | {res.answer:<24} | {res.confidence:<5.2f} | {res.latency_ms:<6.1f} ms | {ground_status}")
    print("=" * 85)


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEMINI_API_KEY", "")
    run_3_more_queries(key)
