"""Test 5 Grounded VQA Queries with Gemini 2.0 Flash."""

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
    # 1. Check local keyframe folders
    local_paths = [
        Path(f"keyframes/{video_id}/{frame_num:03d}.jpg"),
        Path(f"keyframes/{video_id}/{frame_num:04d}.jpg"),
        Path(f"keyframes/{video_id}/{frame_num:05d}.jpg"),
        Path(f"keyframes/{video_id}/{frame_num}.jpg"),
        Path(f"data/keyframes/{video_id}/{frame_num:03d}.jpg"),
        Path(f"data/keyframes/{video_id}/{frame_num:04d}.jpg"),
        Path(f"data/keyframes/{video_id}/{frame_num:05d}.jpg"),
        Path(f"data/keyframes/{video_id}/{frame_num}.jpg"),
    ]
    for lp in local_paths:
        if lp.exists():
            return Image.open(lp)

    # 2. Check zip files in data/keyframes and data/
    prefix = video_id.split("_")[0]  # e.g. L21, L22
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

    # Fallback to dummy image if dataset zip is not downloaded
    logger.warning("Frame %d of %s not found in local/zip. Using blank frame placeholder.", frame_num, video_id)
    return Image.new("RGB", (320, 240), color="gray")



def run_5_test_cases(api_key: str):
    print("=" * 85)
    print("       AIC 2026 TASK 2: GROUNDED VQA BENCHMARK (5 QUERIES)")
    print("=" * 85)

    engine = GroundedQAEngine(api_key=api_key, model_name="gemini-3.6-flash")


    test_queries = [
        {
            "num": 1,
            "id": "query-p1-9-qa",
            "video_id": "L21_V003",
            "question": "Đoạn phim ghi lại cảnh những chiếc xe ô tô lội nước, chiếc xe màu vàng, màu đỏ và màu đen lần lượt chuẩn bị đi qua cầu. Con số được ghi trên biển báo bên trái của cây cầu là bao nhiêu?",
            "frames": [240, 248, 252, 258],
            "expected_evidence": 252,
            "expected_ans": "2,15 hoặc 2.15",
        },
        {
            "num": 2,
            "id": "query-p1-3-qa",
            "video_id": "L21_V007",
            "question": "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loại bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?",
            "frames": [70, 74, 76, 80],
            "expected_evidence": 76,
            "expected_ans": "38.35 hoặc 38,35",
        },
        {
            "num": 3,
            "id": "query-p1-17-qa",
            "video_id": "L22_V008",
            "question": "Đoạn phim ghi lại cảnh sạt lở đất đá tại một con đèo dưới trời mưa lớn. Tên của con đèo được ghi trên biển cảnh báo sạt lở là đèo gì?",
            "frames": [55, 59, 63, 67],
            "expected_evidence": 59,
            "expected_ans": "Đèo Tà Pứa (hoặc Tà Pứa)",
        },
        {
            "num": 4,
            "id": "query-p1-4-qa-color",
            "video_id": "L22_V021",
            "question": "Trong cảnh hai nhân viên của London Zoo đang thực hiện việc cân và ghi nhận số liệu của một con vật, hai nhân viên này đang mặc áo có màu gì?",
            "frames": [109, 110, 111, 112],
            "expected_evidence": 111,
            "expected_ans": "Màu xanh lá cây (Green)",
        },
        {
            "num": 5,
            "id": "query-anti-hallucination-trick",
            "video_id": "L21_V003",
            "question": "Chiếc máy bay trực thăng màu hồng đang bay lơ lửng phía trên cây cầu có số hiệu là bao nhiêu?",
            "frames": [240, 248, 252],
            "expected_evidence": "None (Bẫy chống bịa)",
            "expected_ans": "UNKNOWN (Không có máy bay)",
        },
    ]


    results = []

    for item in test_queries:
        print(f"\n[CÂU {item['num']}] ({item['id']}) - Video: {item['video_id']}")
        print(f"  Câu hỏi:    \"{item['question']}\"")
        print(f"  Kỳ vọng:    Đáp án: {item['expected_ans']} | Frame: {item['expected_evidence']}")

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
    print("                             TỔNG HỢP KẾT QUẢ TEST 5 CÂU")
    print("=" * 85)
    print(f"{'STT':<4} | {'Query ID':<18} | {'Video':<9} | {'Frame':<6} | {'Đáp án VLM':<18} | {'Conf':<5} | {'Latency':<9} | {'Grounding'}")
    print("-" * 85)
    for item, res in results:
        ground_status = "PASSED" if (res.is_grounded if item['num'] != 5 else not res.is_grounded) else "FAILED"
        print(f"{item['num']:<4} | {item['id']:<18} | {res.video_id:<9} | {res.frame_id:<6} | {res.answer:<18} | {res.confidence:<5.2f} | {res.latency_ms:<6.1f} ms | {ground_status}")
    print("=" * 85)


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEMINI_API_KEY", "")
    run_5_test_cases(key)
