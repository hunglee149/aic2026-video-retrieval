"""Test Multi-Frame Context Reasoning on Gemini 3.6 Flash."""

import io
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


def extract_frame(video_id: str, frame_num: int) -> Image.Image:
    prefix = video_id.split("_")[0]
    zip_path = Path(f"data/keyframes/Keyframes_{prefix}.zip")
    target_suffixes = (
        f"{video_id}/{frame_num:03d}.jpg",
        f"{video_id}/{frame_num:04d}.jpg",
        f"{video_id}/{frame_num}.jpg",
    )
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith(target_suffixes):
                with z.open(name) as f:
                    return Image.open(io.BytesIO(f.read()))
    raise FileNotFoundError(f"Frame {frame_num} not found")


def test_multi_frame_temporal_reasoning(api_key: str):
    print("=" * 85)
    print("       KIỂM TRA KHẢ NĂNG SUY LUẬN ĐA KHUNG HÌNH (MULTI-FRAME REASONING)")
    print("=" * 85)

    engine = GroundedQAEngine(api_key=api_key, model_name="gemini-3.6-flash")

    # Đề thi có context dài nhiều sự kiện theo thứ tự thời gian:
    # Sự kiện 1: Cá đặt lên cân (Frame 75-76)
    # Sự kiện 2: Người kéo đuôi con cá khác (Frame 77)
    # Target: Hỏi con số cân trước khi con cá thứ 2 bị kéo
    question = (
        "Đoạn video mở đầu bằng cảnh một con cá ngừ lớn được đặt trên bàn cân điện tử, "
        "ngay sau đó máy quay chuyển sang cảnh một người nhân viên đang kéo đuôi một con cá khác cùng loại. "
        "Dựa vào diễn biến đó, con số hiển thị màu xanh trên màn hình cân khi con cá đầu tiên đang nằm trên cân là bao nhiêu?"
    )

    video_id = "L21_V007"
    frame_sequence = [72, 74, 76, 77, 80]
    
    print(f"\nCâu hỏi ngữ cảnh dài:\n\"{question}\"\n")
    print(f"Chuỗi khung hình cung cấp: {frame_sequence}")

    candidate_frames = [(f, extract_frame(video_id, f)) for f in frame_sequence]

    res: QAResult = engine.answer_query(
        question=question,
        video_id=video_id,
        candidate_frames=candidate_frames,
    )

    print("\n" + "-" * 60)
    print(f"  • Frame bằng chứng VLM chọn: {res.frame_id}")
    print(f"  • Câu trả lời VLM trích xuất: '{res.answer}'")
    print(f"  • Độ tin cậy:                {res.confidence:.2f}")
    print(f"  • Lập luận đa khung hình:    {res.evidence}")
    print(f"  • Dòng CSV nộp bài:          {res.to_submission_row()}")
    print("-" * 60)


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEMINI_API_KEY", "")
    test_multi_frame_temporal_reasoning(key)
