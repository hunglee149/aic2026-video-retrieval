"""Task 2 VQA Benchmark & Real Keyframe Evaluation."""

import logging
import os
import sys
import zipfile
from pathlib import Path
from PIL import Image
import io

from aic.qa.vlm_engine import GroundedQAEngine, QAResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_frame_from_zip(zip_path: Path, video_id: str, frame_num: int) -> Image.Image:
    """Extracts a specific keyframe image directly from a Keyframes zip file."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Keyframe zip not found: {zip_path}")
    
    target_names = [
        f"{video_id}/{frame_num:03d}.jpg",
        f"{video_id}/{frame_num:04d}.jpg",
        f"{video_id}/{frame_num}.jpg",
        f"keyframes/{video_id}/{frame_num:03d}.jpg",
    ]
    
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            for target in target_names:
                if name.endswith(target):
                    with z.open(name) as f:
                        return Image.open(io.BytesIO(f.read()))
    
    raise KeyError(f"Frame {frame_num} of {video_id} not found in {zip_path}")


def run_task2_benchmark():
    """Runs Grounded VQA evaluation on known Round 1 QA benchmark cases."""
    print("=" * 80)
    print("  AIC 2026 TASK 2 (VQA & EVIDENCE GROUNDING) BENCHMARK EVALUATION")
    print("=" * 80)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[WARNING] GEMINI_API_KEY is not set in environment.")
        print("To run live benchmark with Gemini 2.0 Flash, please set GEMINI_API_KEY in your .env or terminal.")
        print("Example: $env:GEMINI_API_KEY='your_api_key'\n")
        return

    engine = GroundedQAEngine(api_key=api_key, model_name="gemini-2.0-flash")

    test_cases = [
        {
            "id": "query-p1-9-qa",
            "question": (
                "Đoạn phim ghi lại cảnh những chiếc xe ô tô lội nước, chiếc xe màu vàng, màu đỏ và màu đen "
                "lần lượt chuẩn bị đi qua cầu. Con số được ghi trên biển báo bên trái của cây cầu là bao nhiêu?"
            ),
            "video_id": "L21_V003",
            "zip_path": Path("data/Keyframes_L21.zip"),
            "frames": [245, 250, 252, 255, 260],
            "expected_evidence_frame": 252,
            "expected_answer_keywords": ["2,15", "2.15"],
        },
        {
            "id": "query-p1-3-qa",
            "question": (
                "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loại bị một người cầm đuôi. "
                "Con số hiển thị cuối cùng trên cân là bao nhiêu?"
            ),
            "video_id": "L21_V007",
            "zip_path": Path("data/Keyframes_L21.zip"),
            "frames": [72, 74, 76, 78, 80],
            "expected_evidence_frame": 76,
            "expected_answer_keywords": ["38.35", "38,35"],
        },
        {
            "id": "trick-negative-qa",
            "question": (
                "Chiếc xe ô tô bay trên bầu trời màu xanh mang biển số gì?"
            ),
            "video_id": "L21_V003",
            "zip_path": Path("data/Keyframes_L21.zip"),
            "frames": [245, 250],
            "expected_evidence_frame": None,
            "expected_answer_keywords": ["UNKNOWN"],
        },
    ]

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n--- Case {idx}: [{tc['id']}] {tc['video_id']} ---")
        print(f"Question: {tc['question']}")
        
        # Load candidate frames
        candidate_images = []
        for f_num in tc["frames"]:
            try:
                img = extract_frame_from_zip(tc["zip_path"], tc["video_id"], f_num)
                candidate_images.append((f_num, img))
            except Exception as e:
                logger.warning("Could not extract frame %d: %s", f_num, e)

        if not candidate_images:
            print(f"[SKIP] Keyframes for {tc['video_id']} not found at {tc['zip_path']}.")
            continue

        print(f"Loaded {len(candidate_images)} candidate frames: {[f[0] for f in candidate_images]}")
        
        # Run inference
        result: QAResult = engine.answer_query(
            question=tc["question"],
            video_id=tc["video_id"],
            candidate_frames=candidate_images,
        )

        print("\n[RESULT REPORT]")
        print(f"  • Video ID:    {result.video_id}")
        print(f"  • Evidence Frame: {result.frame_id}")
        print(f"  • Answer:      '{result.answer}'")
        print(f"  • Confidence:  {result.confidence:.2f}")
        print(f"  • Grounded:    {result.is_grounded}")
        print(f"  • Latency:     {result.latency_ms:.1f} ms")
        print(f"  • Visual Proof: {result.evidence}")
        print(f"  • Submission CSV Row: {result.to_submission_row()}")

        # Check correctness
        matched_expected = any(kw.lower() in result.answer.lower() for kw in tc["expected_answer_keywords"])
        print(f"  • Evaluation:  {'PASSED' if matched_expected else 'FAILED'}")


if __name__ == "__main__":
    run_task2_benchmark()
