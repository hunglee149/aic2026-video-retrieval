"""Comprehensive Task 2 Grounded VQA Benchmark System.

Executes a 25-query benchmark across 6 diverse categories:
- Direct Recognition (5)
- OCR (5)
- Counting (5)
- Temporal Multi-Frame Sequence (5)
- ASR Required (3)
- Insufficient Evidence / Anti-Hallucination (3)

Evaluates in both modes:
1. Oracle / Manual Frames (Pure VLM comprehension)
2. Current Retrieval Frames (Real-world system grounding)

Compares Gemini Flash vs. Local Thinking VLM (Qwen3-VL Thinking).
Generates:
- benchmark_predictions.jsonl / .csv
- Summary Metric Tables
- Head-to-Head Win/Loss Matrix
- Disagreement Error Analysis
- 4-Tier Strategic Conclusion
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from aic.qa.vlm_engine import GroundedQAEngine, QAResult, sanitize_vqa_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_frame_from_store(video_id: str, frame_num: int) -> Image.Image:
    """Extract keyframe image from local dir or zip file."""
    prefix = video_id.split("_")[0]
    
    # 1. Local path
    local_candidates = [
        Path(f"keyframes/{video_id}/{frame_num:03d}.jpg"),
        Path(f"keyframes/{video_id}/{frame_num:04d}.jpg"),
        Path(f"keyframes/{video_id}/{frame_num}.jpg"),
        Path(f"data/keyframes/{video_id}/{frame_num:03d}.jpg"),
    ]
    for lp in local_candidates:
        if lp.exists():
            return Image.open(lp).convert("RGB")

    # 2. Keyframes ZIP files
    zip_candidates = [
        Path(f"data/keyframes/Keyframes_{prefix}.zip"),
        Path(f"data/keyframes/Keyframes_{prefix}_a.zip"),
        Path(f"data/keyframes/Keyframes_{prefix}_b.zip"),
        Path(f"data/keyframes/Keyframes_{prefix}_c.zip"),
        Path(f"data/keyframes/Keyframes_{prefix}_d.zip"),
        Path(f"data/Keyframes_{prefix}.zip"),
    ]
    target_suffixes = (
        f"{video_id}/{frame_num:03d}.jpg",
        f"{video_id}/{frame_num:04d}.jpg",
        f"{video_id}/{frame_num}.jpg",
        f"{video_id}_{frame_num:03d}.jpg",
    )
    for zp in zip_candidates:
        if zp.exists():
            try:
                with zipfile.ZipFile(zp, "r") as z:
                    for name in z.namelist():
                        if name.endswith(target_suffixes):
                            with z.open(name) as f:
                                return Image.open(io.BytesIO(f.read())).convert("RGB")
            except Exception:
                pass

    return Image.new("RGB", (320, 240), color="gray")


def normalize_answer_str(text: str) -> str:
    """Normalize string for strict/flexible comparative evaluation."""
    s = text.lower().strip()
    s = re.sub(r"[,\.](\d+)", r".\1", s)  # normalize 2,15 -> 2.15
    s = re.sub(r"[^\w\.\d\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_answer_correctness(prediction: str, gold: str, aliases: List[str]) -> bool:
    """Evaluate if prediction matches gold or any accepted alias."""
    norm_pred = normalize_answer_str(prediction)
    if not norm_pred:
        return False
    
    all_golds = [gold] + aliases
    for g in all_golds:
        norm_g = normalize_answer_str(g)
        if norm_pred == norm_g:
            return True
        # Substring / number containment for numbers
        if re.match(r"^\d+(\.\d+)?$", norm_g) and norm_g in norm_pred.split():
            return True
        if norm_g in norm_pred or norm_pred in norm_g:
            # High overlap match
            if len(norm_pred) >= 3 and len(norm_g) >= 3:
                return True
    return False


def get_current_vram_gb() -> float:
    """Get peak VRAM in GB if PyTorch with CUDA is active."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def run_full_benchmark(api_key: str, dataset_path: str = "aic/qa/benchmark_dataset.json"):
    print("\n" + "=" * 95)
    print("      AIC 2026 TASK 2: SCIENTIFIC VLM BENCHMARK SUITE (25 QUERIES)")
    print("      Comparing: Gemini 3.6 Flash (Cloud) vs. Qwen3-VL Thinking (Local)")
    print("=" * 95)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\nLoaded {len(dataset)} benchmark queries across 6 categories.\n")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / "benchmark_predictions.jsonl"
    csv_path = out_dir / "benchmark_predictions.csv"

    engine = GroundedQAEngine(api_key=api_key)

    records = []
    models_to_test = ["gemini", "local"]
    modes_to_test = ["oracle", "retrieval"]

    total_runs = len(dataset) * len(models_to_test) * len(modes_to_test)
    run_idx = 0

    for item in dataset:
        qid = item["query_id"]
        qtype = item["question_type"]
        qtext = item["question"]
        if item.get("asr_transcript"):
            qtext_with_asr = f"{qtext} (Ghi chú âm thanh/ASR: \"{item['asr_transcript']}\")"
        else:
            qtext_with_asr = qtext

        vid = item["video_id"]
        gold = item["gold_answer"]
        aliases = item.get("gold_aliases", [])

        for mode in modes_to_test:
            frames_to_use = item["oracle_frames"] if mode == "oracle" else item["retrieval_frames"]
            candidate_images = [(fid, extract_frame_from_store(vid, fid)) for fid in frames_to_use]

            for model_name in models_to_test:
                run_idx += 1
                sys.stdout.write(f"\r[{run_idx}/{total_runs}] Testing {qid} | Mode: {mode:<9} | Model: {model_name:<8}...")
                sys.stdout.flush()

                t0 = time.perf_counter()
                vram_before = get_current_vram_gb()
                err = ""

                try:
                    res: QAResult = engine.answer_query(
                        question=qtext_with_asr,
                        video_id=vid,
                        candidate_frames=candidate_images,
                        provider=model_name,
                    )
                    raw_ans = res.answer
                    latency_sec = res.latency_ms / 1000.0
                    vram_after = get_current_vram_gb()
                    is_correct = check_answer_correctness(raw_ans, gold, aliases)
                    is_supported = res.is_grounded and (res.frame_id in item["oracle_frames"])
                except Exception as e:
                    err = str(e)
                    raw_ans = "ERROR"
                    latency_sec = time.perf_counter() - t0
                    vram_after = vram_before
                    is_correct = False
                    is_supported = False

                norm_ans = normalize_answer_str(raw_ans)

                rec = {
                    "query_id": qid,
                    "question_type": qtype,
                    "evidence_mode": mode,
                    "model": "Gemini 3.6 Flash" if model_name == "gemini" else "Qwen3-VL Thinking",
                    "prompt_version": "v2.0_grounded_structured",
                    "number_of_frames": len(frames_to_use),
                    "gold_answer": gold,
                    "raw_answer": raw_ans,
                    "normalized_answer": norm_ans,
                    "is_correct": is_correct,
                    "is_supported_by_evidence": is_supported,
                    "latency_seconds": round(latency_sec, 2),
                    "peak_vram_gb": round(vram_after, 2),
                    "error": err,
                }
                records.append(rec)

    print("\n\nWriting prediction outputs to results/...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # ---------------------------------------------------------------------------
    # AGGREGATED EVALUATION & TABLES
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("                        BẢNG TỔNG HỢP CHẤT LƯỢNG BẮT BUỘC")
    print("=" * 95)

    def calc_acc(sub_list):
        if not sub_list:
            return 0.0
        return sum(1 for x in sub_list if x["is_correct"]) / len(sub_list) * 100.0

    gemini_recs = [r for r in records if "Gemini" in r["model"]]
    local_recs = [r for r in records if "Qwen" in r["model"]]

    print(f"{'Mô hình':<20} | {'Oracle Acc':<10} | {'Retr. Acc':<10} | {'OCR Acc':<9} | {'Count Acc':<10} | {'Temp Acc':<9} | {'Halluc. Rate':<13} | {'Avg Time':<10} | {'VRAM'}")
    print("-" * 115)

    for m_name, m_list in [("Gemini 3.6 Flash", gemini_recs), ("Qwen3-VL Thinking", local_recs)]:
        or_acc = calc_acc([r for r in m_list if r["evidence_mode"] == "oracle"])
        ret_acc = calc_acc([r for r in m_list if r["evidence_mode"] == "retrieval"])
        ocr_acc = calc_acc([r for r in m_list if r["question_type"] == "ocr"])
        cnt_acc = calc_acc([r for r in m_list if r["question_type"] == "counting"])
        tmp_acc = calc_acc([r for r in m_list if r["question_type"] == "temporal_multi_frame"])
        
        trick_recs = [r for r in m_list if r["question_type"] == "insufficient_evidence"]
        halluc_count = sum(1 for r in trick_recs if r["normalized_answer"] != "unknown" and r["normalized_answer"] != "")
        halluc_rate = (halluc_count / len(trick_recs) * 100.0) if trick_recs else 0.0
        
        avg_time = sum(r["latency_seconds"] for r in m_list) / len(m_list)
        vram_str = "Cloud API" if "Gemini" in m_name else f"{max(r['peak_vram_gb'] for r in m_list):.1f} GB"

        print(f"{m_name:<20} | {or_acc:>8.1f}% | {ret_acc:>8.1f}% | {ocr_acc:>7.1f}% | {cnt_acc:>8.1f}% | {tmp_acc:>7.1f}% | {halluc_rate:>11.1f}% | {avg_time:>7.2f}s | {vram_str}")
    print("=" * 115)

    # ---------------------------------------------------------------------------
    # HEAD-TO-HEAD WIN / LOSS MATRIX
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("               BẢNG ĐỐI ĐẦU TRỰC TIẾP (HEAD-TO-HEAD)")
    print("=" * 65)

    both_correct = 0
    both_wrong = 0
    local_win = 0
    gemini_win = 0
    disagreements = []

    for item in dataset:
        qid = item["query_id"]
        for mode in ["oracle", "retrieval"]:
            g_rec = next(r for r in gemini_recs if r["query_id"] == qid and r["evidence_mode"] == mode)
            l_rec = next(r for r in local_recs if r["query_id"] == qid and r["evidence_mode"] == mode)

            g_ok = g_rec["is_correct"]
            l_ok = l_rec["is_correct"]

            if g_ok and l_ok:
                both_correct += 1
            elif not g_ok and not l_ok:
                both_wrong += 1
            elif l_ok and not g_ok:
                local_win += 1
                disagreements.append(("Qwen Thắng", qid, mode, item["question"], item["gold_answer"], g_rec["raw_answer"], l_rec["raw_answer"]))
            elif g_ok and not l_ok:
                gemini_win += 1
                disagreements.append(("Gemini Thắng", qid, mode, item["question"], item["gold_answer"], g_rec["raw_answer"], l_rec["raw_answer"]))

    total_pairs = both_correct + both_wrong + local_win + gemini_win
    print(f"{'Kết quả trên cùng câu hỏi & frame':<45} | {'Số lượng':<8} | {'Tỷ lệ':<8}")
    print("-" * 65)
    print(f"{'Cả hai mô hình CÙNG ĐÚNG':<45} | {both_correct:>8} | {both_correct/total_pairs*100:>6.1f}%")
    print(f"{'Cả hai mô hình CÙNG SAI':<45} | {both_wrong:>8} | {both_wrong/total_pairs*100:>6.1f}%")
    print(f"{'Gemini ĐÚNG, Qwen Thinking SAI (Gemini Win)':<45} | {gemini_win:>8} | {gemini_win/total_pairs*100:>6.1f}%")
    print(f"{'Qwen Thinking ĐÚNG, Gemini SAI (Local Win)':<45} | {local_win:>8} | {local_win/total_pairs*100:>6.1f}%")
    print("=" * 65)

    # ---------------------------------------------------------------------------
    # DISAGREEMENT ANALYSIS
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("                PHÂN TÍCH CÁC TRƯỜNG HỢP HAI MÔ HÌNH BẤT ĐỒNG")
    print("=" * 95)
    if not disagreements:
        print("Hai mô hình đồng thuận 100% trên các câu hỏi được đánh giá.")
    else:
        for idx, (verdict, qid, mode, qtext, gold, g_ans, l_ans) in enumerate(disagreements[:6], 1):
            print(f"[{idx}] [{verdict}] Query: {qid} (Mode: {mode})")
            print(f"    • Câu hỏi:    \"{qtext}\"")
            print(f"    • Đáp án gốc: \"{gold}\"")
            print(f"    • Gemini:     \"{g_ans}\"")
            print(f"    • Qwen Local: \"{l_ans}\"\n")

    # ---------------------------------------------------------------------------
    # FINAL STRATEGIC CONCLUSION
    # ---------------------------------------------------------------------------
    print("=" * 95)
    print("                         KẾT LUẬN CHIẾN LƯỢC KIẾN TRÚC")
    print("=" * 95)
    
    local_or_acc = calc_acc([r for r in local_recs if r["evidence_mode"] == "oracle"])
    gemini_or_acc = calc_acc([r for r in gemini_recs if r["evidence_mode"] == "oracle"])
    relative_perf = (local_or_acc / gemini_or_acc * 100.0) if gemini_or_acc > 0 else 0.0

    print(f"Độ chính xác tương đối của Local VLM so với Gemini: {relative_perf:.1f}%\n")
    if relative_perf >= 95.0 and local_win >= gemini_win:
        conclusion_type = "1. Thay Gemini hoàn toàn bằng Local Thinking VLM."
        recommendation = "Local VLM đạt độ chính xác vượt trội, chi phí 0đ, không cần API Key/Internet."
    elif relative_perf >= 80.0:
        conclusion_type = "2. Kiến trúc 2 Tầng (Cascade): Local VLM chạy trước, Gemini làm fallback."
        recommendation = "Local VLM xử lý 80%+ các câu hỏi phổ thông nhanh chóng; các câu có confidence < 0.6 sẽ fallback sang Gemini Cloud."
    elif local_win > 0:
        conclusion_type = "3. Router Định Tuyến Theo Loại Câu Hỏi."
        recommendation = "Dùng Local VLM cho các câu Đếm/Nhận diện; Dùng Gemini cho OCR siêu nhỏ và chuỗi thời gian dài."
    else:
        conclusion_type = "4. Giữ Gemini Cloud làm mặc định, Local VLM làm chế độ dự phòng Offline."
        recommendation = "Gemini Cloud cho độ chính xác cao nhất và độ trễ thấp nhất trong điều kiện thi đấu."

    print(f"👉 KẾT LUẬN CHÍNH THỨC: {conclusion_type}")
    print(f"👉 KHUYẾN NGHỊ: {recommendation}\n")
    print("=" * 95)


if __name__ == "__main__":
    k = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEMINI_API_KEY", "")
    run_full_benchmark(k)
