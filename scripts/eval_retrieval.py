#!/usr/bin/env python3
"""Chấm điểm retrieval trên một bộ sanity set đã gán nhãn.

Mục đích: hiện tại không ai biết retrieval tốt hay dở, nên mọi thay đổi đều là
đoán. Có bộ này rồi thì mới nói được "đổi X làm top-5 tăng 6%" thay vì "chắc là
tốt hơn", và mới chỉnh được trọng số RRF có căn cứ.

Chỉ số báo ra:

- **Video Recall@K** — video đúng có nằm trong K ứng viên đầu không.
- **Moment hit@K** — có ứng viên nào vừa đúng video vừa đúng cửa sổ frame không.
  Đây mới là thứ tính điểm thật: đúng video mà sai frame thì R-Score vẫn là 0.
- **Điểm dạng BTC** — ``mean(R@1, R@5, R@20, R@50, R@100)`` theo công thức đề thi.
- **Nguồn nào lập công** — ứng viên đúng do CLIP, SigLIP hay BM25 đưa vào.

Định dạng sanity set: JSONL, mỗi dòng một câu::

    {"query_id": "s01", "text_vi": "cảnh sát giao thông ở ngã tư đông xe",
     "video_id": "L21_V031", "frame": 1500, "window": [1400, 1700],
     "task": "kis", "verified": true}

- ``frame`` và ``window`` là **actual video frame, 0-based**, cùng hệ với
  ``Candidate.start_frame`` (không phải frame 1-based lúc nộp bài).
- Thiếu ``window`` thì lấy ``frame ± --frame-tolerance``.
- ``verified: false`` là phiếu nháp chưa ai kiểm; script chấm riêng hai nhóm và
  **không gộp** phiếu chưa kiểm vào con số chính.

Chạy::

    python scripts/eval_retrieval.py --set eval/sanity_set.jsonl
    python scripts/eval_retrieval.py --set eval/sanity_set.jsonl \\
        --weights clip=1.0,bm25=2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("eval_retrieval")

CUTOFFS = (1, 5, 20, 50, 100)

# ~5 giây ở 30fps. Cửa sổ KIS thật thường rộng hơn nhiều, nên đây là mức chặt.
DEFAULT_FRAME_TOLERANCE = 150


def load_sanity_set(path: Path) -> list[dict]:
    entries = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("//") or raw.startswith("#"):
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            sys.exit(f"{path}:{lineno} JSON hỏng: {exc}")
        missing = [k for k in ("query_id", "text_vi", "video_id") if not entry.get(k)]
        if missing:
            sys.exit(f"{path}:{lineno} thiếu trường: {', '.join(missing)}")
        if entry.get("frame") is None and not entry.get("window"):
            sys.exit(f"{path}:{lineno} cần 'frame' hoặc 'window'")
        entries.append(entry)
    return entries


def window_of(entry: dict, tolerance: int) -> tuple[int, int]:
    window = entry.get("window")
    if window:
        return int(window[0]), int(window[1])
    frame = int(entry["frame"])
    return frame - tolerance, frame + tolerance


def parse_weights(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    weights = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            sys.exit(f"--weights sai định dạng ở '{part}', cần dạng ten=so")
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


def build_retrievers(disable: set[str]):
    """Nạp retriever từ đúng cấu hình .env mà app đang dùng."""
    from aic.ui import app as app_module

    retrievers, statuses = app_module.build_retriever_registry(
        use_dummy=False,
        clip_index=app_module.INDEX_PATH,
        clip_meta=app_module.META_PATH,
        siglip_index=app_module.SIGLIP_INDEX_PATH,
        siglip_meta=app_module.SIGLIP_META_PATH,
        text_index=app_module.TEXT_INDEX_PATH,
        dummy_module=None,
        disable_neural=app_module.DISABLE_NEURAL,
    )
    for slot in statuses:
        logger.info("  %-7s %-9s %s", slot["name"], slot["state"], slot["detail"])

    if disable:
        kept = []
        for retriever in retrievers:
            name = getattr(retriever, "name", "")
            if name in disable:
                logger.info("  bỏ qua %s theo --disable", name)
                continue
            kept.append(retriever)
        retrievers = kept

    if not retrievers:
        sys.exit("Không nạp được nguồn retrieval nào — kiểm tra .env")
    return retrievers, statuses


def evaluate(entries, retrievers, weights, tolerance, limit):
    from aic.core.query_processor import process_query
    from aic.core.types import Query
    from aic.fusion.rank import fuse
    from aic.pipeline import retrieve_and_fuse

    fuse_kwargs = {"weights": weights} if weights else None
    results = []

    for entry in entries:
        query = Query(
            query_id=entry["query_id"],
            text_vi=entry["text_vi"],
            text_en=entry.get("text_en", ""),
            task=entry.get("task", "kis"),
        )
        if not query.text_en:
            process_query(query)

        candidates = retrieve_and_fuse(
            query=query,
            retrievers=retrievers,
            fuse_fn=fuse,
            limit=limit,
            fuse_kwargs=fuse_kwargs,
        )

        gt_video = entry["video_id"]
        low, high = window_of(entry, tolerance)

        video_rank = None
        moment_rank = None
        moment_sources: tuple[str, ...] = ()
        best_distance = None

        for rank, cand in enumerate(candidates, start=1):
            if cand.video_id != gt_video:
                continue
            if video_rank is None:
                video_rank = rank
            frame = cand.representative_frames[0] if cand.representative_frames \
                else cand.start_frame
            if entry.get("frame") is not None:
                distance = abs(int(frame) - int(entry["frame"]))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
            if low <= frame <= high and moment_rank is None:
                moment_rank = rank
                moment_sources = tuple(
                    sorted(k for k in cand.scores if k != "fused")
                )

        results.append(
            {
                "entry": entry,
                "text_en": query.text_en,
                "total": len(candidates),
                "video_rank": video_rank,
                "moment_rank": moment_rank,
                "moment_sources": moment_sources,
                "best_distance": best_distance,
            }
        )
    return results


def recall_at(results, key, cutoff) -> float:
    if not results:
        return 0.0
    hits = sum(1 for r in results if r[key] is not None and r[key] <= cutoff)
    return hits / len(results)


def report(results, label: str) -> None:
    print(f"\n{'=' * 66}")
    print(f"{label}  —  {len(results)} câu")
    print("=" * 66)
    if not results:
        print("  (không có câu nào)")
        return

    print(f"\n  {'K':>4}   {'Video Recall@K':>16}   {'Moment hit@K':>14}")
    for cutoff in CUTOFFS:
        video = recall_at(results, "video_rank", cutoff)
        moment = recall_at(results, "moment_rank", cutoff)
        print(f"  {cutoff:>4}   {video:>15.1%}   {moment:>13.1%}")

    competition = sum(
        recall_at(results, "moment_rank", c) for c in CUTOFFS
    ) / len(CUTOFFS)
    video_score = sum(
        recall_at(results, "video_rank", c) for c in CUTOFFS
    ) / len(CUTOFFS)
    print(f"\n  Điểm dạng BTC (moment) : {competition:.4f}")
    print(f"  Cùng công thức, chỉ xét video (chặn trên): {video_score:.4f}")

    missed = [r for r in results if r["video_rank"] is None]
    print(f"  Không thấy video đúng trong top-{results[0]['total']}: "
          f"{len(missed)}/{len(results)}")

    sources = Counter()
    for r in results:
        if r["moment_sources"]:
            sources[" + ".join(r["moment_sources"])] += 1
    if sources:
        print("\n  Nguồn đưa được moment đúng vào danh sách:")
        for name, count in sources.most_common():
            print(f"    {name:<24} {count:>3} câu")

    distances = [r["best_distance"] for r in results if r["best_distance"] is not None]
    if distances:
        distances.sort()
        median = distances[len(distances) // 2]
        print(f"\n  Khoảng cách frame gần nhất (khi tìm đúng video):")
        print(f"    min={distances[0]}  trung vị={median}  max={distances[-1]}")

    print("\n  Chi tiết từng câu:")
    for r in results:
        entry = r["entry"]
        video = r["video_rank"] if r["video_rank"] else "—"
        moment = r["moment_rank"] if r["moment_rank"] else "—"
        flag = " " if r["moment_rank"] else "!"
        print(f"   {flag} {entry['query_id']:<8} video@{str(video):<5} "
              f"moment@{str(moment):<5} {entry['text_vi'][:44]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True, type=Path, help="file JSONL")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--frame-tolerance", type=int,
                        default=DEFAULT_FRAME_TOLERANCE)
    parser.add_argument("--weights", help="vd: clip=1.0,siglip=1.0,bm25=2.0")
    parser.add_argument("--disable", default="",
                        help="tắt bớt nguồn, vd: bm25 hoặc clip,siglip")
    parser.add_argument("--include-unverified", action="store_true",
                        help="gộp cả phiếu chưa kiểm vào con số chính")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.set.exists():
        sys.exit(f"Không thấy sanity set: {args.set}\n"
                 f"Tạo phiếu nháp bằng: python scripts/make_sanity_drafts.py")

    entries = load_sanity_set(args.set)
    if not entries:
        sys.exit(f"{args.set} rỗng")

    blank = [e for e in entries if not e["text_vi"].strip()]
    if blank:
        logger.warning("Bỏ qua %d phiếu chưa điền mô tả", len(blank))
        entries = [e for e in entries if e["text_vi"].strip()]
    if not entries:
        sys.exit("Chưa có câu nào được điền mô tả — không chấm được gì")

    disable = {s.strip() for s in args.disable.split(",") if s.strip()}
    weights = parse_weights(args.weights)

    logger.info("Nạp retriever:")
    retrievers, _ = build_retrievers(disable)
    logger.info("Chấm %d câu (limit=%d, tolerance=±%d frame)...",
                len(entries), args.limit, args.frame_tolerance)

    results = evaluate(entries, retrievers, weights, args.frame_tolerance,
                       args.limit)

    verified = [r for r in results if r["entry"].get("verified") is True]
    unverified = [r for r in results if r["entry"].get("verified") is not True]

    if args.include_unverified or not verified:
        if not verified and unverified:
            print("\n*** CẢNH BÁO: chưa câu nào được đánh dấu verified:true. ***")
            print("*** Con số dưới đây là của phiếu CHƯA KIỂM, đừng trích dẫn ***")
            print("*** như kết quả chính thức.                                ***")
        report(results, "TẤT CẢ (gồm cả phiếu chưa kiểm)")
    else:
        report(verified, "ĐÃ KIỂM (verified: true)")
        if unverified:
            report(unverified, "CHƯA KIỂM — chỉ tham khảo")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
