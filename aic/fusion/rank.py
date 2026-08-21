from aic.core.types import Candidate

def fuse(runs, limit=100, k=60, weights=None, modalities=None):
    weights = weights or {}
    merged_candidates = {}
    
    for run in runs:
        for rank, candidate in enumerate(run, start=1):
            vid = candidate.video_id
            
            if vid not in merged_candidates:
                merged_candidates[vid] = candidate
                if "fused" not in candidate.scores:
                    candidate.scores["fused"] = 0.0
            else:
                existing = merged_candidates[vid]
                for source, score in candidate.scores.items():
                    if source not in existing.scores:
                        existing.scores[source] = score
                    else:
                        existing.scores[source] = max(existing.scores[source], score)
                
                for ev_key, ev_val in candidate.evidence.items():
                    if ev_key not in existing.evidence:
                        existing.evidence[ev_key] = ev_val
            
            # Tính weight theo cấu hình của người dùng và độ tin cậy của từng nguồn
            weight = 1.0

            # Nếu là BM25 (chứa Caption, OCR, ASR, Summary, Media Info)
            if "bm25" in candidate.scores:
                bm25_val = candidate.scores["bm25"]
                has_asr = "transcript_match" in candidate.evidence and candidate.evidence["transcript_match"]
                has_caption = "caption_match" in candidate.evidence and candidate.evidence["caption_match"]
                has_ocr = "ocr_match" in candidate.evidence and candidate.evidence["ocr_match"]
                has_summary = "summary_match" in candidate.evidence and candidate.evidence["summary_match"]
                has_media = "media_info_match" in candidate.evidence and candidate.evidence["media_info_match"]
                
                custom_text_w = 1.0
                if has_asr:
                    custom_text_w = max(custom_text_w, weights.get("asr", 1.0))
                if has_ocr:
                    custom_text_w = max(custom_text_w, weights.get("ocr", 1.0))
                if has_caption:
                    custom_text_w = max(custom_text_w, weights.get("caption", 1.0))
                if has_summary:
                    custom_text_w = max(custom_text_w, weights.get("summary", 1.0))
                if has_media:
                    custom_text_w = max(custom_text_w, weights.get("media_info", 1.0))
                
                if has_caption or has_ocr or has_asr:
                    weight = 4.0 * (1.0 + bm25_val) * custom_text_w
                elif bm25_val >= 0.7:
                    weight = 2.5 * bm25_val * custom_text_w
                else:
                    weight = 1.0 * bm25_val * custom_text_w

            # Nếu là SigLIP (Visual text-to-image)
            elif "siglip" in candidate.scores:
                siglip_val = candidate.scores["siglip"]
                siglip_user_w = weights.get("siglip", 1.0)
                if siglip_val >= 0.15:
                    weight = 1.5 * siglip_user_w
                elif siglip_val < 0.10:
                    weight = 0.3 * siglip_user_w
                else:
                    weight = 0.8 * siglip_user_w

            # Cộng điểm Confidence-Weighted RRF
            rrf_point = weight / (k + rank)
            merged_candidates[vid].scores["fused"] += rrf_point
                    
    out = list(merged_candidates.values())
    out.sort(key=lambda c: c.scores.get("fused", 0.0), reverse=True)
    out = out[:limit]
            
    dummy_idx = 0
    seen = {c.video_id for c in out}
    while len(out) < limit:
        dummy_vid = f"L00_V{dummy_idx:03d}"
        if dummy_vid not in seen:
            out.append(Candidate(video_id=dummy_vid, start_frame=0, end_frame=0, scores={"fused": 0.0}))
            seen.add(dummy_vid)
        dummy_idx += 1
        
    return out


# Aliases for compatibility
rrf_fuse = fuse
rank = fuse

