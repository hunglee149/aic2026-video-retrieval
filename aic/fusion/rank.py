from aic.core.types import Candidate

def fuse(runs, limit=100, k=60):
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
                
                if "objects" in candidate.evidence:
                    if "objects" not in existing.evidence:
                        existing.evidence["objects"] = []
                    existing.evidence["objects"].extend(candidate.evidence["objects"])
                    existing.evidence["objects"] = list(set(existing.evidence["objects"]))
                
                if "caption" in candidate.evidence and "caption" not in existing.evidence:
                    existing.evidence["caption"] = candidate.evidence["caption"]
            
            # Cộng điểm RRF (Reciprocal Rank Fusion)
            merged_candidates[vid].scores["fused"] += 1.0 / (k + rank)
                    
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

