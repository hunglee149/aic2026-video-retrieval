from aic.core.types import Candidate

def fuse(runs, limit=100):
    merged_candidates = {}
    
    for run in runs:
        for candidate in run:
            vid = candidate.video_id
            if vid not in merged_candidates:
                merged_candidates[vid] = candidate
            else:
                existing = merged_candidates[vid]
                for k, v in candidate.scores.items():
                    if k not in existing.scores:
                        existing.scores[k] = v
                    else:
                        existing.scores[k] = max(existing.scores[k], v)
                
                if "objects" in candidate.evidence:
                    if "objects" not in existing.evidence:
                        existing.evidence["objects"] = []
                    existing.evidence["objects"].extend(candidate.evidence["objects"])
                    existing.evidence["objects"] = list(set(existing.evidence["objects"]))
                
                if "caption" in candidate.evidence and "caption" not in existing.evidence:
                    existing.evidence["caption"] = candidate.evidence["caption"]
                    
    out = list(merged_candidates.values())
    
    for c in out:
        if "fused" not in c.scores:
            c.scores["fused"] = sum(c.scores.values())

    out.sort(key=lambda c: c.best_score, reverse=True)
    out = out[:limit]
            
    dummy_idx = 0
    seen = {c.video_id for c in out}
    while len(out) < limit:
        dummy_vid = f"L00_V{dummy_idx:03d}"
        if dummy_vid not in seen:
            out.append(Candidate(video_id=dummy_vid, start_frame=0, end_frame=0))
            seen.add(dummy_vid)
        dummy_idx += 1
        
    return out

