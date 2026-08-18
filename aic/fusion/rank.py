# NHO FIX LAI NGHEN CAI NAY TEST THOI

def fuse(runs, limit=100):
    seen = set()
    out = []
    for run in runs:
        for candidate in run:
            key = (candidate.video_id, candidate.start_frame)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
    return out[:limit]
