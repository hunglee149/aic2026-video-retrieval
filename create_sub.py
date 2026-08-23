import os
import re
import csv
import zipfile
import pandas as pd
from pathlib import Path

raw_data = """q1 L30_V046 6675 (kiếm frame có kĩ cái kính)
q2 L28_V018 55, 1876, 2151, 2563, 3258, 3452
q3 L21_v007 8846 "37,05"
q4 L22_V021 10961
q5 L26_V035 5095
q6 L26_V023 18160
q7 L26_V041 3765 6852
q8 L26_V171 3:41, 3:47, 3:52
q9 L21_V003 252 "2,15"
q10 L29_V013 7:32
q11 L23_V021 6359
q12 L22_V001: 1:45
q13 L29_V021 6218, 7042, 7811
q14 L26_V171 3:41, 3:47, 3:52
q15 L21_V010 189 "12"
q16 L24-V041 2:46, 3:22, 3:49
q17 L22_V008 5638 "Đèo Tà Pứa"
q18 L26_V389 6193 6422
22 L25_V041: 181
23 L25_V060 19:18
24 L29_V001: 6:08, 5:57
25 L30_V003: 6589"""

map_dir = Path(r"d:\aic2026-video-retrieval\data\batch_01\map-keyframes-aic25-b1\map-keyframes")

fps = 25

def time_to_frame(time_str):
    parts = time_str.split(':')
    if len(parts) == 2:
        m, s = int(parts[0]), int(parts[1])
        return (m * 60 + s) * fps
    return None

def get_closest_keyframe(video_id, target_frame):
    video_id = video_id.upper().replace('-', '_')
    csv_path = map_dir / f"{video_id}.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} does not exist.")
        return target_frame
    
    df = pd.read_csv(csv_path)
    closest_idx = (df['frame_idx'] - target_frame).abs().idxmin()
    return df.loc[closest_idx, 'frame_idx']

queries = []

for line in raw_data.strip().split('\n'):
    line = line.strip()
    if not line: continue
    
    line = re.sub(r'([A-Za-z0-9_]+-[A-Za-z0-9_]+):', r'\1', line)
    line = re.sub(r'([A-Za-z0-9_]+_[A-Za-z0-9_]+):', r'\1', line)
    
    parts = line.split()
    qid = parts[0]
    vid = parts[1].replace('-', '_').upper()
    
    results = []
    for token in parts[2:]:
        token = token.rstrip(',')
        if re.match(r'^\d+:\d+$', token):
            frame = time_to_frame(token)
            results.append(frame)
        elif re.match(r'^\d+$', token):
            results.append(int(token))
            
    kf_results = [get_closest_keyframe(vid, f) for f in results]
    
    queries.append({
        'qid': qid,
        'vid': vid,
        'frames': kf_results
    })

with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for q in queries:
        csv_name = f"submission/{q['qid']}.csv"
        csv_data = ""
        for f in q['frames']:
            csv_data += f"{q['vid']},{f}\n"
        zf.writestr(csv_name, csv_data)
        
print("submission.zip created successfully.")
