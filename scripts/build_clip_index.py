import os
import glob
import json
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from tqdm import tqdm

def build_index():
    base_dir = Path("data/batch_01")
    features_dir = base_dir / "clip-features-32-aic25-b1" / "clip-features-32"
    map_dir = base_dir / "map-keyframes-aic25-b1" / "map-keyframes"
    
    out_dir = Path("local")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = out_dir / "clip_faiss.index"
    meta_path = out_dir / "clip_metadata.json"
    
    dim = 512
    index = faiss.IndexFlatIP(dim)
    metadata = []
    
    csv_files = sorted(glob.glob(str(map_dir / "*.csv")))
    print(f"Found {len(csv_files)} CSV mapping files.")
    
    for csv_file in tqdm(csv_files, desc="Processing videos"):
        vid = Path(csv_file).stem
        npy_file = features_dir / f"{vid}.npy"
        
        if not npy_file.exists():
            print(f"Warning: Missing {npy_file}")
            continue
            
        # Read mappings
        df = pd.read_csv(csv_file)
        
        # Read features
        feats = np.load(npy_file)
        
        if len(df) != len(feats):
            print(f"Warning: {vid} has {len(df)} mapped frames but {len(feats)} features")
            # Truncate to the minimum
            min_len = min(len(df), len(feats))
            df = df.iloc[:min_len]
            feats = feats[:min_len]
            
        # Normalize features for inner product search
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        feats_normalized = feats / norms
        feats_normalized = feats_normalized.astype(np.float32)
        
        # Add to index
        index.add(feats_normalized)
        
        # Add to metadata
        for _, row in df.iterrows():
            metadata.append({
                "video_id": vid,
                "keyframe_num": int(row["n"]),
                "frame_idx": int(row["frame_idx"]),
                "pts_time": float(row["pts_time"])
            })
            
    print(f"Total vectors added: {index.ntotal}")
    print(f"Total metadata entries: {len(metadata)}")
    
    print(f"Writing index to {index_path}...")
    faiss.write_index(index, str(index_path))
    
    print(f"Writing metadata to {meta_path}...")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, separators=(',', ':')) # Compact json
        
    print("Done!")

if __name__ == "__main__":
    build_index()
