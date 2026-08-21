import os
from pathlib import Path

# Tự động dùng ổ D: làm thư mục cache cho HuggingFace models để tiết kiệm ổ C:
hf_d_cache = Path("D:/huggingface_cache")
if hf_d_cache.exists():
    os.environ.setdefault("HF_HOME", str(hf_d_cache))
