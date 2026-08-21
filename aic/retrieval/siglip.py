"""SigLIP retriever — tìm keyframe bằng SigLIP2 embeddings.

Tương tự clip.py nhưng dùng SigLIP2 text encoder từ HuggingFace transformers.
SigLIP2 cho embedding ~1152-dim, mạnh hơn CLIP ở semantic matching.

Usage:
    retriever = build_siglip_retriever("local/siglip_faiss.index", "local/siglip_metadata.json")
    candidates = retriever.search(query, k=100)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .faiss_retriever import FaissRetriever

logger = logging.getLogger(__name__)

# SigLIP2 model name on HuggingFace (1152-dim)
DEFAULT_SIGLIP_MODEL = "google/siglip2-so400m-patch14-224"

_siglip_cache: dict = {}


def _get_siglip_encoder(model_name: str = DEFAULT_SIGLIP_MODEL):
    """Load SigLIP text encoder (chỉ nạp text tower để tiết kiệm RAM & siêu tốc)."""
    if model_name not in _siglip_cache:
        import torch
        from transformers import AutoTokenizer, SiglipTextModel

        logger.info("Loading SigLIP text model: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = SiglipTextModel.from_pretrained(model_name)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logger.info("  → SigLIP text encoder ready on %s", device)

        _siglip_cache[model_name] = (model, tokenizer, device)

    return _siglip_cache[model_name]


def make_siglip_encode_fn(model_name: str = DEFAULT_SIGLIP_MODEL):
    """Tạo hàm encode text → numpy vector cho SigLIP2 (1152 chiều)."""
    import torch

    def encode(text: str) -> np.ndarray:
        model, tokenizer, device = _get_siglip_encoder(model_name)
        inputs = tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            feat = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().flatten()

    return encode


def build_siglip_retriever(
    index_path: str | Path,
    metadata_path: str | Path,
    model_name: str = DEFAULT_SIGLIP_MODEL,
) -> FaissRetriever:
    """Tạo SigLIP retriever sẵn sàng search.

    Parameters
    ----------
    index_path : path tới ``siglip_faiss.index``
    metadata_path : path tới ``siglip_metadata.json``
    """
    encode_fn = make_siglip_encode_fn(model_name)
    return FaissRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        encode_fn=encode_fn,
        name="siglip",
    )
