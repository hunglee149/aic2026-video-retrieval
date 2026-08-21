"""CLIP retriever — tìm keyframe bằng CLIP ViT-B/32.

Sử dụng FAISS index + metadata từ local/.
Text encoder dùng open-clip để encode query tiếng Anh → vector 512-dim.

Usage:
    retriever = build_clip_retriever("local/clip_faiss.index", "local/clip_metadata.json")
    candidates = retriever.search(query, k=100)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core.types import Query
from .faiss_retriever import FaissRetriever

logger = logging.getLogger(__name__)

# Defaults — có thể override
DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"

# Cache để không load model nhiều lần
_clip_model_cache: dict = {}


def _get_clip_encoder(
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
):
    """Load CLIP text encoder via open_clip hoặc transformers."""
    cache_key = (model_name, pretrained)
    if cache_key not in _clip_model_cache:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            import open_clip

            logger.info("Loading CLIP model via open_clip: %s / %s", model_name, pretrained)
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval().to(device)
            encode_func = lambda t: model.encode_text(tokenizer([t]).to(device))
        except Exception:
            from transformers import CLIPModel, CLIPTokenizer

            hf_name = "openai/clip-vit-base-patch32"
            logger.info("Loading CLIP model via transformers: %s", hf_name)
            tokenizer = CLIPTokenizer.from_pretrained(hf_name)
            model = CLIPModel.from_pretrained(hf_name)
            model.eval().to(device)

            def encode_func(t):
                inputs = tokenizer(
                    [t], return_tensors="pt", padding=True, truncation=True
                ).to(device)
                return model.get_text_features(**inputs)

        logger.info("  → CLIP ready on %s", device)
        _clip_model_cache[cache_key] = (encode_func, device)

    return _clip_model_cache[cache_key]


def make_clip_encode_fn(
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
):
    """Tạo hàm encode text → numpy vector cho CLIP."""
    import torch

    def encode(text: str) -> np.ndarray:
        encode_func, device = _get_clip_encoder(model_name, pretrained)
        with torch.no_grad():
            features = encode_func(text)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    return encode


def build_clip_retriever(
    index_path: str | Path,
    metadata_path: str | Path,
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> FaissRetriever:
    """Tạo CLIP retriever sẵn sàng search.

    Parameters
    ----------
    index_path : path tới ``clip_faiss.index``
    metadata_path : path tới ``clip_metadata.json``
    """
    encode_fn = make_clip_encode_fn(model_name, pretrained)
    return FaissRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        encode_fn=encode_fn,
        name="clip",
    )
