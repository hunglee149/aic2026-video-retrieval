"""Tiện ích dùng chung cho các text encoder chạy bằng open_clip.

Một chỗ duy nhất lo bốn việc dễ làm sai:

- chọn device (``auto``/``cuda``/``cpu``), có fallback khi CUDA hết VRAM;
- nạp model đúng **một lần**, an toàn khi nhiều request gọi song song;
- bỏ vision tower vì retrieval chỉ encode text — SO400M nặng vài GB, giữ lại
  tower không dùng là cách nhanh nhất để hết VRAM;
- đo số chiều embedding thật để đối chiếu với index trước khi search.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

_cache: dict[tuple, tuple] = {}
_cache_lock = threading.Lock()


def resolve_device(preference: str = "auto") -> str:
    """``auto`` → cuda nếu có, ngược lại cpu."""
    import torch

    choice = (preference or "auto").strip().lower()
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Yêu cầu device=cuda nhưng PyTorch không thấy GPU")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def apply_hf_cache_dir() -> str | None:
    """Trỏ cache HuggingFace theo ``AIC_HF_CACHE_DIR`` nếu người dùng đặt.

    Không có giá trị thì để nguyên mặc định của HuggingFace. Không hardcode
    đường dẫn tuyệt đối của máy nào cả.
    """
    raw = os.environ.get("AIC_HF_CACHE_DIR", "").strip()
    if not raw:
        return None
    os.environ.setdefault("HF_HOME", raw)
    return raw


def neural_disabled() -> bool:
    return os.environ.get("AIC_DISABLE_NEURAL", "0").strip() == "1"


def _load_open_clip(model_name: str, pretrained: str | None, device: str):
    import open_clip
    import torch

    apply_hf_cache_dir()
    logger.info("Loading open_clip model %s (pretrained=%s)", model_name, pretrained)
    if pretrained:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
    else:
        model, _ = open_clip.create_model_from_pretrained(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    # Chỉ cần text tower.
    if hasattr(model, "visual"):
        try:
            del model.visual
        except Exception as exc:  # pragma: no cover - phụ thuộc kiến trúc model
            logger.debug("Không bỏ được vision tower của %s: %s", model_name, exc)

    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA hết VRAM cho %s; chuyển sang CPU", model_name)
        torch.cuda.empty_cache()
        device = "cpu"
        model = model.to(device)

    logger.info("  → %s sẵn sàng trên %s", model_name, device)
    return model, tokenizer, device


def get_open_clip_encoder(
    model_name: str,
    pretrained: str | None = None,
    device_preference: str = "auto",
):
    """Trả về ``(encode_fn, info)``; model được cache theo (name, pretrained)."""
    if neural_disabled():
        raise RuntimeError("AIC_DISABLE_NEURAL=1 — model neural đã bị tắt")

    key = (model_name, pretrained)
    with _cache_lock:
        if key not in _cache:
            device = resolve_device(device_preference)
            _cache[key] = _load_open_clip(model_name, pretrained, device)
        model, tokenizer, device = _cache[key]

    import torch

    def encode(text: str) -> np.ndarray:
        tokens = tokenizer([text]).to(device)
        with torch.inference_mode():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().astype(np.float32).flatten()

    info = {"model": model_name, "pretrained": pretrained, "device": device}
    return encode, info


def probe_dim(encode_fn) -> int:
    """Số chiều embedding thật của encoder, đo bằng một lần encode."""
    vector = np.asarray(encode_fn("a photo"), dtype=np.float32).reshape(-1)
    return int(vector.shape[0])


def reset_encoder_cache() -> None:
    """Xoá cache model — dùng trong test, không dùng lúc chạy thật."""
    with _cache_lock:
        _cache.clear()
