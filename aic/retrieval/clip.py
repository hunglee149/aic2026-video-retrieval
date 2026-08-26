"""CLIP retriever — tìm keyframe bằng CLIP ViT-B/32.

Index đang dùng được dựng từ bộ ``clip-features-32`` BTC phát (``clip-ViT-B-32``,
trọng số OpenAI, 512 chiều, đã L2-normalize). Text tower phải là **đúng model
đó** thì query mới nằm cùng embedding space; đổi sang encoder khác sẽ vẫn chạy
và vẫn trả kết quả, chỉ là kết quả vô nghĩa — nên ở đây kiểm tra số chiều ngay
lúc khởi tạo thay vì bắt Exception rồi thử encoder khác.

Usage::

    retriever = build_clip_retriever("local/clip_faiss.index",
                                     "local/clip_metadata.json")
    candidates = retriever.search(query, limit=100)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .encoders import get_open_clip_encoder, probe_dim
from .faiss_retriever import FaissRetriever

logger = logging.getLogger(__name__)

# Phải là biến quickgelu để khớp cách index được dựng. Trọng số 'openai' huấn
# luyện với QuickGELU, nhưng config "ViT-B-32" của open_clip hiện đại mặc định
# GELU thường, nên cặp ("ViT-B-32", "openai") vẫn nạp được mà ra embedding lệch.
# Đo lại trên chính clip-features BTC phát (nguồn dựng index): image tower
# quickgelu khớp cosine 0.9994, bản thường chỉ 0.9548 — tức index nằm trong
# không gian của quickgelu, nên text tower cũng phải là quickgelu.
# Ghi chú trung thực: trên probe nhãn yếu ở mức video (top-10, 8 truy vấn) hai
# biến thể cho kết quả ngang nhau; chọn quickgelu vì đây là cặp đúng, không
# phải vì đã đo được cải thiện end-to-end.
DEFAULT_CLIP_MODEL = os.environ.get("AIC_CLIP_MODEL", "ViT-B-32-quickgelu")
DEFAULT_CLIP_PRETRAINED = os.environ.get("AIC_CLIP_PRETRAINED", "openai")
DEFAULT_CLIP_DEVICE = os.environ.get("AIC_CLIP_DEVICE", "auto")


def make_clip_encode_fn(
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
    device_preference: str = DEFAULT_CLIP_DEVICE,
):
    """Hàm encode text → numpy vector đã normalize."""
    encode, _info = get_open_clip_encoder(model_name, pretrained, device_preference)
    return encode


def build_clip_retriever(
    index_path: str | Path,
    metadata_path: str | Path,
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
    device_preference: str = DEFAULT_CLIP_DEVICE,
) -> FaissRetriever:
    """Tạo CLIP retriever, đã kiểm tra index khớp encoder."""
    encode, info = get_open_clip_encoder(model_name, pretrained, device_preference)
    dim = probe_dim(encode)
    logger.info(
        "CLIP encoder: model=%s pretrained=%s device=%s dim=%d",
        info["model"],
        info["pretrained"],
        info["device"],
        dim,
    )
    retriever = FaissRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        encode_fn=encode,
        name="clip",
        expected_dim=dim,
    )
    retriever.encoder_info = dict(info, dim=dim)
    return retriever
