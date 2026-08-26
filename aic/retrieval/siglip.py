"""SigLIP2 retriever — kênh visual thứ hai bên cạnh CLIP.

**Model phải khớp cách index được dựng.** Index ``siglip_faiss.index`` trong repo
này (177.321 vector, 1152 chiều) được sinh bởi notebook
``drive/notebooks/trake_indexing.ipynb`` bằng::

    open_clip.create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP2')

Nên query cũng phải đi qua đúng model đó qua open_clip. Dùng
``transformers.SiglipTextModel`` với ``google/siglip2-*`` là một implementation
khác: nó vẫn cho ra vector 1152 chiều và vẫn search được, nhưng không cùng
embedding space nên kết quả là rác im lặng. Số chiều trùng nhau không đủ để kết
luận là cùng model — vì thế build sẽ đối chiếu dimension và fail rõ ràng.

Cấu hình::

    AIC_SIGLIP_INDEX_PATH   đường dẫn siglip_faiss.index
    AIC_SIGLIP_META_PATH    đường dẫn siglip_metadata.json
    AIC_SIGLIP_MODEL        mặc định hf-hub:timm/ViT-SO400M-14-SigLIP2
    AIC_SIGLIP_DEVICE       auto | cuda | cpu
    AIC_HF_CACHE_DIR        (tuỳ chọn) thư mục cache HuggingFace
    AIC_DISABLE_NEURAL      (tuỳ chọn) =1 để tắt mọi model neural
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .encoders import get_open_clip_encoder, probe_dim
from .faiss_retriever import FaissRetriever

logger = logging.getLogger(__name__)

DEFAULT_SIGLIP_MODEL = os.environ.get(
    "AIC_SIGLIP_MODEL", "hf-hub:timm/ViT-SO400M-14-SigLIP2"
)
DEFAULT_SIGLIP_DEVICE = os.environ.get("AIC_SIGLIP_DEVICE", "auto")


def make_siglip_encode_fn(
    model_name: str = DEFAULT_SIGLIP_MODEL,
    device_preference: str = DEFAULT_SIGLIP_DEVICE,
):
    """Hàm encode text → numpy vector 1152 chiều đã normalize."""
    encode, _info = get_open_clip_encoder(model_name, None, device_preference)
    return encode


def build_siglip_retriever(
    index_path: str | Path,
    metadata_path: str | Path,
    model_name: str = DEFAULT_SIGLIP_MODEL,
    device_preference: str = DEFAULT_SIGLIP_DEVICE,
) -> FaissRetriever:
    """Tạo SigLIP retriever, đã kiểm tra index khớp encoder."""
    encode, info = get_open_clip_encoder(model_name, None, device_preference)
    dim = probe_dim(encode)
    logger.info(
        "SigLIP encoder: model=%s device=%s dim=%d",
        info["model"],
        info["device"],
        dim,
    )
    retriever = FaissRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        encode_fn=encode,
        name="siglip",
        expected_dim=dim,
    )
    retriever.encoder_info = dict(info, dim=dim)
    return retriever
