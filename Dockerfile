FROM python:3.11-slim-bookworm

ARG HF_DATASET_BASE="https://huggingface.co/datasets/manhha2502/fullhd/resolve/main"
ARG INCLUDE_BM25=1
ARG PRELOAD_MODELS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HOME=/home/appuser \
    XDG_CACHE_HOME=/home/appuser/.cache \
    HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch \
    AIC_HF_CACHE_DIR=/home/appuser/.cache/huggingface \
    AIC_USE_DUMMY=0 \
    AIC_USE_CLOUD_MEDIA=1 \
    AIC_HF_DATASET_URL="https://huggingface.co/datasets/manhha2502/fullhd/resolve/main" \
    AIC_INDEX_PATH=/app/local/clip_faiss.index \
    AIC_META_PATH=/app/local/clip_metadata.json \
    AIC_TEXT_INDEX_PATH=/app/data/input/input/index/text_search_index.pkl \
    AIC_VIDEO_METADATA_PATH=/app/local/video_metadata.json \
    AIC_CLIP_DEVICE=cpu \
    AIC_TRANSLATION_DEVICE=cpu \
    AIC_ENABLE_BM25=1 \
    AIC_DISABLE_NEURAL=0 \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first. This avoids pulling CUDA/NVIDIA runtime wheels
# that Azure's CPU Consumption profile cannot use.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY . /app

# Download only the compact retrieval assets from the public HF dataset.
# Videos and keyframes are NOT copied into the container.
RUN mkdir -p /app/local /app/data/input/input/index \
    && curl -fL --retry 6 --retry-all-errors --connect-timeout 30 \
       "$HF_DATASET_BASE/local/clip_faiss.index" \
       -o /app/local/clip_faiss.index \
    && curl -fL --retry 6 --retry-all-errors --connect-timeout 30 \
       "$HF_DATASET_BASE/local/clip_metadata.json" \
       -o /app/local/clip_metadata.json \
    && curl -fL --retry 6 --retry-all-errors --connect-timeout 30 \
       "$HF_DATASET_BASE/local/video_metadata.json" \
       -o /app/local/video_metadata.json \
    && if [ "$INCLUDE_BM25" = "1" ]; then \
         curl -fL --retry 6 --retry-all-errors --connect-timeout 30 \
           "$HF_DATASET_BASE/data/input/input/index/text_search_index.pkl" \
           -o /app/data/input/input/index/text_search_index.pkl; \
       fi \
    && test "$(stat -c%s /app/local/clip_faiss.index)" -gt 300000000 \
    && test "$(stat -c%s /app/local/clip_metadata.json)" -gt 10000000 \
    && test "$(stat -c%s /app/local/video_metadata.json)" -gt 50000 \
    && echo "fa7a55b3c7636aa896ea779b363ebd7b30ad9d103cbb20429a70e86a8e819140  /app/local/clip_faiss.index" | sha256sum -c - \
    && echo "5300a31f9c25123b834801f1389290f893bf85925bfadcfcacddd4ed68dc8acf  /app/local/clip_metadata.json" | sha256sum -c - \
    && if [ "$INCLUDE_BM25" = "1" ]; then \
         echo "aa0b131efd2796e5415f4828012cb55b68aeb99ed7e843ac342fcd6a0b94c5ec  /app/data/input/input/index/text_search_index.pkl" | sha256sum -c -; \
       fi

# Run as non-root and prefetch model weights into this user's cache. The cache
# becomes part of the image, so scale-from-zero does not need to download the
# translation/CLIP models again.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser

USER appuser
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python /app/deploy/prefetch_models.py; fi

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn aic.ui.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]