"""Local Vietnamese-to-English translation backed by Hugging Face."""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-vi-en"
_MODEL_LOAD_LOCK = threading.Lock()


class LocalTranslator:
    """Load OPUS-MT once and provide thread-safe translation inference."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("AIC_TRANSLATION_DEVICE=cuda but CUDA is unavailable")
        if device not in {"cpu", "cuda"}:
            raise ValueError("AIC_TRANSLATION_DEVICE must be auto, cpu, or cuda")

        logger.info("Loading translation model %s on %s", model_name, device)
        self.model_name = model_name
        self.device = torch.device(device)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()
        self._inference_lock = threading.Lock()
        logger.info("Translation model ready on %s", self.device)

    def translate(self, text_vi: str) -> str:
        text_vi = text_vi.strip()
        if not text_vi:
            return ""

        encoded = self._tokenizer(
            text_vi,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with self._inference_lock, self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_length=256,
                num_beams=4,
                early_stopping=True,
            )
        translated = self._tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        if not translated:
            raise RuntimeError("Local translation model returned an empty result")
        return translated


@lru_cache(maxsize=1)
def _load_translator(model_name: str, device: str) -> LocalTranslator:
    return LocalTranslator(model_name=model_name, device=device)


def get_local_translator() -> LocalTranslator:
    """Return one process-wide translator, even under concurrent first requests."""
    model_name = os.environ.get("AIC_TRANSLATION_MODEL", DEFAULT_MODEL)
    device = os.environ.get("AIC_TRANSLATION_DEVICE", "auto").lower()
    with _MODEL_LOAD_LOCK:
        return _load_translator(model_name, device)


@lru_cache(maxsize=256)
def translate_text(text_vi: str) -> str:
    """Translate Vietnamese text to English using the cached local model."""
    return get_local_translator().translate(text_vi)
