"""Download model weights during the Docker build so cold starts stay offline-fast."""
from __future__ import annotations

import gc
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def prefetch_clip() -> None:
    import open_clip

    model_name = os.environ.get("AIC_CLIP_MODEL", "ViT-B-32-quickgelu")
    pretrained = os.environ.get("AIC_CLIP_PRETRAINED", "openai")
    print(f"[prefetch] CLIP {model_name} ({pretrained})")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    del model
    gc.collect()


def prefetch_translation() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = os.environ.get(
        "AIC_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-vi-en"
    )
    print(f"[prefetch] translation {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    del tokenizer, model
    gc.collect()


if __name__ == "__main__":
    prefetch_clip()
    prefetch_translation()
    print("[prefetch] done")
