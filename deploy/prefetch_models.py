import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import open_clip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prefetch_models")

def main():
    logger.info("Prefetching Helsinki-NLP translation model...")
    translation_model = "Helsinki-NLP/opus-mt-vi-en"
    AutoTokenizer.from_pretrained(translation_model)
    AutoModelForSeq2SeqLM.from_pretrained(translation_model)

    logger.info("Prefetching open_clip ViT-B-32-quickgelu model...")
    open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    open_clip.get_tokenizer("ViT-B-32-quickgelu")

    logger.info("Prefetching open_clip SigLIP hf-hub:timm/ViT-SO400M-14-SigLIP2 model...")
    open_clip.create_model_from_pretrained("hf-hub:timm/ViT-SO400M-14-SigLIP2")
    open_clip.get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP2")

    logger.info("All model weights prefetched successfully!")

if __name__ == "__main__":
    main()
