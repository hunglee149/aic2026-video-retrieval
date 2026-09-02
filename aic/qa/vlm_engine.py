"""Grounded Visual Question Answering (Task 2) Engine.

Supports both:
1. Cloud VLM: Google Gemini Flash API (Ultra-fast, high accuracy).
2. Local Thinking VLM: Qwen3-VL Thinking / Qwen2.5-VL (Offline, deep reasoning extraction).

Input:
  - question (str): Query / Question in Vietnamese or English.
  - video_id (str): Candidate video identifier (e.g. "L21_V003").
  - candidate_frames (list): List of (frame_id, image_data) where image_data can be PIL Image, Path, or bytes.

Output (QAResult):
  - video_id (str): e.g. "L21_V003"
  - frame_id (int): Selected evidence frame id
  - answer (str): Concise answer (<= 100 chars, BTC compliant)
  - confidence (float): 0.0 to 1.0
  - evidence (str): Grounded visual reasoning / Thinking chain
  - latency_ms (float): Execution time in milliseconds
  - is_grounded (bool): Whether the answer is verified with high confidence
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union
from PIL import Image

logger = logging.getLogger(__name__)

# Maximum answer length specified by BTC rules
MAX_ANSWER_LENGTH = 100


@dataclass
class QAResult:
    """Structured result for a Task 2 VQA query."""

    video_id: str
    frame_id: int
    answer: str
    confidence: float
    evidence: str
    latency_ms: float = 0.0
    is_grounded: bool = True
    provider: str = "gemini"
    thinking_process: Optional[str] = None
    raw_response: dict = field(default_factory=dict)

    def to_submission_row(self) -> str:
        """Format as BTC standard CSV row: <video_name>, <frame_id>, "<answer>"."""
        ans = self.answer.replace('"', '""').strip()
        return f'{self.video_id}, {self.frame_id}, "{ans}"'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_vqa_answer(answer: str) -> str:
    """Sanitize answer text according to BTC AI Challenge rules."""
    ans = answer.strip().strip('"\'`')
    ans = re.sub(r"\s+", " ", ans)
    if len(ans) > MAX_ANSWER_LENGTH:
        ans = ans[:MAX_ANSWER_LENGTH].rsplit(" ", 1)[0]
    return ans.strip()


def parse_thinking_output(raw_text: str) -> tuple[str, str]:
    """Extract <think>...</think> chain of thought and output text."""
    think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        final_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
        return thinking, final_text
    return "", raw_text.strip()


def extract_json_payload(text: str) -> dict:
    """Safely extract JSON object from markdown code blocks or text."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed[0] if parsed else {}
        elif isinstance(parsed, dict):
            return parsed
    except Exception:
        # Fallback: search for first { ... } block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return {}


class BaseVLMProvider:
    """Abstract interface for VLM providers."""

    def answer_query(
        self,
        question: str,
        video_id: str,
        candidate_frames: Sequence[tuple[int, Any]],
        min_confidence_threshold: float = 0.6,
    ) -> QAResult:
        raise NotImplementedError


class GeminiCloudProvider(BaseVLMProvider):
    """Google Gemini Flash Cloud API Provider."""

    SYSTEM_INSTRUCTION = (
        "You are an expert AI Visual Inspector for Video Retrieval Evaluation (AI Challenge).\n"
        "Your task is to accurately answer questions about video keyframes with 100% visual grounding.\n\n"
        "RULES:\n"
        "1. STRICT FACTUAL ACCURACY: Base your answer EXCLUSIVELY on visual evidence visible in the provided keyframes.\n"
        "2. ZERO HALLUCINATION: If the visual entity, text, or action is NOT clearly visible, or if the question is impossible to answer from the frames, you MUST set answer to 'UNKNOWN' and confidence to 0.0.\n"
        "3. MULTI-FRAME TEMPORAL ALIGNMENT: Identify which exact keyframe (by its frame_id) provides the clearest evidence for the answer.\n"
        "4. CONCISE ANSWER: The answer must be short, factual, and strictly under 100 characters.\n"
        "5. OUTPUT FORMAT: Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "frame_id": <int: frame number containing evidence>,\n'
        '  "answer": "<string: concise factual answer or UNKNOWN>",\n'
        '  "confidence": <float: 0.0 to 1.0>,\n'
        '  "evidence": "<string: brief explanation of visual evidence visible in that frame>"\n'
        "}"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.6-flash",
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please provide api_key or set GEMINI_API_KEY environment variable."
            )

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._backend = "genai"
            return self._client
        except (ImportError, AttributeError):
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai
            self._backend = "generativeai"
            return self._client

    def answer_query(
        self,
        question: str,
        video_id: str,
        candidate_frames: Sequence[tuple[int, Any]],
        min_confidence_threshold: float = 0.6,
    ) -> QAResult:
        start_time = time.perf_counter()
        client = self._get_client()

        # Build prompt & image parts
        contents = []
        prompt_intro = (
            f"Video ID: {video_id}\n"
            f"Question / Query: {question}\n\n"
            f"Examine the following {len(candidate_frames)} candidate keyframes from {video_id} and determine the correct answer and the exact frame_id showing the visual evidence:\n"
        )
        contents.append(prompt_intro)

        for frame_id, img_source in candidate_frames:
            pil_img = self._to_pil_image(img_source)
            contents.append(f"--- Keyframe ID: {frame_id} ---")
            contents.append(pil_img)

        contents.append(
            "\nBased on the keyframes above, output the single JSON object containing "
            "frame_id, answer (<= 100 chars), confidence (0.0-1.0), and evidence."
        )

        max_retries = 3
        delay = 2.0
        best_frame_id = candidate_frames[0][0] if candidate_frames else 1
        answer = "UNKNOWN"
        confidence = 0.0
        evidence = ""
        is_grounded = False
        raw_text = ""

        for attempt in range(max_retries):
            try:
                if self._backend == "genai":
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config={
                            "system_instruction": self.SYSTEM_INSTRUCTION,
                            "temperature": self.temperature,
                            "response_mime_type": "application/json",
                        },
                    )
                    raw_text = response.text or ""
                else:
                    model = client.GenerativeModel(
                        model_name=self.model_name,
                        system_instruction=self.SYSTEM_INSTRUCTION,
                        generation_config={"temperature": self.temperature, "response_mime_type": "application/json"},
                    )
                    response = model.generate_content(contents)
                    raw_text = response.text or ""

                parsed = extract_json_payload(raw_text)
                best_frame_id = int(parsed.get("frame_id", candidate_frames[0][0]))
                answer = sanitize_vqa_answer(str(parsed.get("answer", "UNKNOWN")))
                confidence = float(parsed.get("confidence", 0.0))
                evidence = str(parsed.get("evidence", ""))
                is_grounded = confidence >= min_confidence_threshold and answer.upper() != "UNKNOWN"
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "quota" in err_msg.lower() or "rate" in err_msg.lower():
                    logger.warning("Gemini Rate limit (429) hit, retrying in %.1fs (attempt %d/%d)...", delay, attempt + 1, max_retries)
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    logger.error("Gemini VLM inference failed for %s: %s", video_id, e)
                    evidence = f"Inference error: {e}"
                    break


        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return QAResult(
            video_id=video_id,
            frame_id=best_frame_id,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            latency_ms=latency_ms,
            is_grounded=is_grounded,
            provider="gemini",
            raw_response={"raw_text": raw_text},
        )

    def _to_pil_image(self, src: Any) -> Image.Image:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        if isinstance(src, (str, Path)):
            return Image.open(src).convert("RGB")
        if isinstance(src, bytes):
            return Image.open(io.BytesIO(src)).convert("RGB")
        raise ValueError(f"Unsupported image type: {type(src)}")


class LocalThinkingVLMProvider(BaseVLMProvider):
    """Local Qwen-VL Thinking / Reasoning Vision-Language Model Provider.
    
    Features:
    - Lazy Loading: Loaded only when requested.
    - Deep Reasoning: Extracts <think>...</think> rationale to understand complex multi-frame context.
    - Offline & Privacy: Zero external API calls.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name or os.environ.get("AIC_LOCAL_VLM_MODEL", "Qwen/Qwen3-VL-4B-Thinking")
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return self._model, self._processor

        logger.info("Loading Local Thinking VLM (%s)...", self.model_name)
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device == "cuda" else torch.float32

            processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device != "cuda":
                model.to(device)

            self._model = model
            self._processor = processor
            logger.info("Local Thinking VLM loaded successfully on %s", device)
            return self._model, self._processor
        except Exception as e:
            logger.error("Failed to load local model %s: %s", self.model_name, e)
            raise RuntimeError(f"Local Thinking VLM loading error: {e}")

    def answer_query(
        self,
        question: str,
        video_id: str,
        candidate_frames: Sequence[tuple[int, Any]],
        min_confidence_threshold: float = 0.6,
    ) -> QAResult:
        start_time = time.perf_counter()
        
        try:
            model, processor = self._load_model()
            images = [self._to_pil_image(img_src) for _, img_src in candidate_frames]
            frame_ids = [fid for fid, _ in candidate_frames]

            # Build prompt with thinking instruction
            prompt = (
                f"You are evaluating video {video_id} across candidate frames {frame_ids}.\n"
                f"Question: {question}\n\n"
                f"First think through all visual details, sequence of events, and timestamps inside <think>...</think>.\n"
                f"Then output the final JSON answer:\n"
                f'{{"frame_id": <int: frame number>, "answer": "<concise answer <= 100 chars or UNKNOWN>", "confidence": <float: 0.0 to 1.0>, "evidence": "<brief reason>"}}'
            )

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}],
                }
            ]

            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=images, padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.01)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            thinking, final_text = parse_thinking_output(raw_text)
            parsed = extract_json_payload(final_text)

            best_frame_id = int(parsed.get("frame_id", frame_ids[0]))
            answer = sanitize_vqa_answer(str(parsed.get("answer", "UNKNOWN")))
            confidence = float(parsed.get("confidence", 0.85 if answer.upper() != "UNKNOWN" else 0.0))
            evidence = str(parsed.get("evidence", thinking[:200] if thinking else ""))
            is_grounded = confidence >= min_confidence_threshold and answer.upper() != "UNKNOWN"

        except Exception as e:
            logger.error("Local Thinking VLM inference failed: %s", e)
            best_frame_id = candidate_frames[0][0] if candidate_frames else 1
            answer = "UNKNOWN"
            confidence = 0.0
            evidence = f"Local VLM Error: {e}"
            thinking = ""
            is_grounded = False
            raw_text = ""

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return QAResult(
            video_id=video_id,
            frame_id=best_frame_id,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            latency_ms=latency_ms,
            is_grounded=is_grounded,
            provider="local_thinking",
            thinking_process=thinking,
            raw_response={"raw_text": raw_text},
        )

    def _to_pil_image(self, src: Any) -> Image.Image:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        if isinstance(src, (str, Path)):
            return Image.open(src).convert("RGB")
        if isinstance(src, bytes):
            return Image.open(io.BytesIO(src)).convert("RGB")
        raise ValueError(f"Unsupported image type: {type(src)}")


class GroundedQAEngine:
    """Unified Grounded VQA Engine with support for both Cloud & Local Thinking models."""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.6-flash",
        local_model_name: Optional[str] = None,
        temperature: float = 0.0,
        min_confidence_threshold: float = 0.6,
    ):
        self.provider_type = provider
        self.min_confidence_threshold = min_confidence_threshold
        
        self.gemini_provider = GeminiCloudProvider(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
        self.local_provider = LocalThinkingVLMProvider(
            model_name=local_model_name,
        )

    def answer_query(
        self,
        question: str,
        video_id: str,
        candidate_frames: Sequence[tuple[int, Any]],
        provider: Optional[str] = None,
    ) -> QAResult:
        """Answers query using chosen provider ('gemini' or 'local')."""
        chosen_provider = provider or self.provider_type
        if chosen_provider in ("local", "local_thinking", "qwen"):
            return self.local_provider.answer_query(
                question=question,
                video_id=video_id,
                candidate_frames=candidate_frames,
                min_confidence_threshold=self.min_confidence_threshold,
            )
        return self.gemini_provider.answer_query(
            question=question,
            video_id=video_id,
            candidate_frames=candidate_frames,
            min_confidence_threshold=self.min_confidence_threshold,
        )


def answer_video_qa(
    question: str,
    video_id: str,
    candidate_frames: Sequence[tuple[int, Any]],
    api_key: Optional[str] = None,
    provider: str = "gemini",
    local_model_name: Optional[str] = None,
) -> QAResult:
    """Convenience helper for answering Task 2 Q&A query."""
    engine = GroundedQAEngine(
        provider=provider,
        api_key=api_key,
        local_model_name=local_model_name,
    )
    return engine.answer_query(question=question, video_id=video_id, candidate_frames=candidate_frames)
