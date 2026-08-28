"""Grounded Visual Question Answering (Task 2) Engine powered by Gemini 2.0 Flash.

Input:
  - question (str): Query / Question in Vietnamese or English.
  - video_id (str): Candidate video identifier (e.g. "L21_V003").
  - candidate_frames (list): List of (frame_id, image_data) where image_data can be PIL Image, Path, or bytes.

Output (QAResult):
  - video_id (str): e.g. "L21_V003"
  - frame_id (int): Selected evidence frame id
  - answer (str): Concise answer (<= 100 chars, BTC compliant)
  - confidence (float): 0.0 to 1.0
  - evidence (str): Grounded visual reason
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
    raw_response: dict = field(default_factory=dict)

    def to_submission_row(self) -> str:
        """Format as BTC standard CSV row: <video_name>, <frame_id>, "<answer>"."""
        ans = self.answer.replace('"', '""').strip()
        return f'{self.video_id}, {self.frame_id}, "{ans}"'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundedQAEngine:
    """Multi-modal Grounded Visual Question Answering Engine.
    
    Features:
    - Zero-Hallucination Guardrails: Forces model to verify visual evidence or return UNKNOWN.
    - Multi-frame Temporal Grounding: Selects the exact frame proving the answer.
    - Automated latency measurement & detailed logging.
    - BTC formatting rules compliance (escaped CSV, <= 100 chars).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.6-flash",
        temperature: float = 0.0,
        min_confidence_threshold: float = 0.6,
    ):


        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.min_confidence_threshold = min_confidence_threshold
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
        candidate_frames: Sequence[tuple[int, Union[str, Path, bytes, Any]]],
    ) -> QAResult:
        """Answers a VQA question given candidate frames from a video.
        
        Args:
            question: Search query / Question text in Vietnamese or English.
            video_id: Candidate video ID.
            candidate_frames: List of tuples (frame_id, image_source)
            
        Returns:
            QAResult with video_id, frame_id, answer, confidence, evidence, latency_ms.
        """
        start_time = time.perf_counter()

        if not candidate_frames:
            latency = (time.perf_counter() - start_time) * 1000.0
            return QAResult(
                video_id=video_id,
                frame_id=0,
                answer="UNKNOWN",
                confidence=0.0,
                evidence="No candidate frames provided for analysis.",
                latency_ms=round(latency, 2),
                is_grounded=False,
            )

        try:
            client = self._get_client()
        except Exception as e:
            logger.warning("VLM Client initialization failed: %s", e)
            latency = (time.perf_counter() - start_time) * 1000.0
            return QAResult(
                video_id=video_id,
                frame_id=candidate_frames[0][0],
                answer="ERROR_INITIALIZING_CLIENT",
                confidence=0.0,
                evidence=f"Client initialization error: {e}",
                latency_ms=round(latency, 2),
                is_grounded=False,
            )

        # Prepare images with PIL or bytes
        contents = []
        system_instruction = (
            "You are a precise Video Question Answering Judge for the AI Challenge Video Retrieval Competition.\n"
            "Your task is to analyze the provided candidate video keyframes and answer the question with 100% VISUAL GROUNDING.\n\n"
            "CRITICAL RULES TO PREVENT HALLUCINATION:\n"
            "1. GROUNDING TRUTH: ONLY answer based on what is clearly and unmistakably visible in the provided frames. DO NOT assume, guess, or extrapolate.\n"
            "2. FRAME SELECTION: Identify the exact frame_id (from the labeled frames) that provides the clearest visual proof for your answer.\n"
            "3. CONCISE ANSWER: The answer must be extremely concise (e.g. a specific number, text on a sign, name of an object/person, color). Maximum 100 characters.\n"
            "4. NUMERICAL FORMATTING: For decimal numbers, keep standard format (e.g., '2,15' or '38.35') exactly as displayed on signs/scales.\n"
            "5. UNKNOWN RULE: If the visual detail is too blurry, occluded, ambiguous, or not present in any of the frames, you MUST set 'answer' to 'UNKNOWN' and 'confidence' to a value below 0.3.\n\n"
            "OUTPUT FORMAT (STRICT JSON ONLY):\n"
            "{\n"
            '  "frame_id": <int: frame number of best evidence>,\n'
            '  "answer": "<string: short answer <= 100 chars>",\n'
            '  "confidence": <float: 0.0 to 1.0>,\n'
            '  "evidence": "<string: concise explanation of visual proof and location in the frame>"\n'
            "}"
        )

        contents.append(f"Question: {question}\nVideo ID: {video_id}\nPlease examine the following {len(candidate_frames)} candidate keyframes:")

        for frame_id, img_data in candidate_frames:
            try:
                pil_img = self._load_image(img_data)
                contents.append(f"\n--- Keyframe ID: {frame_id} ---")
                contents.append(pil_img)
            except Exception as load_err:
                logger.warning("Failed to load frame %s (%s): %s", frame_id, video_id, load_err)

        try:
            backend = getattr(self, "_backend", "genai")
            if backend == "genai" and hasattr(client, "models"):
                try:
                    from google.genai import types
                    config = types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.temperature,
                        response_mime_type="application/json",
                    )
                except (ImportError, AttributeError):
                    config = {
                        "system_instruction": system_instruction,
                        "temperature": self.temperature,
                        "response_mime_type": "application/json",
                    }
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                raw_text = response.text.strip() if response.text else "{}"
            else:
                import google.generativeai as genai
                model = client.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction,
                    generation_config={"temperature": self.temperature, "response_mime_type": "application/json"},
                )
                response = model.generate_content(contents)
                raw_text = response.text.strip() if response.text else "{}"



            latency = (time.perf_counter() - start_time) * 1000.0
            
            # Clean possible markdown wrapping
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```[a-zA-Z]*\n", "", raw_text)
                raw_text = re.sub(r"\n```$", "", raw_text)

            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
            elif not isinstance(parsed, dict):
                parsed = {}

            best_frame_id = int(parsed.get("frame_id", candidate_frames[0][0]))
            answer = str(parsed.get("answer", "UNKNOWN")).strip()
            confidence = float(parsed.get("confidence", 0.0))
            evidence = str(parsed.get("evidence", ""))


            # Format & sanitize answer
            answer = self._sanitize_answer(answer)
            is_grounded = confidence >= self.min_confidence_threshold and answer.upper() != "UNKNOWN"

            logger.info(
                "Task2 QA [%s] -> Frame: %d | Ans: '%s' | Conf: %.2f | Latency: %.1fms",
                video_id, best_frame_id, answer, confidence, latency
            )

            return QAResult(
                video_id=video_id,
                frame_id=best_frame_id,
                answer=answer,
                confidence=confidence,
                evidence=evidence,
                latency_ms=round(latency, 2),
                is_grounded=is_grounded,
                raw_response=parsed,
            )

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            logger.error("VLM inference failed for %s: %s", video_id, e)
            return QAResult(
                video_id=video_id,
                frame_id=candidate_frames[0][0],
                answer="UNKNOWN",
                confidence=0.0,
                evidence=f"Inference error: {e}",
                latency_ms=round(latency, 2),
                is_grounded=False,
            )

    def _load_image(self, img_data: Any):
        """Convert path, bytes, or PIL image into a valid PIL Image."""
        from PIL import Image

        if isinstance(img_data, Image.Image):
            return img_data
        if isinstance(img_data, (str, Path)):
            return Image.open(img_data)
        if isinstance(img_data, (bytes, bytearray)):
            return Image.open(io.BytesIO(img_data))
        raise ValueError(f"Unsupported image type: {type(img_data)}")

    def _sanitize_answer(self, text: str) -> str:
        """Sanitizes and bounds answer string according to BTC regulations."""
        # Trim leading/trailing whitespace & quotes
        text = text.strip().strip('"').strip("'").strip()
        
        # Collapse multiple internal spaces
        text = re.sub(r"\s+", " ", text)
        
        # Enforce maximum character length limit (100 characters)
        if len(text) > MAX_ANSWER_LENGTH:
            text = text[:MAX_ANSWER_LENGTH].rstrip()

        return text



def answer_video_qa(
    question: str,
    video_id: str,
    candidate_frames: Sequence[tuple[int, Union[str, Path, bytes, Any]]],
    api_key: Optional[str] = None,
) -> QAResult:
    """Helper function to run Grounded QA directly."""
    engine = GroundedQAEngine(api_key=api_key)
    return engine.answer_query(question=question, video_id=video_id, candidate_frames=candidate_frames)
