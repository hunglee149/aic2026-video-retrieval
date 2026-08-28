"""VQA & Evidence Grounding Engine for Video Retrieval (Task 2)."""

from .vlm_engine import GroundedQAEngine, QAResult, answer_video_qa

__all__ = ["GroundedQAEngine", "QAResult", "answer_video_qa"]
