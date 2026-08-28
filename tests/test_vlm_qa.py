"""Unit tests for Grounded Visual Question Answering (Task 2)."""

import json
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from aic.qa.vlm_engine import GroundedQAEngine, QAResult, answer_video_qa


def test_qa_result_formatting():
    """Verify BTC CSV submission format for Q&A task."""
    res = QAResult(
        video_id="L21_V003",
        frame_id=252,
        answer="2,15",
        confidence=0.98,
        evidence="Sign on bridge pier displays 2,15",
    )
    # Format: <video_name>, <frame_id>, "<answer>"
    row = res.to_submission_row()
    assert row == 'L21_V003, 252, "2,15"'


def test_qa_result_escaping_internal_quotes():
    """Verify double quote escaping inside answer strings."""
    res = QAResult(
        video_id="L22_V008",
        frame_id=59,
        answer='Đèo "Tà Pứa"',
        confidence=0.95,
        evidence="Landslide sign",
    )
    row = res.to_submission_row()
    assert row == 'L22_V008, 59, "Đèo ""Tà Pứa"""'


def test_answer_sanitization():
    """Verify character limit (<= 100 chars) and space normalization."""
    engine = GroundedQAEngine()
    long_answer = "a" * 150
    sanitized = engine._sanitize_answer(long_answer)
    assert len(sanitized) == 100

    quoted = '"  38.35  "'
    assert engine._sanitize_answer(quoted) == "38.35"


def test_empty_candidate_frames():
    """Verify safe fallback when no candidate frames are provided."""
    engine = GroundedQAEngine()
    res = engine.answer_query("Con cá nặng bao nhiêu?", "L21_V007", [])
    assert res.video_id == "L21_V007"
    assert res.answer == "UNKNOWN"
    assert res.confidence == 0.0
    assert not res.is_grounded
    assert res.latency_ms >= 0


def test_anti_hallucination_unknown():
    """Verify that UNKNOWN response sets is_grounded to False."""
    engine = GroundedQAEngine(min_confidence_threshold=0.6)
    
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "frame_id": 100,
        "answer": "UNKNOWN",
        "confidence": 0.2,
        "evidence": "Sign is too blurry to read numbers.",
    })
    
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(engine, "_get_client", return_value=mock_client):
        img = Image.new("RGB", (64, 64), color="red")
        res = engine.answer_query("Con số là bao nhiêu?", "L21_V003", [(100, img)])

        assert res.answer == "UNKNOWN"
        assert res.confidence == 0.2
        assert not res.is_grounded
        assert res.latency_ms > 0
        assert "blurry" in res.evidence


def test_successful_grounded_qa():
    """Verify complete grounded QA pipeline with mock response."""
    engine = GroundedQAEngine(min_confidence_threshold=0.6)
    
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "frame_id": 252,
        "answer": "2,15",
        "confidence": 0.96,
        "evidence": "Clear circular height limit sign showing 2,15 on the left pier.",
    })
    
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(engine, "_get_client", return_value=mock_client):
        img1 = Image.new("RGB", (64, 64), color="blue")
        img2 = Image.new("RGB", (64, 64), color="yellow")
        res = engine.answer_query(
            question="Con số trên biển báo bên trái cầu là bao nhiêu?",
            video_id="L21_V003",
            candidate_frames=[(200, img1), (252, img2)],
        )

        assert res.video_id == "L21_V003"
        assert res.frame_id == 252
        assert res.answer == "2,15"
        assert res.confidence == 0.96
        assert res.is_grounded
        assert res.latency_ms > 0
        assert res.to_submission_row() == 'L21_V003, 252, "2,15"'
