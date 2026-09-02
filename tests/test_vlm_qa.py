"""Unit tests for Grounded Visual Question Answering (Task 2)."""

import json
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from aic.qa.vlm_engine import (
    GroundedQAEngine,
    LocalThinkingVLMProvider,
    QAResult,
    answer_video_qa,
    parse_thinking_output,
    sanitize_vqa_answer,
)


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
    long_answer = "a " * 150
    sanitized = sanitize_vqa_answer(long_answer)
    assert len(sanitized) <= 100

    quoted = '"  38.35  "'
    assert sanitize_vqa_answer(quoted) == "38.35"


def test_parse_thinking_output():
    """Verify extraction of <think>...</think> chain of thought."""
    raw = "<think>\nIn keyframe 76, the scale displays 38.35 in green.\n</think>\n```json\n{\"frame_id\": 76, \"answer\": \"38.35\", \"confidence\": 0.98}\n```"
    thinking, final_text = parse_thinking_output(raw)
    assert "displays 38.35" in thinking
    assert "<think>" not in final_text
    assert '"answer": "38.35"' in final_text


def test_anti_hallucination_unknown():
    """Verify that UNKNOWN response sets is_grounded to False."""
    engine = GroundedQAEngine(provider="gemini", min_confidence_threshold=0.6)
    
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "frame_id": 100,
        "answer": "UNKNOWN",
        "confidence": 0.2,
        "evidence": "Sign is too blurry to read numbers.",
    })
    
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(engine.gemini_provider, "_get_client", return_value=mock_client):
        engine.gemini_provider._backend = "genai"
        img = Image.new("RGB", (64, 64), color="red")
        res = engine.answer_query("Con số là bao nhiêu?", "L21_V003", [(100, img)])

        assert res.answer == "UNKNOWN"
        assert res.confidence == 0.2
        assert not res.is_grounded
        assert res.latency_ms > 0
        assert "blurry" in res.evidence


def test_local_thinking_vlm_mock():
    """Verify Local Thinking VLM provider execution and rationale extraction."""
    local_provider = LocalThinkingVLMProvider(model_name="Qwen/Qwen3-VL-4B-Thinking")

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = MagicMock()

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "prompt"
    mock_processor.return_value = MagicMock(to=lambda d: MagicMock(input_ids=[[1, 2, 3]]))
    
    # Simulate thinking model output
    mock_processor.batch_decode.return_value = [
        "<think>Frame 97 shows exercise touch toes</think>\n"
        '{"frame_id": 97, "answer": "Gập người chạm chân", "confidence": 0.95, "evidence": "Frame 97"}'
    ]

    with patch.object(local_provider, "_load_model", return_value=(mock_model, mock_processor)):
        img = Image.new("RGB", (64, 64), color="blue")
        res = local_provider.answer_query("Tập thể dục gì?", "L30_V046", [(97, img)])

        assert res.answer == "Gập người chạm chân"
        assert res.frame_id == 97
        assert res.confidence == 0.95
        assert res.is_grounded
        assert res.provider == "local_thinking"
        assert "touch toes" in res.thinking_process


def test_api_qa_answer_endpoint():
    """Test FastAPI /api/qa/answer endpoint."""
    from fastapi.testclient import TestClient
    from aic.ui.app import app

    client = TestClient(app)
    
    with patch("aic.qa.vlm_engine.GroundedQAEngine.answer_query") as mock_answer:
        mock_answer.return_value = QAResult(
            video_id="L21_V003",
            frame_id=252,
            answer="2,15",
            confidence=0.98,
            evidence="Sign on bridge",
            is_grounded=True,
        )

        response = client.post(
            "/api/qa/answer",
            json={
                "question": "Con số trên cầu?",
                "video_id": "L21_V003",
                "frame_ids": [252],
                "provider": "gemini",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["result"]["answer"] == "2,15"
        assert data["submission_row"] == 'L21_V003, 252, "2,15"'
