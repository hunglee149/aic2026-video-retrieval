"""Tests cho module Gemini integration & fallback mechanism."""

from unittest.mock import MagicMock, patch

import pytest

from aic.core.gemini import (
    DEFAULT_GEMINI_MODEL,
    expand_query_with_gemini,
    get_gemini_api_key,
    get_gemini_model,
    is_gemini_available,
    translate_with_gemini,
)
from aic.core.query_processor import process_query
from aic.core.types import Query


class TestGeminiConfig:
    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        assert get_gemini_model() == DEFAULT_GEMINI_MODEL

    def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-preview")
        assert get_gemini_model() == "gemini-3.6-preview"

    def test_availability(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert not is_gemini_available()

        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_123")
        assert is_gemini_available()
        assert get_gemini_api_key() == "dummy_key_123"


class TestGeminiTranslation:
    def test_translate_without_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert translate_with_gemini("người đi xe máy") is None

    @patch("aic.core.gemini._call_gemini_api")
    def test_translate_success(self, mock_api, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        mock_api.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "A person riding a motorcycle"}]
                    }
                }
            ]
        }
        res = translate_with_gemini("người đi xe máy")
        assert res == "A person riding a motorcycle"

    @patch("aic.core.gemini._call_gemini_api")
    def test_translate_api_error_returns_none(self, mock_api, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        mock_api.return_value = None
        assert translate_with_gemini("người đi xe máy") is None


class TestGeminiQueryExpansion:
    @patch("aic.core.gemini._call_gemini_api")
    def test_expand_query_success(self, mock_api, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        mock_api.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"text_en": "Fire truck speeding down street", "expanded_vi": ["xe cứu hỏa", "xe chữa cháy"], "expanded_en": ["fire engine", "firetruck"], "objects": ["Car", "Vehicle"]}'
                            }
                        ]
                    }
                }
            ]
        }
        res = expand_query_with_gemini("xe cứu hỏa đang chạy trên đường phố", task="kis")
        assert res is not None
        assert res["text_en"] == "Fire truck speeding down street"
        assert "xe chữa cháy" in res["expanded_vi"]
        assert "firetruck" in res["expanded_en"]
        assert "Car" in res["objects"]

    @patch("aic.core.gemini._call_gemini_api")
    def test_expand_query_with_markdown_fence(self, mock_api, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        mock_api.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '```json\n{"text_en": "A dog running", "expanded_vi": ["chó chạy"], "expanded_en": ["running dog"], "objects": ["Dog"]}\n```'
                            }
                        ]
                    }
                }
            ]
        }
        res = expand_query_with_gemini("chó đang chạy")
        assert res is not None
        assert res["text_en"] == "A dog running"
        assert res["objects"] == ["Dog"]

    @patch("aic.core.gemini._call_gemini_api")
    def test_expand_query_invalid_json_returns_none(self, mock_api, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        mock_api.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "Not a JSON response"}]}}
            ]
        }
        res = expand_query_with_gemini("test invalid")
        assert res is None


class TestProcessQueryWithGemini:
    @patch("aic.core.gemini.expand_query_with_gemini")
    def test_process_query_auto_uses_gemini(self, mock_expand, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        mock_expand.return_value = {
            "text_en": "Person holding laptop",
            "expanded_vi": ["máy tính", "laptop"],
            "expanded_en": ["notebook", "computer"],
            "objects": ["Person", "Computer"],
        }

        q = Query(query_id="q1", text_vi="người đang cầm máy tính")
        processed = process_query(q)

        assert processed.text_en == "Person holding laptop"
        assert "laptop" in processed.expanded_vi
        assert "Computer" in processed.objects

    @patch("aic.core.gemini.expand_query_with_gemini")
    def test_process_query_falls_back_when_gemini_fails(self, mock_expand, monkeypatch):
        import aic.core.query_processor as processor

        monkeypatch.setattr(
            processor,
            "_local_translate",
            lambda text: "a person driving a car",
            raising=False,
        )
        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        mock_expand.return_value = None  # Giả lập Gemini lỗi hoặc timeout

        q = Query(query_id="q2", text_vi="người đang lái xe ô tô", text_en="")
        processed = process_query(q)

        # Fallback phải chạy xong mà không văng exception
        assert processed.text_vi == "người đang lái xe ô tô"
        assert processed.text_en == "a person driving a car"
        assert "Car" in processed.objects or "Person" in processed.objects


class TestApiTranslateWithGemini:
    @patch("aic.core.gemini.translate_with_gemini")
    def test_api_translate_uses_gemini(self, mock_trans, monkeypatch):
        from fastapi.testclient import TestClient
        from aic.ui.app import app

        monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
        mock_trans.return_value = "A dog walking in the park"

        client = TestClient(app)
        resp = client.post("/api/translate", json={"text_vi": "chó đi dạo trong công viên"})
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"ok": True, "text_en": "A dog walking in the park"}

    @patch("aic.ui.app.translate_text")
    def test_api_translate_falls_back_to_local(self, mock_local, monkeypatch):
        from fastapi.testclient import TestClient
        from aic.ui.app import app

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        mock_local.return_value = "local translation result"

        client = TestClient(app)
        resp = client.post("/api/translate", json={"text_vi": "câu tiếng việt"})
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"ok": True, "text_en": "local translation result"}

