"""Tests for the lazy, cached local translation adapter."""

import aic.core.local_translation as local_translation


def test_translator_is_loaded_once_and_results_are_cached(monkeypatch):
    created = []
    translated = []

    class FakeTranslator:
        def __init__(self, model_name, device):
            created.append((model_name, device))

        def translate(self, text_vi):
            translated.append(text_vi)
            return "a woman wearing a red shirt"

    local_translation._load_translator.cache_clear()
    local_translation.translate_text.cache_clear()
    monkeypatch.setattr(local_translation, "LocalTranslator", FakeTranslator)
    monkeypatch.setenv("AIC_TRANSLATION_DEVICE", "cpu")

    try:
        first = local_translation.translate_text("một người phụ nữ mặc áo đỏ")
        second = local_translation.translate_text("một người phụ nữ mặc áo đỏ")

        assert first == second == "a woman wearing a red shirt"
        assert created == [(local_translation.DEFAULT_MODEL, "cpu")]
        assert translated == ["một người phụ nữ mặc áo đỏ"]
    finally:
        local_translation.translate_text.cache_clear()
        local_translation._load_translator.cache_clear()


def test_translate_endpoint_uses_local_translation(monkeypatch):
    import aic.ui.app as ui_app

    monkeypatch.setattr(ui_app, "translate_text", lambda text: "a red car")

    result = ui_app.translate(ui_app.TranslateRequest(text_vi="một chiếc xe màu đỏ"))

    assert result == {"text_en": "a red car", "ok": True}


def test_missing_sentencepiece_gives_actionable_error(monkeypatch):
    """Thiếu sentencepiece thì transformers ném ra một trang dài tên class.

    Thông báo đó không nói được gì cho người dùng — sự cố thật đã tốn cả buổi
    vì server chạy bằng .venv thiếu gói, mà lỗi chỉ ghi 'Unrecognized
    configuration class MarianConfig'. Ở đây phải nói thẳng thiếu gói nào,
    Python nào, và cài bằng lệnh gì.
    """
    monkeypatch.setattr(
        local_translation,
        "_missing_tokenizer_dependencies",
        lambda: ["sentencepiece"],
    )

    error = local_translation._dependency_error(
        "Helsinki-NLP/opus-mt-vi-en",
        Exception("Unrecognized configuration class MarianConfig ..."),
    )

    assert error is not None
    message = str(error)
    assert "sentencepiece" in message
    assert "pip install" in message
    assert "Helsinki-NLP/opus-mt-vi-en" in message


def test_no_extra_error_when_dependencies_present(monkeypatch):
    """Đủ gói mà vẫn lỗi thì đừng che lỗi gốc bằng thông báo sai."""
    monkeypatch.setattr(
        local_translation, "_missing_tokenizer_dependencies", lambda: []
    )

    assert local_translation._dependency_error("m", Exception("boom")) is None


def test_missing_dependency_check_reports_real_absence(monkeypatch):
    monkeypatch.setattr(
        local_translation.importlib.util,
        "find_spec",
        lambda name: None if name == "sentencepiece" else object(),
    )

    assert local_translation._missing_tokenizer_dependencies() == ["sentencepiece"]
