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
