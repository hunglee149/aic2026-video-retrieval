"""Tích hợp Google Gemini (LLM & Multimodal) cho AIC 2026.

Gọi trực tiếp qua REST API chính thức của Google AI Studio bằng `httpx`.
Không cần cài đặt thêm thư viện nặng, không tốn RAM/CPU trên Codespace.

Cấu hình qua biến môi trường:
    GEMINI_API_KEY: API key từ https://aistudio.google.com
    GEMINI_MODEL: Mặc định 'gemini-2.5-flash' (hoặc 'gemini-2.0-flash')
    GEMINI_TIMEOUT: Thời gian timeout (giây), mặc định 8.0s
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_TIMEOUT = 8.0
GEMINI_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _read_key_from_env_file(key_name: str) -> str:
    from pathlib import Path
    env_file = Path(".env")
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == key_name:
                        return v.strip().strip("'\"")
        except Exception:
            pass
    return ""


def get_gemini_api_key() -> str:
    """Lấy API key từ biến môi trường GEMINI_API_KEY hoặc GOOGLE_API_KEY, fallback đọc file .env."""
    key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key and not os.environ.get("PYTEST_CURRENT_TEST"):
        key = _read_key_from_env_file("GEMINI_API_KEY") or _read_key_from_env_file("GOOGLE_API_KEY")
        if key:
            os.environ["GEMINI_API_KEY"] = key
    return key


def get_gemini_model() -> str:
    """Lấy tên model phiên bản mới nhất từ biến môi trường, mặc định gemini-3.6-flash."""
    model = os.environ.get("GEMINI_MODEL", "").strip()
    if not model and not os.environ.get("PYTEST_CURRENT_TEST"):
        model = _read_key_from_env_file("GEMINI_MODEL")
        if model:
            os.environ["GEMINI_MODEL"] = model
    return model or DEFAULT_GEMINI_MODEL


def is_gemini_available() -> bool:
    """Kiểm tra xem Gemini đã được cấu hình API Key chưa."""
    return bool(get_gemini_api_key())


def _call_gemini_api(
    payload: dict[str, Any],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Optional[dict]:
    """Hàm lõi gửi request POST tới Google Generative Language API."""
    # Tránh gọi ra ngoài mạng trong quá trình chạy test tự động
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("AIC_LIVE_GEMINI_TEST"):
        return None

    key = api_key or get_gemini_api_key()
    if not key:
        return None

    mod = model or get_gemini_model()
    url = GEMINI_API_ENDPOINT.format(model=mod)
    params = {"key": key}
    t = timeout or float(os.environ.get("GEMINI_TIMEOUT", str(DEFAULT_TIMEOUT)))

    try:
        with httpx.Client(timeout=t) as client:
            resp = client.post(url, params=params, json=payload)
            if resp.status_code != 200:
                logger.warning(
                    "Gemini API returned status %d for model %s: %s",
                    resp.status_code,
                    mod,
                    resp.text[:200],
                )
                return None
            return resp.json()
    except Exception as exc:
        logger.warning("Lỗi kết nối Gemini API (%s): %s", mod, exc)
        return None


def translate_with_gemini(text_vi: str) -> Optional[str]:
    """Dịch câu hỏi tiếng Việt sang tiếng Anh tự nhiên, chuẩn xác cho video retrieval."""
    if not text_vi or not text_vi.strip() or not is_gemini_available():
        return None

    prompt = (
        "Translate the following Vietnamese video query into a natural, descriptive English sentence "
        "optimized for text-to-video search (CLIP/SigLIP). "
        "Preserve key subjects, actions, colors, and contextual details accurately. "
        "Return ONLY the translated English sentence with no extra explanation or quotes.\n\n"
        f"Vietnamese: {text_vi.strip()}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 200,
        },
    }

    data = _call_gemini_api(payload)
    if not data:
        return None

    try:
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts and "text" in parts[0]:
                translated = parts[0]["text"].strip()
                return translated.strip('"\n ')
    except Exception as exc:
        logger.warning("Lỗi parse kết quả dịch từ Gemini: %s", exc)

    return None


def expand_query_with_gemini(text_vi: str, task: str = "kis") -> Optional[dict]:
    """Mở rộng câu truy vấn bằng Gemini dưới dạng JSON có cấu trúc.

    Trả về:
    {
        "text_en": "bản dịch tiếng Anh chuẩn",
        "expanded_vi": ["từ đồng nghĩa 1", "từ đồng nghĩa 2", ...],
        "expanded_en": ["synonym 1", "synonym 2", ...],
        "objects": ["Person", "Car", ...]
    }
    """
    if not text_vi or not text_vi.strip() or not is_gemini_available():
        return None

    system_instruction = (
        "You are an expert AI for the AI Challenge (AIC) Video Retrieval system. "
        "Analyze the given Vietnamese video search query and extract search parameters in JSON format.\n"
        "Return a JSON object with EXACTLY these keys:\n"
        '- "text_en": Accurate, descriptive English translation of the query optimized for CLIP/SigLIP.\n'
        '- "expanded_vi": List of 3-6 Vietnamese synonym phrases or closely related keywords for BM25 text retrieval.\n'
        '- "expanded_en": List of 3-6 English visual synonym phrases for multimodal retrieval.\n'
        '- "objects": List of recognized OpenImages entity names (e.g. Person, Car, Motorcycle, Bicycle, Bus, Dog, Cat, Building, Tree, Food, Chair, Table, Television, Computer, Telephone, Vehicle).\n'
        "Return ONLY the valid JSON object without any markdown wrapping or backticks."
    )

    user_content = f"Task type: {task}\nVietnamese query: {text_vi.strip()}"

    payload = {
        "contents": [
            {"parts": [{"text": f"{system_instruction}\n\nQuery:\n{user_content}"}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "maxOutputTokens": 400,
        },
    }

    data = _call_gemini_api(payload)
    if not data:
        return None

    try:
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts and "text" in parts[0]:
                raw_text = parts[0]["text"].strip()
                # Remove markdown fences if model included them
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`")
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:].strip()
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict) and "text_en" in parsed:
                    return {
                        "text_en": str(parsed.get("text_en", "")).strip(),
                        "expanded_vi": [str(x) for x in parsed.get("expanded_vi", []) if x],
                        "expanded_en": [str(x) for x in parsed.get("expanded_en", []) if x],
                        "objects": [str(x) for x in parsed.get("objects", []) if x],
                    }
    except Exception as exc:
        logger.warning("Lỗi parse JSON mở rộng query từ Gemini: %s", exc)

    return None
