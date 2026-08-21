"""Query processor — đọc file query từ BTC và tạo Query objects.

BTC cung cấp query dưới dạng file text, ví dụ:
    pack1_q3_kis.txt — Textual KIS
    pack1_q1_qa.txt  — QA (có 2 dòng: mô tả + câu hỏi)
    pack1_q2_trake.txt — TRAKE (nhiều events)

Quy ước query_id = tên file bỏ đuôi, ví dụ "pack1_q3_kis".
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from .types import Query

logger = logging.getLogger(__name__)

# Map suffix → task type
_TASK_PATTERNS = {
    "kis": "kis",
    "qa": "qa",
    "trake": "trake",
}


def detect_task(filename: str) -> str:
    """Đoán task type từ tên file.

    Ví dụ: 'pack1_q3_kis.txt' → 'kis'
    """
    stem = Path(filename).stem.lower()
    for suffix, task in _TASK_PATTERNS.items():
        if suffix in stem:
            return task
    logger.warning("Cannot detect task from '%s', defaulting to 'kis'", filename)
    return "kis"


def parse_query_file(filepath: str | Path) -> Query:
    """Đọc một file query từ BTC, trả về Query object.

    Format:
        KIS: 1 đoạn mô tả (có thể nhiều dòng)
        QA: dòng 1 = mô tả, dòng cuối bắt đầu bằng "?" hoặc là câu hỏi
        TRAKE: nhiều events, mỗi event một dòng hoặc đánh số
    """
    filepath = Path(filepath)
    query_id = filepath.stem
    task = detect_task(filepath.name)

    text = filepath.read_text(encoding="utf-8").strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    n_events = 1
    if task == "trake":
        # Đếm số events (mỗi dòng hoặc đánh số 1., 2., ...)
        n_events = len(lines)
        # Nếu dòng đầu là mô tả chung, events bắt đầu từ dòng 2
        if len(lines) > 1 and not re.match(r"^\d+[\.\)]", lines[0]):
            n_events = len(lines) - 1

    return Query(
        query_id=query_id,
        text_vi=text,
        text_en="",  # sẽ điền bằng translate()
        task=task,
        n_events=n_events,
    )


def parse_query_dir(dirpath: str | Path) -> list[Query]:
    """Đọc toàn bộ file .txt trong thư mục, trả về list[Query]."""
    dirpath = Path(dirpath)
    queries = []
    for f in sorted(dirpath.glob("*.txt")):
        try:
            q = parse_query_file(f)
            queries.append(q)
        except Exception as e:
            logger.error("Error parsing %s: %s", f, e)
    logger.info("Parsed %d queries from %s", len(queries), dirpath)
    return queries


def translate_query(query: Query, translate_fn=None) -> Query:
    """Dịch query.text_vi sang tiếng Anh, ghi vào query.text_en.

    translate_fn: callable nhận str → str. 
    Nếu None, dùng Gemini API (cần google-genai).
    """
    if query.text_en:
        return query  # đã có bản dịch

    if translate_fn is None:
        translate_fn = _gemini_translate

    try:
        query.text_en = translate_fn(query.text_vi)
        logger.info("Translated [%s]: %s", query.query_id, query.text_en[:80])
    except Exception as e:
        logger.warning("Translation failed for %s: %s", query.query_id, e)
        query.text_en = query.text_vi  # fallback: giữ nguyên tiếng Việt

    return query


def process_query(query: Query, llm_fn=None) -> Query:
    """Xử lý toàn diện Query: Dịch + Mở rộng từ đồng nghĩa + Trích xuất Object.

    Điền đầy đủ:
    - query.text_en: Dịch chuẩn cho CLIP/SigLIP
    - query.expanded_vi / query.expanded_en: Từ đồng nghĩa cho BM25
    - query.objects: Danh sách Object detection entities cho ObjectFilter
    """
    if llm_fn is None:
        llm_fn = _gemini_process_query

    try:
        res = llm_fn(query.text_vi)
        if isinstance(res, dict):
            if not query.text_en:
                query.text_en = res.get("text_en", "")
            query.expanded_vi = res.get("expanded_vi", [])
            query.expanded_en = res.get("expanded_en", [])
            query.objects = res.get("objects", [])
            logger.info("Processed [%s]: en='%s', objects=%s",
                        query.query_id, query.text_en[:50], query.objects)
            return query
    except Exception as e:
        logger.warning("Full query processing failed for %s: %s. Using fallback.", query.query_id, e)

    # Fallback to standard translation and rule-based extraction
    translate_query(query)
    _rule_based_extract(query)
    return query


def _rule_based_extract(query: Query) -> None:
    """Fallback rule-based extraction of objects and synonyms when offline."""
    text_lower = f"{query.text_vi} {query.text_en}".lower()
    
    # Common object keywords mapping to OpenImages entities
    obj_rules = {
        "người": "Person", "man": "Person", "woman": "Person", "person": "Person",
        "xe": "Car", "ô tô": "Car", "car": "Car", "vehicle": "Vehicle",
        "xe máy": "Motorcycle", "motorcycle": "Motorcycle", "motorbike": "Motorcycle",
        "xe đạp": "Bicycle", "bicycle": "Bicycle", "bike": "Bicycle",
        "xe buýt": "Bus", "bus": "Bus",
        "chó": "Dog", "dog": "Dog",
        "mèo": "Cat", "cat": "Cat",
        "cây": "Tree", "tree": "Tree",
        "nhà": "Building", "tòa nhà": "Building", "building": "Building",
        "thức ăn": "Food", "món ăn": "Food", "nấu ăn": "Food", "food": "Food",
        "bàn": "Table", "table": "Table",
        "ghế": "Chair", "chair": "Chair",
        "điện thoại": "Telephone", "phone": "Telephone",
        "máy tính": "Computer", "laptop": "Computer",
        "tivi": "Television", "tv": "Television",
    }
    
    found_objects = set()
    for kw, entity in obj_rules.items():
        if kw in text_lower:
            found_objects.add(entity)
    query.objects = sorted(found_objects)


def _format_as_caption(text_en: str) -> str:
    """Định dạng bản dịch tiếng Anh theo đúng cấu trúc Caption của tập dữ liệu."""
    text_en = text_en.strip()
    if not text_en:
        return ""
    
    # Chuẩn hóa chữ cái đầu
    text_clean = text_en[0].lower() + text_en[1:] if len(text_en) > 1 else text_en.lower()
    
    # Kiểm tra xem đã có prefix mô tả góc quay chưa
    prefixes = ("a shot of", "a medium shot", "a close-up", "a wide angle", "a wide shot",
                "an aerial", "a first-person", "in this frame", "a split-screen", "a view of")
    if any(text_clean.startswith(p) for p in prefixes):
        return text_en[0].upper() + text_en[1:]

    # Thêm prefix chuẩn theo văn phong của mô hình Captioning
    return f"A shot of {text_clean}"


def _gemini_process_query(text_vi: str) -> dict:
    """Dùng Gemini phân tích mở rộng query theo phong cách video captioning chuẩn."""
    import json
    if os.environ.get("GEMINI_API_KEY"):
        try:
            from google import genai
            client = genai.Client()
            prompt = (
                "You are an expert in AI Challenge Video Retrieval.\n"
                "Convert this Vietnamese search query into an expanded Video Caption description matching video dataset style.\n"
                "Dataset caption style examples:\n"
                "- 'A medium shot of a news studio with two news anchors standing behind a white desk...'\n"
                "- 'A close-up shot of a chef preparing ingredients in a kitchen with cooking utensils...'\n"
                "- 'An aerial wide shot of a river with boats moving on water at sunset...'\n\n"
                f"Vietnamese Query: \"{text_vi}\"\n\n"
                "Output ONLY a JSON object with these keys:\n"
                "- text_en: Detailed English description in caption style starting with 'A shot of / A medium shot of / A close-up shot of'\n"
                "- expanded_vi: List of 4-6 Vietnamese synonyms and related keywords\n"
                "- expanded_en: List of 4-6 English synonyms and visual elements\n\n"
                "JSON format:"
            )
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            txt = response.text.strip()
            if txt.startswith("```"):
                txt = re.sub(r"^```[a-zA-Z]*\n", "", txt)
                txt = re.sub(r"\n```$", "", txt)
            return json.loads(txt)
        except Exception as e:
            logger.warning("Gemini query processing failed: %s", e)

    # Fallback khi offline hoặc không có API key
    text_en = _gemini_translate(text_vi)
    return {
        "text_en": text_en,
        "expanded_vi": [],
        "expanded_en": [],
    }


def _gemini_translate(text_vi: str) -> str:
    """Dịch tiếng Việt → tiếng Anh theo phong cách caption của tập dữ liệu."""
    # 1. Thử Google Gemini nếu có API key
    if os.environ.get("GEMINI_API_KEY"):
        try:
            from google import genai
            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=(
                    "Translate the following Vietnamese video retrieval query into a detailed visual caption in English. "
                    "Format in dataset caption style starting with 'A shot of / A medium shot of / A close-up shot of' "
                    "describing the visual scene, subjects, and actions. Output ONLY the English caption.\n\n"
                    f"{text_vi}"
                ),
            )
            if response.text:
                return response.text.strip()
        except Exception:
            pass

    # 2. Thử deep_translator (miễn phí, không cần API key, cực nhanh)
    try:
        from deep_translator import GoogleTranslator
        raw_en = GoogleTranslator(source="vi", target="en").translate(text_vi)
        if raw_en:
            return _format_as_caption(raw_en.strip())
    except Exception as e:
        logger.debug("deep_translator failed: %s", e)

    return _format_as_caption(text_vi)


def make_query(
    query_id: str,
    text_vi: str,
    text_en: str = "",
    task: str = "kis",
    n_events: int = 1,
    modalities: list[str] = None,
    weights: dict[str, float] = None,
) -> Query:
    """Helper để tạo Query nhanh (cho notebook / testing)."""
    return Query(
        query_id=query_id,
        text_vi=text_vi,
        text_en=text_en,
        task=task,
        n_events=n_events,
        modalities=modalities if modalities is not None else ["siglip", "caption", "ocr", "asr", "summary", "media_info"],
        weights=weights or {},
    )
