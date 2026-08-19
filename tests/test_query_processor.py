"""Tests cho query processor — parse files + detect task."""

from pathlib import Path

import pytest

from aic.core.query_processor import (
    detect_task,
    make_query,
    parse_query_file,
    translate_query,
)
from aic.core.types import Query


class TestDetectTask:
    def test_kis(self):
        assert detect_task("pack1_q3_kis.txt") == "kis"

    def test_qa(self):
        assert detect_task("pack1_q1_qa.txt") == "qa"

    def test_trake(self):
        assert detect_task("pack1_q2_trake.txt") == "trake"

    def test_unknown_defaults_kis(self):
        assert detect_task("random_query.txt") == "kis"

    def test_case_insensitive(self):
        assert detect_task("PACK1_Q3_KIS.TXT") == "kis"


class TestParseQueryFile:
    def test_parse_kis(self, tmp_path):
        f = tmp_path / "pack1_q1_kis.txt"
        f.write_text("Tìm đoạn video có người đàn ông mặc áo xanh", encoding="utf-8")

        q = parse_query_file(f)

        assert q.query_id == "pack1_q1_kis"
        assert q.task == "kis"
        assert "áo xanh" in q.text_vi
        assert q.text_en == ""
        assert q.n_events == 1

    def test_parse_trake_counts_events(self, tmp_path):
        f = tmp_path / "pack1_q2_trake.txt"
        f.write_text(
            "Tìm video có các sự kiện sau:\n"
            "1. Người phụ nữ bước vào phòng\n"
            "2. Cô ấy ngồi xuống ghế\n"
            "3. Cô ấy mở laptop\n",
            encoding="utf-8",
        )

        q = parse_query_file(f)

        assert q.task == "trake"
        assert q.n_events == 3  # 3 events (dòng 1 là mô tả chung)


class TestMakeQuery:
    def test_basic(self):
        q = make_query("test_q1", "xin chào", text_en="hello", task="kis")
        assert q.query_id == "test_q1"
        assert q.for_clip() == "hello"
        assert q.for_text() == "xin chào"


class TestTranslateQuery:
    def test_already_translated(self):
        q = make_query("q1", "tiếng Việt", text_en="English version")
        result = translate_query(q)
        assert result.text_en == "English version"  # không đổi

    def test_custom_translate_fn(self):
        q = make_query("q1", "xin chào")
        result = translate_query(q, translate_fn=lambda x: "hello")
        assert result.text_en == "hello"

    def test_fallback_on_error(self):
        q = make_query("q1", "tiếng Việt")

        def failing_fn(x):
            raise RuntimeError("API error")

        result = translate_query(q, translate_fn=failing_fn)
        assert result.text_en == "tiếng Việt"  # fallback giữ nguyên
