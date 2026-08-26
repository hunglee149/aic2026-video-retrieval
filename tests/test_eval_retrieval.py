"""Tests cho bộ chấm điểm retrieval (`scripts/eval_retrieval.py`).

Chỉ chấm phần logic thuần — đọc file, tính cửa sổ frame, tính Recall@K. Phần
gọi retriever thật nằm ngoài phạm vi unit test vì nó cần index vài trăm MB.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "eval_retrieval.py"
_spec = importlib.util.spec_from_file_location("eval_retrieval", _MODULE_PATH)
eval_retrieval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_retrieval)


def write_set(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "set.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


BASE_ROW = {
    "query_id": "s01",
    "text_vi": "cảnh sát giao thông ở ngã tư",
    "video_id": "L21_V031",
    "frame": 1500,
}


class TestLoadSanitySet:
    def test_reads_entries(self, tmp_path):
        path = write_set(tmp_path, [BASE_ROW, dict(BASE_ROW, query_id="s02")])

        entries = eval_retrieval.load_sanity_set(path)

        assert [e["query_id"] for e in entries] == ["s01", "s02"]

    def test_skips_blank_and_comment_lines(self, tmp_path):
        path = tmp_path / "set.jsonl"
        path.write_text(
            "# ghi chú\n\n" + json.dumps(BASE_ROW, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        assert len(eval_retrieval.load_sanity_set(path)) == 1

    def test_rejects_missing_ground_truth_fields(self, tmp_path):
        path = write_set(tmp_path, [{"query_id": "s01", "text_vi": "x"}])

        with pytest.raises(SystemExit, match="video_id"):
            eval_retrieval.load_sanity_set(path)

    def test_rejects_entry_without_frame_or_window(self, tmp_path):
        path = write_set(tmp_path, [{k: v for k, v in BASE_ROW.items()
                                     if k != "frame"}])

        with pytest.raises(SystemExit, match="frame"):
            eval_retrieval.load_sanity_set(path)

    def test_accepts_window_without_frame(self, tmp_path):
        row = {k: v for k, v in BASE_ROW.items() if k != "frame"}
        row["window"] = [1400, 1700]
        path = write_set(tmp_path, [row])

        assert eval_retrieval.load_sanity_set(path)[0]["window"] == [1400, 1700]

    def test_reports_broken_json_with_line_number(self, tmp_path):
        path = tmp_path / "set.jsonl"
        path.write_text('{"query_id": "s01"\n', encoding="utf-8")

        with pytest.raises(SystemExit, match="set.jsonl:1"):
            eval_retrieval.load_sanity_set(path)


class TestWindow:
    def test_explicit_window_wins(self):
        entry = dict(BASE_ROW, window=[100, 200])

        assert eval_retrieval.window_of(entry, tolerance=999) == (100, 200)

    def test_falls_back_to_frame_plus_tolerance(self):
        assert eval_retrieval.window_of(BASE_ROW, tolerance=150) == (1350, 1650)


class TestRecall:
    def _results(self, ranks):
        return [{"video_rank": r} for r in ranks]

    def test_counts_hits_within_cutoff(self):
        results = self._results([1, 3, 30, None])

        assert eval_retrieval.recall_at(results, "video_rank", 1) == 0.25
        assert eval_retrieval.recall_at(results, "video_rank", 5) == 0.5
        assert eval_retrieval.recall_at(results, "video_rank", 100) == 0.75

    def test_missing_rank_never_counts(self):
        results = self._results([None, None])

        assert eval_retrieval.recall_at(results, "video_rank", 100) == 0.0

    def test_empty_input_is_zero_not_crash(self):
        assert eval_retrieval.recall_at([], "video_rank", 5) == 0.0

    def test_cutoffs_match_competition_formula(self):
        assert eval_retrieval.CUTOFFS == (1, 5, 20, 50, 100)


class TestWeights:
    def test_parses_pairs(self):
        assert eval_retrieval.parse_weights("clip=1.5,bm25=2") == {
            "clip": 1.5,
            "bm25": 2.0,
        }

    def test_none_stays_none(self):
        assert eval_retrieval.parse_weights(None) is None
        assert eval_retrieval.parse_weights("") is None

    def test_rejects_malformed_pair(self):
        with pytest.raises(SystemExit, match="ten=so"):
            eval_retrieval.parse_weights("clip")
