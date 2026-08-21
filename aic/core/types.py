from dataclasses import dataclass, field


@dataclass
class Query:
    """Câu hỏi của BTC, kèm bản tiếng Anh do người vận hành tự gõ.

    Quy ước: query_id ph là tên file câu hỏi của BTC sau khi bỏ phần đuôi.
    Ví dụ BTC gửi pack1_q3_kis.txt thì query_id = "pack1_q3_kis".

    Lý do: BTC yêu cầu tên file CSV nộp lên khớp tên file câu hỏi. Đặt trùng
    ngay từ đây thì đúng theo cấu tạo, không phải ánh xạ lại ở bước cuối và
    không có chỗ nào để lệch.
    """

    query_id: str
    text_vi: str
    text_en: str = ""
    expanded_en: list[str] = field(default_factory=list)
    expanded_vi: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    task: str = "kis"  # kis | qa | trake
    n_events: int = 1
    modalities: list[str] = field(default_factory=lambda: ["siglip", "caption", "ocr", "asr", "summary", "media_info"])
    weights: dict[str, float] = field(default_factory=dict)

    def for_clip(self) -> str:
        return self.text_en or self.text_vi

    def for_text(self) -> str:
        return self.text_vi

    def for_bm25(self) -> str:
        """Chuỗi text kết hợp song ngữ + từ đồng nghĩa mở rộng cho BM25."""
        parts = [self.text_vi]
        if self.text_en:
            parts.append(self.text_en)
        if self.expanded_vi:
            parts.extend(self.expanded_vi)
        if self.expanded_en:
            parts.extend(self.expanded_en)
        return " ".join(parts).strip()


@dataclass
class Candidate:
    # Output tầng tìm kiếm

    video_id: str
    start_frame: int
    end_frame: int
    representative_frames: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)  # {"clip": 0.76, "transcript": 0.41}
    evidence: dict = field(default_factory=dict)  # {"objects": [...], "caption": "..."}

    @property
    def best_score(self) -> float:
        if "fused" in self.scores:
            return self.scores["fused"]
        return max(self.scores.values(), default=0.0)

    def middle_frame(self) -> int:
        mid = (self.start_frame + self.end_frame) // 2
        if not self.representative_frames:
            return mid
        return min(self.representative_frames, key=lambda f: abs(f - mid))


@dataclass
class Answer:
    
    query_id: str
    rank: int
    video_id: str
    frames: list
    answer: str = ""
