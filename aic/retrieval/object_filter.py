"""Object-based soft filter / scorer.

Đọc dữ liệu từ local/objects/ và dùng để cộng/trừ điểm cho Candidate
dựa trên object detection kết quả.

Không loại bỏ candidate (hard filter) mà chỉ tăng/giảm điểm (soft scoring)
để tránh false negative.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from ..core.types import Candidate

logger = logging.getLogger(__name__)

# Common object entity -> readable name mapping (from Open Images)
ENTITY_MAP = {
    "Person": "person",
    "Human face": "person",
    "Human body": "person",
    "Human head": "person",
    "Human hand": "person",
    "Human arm": "person",
    "Human leg": "person",
    "Man": "person",
    "Woman": "person",
    "Boy": "person",
    "Girl": "person",
    "Car": "car",
    "Vehicle": "vehicle",
    "Land vehicle": "vehicle",
    "Motorcycle": "motorcycle",
    "Bicycle": "bicycle",
    "Bus": "bus",
    "Truck": "truck",
    "Dog": "dog",
    "Cat": "cat",
    "Tree": "tree",
    "Building": "building",
    "House": "building",
    "Food": "food",
    "Table": "table",
    "Chair": "chair",
    "Telephone": "phone",
    "Mobile phone": "phone",
    "Book": "book",
    "Computer": "computer",
    "Laptop": "computer",
    "Television": "tv",
}


class ObjectFilter:
    """Load object detection data và cung cấp scoring API.

    Hỗ trợ load trực tiếp từ file ``objects_index.pkl`` (nhanh 0.1s)
    hoặc đọc từ thư mục ``objects/``.

    Parameters
    ----------
    index_path : đường dẫn tới file ``objects_index.pkl`` (nếu có)
    objects_dir : thư mục fallback nếu chưa có file .pkl
    confidence_threshold : chỉ giữ detection có score >= threshold
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        objects_dir: str | Path | None = None,
        confidence_threshold: float = 0.4,
    ):
        self.confidence_threshold = confidence_threshold
        self._data: dict[str, dict[int, dict[str, float]]] = {}
        self.objects_dir = Path(objects_dir) if objects_dir else None

        # 1. Try loading from objects_index.pkl first
        if index_path and Path(index_path).exists():
            logger.info("Loading unified objects index from %s", index_path)
            import pickle
            with open(index_path, "rb") as f:
                self._data = pickle.load(f)
            logger.info("  → Loaded object detections for %d videos", len(self._data))
        elif objects_dir and Path(objects_dir).exists():
            # Check if objects_index.pkl exists in index folder
            alt_pkl = Path(objects_dir).parent / "index" / "objects_index.pkl"
            if alt_pkl.exists():
                logger.info("Loading unified objects index from %s", alt_pkl)
                import pickle
                with open(alt_pkl, "rb") as f:
                    self._data = pickle.load(f)
                logger.info("  → Loaded object detections for %d videos", len(self._data))
            else:
                logger.info("ObjectFilter initialized from directory %s (threshold=%.2f)",
                            self.objects_dir, confidence_threshold)
        else:
            logger.warning("ObjectFilter initialized without valid index or directory")

    def _load_video_objects(self, video_id: str) -> dict[int, dict[str, float]]:
        """Load tất cả object detection cho 1 video.

        Returns: dict[keyframe_num → dict[entity_name → score]]
        """
        if video_id in self._data:
            return self._data[video_id]

        if not self.objects_dir or not self.objects_dir.exists():
            return {}

        batch = video_id.split("_")[0]
        video_dir = self.objects_dir / batch / video_id
        result: dict[int, dict[str, float]] = {}

        if not video_dir.exists():
            self._data[video_id] = result
            return result

        for json_file in video_dir.glob("*.json"):
            try:
                kf_num = int(json_file.stem)
            except ValueError:
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            entities = data.get("detection_class_entities", [])
            scores = data.get("detection_scores", [])

            entity_scores: dict[str, float] = {}
            for entity, score_str in zip(entities, scores):
                try:
                    score = float(score_str)
                except (ValueError, TypeError):
                    continue
                if score >= self.confidence_threshold:
                    if entity not in entity_scores or score > entity_scores[entity]:
                        entity_scores[entity] = round(score, 4)

            if entity_scores:
                result[kf_num] = entity_scores

        self._data[video_id] = result
        return result

    def get_objects_for_keyframe(
        self, video_id: str, keyframe_num: int
    ) -> dict[str, float]:
        """Lấy dict {entity_name: score} cho 1 keyframe cụ thể."""
        video_objs = self._load_video_objects(video_id)
        return video_objs.get(keyframe_num, {})

    def get_object_summary(self, video_id: str) -> Counter:
        """Tổng hợp tất cả objects xuất hiện trong video."""
        video_objs = self._load_video_objects(video_id)
        counter = Counter()
        for kf_objs in video_objs.values():
            counter.update(kf_objs.keys())
        return counter

    def score_candidate(
        self,
        candidate: Candidate,
        query_objects: list[str],
    ) -> float:
        """Tính bonus score cho candidate dựa trên object overlap + detection confidence.

        Returns
        -------
        float: bonus score trong [0, 1]
        """
        if not query_objects:
            return 0.0

        video_objs = self._load_video_objects(candidate.video_id)
        if not video_objs:
            return 0.0

        # Tìm keyframe gần nhất với start_frame
        kf_num = candidate.start_frame
        if kf_num in video_objs:
            kf_entities = video_objs[kf_num]
        else:
            # Fallback: gộp tất cả keyframes của video
            kf_entities = {}
            for objs in video_objs.values():
                for ent, sc in objs.items():
                    if ent not in kf_entities or sc > kf_entities[ent]:
                        kf_entities[ent] = sc

        if not kf_entities:
            return 0.0

        # Map canonical entity names -> max confidence
        detected_canonical: dict[str, float] = {}
        for entity, score in kf_entities.items():
            canonical = ENTITY_MAP.get(entity, entity.lower())
            if canonical not in detected_canonical or score > detected_canonical[canonical]:
                detected_canonical[canonical] = score

        # Check query objects
        matched_scores = []
        for obj in query_objects:
            canonical = ENTITY_MAP.get(obj, obj.lower())
            if canonical in detected_canonical:
                matched_scores.append(detected_canonical[canonical])

        if not matched_scores:
            return 0.0

        # Score = (tỷ lệ object khớp) * (trung bình confidence của object khớp)
        overlap_ratio = len(matched_scores) / len(query_objects)
        avg_confidence = sum(matched_scores) / len(matched_scores)
        return round(overlap_ratio * avg_confidence, 4)

    def apply_scores(
        self,
        candidates: list[Candidate],
        query_objects: list[str],
        score_key: str = "object_match",
        weight: float = 0.3,
    ) -> list[Candidate]:
        """Cộng bonus score vào tất cả candidates."""
        if not query_objects:
            return candidates

        for cand in candidates:
            bonus = self.score_candidate(cand, query_objects)
            cand.scores[score_key] = round(bonus * weight, 6)

            if bonus > 0:
                video_objs = self._load_video_objects(cand.video_id)
                all_entities = set()
                for objs in video_objs.values():
                    all_entities.update(objs.keys())
                cand.evidence["objects"] = sorted(all_entities)[:10]

        return candidates


def build_object_filter(
    index_dir: str | Path = "local/index",
    confidence_threshold: float = 0.4,
) -> ObjectFilter:
    """Factory function — tạo ObjectFilter từ file objects_index.pkl."""
    index_dir = Path(index_dir)
    pkl_path = index_dir / "objects_index.pkl"
    return ObjectFilter(
        index_path=pkl_path if pkl_path.exists() else None,
        objects_dir=index_dir.parent / "objects",
        confidence_threshold=confidence_threshold,
    )
