from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from ai_model_comparator.data.models import DetectionRecord
from ai_model_comparator.detection.tiling import TileSpec


TARGET_CLASSES = {"full_leaf", "cut_leaf", "black_sigatoka", "panama"}


@dataclass(slots=True)
class DetectionContext:
    image_name: str
    image_width: int
    image_height: int

    def pixel_to_geo(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        return 0.0, 0.0


class GeminiDetectionProvider:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite") -> None:
        self.api_key = api_key
        self.model_name = model_name

    def detect_tile(self, tile_image: Image.Image, tile: TileSpec, context: DetectionContext) -> list[DetectionRecord]:
        if not self.api_key:
            return []
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai to use Gemini detection: pip install google-genai") from exc

        client = genai.Client(api_key=self.api_key)
        prompt = _gemini_prompt(tile)
        response = client.models.generate_content(
            model=self.model_name,
            contents=[tile_image, prompt],
        )
        text = getattr(response, "text", "") or ""
        payload = _extract_json_payload(text)
        return _records_from_gemini_payload(payload, tile, context)


class MusaYoloPipelineProvider:
    def __init__(self, leaf_model_path: str, disease_model_path: str) -> None:
        self.leaf_model_path = leaf_model_path
        self.disease_model_path = disease_model_path
        self._leaf_model = None
        self._disease_model = None

    def detect_tile(self, tile_image: Image.Image, tile: TileSpec, context: DetectionContext) -> list[DetectionRecord]:
        if not self.leaf_model_path or not self.disease_model_path:
            return []
        if not Path(self.leaf_model_path).exists():
            raise RuntimeError(f"Leaf model path does not exist: {self.leaf_model_path}")
        if not Path(self.disease_model_path).exists():
            raise RuntimeError(f"Disease model path does not exist: {self.disease_model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics to use local model detection: pip install ultralytics") from exc

        if self._leaf_model is None:
            self._leaf_model = YOLO(self.leaf_model_path)
        if self._disease_model is None:
            self._disease_model = YOLO(self.disease_model_path)

        leaf_result = self._leaf_model.predict(tile_image, imgsz=512, conf=0.25, verbose=False)[0]
        disease_result = self._disease_model.predict(tile_image, imgsz=512, conf=0.25, verbose=False)[0]
        leaf_records = self._parse_result(leaf_result, tile, context, {"full_leaf", "cut_leaf"}, "leaf")
        disease_candidates = self._parse_result(disease_result, tile, context, {"black_sigatoka", "panama"}, "disease")
        accepted_diseases = self._funnel_diseases_through_leaves(disease_candidates, leaf_records)
        self._mark_leaf_health(leaf_records, accepted_diseases)
        return leaf_records + accepted_diseases

    def _parse_result(
        self,
        result,
        tile: TileSpec,
        context: DetectionContext,
        allowed_classes: set[str],
        stage: str,
    ) -> list[DetectionRecord]:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        records: list[DetectionRecord] = []
        for index, box in enumerate(boxes):
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            class_index = int(box.cls[0].item()) if box.cls is not None else -1
            class_name = _normalize_class_name(str(names.get(class_index, class_index)))
            if class_name not in allowed_classes:
                continue
            confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
            pixel_x = tile.x + ((xyxy[0] + xyxy[2]) / 2)
            pixel_y = tile.y + ((xyxy[1] + xyxy[3]) / 2)
            latitude, longitude = context.pixel_to_geo(pixel_x, pixel_y)
            records.append(
                DetectionRecord(
                    id=f"trained_pipeline-{stage}-{tile.name}-{index + 1}",
                    image_name=context.image_name,
                    class_name=class_name,
                    latitude=latitude,
                    longitude=longitude,
                    confidence=confidence,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    health=None if class_name == "full_leaf" else _health_for_class(class_name),
                    source="trained_pipeline",
                    layer_keys=[class_name, stage],
                    metadata={
                        "tile": tile.name,
                        "georeferenced": False,
                        "pipeline": "musa_two_stage_yolo",
                        "stage": stage,
                        "tile_origin": [tile.x, tile.y],
                        "bbox_tile": xyxy,
                        "bbox_image": [tile.x + xyxy[0], tile.y + xyxy[1], tile.x + xyxy[2], tile.y + xyxy[3]],
                    },
                )
            )
        return records

    def _funnel_diseases_through_leaves(
        self,
        disease_records: list[DetectionRecord],
        leaf_records: list[DetectionRecord],
    ) -> list[DetectionRecord]:
        match_leaves = [leaf for leaf in leaf_records if leaf.class_name == "full_leaf"]
        accepted: list[DetectionRecord] = []
        for disease in disease_records:
            related_leaf_id = _find_related_leaf_id(disease.pixel_x, disease.pixel_y, match_leaves)
            if related_leaf_id is None:
                disease.health = "rejected"
                disease.source = "trained_pipeline_rejected"
                disease.metadata["rejection_reason"] = "outside_full_leaf_filter"
                continue
            disease.health = "diseased"
            disease.related_leaf_id = related_leaf_id
            disease.layer_keys = [disease.class_name, "disease", "diseased_leaf"]
            accepted.append(disease)
        return accepted

    def _mark_leaf_health(
        self,
        leaf_records: list[DetectionRecord],
        accepted_diseases: list[DetectionRecord],
    ) -> None:
        diseased_leaf_ids = {disease.related_leaf_id for disease in accepted_diseases if disease.related_leaf_id}
        for leaf in leaf_records:
            if leaf.class_name == "cut_leaf":
                leaf.health = "cut"
                leaf.layer_keys = ["cut_leaf"]
            elif leaf.id in diseased_leaf_ids:
                leaf.health = "diseased"
                leaf.layer_keys = ["full_leaf", "diseased_leaf"]
            else:
                leaf.health = "healthy"
                leaf.layer_keys = ["full_leaf", "healthy_leaf"]


def encode_tile_png(tile_image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    tile_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _gemini_prompt(tile: TileSpec) -> str:
    return (
        "Analyze this 512x512 or edge tile from a high-resolution banana plantation image. "
        "Detect only these classes: full_leaf, cut_leaf, black_sigatoka, panama. "
        "Return strict JSON only with this shape: "
        '{"detections":[{"class_name":"black_sigatoka","confidence":0.0,'
        '"bbox":[x_min,y_min,x_max,y_max],"notes":"short evidence"}]}. '
        "Coordinates must be pixel coordinates inside this tile, where x/y start at 0. "
        "If no target class is visible, return {\"detections\":[]}. "
        f"Tile id: {tile.name}."
    )


def _extract_json_payload(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return {"detections": []}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"detections": []}


def _records_from_gemini_payload(payload: dict, tile: TileSpec, context: DetectionContext) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    for index, item in enumerate(payload.get("detections", [])):
        class_name = _normalize_class_name(str(item.get("class_name", "")))
        if class_name not in TARGET_CLASSES:
            continue
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        pixel_x = tile.x + ((x1 + x2) / 2)
        pixel_y = tile.y + ((y1 + y2) / 2)
        latitude, longitude = context.pixel_to_geo(pixel_x, pixel_y)
        confidence = float(item.get("confidence", 0.0) or 0.0)
        records.append(
            DetectionRecord(
                id=f"gemini-{tile.name}-{index + 1}",
                image_name=context.image_name,
                class_name=class_name,
                latitude=latitude,
                longitude=longitude,
                confidence=max(0.0, min(confidence, 1.0)),
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                health=_health_for_class(class_name),
                source="gemini",
                layer_keys=[class_name],
                metadata={
                    "tile": tile.name,
                    "georeferenced": False,
                    "tile_origin": [tile.x, tile.y],
                    "bbox_tile": [x1, y1, x2, y2],
                    "bbox_image": [tile.x + x1, tile.y + y1, tile.x + x2, tile.y + y2],
                    "notes": str(item.get("notes", "")),
                    "model": "gemini-2.5-flash-lite",
                },
            )
        )
    return records


def _find_related_leaf_id(pixel_x: float, pixel_y: float, leaf_records: list[DetectionRecord]) -> str | None:
    for leaf in leaf_records:
        bbox = leaf.metadata.get("bbox_image", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if x1 <= pixel_x <= x2 and y1 <= pixel_y <= y2:
            return leaf.id
    return None


def _health_for_class(class_name: str) -> str:
    if class_name in {"black_sigatoka", "panama"}:
        return "diseased"
    if class_name == "cut_leaf":
        return "cut"
    return "healthy"


def _normalize_class_name(value: str) -> str:
    cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "black": "black_sigatoka",
        "sigatoka": "black_sigatoka",
        "black_sikatoka": "black_sigatoka",
        "fusarium": "panama",
        "fusarium_wilt": "panama",
        "panama_disease": "panama",
        "full": "full_leaf",
        "fullleaf": "full_leaf",
        "cut": "cut_leaf",
        "cutleaf": "cut_leaf",
    }
    return aliases.get(cleaned, cleaned)
