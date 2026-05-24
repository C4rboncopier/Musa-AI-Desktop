from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from banana_mapper.detection import _generate_tiles

from .models import CLASSES, DetectionPoint, InferenceRun


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TileProgressCallback = Callable[[int, int, tuple[int, int, int, int], str], None]


class GeminiVisionProvider:
    """Small REST client for Gemini image understanding experiments."""

    def __init__(self, api_key: str, model_name: str = DEFAULT_GEMINI_MODEL) -> None:
        self.api_key = api_key.strip()
        self.model_name = model_name.strip() or DEFAULT_GEMINI_MODEL

    def detect_points(
        self,
        image_path: str | Path,
        *,
        slice_size: int = 512,
        slice_overlap: int = 96,
        max_tiles: int | None = None,
        allowed_classes: tuple[str, ...] = CLASSES,
        progress_callback: TileProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> InferenceRun:
        if not self.api_key:
            raise RuntimeError("Gemini API key is required.")

        path = Path(image_path)
        started = time.perf_counter()
        with Image.open(path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            all_tiles = _generate_tiles(original_width, original_height, slice_size, slice_overlap)
            tiles = all_tiles
            raw_points: list[DetectionPoint] = []
            warnings: list[str] = []
            raw_responses: list[dict[str, Any]] = []
            usage_total: dict[str, int] = {}
            failed_tiles = 0
            cancelled = False
            if max_tiles is not None and max_tiles > 0 and max_tiles < len(all_tiles):
                tiles = all_tiles[:max_tiles]
                warnings.append(f"Temporary tile limit active: processed first {len(tiles)} of {len(all_tiles)} tiles.")

            for tile_index, (x0, y0, x1, y1) in enumerate(tiles, start=1):
                if should_stop and should_stop():
                    cancelled = True
                    warnings.append("Gemini mapping cancelled before all tiles were processed.")
                    break
                if progress_callback:
                    progress_callback(tile_index, len(tiles), (x0, y0, x1, y1), "Sending tile to Gemini")
                tile = image.crop((x0, y0, x1, y1))
                request_bytes, request_width, request_height = _encode_jpeg(tile)
                prompt = _detection_prompt(request_width, request_height, tile_index, len(tiles), allowed_classes)
                payload = _gemini_payload(prompt, request_bytes)
                try:
                    response = self._post_json(payload)
                    text = _extract_text(response)
                    parsed = _parse_json_text(text)
                    tile_points = _points_from_response(
                        parsed,
                        original_width=request_width,
                        original_height=request_height,
                        request_width=request_width,
                        request_height=request_height,
                        allowed_classes=allowed_classes,
                    )
                    for point in tile_points:
                        point.x += x0
                        point.y += y0
                        point.id = f"gemini-t{tile_index}-{len(raw_points) + 1}"
                        point.raw["tile_index"] = tile_index
                        point.raw["tile_bounds"] = [x0, y0, x1, y1]
                        raw_points.append(point)
                    usage = dict(response.get("usageMetadata", {}) or {})
                    _merge_usage(usage_total, usage)
                    raw_responses.append(
                        {
                            "tile_index": tile_index,
                            "bounds": [x0, y0, x1, y1],
                            "text": text,
                            "usage": usage,
                        }
                    )
                except Exception as exc:
                    if should_stop and should_stop():
                        cancelled = True
                        warnings.append("Gemini mapping cancelled during API processing.")
                        break
                    failed_tiles += 1
                    warnings.append(f"Gemini failed on tile {tile_index}: {exc}")

        duration_ms = int((time.perf_counter() - started) * 1000)
        points = _dedupe_pixel_points(raw_points, radius_px=max(8.0, slice_overlap * 0.35))
        for index, point in enumerate(points, start=1):
            point.id = f"gemini-{index}"

        return InferenceRun(
            source="gemini",
            image_path=str(path),
            image_width=original_width,
            image_height=original_height,
            duration_ms=duration_ms,
            model_name=f"{self.model_name} ({', '.join(allowed_classes)})",
            points=points,
            raw_response=json.dumps(raw_responses, ensure_ascii=True),
            usage=usage_total,
            warnings=warnings,
            tile_count=len(tiles),
            failed_tiles=failed_tiles,
            raw_detection_count=len(raw_points),
            deduped_detection_count=len(points),
            cancelled=cancelled,
        )

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = GEMINI_ENDPOINT.format(model=self.model_name)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc


def _encode_jpeg(image: Image.Image) -> tuple[bytes, int, int]:
    import io

    request_width, request_height = image.size
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return buffer.getvalue(), request_width, request_height


def _detection_prompt(width: int, height: int, tile_index: int, tile_count: int, allowed_classes: tuple[str, ...]) -> str:
    classes = ", ".join(allowed_classes)
    schema_labels = " | ".join(allowed_classes)
    return f"""
You are evaluating RGB drone image tile {tile_index}/{tile_count} for a banana plantation research benchmark.
Detect visible instances of these classes only: {classes}.

Return JSON only, using this exact schema:
{{
  "detections": [
    {{
      "class_name": "{schema_labels}",
      "normalized_x": 0.0,
      "normalized_y": 0.0,
      "confidence": 0.0,
      "description": "short visual reason"
    }}
  ],
  "notes": "short note about uncertainty"
}}

Rules:
- Output one center point per detected object or disease manifestation.
- Do not output boxes or polygons.
- Do not return any class outside this allowed class list: {classes}.
- normalized_x and normalized_y must be relative to the provided {width}x{height} image, with origin at the top-left.
- Use values from 0.0 to 1.0.
- confidence is your visual confidence estimate from 0.0 to 1.0.
- If unsure, omit the detection instead of guessing.
- If many detections exist, keep the most visually confident 300 total.
""".strip()


def _gemini_payload(prompt: str, image_bytes: bytes) -> dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    }


def _extract_text(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    chunks = [str(part.get("text", "")) for part in parts if part.get("text")]
    return "\n".join(chunks).strip()


def _parse_json_text(text: str) -> dict[str, Any]:
    if not text:
        return {"detections": []}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _points_from_response(
    parsed: dict[str, Any],
    *,
    original_width: int,
    original_height: int,
    request_width: int,
    request_height: int,
    allowed_classes: tuple[str, ...] = CLASSES,
) -> list[DetectionPoint]:
    detections = parsed.get("detections") or parsed.get("points") or []
    points: list[DetectionPoint] = []
    for index, item in enumerate(detections, start=1):
        if not isinstance(item, dict):
            continue
        class_name = str(item.get("class_name") or item.get("class") or "").strip().lower()
        class_name = class_name.replace(" ", "_").replace("-", "_")
        if class_name not in allowed_classes:
            continue

        x, y = _coordinate_from_item(
            item,
            original_width=original_width,
            original_height=original_height,
            request_width=request_width,
            request_height=request_height,
        )
        if x is None or y is None:
            continue
        points.append(
            DetectionPoint(
                id=f"gemini-{index}",
                source="gemini",
                class_name=class_name,
                x=max(0.0, min(float(original_width), x)),
                y=max(0.0, min(float(original_height), y)),
                confidence=_safe_float(item.get("confidence")),
                description=str(item.get("description") or item.get("reasoning") or ""),
                raw=item,
            )
        )
    return points


def _coordinate_from_item(
    item: dict[str, Any],
    *,
    original_width: int,
    original_height: int,
    request_width: int,
    request_height: int,
) -> tuple[float | None, float | None]:
    nx = _safe_float(item.get("normalized_x"))
    ny = _safe_float(item.get("normalized_y"))
    if nx is not None and ny is not None:
        return nx * original_width, ny * original_height

    x = _safe_float(item.get("x"))
    y = _safe_float(item.get("y"))
    if x is None or y is None:
        center = item.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            x = _safe_float(center[0])
            y = _safe_float(center[1])
    if x is None or y is None:
        return None, None

    if 0 <= x <= 1 and 0 <= y <= 1:
        return x * original_width, y * original_height
    return x * (original_width / max(1, request_width)), y * (original_height / max(1, request_height))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_usage(target: dict[str, int], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        try:
            target[key] = int(target.get(key, 0)) + int(value)
        except (TypeError, ValueError):
            continue


def _dedupe_pixel_points(points: list[DetectionPoint], radius_px: float) -> list[DetectionPoint]:
    import math

    unique: list[DetectionPoint] = []
    for point in sorted(points, key=lambda item: item.confidence or 0.0, reverse=True):
        duplicate = None
        for existing in unique:
            if existing.class_name != point.class_name:
                continue
            if math.hypot(existing.x - point.x, existing.y - point.y) <= radius_px:
                duplicate = existing
                break
        if duplicate is None:
            unique.append(point)
        elif (point.confidence or 0.0) > (duplicate.confidence or 0.0):
            duplicate.x = point.x
            duplicate.y = point.y
            duplicate.confidence = point.confidence
            duplicate.raw = point.raw
    unique.sort(key=lambda item: (item.class_name, item.y, item.x))
    return unique


def estimate_gemini_flash_cost(usage: dict[str, Any]) -> float | None:
    """Rough Gemini 2.5 Flash-Lite text-token estimate, excluding image pricing nuances."""
    prompt_tokens = _safe_float(usage.get("promptTokenCount")) or 0.0
    output_tokens = _safe_float(usage.get("candidatesTokenCount")) or 0.0
    if prompt_tokens <= 0 and output_tokens <= 0:
        return None
    return (prompt_tokens / 1_000_000 * 0.10) + (output_tokens / 1_000_000 * 0.40)
