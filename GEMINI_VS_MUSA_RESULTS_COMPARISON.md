# Gemini Pipeline vs MUSA-AI Pipeline Results Comparison

This document summarizes the experimental benchmark results for the Gemini vision pipeline compared with the local MUSA-AI pipeline. It includes two Gemini prompt versions:

1. **Current simplified Gemini prompt**: completed successfully, but produced less accurate and less spatially consistent results than MUSA-AI.
2. **Improved Gemini prompt**: designed to be stricter and more accurate, but repeatedly failed in testing due to API timeout / response failure behavior.

The test used the same tiled AI mapping workflow with a temporary tile limit of 1 tile out of 130 total image tiles.

## Test Context

| Item | Value |
|---|---|
| Experiment type | Single-image tiled AI mapping benchmark |
| Run mode | Gemini and MUSA-AI started simultaneously |
| Total available tiles | 130 |
| Temporary tile limit | 1 tile |
| Processed tile bounds | `(0, 0)` to `(512, 512)` |
| Detection classes | `full_leaf`, `cut_leaf`, `black_sigatoka`, `panama` |
| Gemini model | `gemini-2.5-flash-lite` |
| Current Gemini version | Simplified prompt and JSON point-output pipeline |
| Improved Gemini version | Stricter class-specific prompt with richer validation requirements |
| MUSA-AI models | `best.pt + best.pt` |

## Result Images for Presentation

Add the two output screenshots below when preparing the presentation deck.

| Pipeline | Result Image | Notes |
|---|---|---|
| Gemini | `[Insert Gemini result screenshot here]` | Shows Gemini-generated point detections on the scanned tile. |
| MUSA-AI | `[Insert MUSA-AI result screenshot here]` | Shows local YOLO-generated point detections on the same scanned tile. |

## Main Result Summary

| Evaluation Point | Current Simplified Gemini | Improved Gemini Prompt | MUSA-AI |
|---|---|---|---|
| Completed selected tile | Yes | No, failed during API processing | Yes |
| Returned usable map points | Yes | No | Yes |
| Final detections | 15 | 0 | 20 |
| Main issue | Lower spatial agreement and slower runtime | Repeated timeout / failed response behavior | None observed in this test |
| Practical role | Experimental comparison only | Not usable in current setup | Primary AI mapping pipeline |

## Current Simplified Gemini vs MUSA-AI Metrics

| Metric | Current Simplified Gemini | MUSA-AI Pipeline |
|---|---:|---:|
| Status | Completed | Completed |
| Total model time spent | 90.26 s | 0.57 s |
| Total benchmark wall-clock time | 90.28 s | 90.28 s shared run |
| Tiles processed | 1 / 1 | 1 / 1 |
| Failed tiles | 0 | 0 |
| Average time per tile | 90,257.0 ms | 572.0 ms |
| Raw detections before dedupe | 15 | 22 |
| Detections after dedupe | 15 | 20 |
| Total final detections | 15 | 20 |
| Main bottleneck | Remote API latency | Local inference runtime |

## Current Simplified Gemini Class Count Comparison

| Class | Gemini Count | MUSA-AI Count | Difference |
|---|---:|---:|---:|
| `full_leaf` | 15 | 20 | MUSA-AI +5 |
| `cut_leaf` | 0 | 0 | 0 |
| `black_sigatoka` | 0 | 0 | 0 |
| `panama` | 0 | 0 | 0 |
| **Total** | **15** | **20** | **MUSA-AI +5** |

## Current Simplified Gemini Runtime Comparison

| Runtime Metric | Gemini Pipeline | MUSA-AI Pipeline | Difference |
|---|---:|---:|---|
| Model time | 90.26 s | 0.57 s | Gemini was about 158x slower |
| Average time per tile | 90,257.0 ms | 572.0 ms | Gemini was about 158x slower per tile |
| Tile completion | 1 successful tile | 1 successful tile | Both completed the selected tile |
| Inference type | Remote API call | Local YOLO inference | Gemini depends on internet/API latency |

## Current Simplified Gemini Similarity Metrics

| Metric | Value |
|---|---:|
| Overall similarity within 40px | 0.300 |
| Mean matched distance | 25.00 px |
| Matched `full_leaf` points | 6 |
| Gemini `full_leaf` count | 15 |
| MUSA-AI `full_leaf` count | 20 |

## Per-Class Similarity

| Class | Gemini Count | MUSA-AI Count | Matched Count | Similarity |
|---|---:|---:|---:|---:|
| `full_leaf` | 15 | 20 | 6 | 0.300 |
| `cut_leaf` | 0 | 0 | 0 | 0.000 |
| `black_sigatoka` | 0 | 0 | 0 | 0.000 |
| `panama` | 0 | 0 | 0 | 0.000 |

## Gemini API Usage and Cost

| Usage Metric | Value |
|---|---:|
| Prompt tokens | 565 |
| Candidate/output tokens | 1,041 |
| Total tokens | 1,606 |
| Rough Gemini Flash token estimate | `$0.000473` |

Note: This cost estimate only reflects the token estimate reported by the application. It does not necessarily represent all real-world API billing factors for image-based requests.

## Improved Gemini Prompt Failure Result

The improved Gemini prompt was intended to make detections more accurate by adding class-specific rules, stricter exclusions, coordinate cross-checking, and richer JSON fields. However, in this test setup, the improved prompt repeatedly failed and did not produce usable detections.

| Metric | Improved Gemini Prompt Result |
|---|---:|
| Status | Completed, but tile failed |
| Total model time spent | 205.53 s |
| Tiles processed | 0 / 1 |
| Failed tiles | 1 |
| Average time per tile | 205,525.0 ms |
| Raw detections before dedupe | 0 |
| Detections after dedupe | 0 |
| `full_leaf` | 0 |
| `cut_leaf` | 0 |
| `black_sigatoka` | 0 |
| `panama` | 0 |
| Total detections | 0 |
| Warning count | 2 |

### Improved Gemini Failure Warnings

| Warning | Message |
|---:|---|
| 1 | Temporary tile limit active: processed first 1 of 130 tiles. |
| 2 | Gemini failed on tile 1: The read operation timed out. |

### Improved Gemini Failure Logs

| Step | Log Message |
|---:|---|
| 1 | Gemini tiled mapping started. |
| 2 | Gemini: Sending tile to Gemini - Tile 1/1, px `(0, 0)` to `(512, 512)` |
| 3 | Gemini finished with 0 point(s). |
| 4 | Cancellation requested. Current tile/request may finish before workers stop. |
| 5 | Total benchmark wall-clock time: 205.54s |

## Prompt Version Comparison

| Prompt Version | Goal | Observed Behavior |
|---|---|---|
| Current simplified prompt | Keep the Gemini request simple enough to return usable JSON point detections | Completed successfully, but less accurate than MUSA-AI |
| Improved prompt | Increase accuracy with stricter class definitions, exclusions, coordinate validation, and tile quality metadata | Repeatedly failed in the current setup and returned no usable detections |

## Current Simplified Gemini Prompt

This is the current working Gemini prompt structure used by the application.

```text
You are evaluating RGB drone image tile {tile_index}/{tile_count} for a banana plantation research benchmark.
Detect visible instances of these classes only: {classes}.

Return JSON only, using this exact schema:
{
  "detections": [
    {
      "class_name": "{schema_labels}",
      "normalized_x": 0.0,
      "normalized_y": 0.0,
      "confidence": 0.0,
      "description": "short visual reason"
    }
  ],
  "notes": "short note about uncertainty"
}

Rules:
- Output one center point per detected object or disease manifestation.
- Do not output boxes or polygons.
- Do not return any class outside this allowed class list: {classes}.
- normalized_x and normalized_y must be relative to the provided {width}x{height} image, with origin at the top-left.
- Use values from 0.0 to 1.0.
- confidence is your visual confidence estimate from 0.0 to 1.0.
- If unsure, omit the detection instead of guessing.
- If many detections exist, keep the most visually confident 300 total.
```

## Improved Gemini Prompt Used During Failed Experiment

This was the stricter Gemini prompt structure tested earlier. It was conceptually better for accuracy, but in practice it caused the Gemini pipeline to fail in the current setup.

```text
You are analyzing one {width}x{height} RGB drone image tile from a banana plantation.
This is tile {tile_index}/{tile_count} in a sliced orthomosaic benchmark.

Allowed classes:
- full_leaf: mostly complete visible banana leaf blade, not generic canopy texture.
- cut_leaf: banana leaf blade that is visibly torn, clipped, truncated, broken, or missing a major section.
- black_sigatoka: dark brown or black streaks, spots, or necrotic lesions visible on banana leaf tissue.
- panama: clear yellowing or wilting stress pattern; only mark when the visible symptom is clear.

Task:
Return point detections only for clearly visible instances of the allowed classes: {classes}.
Each detection must be one point at the visual center of the object or symptom.

Return JSON only, using this exact schema:
{
  "detections": [
    {
      "class_name": "{schema_labels}",
      "normalized_x": 0.0,
      "normalized_y": 0.0,
      "pixel_x": 0,
      "pixel_y": 0,
      "confidence": 0.0,
      "evidence": "short visual evidence",
      "uncertainty_reason": ""
    }
  ],
  "tile_quality": {
    "blur": "low | medium | high",
    "lighting": "good | uneven | poor",
    "occlusion": "low | medium | high"
  },
  "notes": "short note about uncertainty"
}

Strict exclusions:
- Do not detect classes outside the allowed list: {classes}.
- Do not mark soil, shadows, roads, stems, trunks, non-banana plants, or generic vegetation.
- Do not infer disease from location or context alone.
- Do not mark vague discoloration unless it matches one of the allowed disease symptoms.

Coordinate rules:
- Output one center point per detected object or disease manifestation.
- Do not output boxes or polygons.
- normalized_x and normalized_y must be relative to the provided {width}x{height} image, with origin at the top-left.
- Use values from 0.0 to 1.0.
- pixel_x and pixel_y must be integer pixel coordinates inside this tile.
- normalized_x * {width} should approximately match pixel_x.
- normalized_y * {height} should approximately match pixel_y.

Confidence rules:
- confidence is a visual confidence estimate from 0.0 to 1.0.
- full_leaf: include only if confidence is at least 0.60.
- cut_leaf: include only if confidence is at least 0.65.
- black_sigatoka: include only if confidence is at least 0.70.
- panama: include only if confidence is at least 0.80.
- If unsure, omit the detection instead of guessing.
- If many detections exist, keep the most visually confident 300 total.
```

## Warning Comparison

| Pipeline / Prompt | Warning Count | Warnings |
|---|---:|---|
| Current simplified Gemini | 1 | Temporary tile limit active: processed first 1 of 130 tiles. |
| Improved Gemini prompt | 2 | Temporary tile limit active: processed first 1 of 130 tiles. Gemini failed on tile 1: The read operation timed out. |
| MUSA-AI | 1 | Temporary tile limit active: processed first 1 of 130 tiles. |

## Combined Logs for Successful Current Gemini Run

| Step | Log Message |
|---:|---|
| 1 | Gemini and MUSA-AI tiled mapping started simultaneously. |
| 2 | Gemini: Sending tile to Gemini - Tile 1/1, px `(0, 0)` to `(512, 512)` |
| 3 | MUSA-AI: Running local YOLO tile inference - Tile 1/1, px `(0, 0)` to `(512, 512)` |
| 4 | MUSA-AI finished with 20 point(s). |
| 5 | Gemini finished with 15 point(s). |
| 6 | Cancellation requested. Current tile/request may finish before workers stop. |
| 7 | Total benchmark wall-clock time: 90.28s |

## Interpretation

The current simplified Gemini pipeline completed successfully and produced usable point detections. It detected 15 `full_leaf` points, while MUSA-AI detected 20 `full_leaf` points on the same tile. However, only 6 Gemini detections matched MUSA-AI detections within a 40px radius, resulting in an overall similarity score of 0.300. This means the current Gemini pipeline is functional but not highly accurate or spatially consistent compared with the local MUSA-AI model.

The improved Gemini prompt was designed to improve accuracy, but it repeatedly failed in the current experimental setup. This creates a practical tradeoff: the simpler prompt works but is less accurate, while the stricter prompt is theoretically better but not operationally reliable in this workflow.

The runtime difference is also large. Gemini required 90.26 seconds for one successful tile, while MUSA-AI required only 0.57 seconds. This makes Gemini approximately 158 times slower than the local model for this test.

## Technical Takeaway

| Evaluation Area | Current Simplified Gemini | Improved Gemini Prompt | MUSA-AI Pipeline |
|---|---|---|---|
| Practicality for tiled mapping | Works for one tile, but slow and less accurate | Failed in current setup | Stronger and much faster |
| Accuracy / spatial agreement | Low-to-moderate; similarity 0.300 | No usable output | Stronger reference output for this test |
| Speed | Very slow due to remote API latency | Slower/failing due to heavier request behavior | Fast local inference |
| Output reliability | Produced usable points in this run | No usable detections due to timeout | Produced usable points |
| Detection density | 15 final points | 0 final points | 20 final points |
| Offline capability | No | No | Yes, if local model/runtime is available |
| Scaling risk | High latency and API dependency | Very high reliability risk | More predictable for batch processing |
| Best role in the system | Experimental benchmark or auxiliary layer | Not recommended in current form | Core detection and mapping pipeline |

## Conclusion

The current simplified Gemini pipeline can generate usable point detections, but the results are not as accurate or spatially consistent as MUSA-AI. The improved Gemini prompt was intended to increase accuracy, but it repeatedly failed in the current setup, producing no usable detections.

This supports the main architectural conclusion: Gemini should not replace the MUSA-AI local YOLO pipeline as the primary detector. Gemini may still be useful as an experimental comparison layer, report-generation assistant, or qualitative review tool, but MUSA-AI should remain the core AI mapping pipeline for scalable drone imagery analysis.
