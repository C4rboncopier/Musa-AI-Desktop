# Experimental AI Comparison App

This standalone tool compares two single-image detection approaches:

1. Gemini Vision point detection using a user-provided Gemini API key.
2. Local MUSA-AI YOLO/YOLOv8-seg model detection using user-selected `.pt` weights.

It does not replace the main Musa AI GeoTIFF pipeline. It is only for feasibility testing, visual comparison, and research/demo benchmarking.

## Run

```powershell
python run_experimental_comparison.py
```

If `python` is not on PATH, run it with the Python interpreter used for the main Musa AI application.

## Workflow

1. Select one RGB image, ideally a 4K image extracted from a stitched GeoTIFF orthomosaic.
2. Select the MUSA-AI leaf model path and/or disease model path.
3. Paste a Gemini API key.
4. Choose whether the local MUSA-AI run should use Leaf + Disease, Leaf only, or Disease only mode.
5. Choose whether MUSA-AI should keep disease points even when they are outside detected leaf polygons.
6. Optionally set a temporary tile limit for quick tests. `No limit` processes the full image.
7. Run Gemini, MUSA-AI, or both. Both pipelines use the same 512x512 overlapping tiled scan pattern.
8. Compare point overlays in the interactive Leaflet/OpenStreetMap image map.
9. Export the comparison JSON for later analysis.

## Outputs Compared

The app displays point markers for:

- `full_leaf`
- `cut_leaf`
- `black_sigatoka`
- `panama`

Each marker popup shows:

- class name
- source model
- confidence when available
- X/Y image coordinate
- Gemini description when available

## Map Note

The app displays an OpenStreetMap basemap immediately. When the selected RGB
image contains GPS EXIF metadata plus enough camera metadata to estimate a
ground footprint, the image overlay is placed around the GPS center. The app
prefers relative drone altitude when estimating footprint; plain GPS altitude is
usually height above sea level, so it is not used as above-ground altitude by
default. If the image has GPS but no usable footprint, the app places a
120-meter-wide fallback extent around the GPS point. If no GPS exists, it uses a
neutral Philippines-centered demo extent.

If the scale is still wrong, enter a manual ground footprint width and height in
meters, then click Apply Scale.

This is still an approximation for single JPEG/PNG images. True orthomosaic
alignment should use GeoTIFF geotransform/CRS metadata. Also, yaw/direction is
shown in the metadata summary when available, but the experimental Leaflet image
overlay is axis-aligned and does not rotate the image yet.

## Metrics

The app reports:

- detections per class
- total detections
- inference/API runtime
- total model time spent
- tiles processed and failed tile count
- average runtime per tile
- raw detections before deduplication
- final detections after deduplication
- Gemini token usage when returned by the API
- rough Gemini Flash token cost estimate when usage data is available
- simple point-overlap similarity between Gemini and MUSA-AI within a pixel radius

## Tiled Benchmarking

Both Gemini and MUSA-AI run through the image in 512x512 overlapping tiles with
the same overlap value used by the Musa AI GeoTIFF mapping pipeline. A loading
dialog shows the current tile number and pixel bounds, and the active tile is
drawn on the map while inference is running.

The temporary tile limit is useful for fast API and speed tests. For example,
if the image produces 120 tiles and the limit is set to 2, each selected
pipeline processes only the first 2 tiles, deduplicates those partial outputs,
and displays the partial result immediately.

The Gemini default model is `gemini-2.5-flash-lite`. `Run Both` starts Gemini
and MUSA-AI simultaneously. The loading dialog includes a Cancel button; when
cancelled, each worker stops after the current tile/request and returns whatever
partial detections have already been collected.

For the local MUSA-AI pipeline, the disease model can be a normal YOLOv8 detect
model. The experiment includes a "Keep disease points outside leaves" option so
the disease detector can be evaluated independently from the leaf segmentation
filter.

The local MUSA-AI pipeline can also run the leaf and disease models separately:

- `Leaf + Disease`: runs both local models and outputs all selected local classes.
- `Leaf only`: requires only the YOLOv8-seg leaf model path and outputs leaf center points.
- `Disease only`: requires only the YOLOv8 disease model path and outputs disease center points. In this mode, disease points are kept even when the leaf filter checkbox is off because no leaf polygons are available for filtering.

The same mode is also applied to Gemini. In `Leaf only`, Gemini is prompted for
only `full_leaf` and `cut_leaf`. In `Disease only`, Gemini is prompted for only
`black_sigatoka` and `panama`, and parsed Gemini results outside the selected
class group are discarded.

During `Run Both`, the map shows independent live tile outlines for each
pipeline: Gemini uses a cyan inset outline and MUSA-AI uses an orange outline.
The loading popup keeps a visible status line for each pipeline and marks a
pipeline as done, cancelled, or failed while the other pipeline continues.

The app remembers the selected leaf model path, disease model path, Gemini API
key, Gemini model name, local model mode, confidence, device, tile limit, and
disease filtering option through local Qt settings so they are restored the next
time the experimental app opens.

## Architecture

The implementation is intentionally separate from the production app:

- `experimental_ai_comparison/app.py`: PyQt6 UI and workflow orchestration.
- `experimental_ai_comparison/map_view.html`: Leaflet image-map viewer.
- `experimental_ai_comparison/gemini_provider.py`: Gemini REST provider.
- `experimental_ai_comparison/musa_inference.py`: local YOLO adapter.
- `experimental_ai_comparison/comparison.py`: point matching and metrics.
- `experimental_ai_comparison/models.py`: shared data models.
- `experimental_ai_comparison/workers.py`: background inference threads.
