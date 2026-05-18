# Musa AI

Musa AI is a PyQt6 desktop geospatial AI mapping platform for drone
orthomosaic visualization, banana plantation monitoring, and disease detection
workflows.

## What Changed

- Project dashboard with recent projects, search, quick actions, status metrics,
  and project cards.
- Project cards support opening, editing the title/description, and deleting
  project metadata from the local database.
- Project-first workflow: create or open a project before importing datasets or
  running AI mapping.
- SQLite metadata store at `.banana_mapper/banana_mapper.sqlite3` by default.
- System-managed project outputs under `SystemOutput/<project-name>_<project-id>/`.
- Per-project GeoTIFF preview cache under each project's `cache/geotiff/` folder.
- Filepath-only persistence: GeoTIFFs, drone images, model weights, and exports
  remain on the local filesystem.
- Automatic missing-path validation for linked assets, AI models, exports, and
  result files.
- Project opening popup with automatic restoration of linked GeoTIFF previews and
  the latest saved analysis results.
- Working AI model manager for preferred leaf and disease model paths. New
  projects automatically inherit those preferred models when the files still
  exist.
- Modular code structure:
  - `banana_mapper/core/` for SQLite repositories and project models.
  - `banana_mapper/ui/` for dashboard, workspace, dialogs, and reusable widgets.
  - Existing GeoTIFF, AI detection, worker, and map bridge modules remain focused
    on processing and map integration.
- Professional GIS workspace with layer controls, metadata inspector, asset
  manager, AI workflow panel, coordinate display, map controls, logs, loading
  dialogs, and status messages.
- Heatmap rendering has been removed. AI visualization now focuses on GeoTIFF
  overlays, point-based detection markers, and layer visibility controls.
- AI mapping automatically writes each run into the project's managed output
  folder, with JSON, CSV, and Excel `.xlsx` result files available in the
  project Output Manager.

## Setup

Install Python 3.10+ and dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
python main.py
```

The app vendors Leaflet locally. OpenStreetMap basemap tiles still require an
internet connection.

## Workflow

1. Open the dashboard.
2. Create or select a project.
   Existing projects show an opening popup while linked data is restored.
   Use the project card actions to edit the title/description or delete the
   project record.
3. Use **Manage Models** to choose preferred AI model weights, or link models
   inside a specific project.
4. Link a stitched GeoTIFF, drone image folder, and AI model weights.
5. Import the GeoTIFF to render it on the GIS map.
6. Run AI Mapping from either the linked image folder or the current GeoTIFF.
   Outputs are routed automatically to the project's managed folder.
7. Review point layers, counts, metadata, outputs, and logs in the project workspace.

When a project is opened again, Musa AI automatically reloads its linked
GeoTIFF if the filepath still exists and restores the latest saved JSON/CSV
analysis result overlays.

## Local Database Policy

The SQLite database stores only metadata:

- Project names, descriptions, settings, and timestamps
- Local filepaths for GeoTIFFs, image folders, AI models, exports, and results
- Processing configuration metadata
- Detection result summaries and exported JSON/CSV paths

It does not store image binaries, GeoTIFF binaries, or AI model weights.
Generated result files are stored on disk in the system-managed `SystemOutput`
folder and are intentionally excluded from Git.
Reusable GeoTIFF preview caches are stored per project, not in a shared app
cache, so each project remains isolated and portable.
