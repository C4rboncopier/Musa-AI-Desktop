# Musa AI Release and Deployment Guide

This guide describes the recommended path for turning Musa AI into a Windows
desktop app that users can download, install, update, and configure with their
own API keys.

## Recommended Packaging Model

Use a two-step release pipeline:

1. Build the application with PyInstaller as a one-folder Windows app.
2. Wrap the PyInstaller output folder in an installer with Inno Setup or NSIS.

Recommended public installer name:

```text
MusaAI-Setup-1.0.0.exe
```

Recommended installed app layout:

```text
C:\Program Files\Musa AI\
  Musa AI.exe
  _internal\
  banana_mapper\
  vendor\
```

Recommended user data layout:

```text
%LOCALAPPDATA%\Musa AI\
  banana_mapper.sqlite3
  google_maps.env
  cache\
```

The app already stores packaged-build user data under `%LOCALAPPDATA%\Musa AI`.
Development builds still use `.banana_mapper` in the workspace unless
`BANANA_MAPPER_HOME` is set.

## Secret and API Key Policy

Do not ship a shared Google Maps Platform API key inside the installer.

Musa AI should use a user-provided API key:

1. The user creates their own Google Maps Platform key.
2. The user opens Settings > Map Services.
3. The user pastes and saves their own key.
4. Musa AI stores it locally as:

```text
%LOCALAPPDATA%\Musa AI\google_maps.env
```

This keeps your billing account safe. The key is local to the user and is not
included in Git, the installer, or the source code.

Recommended Google API restrictions:

- API restriction: `Map Tiles API`
- Budget alerts enabled
- Quotas configured for daily or monthly protection
- Application restriction where practical for the deployment environment

## Build Machine Setup

Use a clean Windows build machine or Windows VM.

Install:

- Python 3.10 or newer
- Git
- Visual C++ Redistributable build/runtime tools if required by geospatial libs
- PyInstaller
- Inno Setup or NSIS

Create a clean virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install pyinstaller
```

Run a local sanity check:

```powershell
python main.py
```

## PyInstaller Build

Build as a one-folder app. Avoid `--onefile` for this project because PyQt
WebEngine, rasterio/GDAL, PyTorch, and model/runtime assets are large and more
stable as a folder bundle.

Example command:

```powershell
pyinstaller main.py `
  --name "Musa AI" `
  --windowed `
  --onedir `
  --clean `
  --add-data "banana_mapper\map_view.html;banana_mapper" `
  --add-data "banana_mapper\vendor;banana_mapper\vendor"
```

Expected output:

```text
dist\Musa AI\Musa AI.exe
```

Test the app from `dist\Musa AI\` before creating the installer.

## AI Runtime Strategy

PyTorch and CUDA packages are large and hardware-sensitive. The safest release
strategy is:

1. Ship the core GUI app with normal dependencies.
2. Let the Hardware Check page detect the user's CPU/GPU status.
3. Offer setup guidance for CPU or NVIDIA GPU PyTorch.
4. Keep model weights outside Git and outside the base installer unless you have
   redistribution rights and a clear model versioning policy.

Release options:

- CPU-only package: easiest and smaller, but slower AI processing.
- GPU runtime package: larger and more fragile, but faster on supported NVIDIA
  systems.
- Separate runtime installers: best long-term option for production.

For GPU support, users need a compatible NVIDIA driver and the correct PyTorch
build selected from the official PyTorch install selector.

## Installer Creation

Use Inno Setup or NSIS to package the entire PyInstaller output folder.

Installer should:

- Install to `C:\Program Files\Musa AI`.
- Create a Start Menu shortcut.
- Optionally create a Desktop shortcut.
- Never write API keys during installation.
- Never install generated project output into the app folder.
- Preserve `%LOCALAPPDATA%\Musa AI` during uninstall unless the user explicitly
  chooses to remove user data.

Installer output:

```text
MusaAI-Setup-1.0.0.exe
```

## Pre-Release Checklist

Before shipping a release:

1. Update the version number in release notes and installer metadata.
2. Create a clean virtual environment.
3. Install requirements from scratch.
4. Run the app locally from source.
5. Build with PyInstaller.
6. Run the app from `dist\Musa AI`.
7. Test on a clean Windows machine or VM.
8. Create a new project.
9. Import a GeoTIFF.
10. Switch between OpenStreetMap and Google Satellite.
11. Confirm Google Satellite prompts for a user key when none is configured.
12. Save a user Google API key in Settings and verify Google Satellite loads.
13. Run Hardware Check.
14. Verify project outputs are written to `SystemOutput` or the configured output
    root.
15. Verify no `.env`, `google_maps.env`, model weights, GeoTIFFs, or output files
    are included in the installer unless intentionally bundled.

## Updating to a New Version

Recommended version workflow:

1. Create a release branch:

```powershell
git checkout -b release/1.1.0
```

2. Update release notes or changelog.
3. Run tests and manual checks.
4. Build a fresh PyInstaller bundle.
5. Create a fresh installer with the new version number.
6. Install over the previous version on a test machine.
7. Confirm existing user data still loads from:

```text
%LOCALAPPDATA%\Musa AI
```

8. Confirm existing projects, preferences, and user-provided Google API key still
   work.
9. Confirm uninstall/reinstall does not accidentally delete user projects or
   outputs.
10. Tag the release:

```powershell
git tag v1.1.0
git push origin v1.1.0
```

## Updating Dependencies

When updating dependencies:

1. Update `requirements.txt`.
2. Rebuild the virtual environment from scratch.
3. Run source app tests.
4. Run PyInstaller build.
5. Test on a clean machine.
6. Pay special attention to:
   - PyQt6 and PyQt6-WebEngine compatibility
   - rasterio/GDAL DLL packaging
   - PyTorch package size and CUDA compatibility
   - ultralytics model loading

Avoid changing PyTorch/CUDA versions in a patch release unless the release is
specifically meant to update the AI runtime.

## Rollback Plan

Keep the previous installer available:

```text
MusaAI-Setup-1.0.0.exe
MusaAI-Setup-1.1.0.exe
```

If a release fails:

1. Ask users to uninstall the new app only.
2. Preserve `%LOCALAPPDATA%\Musa AI`.
3. Reinstall the previous version.
4. Confirm projects and settings load normally.

## Notes for Production Hardening

Before a public release, consider adding:

- App version display in Settings.
- Automatic update checking.
- Signed Windows installer and executable.
- Crash log collection with user consent.
- Separate AI runtime manager.
- A clearer Google Maps key validation flow.
- Export/import backup for project metadata.
