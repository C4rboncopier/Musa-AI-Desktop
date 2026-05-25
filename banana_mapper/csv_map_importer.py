from __future__ import annotations

import csv
from pathlib import Path


class CsvImportError(Exception):
    """Custom exception raised for invalid CSV import structure or data."""
    pass


def parse_csv_coordinates(file_path: str | Path) -> list[dict]:
    """Parse coordinates from a CSV file and extract Name, Latitude, and Longitude.

    Validates that:
    - The required columns 'Name', 'Latitude', and 'Longitude' are present.
    - All rows have valid numeric values for Latitude and Longitude.
    - Latitude is within the range [-90.0, 90.0].
    - Longitude is within the range [-180.0, 180.0].

    Raises:
        CsvImportError: If validation fails or columns are missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise CsvImportError("The CSV file does not exist.")

    records = []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            # We use DictReader to handle columns by header names
            reader = csv.DictReader(handle)
            
            # Read first row or check fieldnames
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise CsvImportError("The CSV file is empty or has no header row.")

            # Validate that the exact columns are present
            required_cols = {"Name", "Latitude", "Longitude"}
            missing_cols = required_cols - set(fieldnames)
            if missing_cols:
                raise CsvImportError(
                    f"Missing required column(s): {', '.join(sorted(missing_cols))}. "
                    f"Found columns: {', '.join(fieldnames)}"
                )

            for line_num, row in enumerate(reader, start=2):
                # Skip completely empty rows
                if not any(row.values()):
                    continue

                name_val = row.get("Name")
                lat_val = row.get("Latitude")
                lon_val = row.get("Longitude")

                if name_val is None or lat_val is None or lon_val is None:
                    raise CsvImportError(
                        f"Line {line_num}: Row has missing column values."
                    )

                name = name_val.strip()
                if not name:
                    raise CsvImportError(
                        f"Line {line_num}: 'Name' column cannot be empty."
                    )

                # Validate Latitude
                try:
                    lat = float(lat_val.strip())
                except ValueError:
                    raise CsvImportError(
                        f"Line {line_num}: Invalid float format for Latitude: '{lat_val}'"
                    )

                if not (-90.0 <= lat <= 90.0):
                    raise CsvImportError(
                        f"Line {line_num}: Latitude out of bounds [-90, 90]: {lat}"
                    )

                # Validate Longitude
                try:
                    lon = float(lon_val.strip())
                except ValueError:
                    raise CsvImportError(
                        f"Line {line_num}: Invalid float format for Longitude: '{lon_val}'"
                    )

                if not (-180.0 <= lon <= 180.0):
                    raise CsvImportError(
                        f"Line {line_num}: Longitude out of bounds [-180, 180]: {lon}"
                    )

                records.append({
                    "name": name,
                    "latitude": lat,
                    "longitude": lon
                })

    except CsvImportError:
        # Re-raise our domain validation errors
        raise
    except Exception as exc:
        raise CsvImportError(f"Failed to read CSV file: {exc}") from exc

    if not records:
        raise CsvImportError("No valid coordinate rows found in the CSV file.")

    return records
