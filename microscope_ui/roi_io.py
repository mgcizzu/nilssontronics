from __future__ import annotations

import json
from pathlib import Path

from .models import FocusPoint, ScanRegion


ROI_FILE_VERSION = 1


def save_roi_set(
    path: Path | str,
    regions: tuple[ScanRegion, ...],
    focus_points: tuple[FocusPoint, ...],
) -> None:
    payload = {
        "format": "microscope-scan-console-roi",
        "version": ROI_FILE_VERSION,
        "coordinate_system": "stage_micrometres",
        "regions": [
            {
                "shape": region.shape,
                "vertices_um": [[x, y] for x, y in region.vertices],
            }
            for region in regions
        ],
        "focus_points": [
            {"x_um": point.x_um, "y_um": point.y_um, "z_um": point.z_um}
            for point in focus_points
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_roi_set(path: Path | str) -> tuple[tuple[ScanRegion, ...], tuple[FocusPoint, ...]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "microscope-scan-console-roi":
        raise ValueError("This is not a Microscope Scan Console ROI file")
    if payload.get("version") != ROI_FILE_VERSION:
        raise ValueError(f"Unsupported ROI file version: {payload.get('version')}")
    if payload.get("coordinate_system") != "stage_micrometres":
        raise ValueError("ROI coordinates must be stage micrometres")

    regions = []
    for item in payload.get("regions", []):
        try:
            points = tuple((float(point[0]), float(point[1])) for point in item["vertices_um"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError("An ROI contains invalid vertices") from exc
        if len(points) < 3:
            raise ValueError("Each ROI needs at least three vertices")
        if item.get("shape") == "rectangle" and len(points) == 4:
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            regions.append(
                ScanRegion(
                    min(x_values),
                    min(y_values),
                    max(x_values) - min(x_values),
                    max(y_values) - min(y_values),
                )
            )
        else:
            regions.append(ScanRegion.from_polygon(points))

    focus_points = []
    for item in payload.get("focus_points", []):
        try:
            focus_points.append(
                FocusPoint(float(item["x_um"]), float(item["y_um"]), float(item["z_um"]))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A focus point contains invalid coordinates") from exc
    return tuple(regions), tuple(focus_points)
