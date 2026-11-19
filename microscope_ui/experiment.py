"""Experiment → Cycle model for cyclic imaging, persisted as ``experiment.json``.

An experiment fixes the objective, the ROI set, the overlap and therefore the
tile list (indices, grid rows/columns and acquisition order) once. Every cycle
replays exactly that tile list. Between cycles the operator may:

* translate the whole ROI/focus-point set (slide re-mounted slightly shifted);
* refocus each stored focus point (tissue Z changes after re-mounting);
* choose channels, exposures and powers freely (decoding happens later).

Layout::

    <output>/<project>/<sample>_<timestamp>/
        experiment.json          this model (atomically replaced, fsynced)
        cycle_01/                one folder per cycle (see mosaic.py)
        cycle_02/
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Literal

from .durability import durable_fsync, durable_fsync_directory
from .models import (
    AcquisitionOptions,
    ChannelSetting,
    FocusPoint,
    ScanRegion,
    Tile,
    ZSettings,
)

EXPERIMENT_FORMAT = "nilssontronics-experiment"
EXPERIMENT_VERSION = 1
EXPERIMENT_FILE = "experiment.json"

CycleStatus = Literal["running", "paused", "complete", "aborted", "failed"]
RESUMABLE = ("running", "paused", "aborted", "failed")


@dataclass(frozen=True)
class CycleRecord:
    number: int
    status: CycleStatus
    channels: tuple[ChannelSetting, ...]
    z: ZSettings
    focus_points: tuple[FocusPoint, ...]
    offset_x_um: float = 0.0
    offset_y_um: float = 0.0
    single_z_um: float | None = None
    started_at: str = ""
    finished_at: str | None = None
    error: str | None = None
    tiles_completed: int = 0
    flat_field: str | None = None
    options: AcquisitionOptions | None = None

    @property
    def folder_name(self) -> str:
        return f"cycle_{self.number:02d}"


@dataclass
class Experiment:
    root: Path
    project: str
    sample: str
    objective_id: str
    overlap_x_percent: float
    overlap_y_percent: float
    regions: tuple[ScanRegion, ...]
    tiles: tuple[Tile, ...]
    options: AcquisitionOptions
    base_focus_points: tuple[FocusPoint, ...] = ()
    created_at: str = field(default_factory=lambda: _now())
    cycles: list[CycleRecord] = field(default_factory=list)
    # Image calibration when cycle 1 started (pixel size, camera, orientation).
    # Every later cycle and every resume must match it; empty for old files.
    calibration: dict = field(default_factory=dict)

    # -- creation & persistence ----------------------------------------------
    @classmethod
    def create(
        cls,
        output_root: Path,
        project: str,
        sample: str,
        objective_id: str,
        overlap_x_percent: float,
        overlap_y_percent: float,
        regions: tuple[ScanRegion, ...],
        tiles: tuple[Tile, ...],
        options: AcquisitionOptions,
        focus_points: tuple[FocusPoint, ...] = (),
        calibration: dict | None = None,
    ) -> "Experiment":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(output_root) / slug(project) / f"{slug(sample)}_{timestamp}"
        root.mkdir(parents=True, exist_ok=False)
        experiment = cls(
            root=root,
            project=project,
            sample=sample,
            objective_id=objective_id,
            overlap_x_percent=overlap_x_percent,
            overlap_y_percent=overlap_y_percent,
            regions=regions,
            tiles=tiles,
            options=options,
            base_focus_points=focus_points,
            calibration=dict(calibration or {}),
        )
        experiment.save()
        return experiment

    @classmethod
    def load(cls, path: Path | str) -> "Experiment":
        path = Path(path)
        if path.is_dir():
            path = path / EXPERIMENT_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != EXPERIMENT_FORMAT:
            raise ValueError("This is not an experiment file")
        if data.get("version") != EXPERIMENT_VERSION:
            raise ValueError(f"Unsupported experiment version: {data.get('version')}")
        return cls(
            root=path.parent,
            project=data["project"],
            sample=data["sample"],
            objective_id=data["objective_id"],
            overlap_x_percent=data["overlap_x_percent"],
            overlap_y_percent=data["overlap_y_percent"],
            regions=tuple(_region_from_dict(item) for item in data["regions"]),
            tiles=tuple(Tile(**item) for item in data["tiles"]),
            options=AcquisitionOptions(**data.get("options", {})),
            base_focus_points=tuple(FocusPoint(**item) for item in data.get("base_focus_points", [])),
            created_at=data.get("created_at", ""),
            cycles=[_cycle_from_dict(item) for item in data.get("cycles", [])],
            calibration=dict(data.get("calibration", {})),
        )

    def to_dict(self) -> dict:
        return {
            "format": EXPERIMENT_FORMAT,
            "version": EXPERIMENT_VERSION,
            "project": self.project,
            "sample": self.sample,
            "created_at": self.created_at,
            "updated_at": _now(),
            "objective_id": self.objective_id,
            "overlap_x_percent": self.overlap_x_percent,
            "overlap_y_percent": self.overlap_y_percent,
            "calibration": self.calibration,
            "coordinate_system": "stage_micrometres (cycle 1 frame; per-cycle offsets below)",
            "regions": [
                {"shape": region.shape, "vertices_um": [list(v) for v in region.vertices]}
                for region in self.regions
            ],
            "tiles": [asdict(tile) for tile in self.tiles],
            "options": asdict(self.options),
            "base_focus_points": [asdict(point) for point in self.base_focus_points],
            "cycles": [_cycle_to_dict(cycle) for cycle in self.cycles],
        }

    def save(self) -> Path:
        path = self.root / EXPERIMENT_FILE
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.flush()
            durable_fsync(handle.fileno())
        temporary.replace(path)
        durable_fsync_directory(self.root)
        return path

    # -- cycles ----------------------------------------------------------------
    @property
    def next_cycle_number(self) -> int:
        return len(self.cycles) + 1

    def cycle(self, number: int) -> CycleRecord:
        return self.cycles[number - 1]

    def cycle_folder(self, number: int) -> Path:
        return self.root / f"cycle_{number:02d}"

    def add_cycle(self, record: CycleRecord) -> CycleRecord:
        if record.number != self.next_cycle_number:
            raise ValueError(f"Expected cycle {self.next_cycle_number}, got {record.number}")
        unfinished = self.resumable_cycle()
        if unfinished is not None:
            raise ValueError(
                f"Cycle {unfinished.number} is not complete; resume or finish it first"
            )
        self.cycles.append(record)
        try:
            self.save()
        except BaseException:
            self.cycles.pop()  # memory never runs ahead of experiment.json
            raise
        return record

    def update_cycle(self, number: int, **changes) -> CycleRecord:
        """Change a cycle record; the in-memory record changes only if the save succeeds."""

        previous = self.cycles[number - 1]
        record = replace(previous, **changes)
        self.cycles[number - 1] = record
        try:
            self.save()
        except BaseException:
            self.cycles[number - 1] = previous
            raise
        return record

    def resumable_cycle(self) -> CycleRecord | None:
        for record in reversed(self.cycles):
            if record.status in RESUMABLE:
                return record
        return None

    def tiles_for_cycle(self, number: int) -> tuple[Tile, ...]:
        record = self.cycle(number)
        return tuple(tile.translated(record.offset_x_um, record.offset_y_um) for tile in self.tiles)

    def regions_with_offset(self, dx_um: float, dy_um: float) -> tuple[ScanRegion, ...]:
        return tuple(region.translated(dx_um, dy_um) for region in self.regions)

    def offset_of(self, regions: tuple[ScanRegion, ...], tolerance_um: float = 1e-3) -> tuple[float, float]:
        """Return the pure XY translation from the experiment ROIs to ``regions``.

        Raises ValueError if the regions differ by anything other than one
        common translation (ROIs must stay fixed across cycles).
        """

        if len(regions) != len(self.regions):
            raise ValueError("The ROI set changed; only moving the whole set is allowed between cycles")
        if not regions:
            return (0.0, 0.0)
        reference = self.regions[0].vertices[0]
        moved = regions[0].vertices[0]
        dx, dy = moved[0] - reference[0], moved[1] - reference[1]
        for original, current in zip(self.regions, regions):
            if len(original.vertices) != len(current.vertices):
                raise ValueError("The ROI set changed; only moving the whole set is allowed between cycles")
            for (ox, oy), (cx, cy) in zip(original.vertices, current.vertices):
                if abs(ox + dx - cx) > tolerance_um or abs(oy + dy - cy) > tolerance_um:
                    raise ValueError(
                        "The ROI set changed shape; only moving the whole set is allowed between cycles"
                    )
        return (dx, dy)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "unnamed"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _region_from_dict(item: dict) -> ScanRegion:
    points = tuple((float(x), float(y)) for x, y in item["vertices_um"])
    if item.get("shape") == "rectangle" and len(points) == 4:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return ScanRegion(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    return ScanRegion.from_polygon(points)


def _cycle_to_dict(cycle: CycleRecord) -> dict:
    data = asdict(cycle)
    data["folder"] = cycle.folder_name
    return data


def _cycle_from_dict(item: dict) -> CycleRecord:
    item = dict(item)
    item.pop("folder", None)
    item["channels"] = tuple(ChannelSetting(**c) for c in item.get("channels", []))
    item["z"] = ZSettings(**item.get("z", {}))
    item["focus_points"] = tuple(FocusPoint(**p) for p in item.get("focus_points", []))
    if item.get("options") is not None:
        item["options"] = AcquisitionOptions(**item["options"])
    return CycleRecord(**item)
