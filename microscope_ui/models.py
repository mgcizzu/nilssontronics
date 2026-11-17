from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Literal


Immersion = Literal["air", "oil", "water", "silicone", "other"]
ZMode = Literal["single", "stack", "focus_map"]


@dataclass(frozen=True)
class Objective:
    id: str
    name: str
    magnification: float
    numerical_aperture: float
    immersion: Immersion
    immersion_refractive_index: float
    turret_position: int
    pixel_size_um: float
    safe_z_min_um: float
    safe_z_max_um: float


@dataclass(frozen=True)
class Channel:
    id: str
    name: str
    color: str
    emission_wavelength_nm: float
    default_exposure_ms: float
    max_exposure_ms: float
    default_power_percent: float
    max_power_percent: float
    manual_prompt: str | None = None


@dataclass(frozen=True)
class StageLimits:
    x_min_um: float
    x_max_um: float
    y_min_um: float
    y_max_um: float
    z_min_um: float
    z_max_um: float


@dataclass(frozen=True)
class MicroscopeConfig:
    microscope_name: str
    camera_width_px: int
    camera_height_px: int
    overlap_x_percent: float
    overlap_y_percent: float
    bytes_per_pixel: int
    objectives: tuple[Objective, ...]
    channels: tuple[Channel, ...]
    stage_limits: StageLimits
    writer_queue_frames: int = 4
    recipes: tuple[dict, ...] = ()

    def objective(self, objective_id: str) -> Objective:
        try:
            return next(item for item in self.objectives if item.id == objective_id)
        except StopIteration as exc:
            raise KeyError(f"Unknown objective: {objective_id}") from exc

    def channel(self, channel_id: str) -> Channel:
        try:
            return next(item for item in self.channels if item.id == channel_id)
        except StopIteration as exc:
            raise KeyError(f"Unknown channel: {channel_id}") from exc


@dataclass(frozen=True)
class ScanRegion:
    x_um: float
    y_um: float
    width_um: float
    height_um: float
    polygon_points_um: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_polygon(cls, points: tuple[tuple[float, float], ...]) -> "ScanRegion":
        if len(points) < 3:
            raise ValueError("A polygon region needs at least three points")
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        return cls(
            x_um=x_min,
            y_um=y_min,
            width_um=x_max - x_min,
            height_um=y_max - y_min,
            polygon_points_um=points,
        )

    @property
    def shape(self) -> str:
        return "polygon" if self.polygon_points_um else "rectangle"

    @property
    def vertices(self) -> tuple[tuple[float, float], ...]:
        if self.polygon_points_um:
            return self.polygon_points_um
        return (
            (self.x_um, self.y_um),
            (self.x_max_um, self.y_um),
            (self.x_max_um, self.y_max_um),
            (self.x_um, self.y_max_um),
        )

    @property
    def x_max_um(self) -> float:
        return self.x_um + self.width_um

    @property
    def y_max_um(self) -> float:
        return self.y_um + self.height_um

    def translated(self, dx_um: float, dy_um: float) -> "ScanRegion":
        if self.polygon_points_um:
            return ScanRegion.from_polygon(
                tuple((x + dx_um, y + dy_um) for x, y in self.polygon_points_um)
            )
        return ScanRegion(
            self.x_um + dx_um,
            self.y_um + dy_um,
            self.width_um,
            self.height_um,
        )


@dataclass(frozen=True)
class FocusPoint:
    x_um: float
    y_um: float
    z_um: float

    def translated_xy(self, dx_um: float, dy_um: float) -> "FocusPoint":
        return FocusPoint(self.x_um + dx_um, self.y_um + dy_um, self.z_um)


@dataclass(frozen=True)
class ChannelSetting:
    channel_id: str
    exposure_ms: float
    power_percent: float


@dataclass(frozen=True)
class ZSettings:
    mode: ZMode = "single"
    bottom_um: float = 0.0
    top_um: float = 0.0
    step_um: float = 1.0
    begin_captured: bool = True
    end_captured: bool = True

    @property
    def plane_count(self) -> int:
        if self.mode != "stack":
            return 1
        if not self.begin_captured or not self.end_captured or self.step_um <= 0:
            return 0
        distance = abs(self.top_um - self.bottom_um)
        if distance == 0:
            return 1
        return math.ceil(distance / self.step_um - 1e-9) + 1


@dataclass(frozen=True)
class AcquisitionPlan:
    project: str
    sample: str
    output_root: Path
    objective_id: str
    channels: tuple[ChannelSetting, ...]
    regions: tuple[ScanRegion, ...]
    z: ZSettings = field(default_factory=ZSettings)
    focus_points: tuple[FocusPoint, ...] = ()
    overlap_x_percent: float = 10.0
    overlap_y_percent: float = 10.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["output_root"] = str(self.output_root)
        return result


@dataclass(frozen=True)
class AcquisitionEstimate:
    tiles: int
    frames: int
    z_planes: int
    duration_seconds: float
    data_bytes: int

    @property
    def duration_label(self) -> str:
        seconds = max(0, round(self.duration_seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @property
    def data_label(self) -> str:
        value = float(self.data_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024
        return f"{value:.1f} TB"
