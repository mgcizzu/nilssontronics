from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Literal


Immersion = Literal["air", "oil", "water", "silicone", "other"]
ZMode = Literal["single", "stack", "focus_map", "surface_stack"]
TileOrder = Literal["raster", "serpentine"]
FocusMethod = Literal["linear", "plane"]

Z_MODES: tuple[str, ...] = ("single", "stack", "focus_map", "surface_stack")
TILE_ORDERS: tuple[str, ...] = ("raster", "serpentine")
FOCUS_METHODS: tuple[str, ...] = ("linear", "plane")


def endpoint_inclusive_plane_count(distance_um: float, step_um: float) -> int:
    """Planes needed to cover ``distance_um`` with spacing no larger than ``step_um``.

    This is the single definition used by the optics helpers, the plan model and
    the acquisition engine so that estimates and real acquisitions always agree.
    """

    if step_um <= 0:
        raise ValueError("Z step must be greater than zero")
    distance_um = abs(distance_um)
    if distance_um == 0:
        return 1
    return math.ceil(distance_um / step_um - 1e-9) + 1


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

    def contains_xy(self, x_um: float, y_um: float) -> bool:
        return (
            self.x_min_um <= x_um <= self.x_max_um
            and self.y_min_um <= y_um <= self.y_max_um
        )

    def contains_z(self, z_um: float) -> bool:
        return self.z_min_um <= z_um <= self.z_max_um


@dataclass(frozen=True)
class Recipe:
    name: str
    objective_id: str
    channels: tuple[str, ...]
    z_mode: ZMode = "single"


@dataclass(frozen=True)
class Timing:
    """Hardware timing used only for estimates (seconds unless stated)."""

    stage_speed_um_per_s: float = 5000.0
    stage_settle_s: float = 0.10
    z_move_s: float = 0.03
    channel_switch_s: float = 0.12
    camera_readout_ms: float = 25.0
    distortion_correction_ms: float = 100.0


@dataclass(frozen=True)
class CameraOrientation:
    """How camera pixels map onto stage axes.

    Applied by the hardware backend so every frame handed to the rest of the
    program already has +column = +stage X and +row = +stage Y.
    """

    flip_x: bool = False
    flip_y: bool = False
    transpose: bool = False


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
    recipes: tuple[Recipe, ...] = ()
    timing: Timing = field(default_factory=Timing)
    camera_orientation: CameraOrientation = field(default_factory=CameraOrientation)
    approach_offset_um: float = 0.0
    simulator_frame_px: int = 256
    fsync_tiles: bool = True
    hardware: dict = field(default_factory=dict)
    # Simulator extras (calibration slide, simulated lens distortion); see backend.py.
    simulator_options: dict = field(default_factory=dict)
    # Folder holding calibration files (distortion corrections); set by load_config.
    calibration_dir: Path | None = None

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

    def recipe(self, name: str) -> Recipe:
        try:
            return next(item for item in self.recipes if item.name == name)
        except StopIteration as exc:
            raise KeyError(f"Unknown recipe: {name}") from exc

    def fov_um(self, objective_id: str) -> tuple[float, float]:
        objective = self.objective(objective_id)
        width_px, height_px = self.oriented_camera_size_px
        return (
            width_px * objective.pixel_size_um,
            height_px * objective.pixel_size_um,
        )

    @property
    def oriented_camera_size_px(self) -> tuple[int, int]:
        """Frame width/height after the configured stage-axis orientation."""

        if self.camera_orientation.transpose:
            return (self.camera_height_px, self.camera_width_px)
        return (self.camera_width_px, self.camera_height_px)

    def safe_z_range(self, objective_id: str) -> tuple[float, float]:
        objective = self.objective(objective_id)
        return (
            max(self.stage_limits.z_min_um, objective.safe_z_min_um),
            min(self.stage_limits.z_max_um, objective.safe_z_max_um),
        )


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
class Tile:
    """One field of view in the experiment's fixed tile grid.

    ``index`` is the acquisition order (0-based) and never changes between
    cycles; ``row``/``col`` are positions in the shared grid.
    """

    index: int
    row: int
    col: int
    x_um: float
    y_um: float

    def translated(self, dx_um: float, dy_um: float) -> "Tile":
        return Tile(self.index, self.row, self.col, self.x_um + dx_um, self.y_um + dy_um)

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x_um, self.y_um)


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
    # Begin/End must be captured explicitly from live Z; a plan built without
    # them is never valid for a Z-stack.
    begin_captured: bool = False
    end_captured: bool = False

    @property
    def plane_count(self) -> int:
        if self.mode not in ("stack", "surface_stack"):
            return 1
        if self.mode == "stack" and (not self.begin_captured or not self.end_captured):
            return 0
        if self.step_um <= 0:
            return 0
        if self.mode == "surface_stack":
            if self.bottom_um > 0 or self.top_um < 0:
                return 0
            below = endpoint_inclusive_plane_count(abs(self.bottom_um), self.step_um)
            above = endpoint_inclusive_plane_count(abs(self.top_um), self.step_um)
            return below + above - 1  # the surface plane belongs to both halves
        return endpoint_inclusive_plane_count(self.top_um - self.bottom_um, self.step_um)

    def stack_positions(self) -> tuple[float, ...]:
        """Absolute positions, or surface-relative offsets, from bottom to top."""

        count = self.plane_count
        if self.mode not in ("stack", "surface_stack") or count == 0:
            return ()
        if self.mode == "surface_stack":
            below_count = endpoint_inclusive_plane_count(abs(self.bottom_um), self.step_um)
            above_count = endpoint_inclusive_plane_count(abs(self.top_um), self.step_um)
            below_spacing = abs(self.bottom_um) / (below_count - 1) if below_count > 1 else 0.0
            above_spacing = abs(self.top_um) / (above_count - 1) if above_count > 1 else 0.0
            below = tuple(self.bottom_um + index * below_spacing for index in range(below_count))
            above = tuple(index * above_spacing for index in range(1, above_count))
            return below + above
        if count == 1:
            return (self.bottom_um,)
        spacing = (self.top_um - self.bottom_um) / (count - 1)
        return tuple(self.bottom_um + index * spacing for index in range(count))


@dataclass(frozen=True)
class AcquisitionOptions:
    """Optional behaviours, all off/conservative by default."""

    tile_order: TileOrder = "raster"
    # Approach every XY target from the same side (needs approach_offset_um > 0
    # in the configuration) to cancel stage backlash.
    consistent_approach: bool = False
    # Close the light path during stage and Z moves (per-frame shutter).
    shutter_per_frame: bool = True
    # Estimate a flat-field profile per channel after each cycle (non-destructive).
    flat_field: bool = False
    # Resample every plane with the objective's calibrated distortion correction
    # before saving (fixed by cycle 1). Optionally keep the camera frames too.
    distortion_correction: bool = False
    keep_uncorrected: bool = True
    focus_method: FocusMethod = "linear"


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
    options: AcquisitionOptions = field(default_factory=AcquisitionOptions)

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
    actual_overlap_x_percent: float = 0.0
    actual_overlap_y_percent: float = 0.0

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
