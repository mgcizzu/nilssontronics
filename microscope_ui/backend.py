from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from .models import CameraOrientation, Channel, MicroscopeConfig, Objective


class HardwareError(RuntimeError):
    """A device refused a command or reported a fault."""


class StageLimitError(HardwareError):
    """A requested move lies outside the configured travel limits.

    Backends must raise this *before* commanding the stage. Moves are never
    silently clamped: an out-of-range target is a planning error.
    """


@dataclass(frozen=True)
class StagePosition:
    x_um: float
    y_um: float
    z_um: float


class MicroscopeBackend(Protocol):
    """Narrow hardware boundary used by the interface and acquisition engine.

    The UI never talks to Micro-Manager device properties directly. Frames
    returned by :meth:`snap` are already oriented so that +column is +stage X
    and +row is +stage Y.
    """

    name: str
    simulated: bool

    @property
    def position(self) -> StagePosition: ...

    @property
    def current_objective_id(self) -> str: ...

    def move_xy(self, x_um: float, y_um: float) -> StagePosition: ...

    def move_z(self, z_um: float) -> StagePosition: ...

    def change_objective(self, objective: Objective) -> None: ...

    def set_channel(self, channel: Channel, exposure_ms: float, power_percent: float) -> None: ...

    def set_shutter_per_frame(self, enabled: bool) -> None: ...

    def set_shutter_open(self, is_open: bool) -> None: ...

    def snap(self) -> np.ndarray: ...

    def close(self) -> None: ...


def check_xy(config: MicroscopeConfig, x_um: float, y_um: float) -> None:
    limits = config.stage_limits
    if not (math.isfinite(x_um) and math.isfinite(y_um)) or not limits.contains_xy(x_um, y_um):
        raise StageLimitError(
            f"XY target ({x_um:.1f}, {y_um:.1f}) µm is outside the stage limits "
            f"X {limits.x_min_um:g}…{limits.x_max_um:g}, Y {limits.y_min_um:g}…{limits.y_max_um:g}"
        )


def check_z(config: MicroscopeConfig, objective_id: str, z_um: float) -> None:
    low, high = config.safe_z_range(objective_id)
    if not math.isfinite(z_um) or not low <= z_um <= high:
        raise StageLimitError(
            f"Z target {z_um:.2f} µm is outside the safe range {low:g}…{high:g} µm "
            f"for objective {objective_id}"
        )


def apply_orientation(frame: np.ndarray, orientation: CameraOrientation) -> np.ndarray:
    """Map raw camera pixels onto stage axes (cheap views, no copies)."""

    if orientation.transpose:
        frame = frame.T
    if orientation.flip_x:
        frame = frame[:, ::-1]
    if orientation.flip_y:
        frame = frame[::-1, :]
    return frame


class SimulatedMicroscope:
    """Deterministic microscope simulator.

    Frames are rendered from a synthetic tissue defined in stage coordinates, so
    overlapping tiles show the same nuclei and stitching/registration code can
    be exercised. The simulated sample has a gently tilted focal surface; frames
    lose contrast when Z is away from it.
    """

    name = "Simulator"
    simulated = True

    def __init__(
        self,
        config: MicroscopeConfig,
        frame_px: int | None = None,
        distortion: dict | None = None,
        calibration_slide: dict | None = None,
    ) -> None:
        self._config = config
        options = config.simulator_options
        # Simulated optics (all off by default): radial lens distortion k1, a per-channel
        # magnification change per nm of emission away from 550 nm, and per-channel shifts.
        self.distortion = dict(options.get("distortion", {}) if distortion is None else distortion)
        # A fluorescent line grid slide somewhere on the stage, visible in every channel.
        slide = options.get("calibration_slide") if calibration_slide is None else calibration_slide
        self.calibration_slide = dict(slide) if slide else None
        self._position = StagePosition(0.0, 0.0, 0.0)
        self._objective_id = config.objectives[0].id
        self._channel: Channel | None = None
        self._exposure_ms = 10.0
        self._power_percent = 0.0
        self.frame_px = frame_px or config.simulator_frame_px
        self.snap_count = 0
        self.shutter_per_frame = True
        self.shutter_open = False
        self.light_on_during_moves = 0
        # Exposures with auto-shutter off and the shutter closed: no light reaches the sample.
        self.dark_snaps = 0
        self.move_log: list[tuple[str, float, float]] = []

    # -- state -------------------------------------------------------------
    @property
    def position(self) -> StagePosition:
        return self._position

    @property
    def current_objective_id(self) -> str:
        return self._objective_id

    @property
    def current_channel(self) -> Channel | None:
        return self._channel

    @property
    def power_percent(self) -> float:
        return self._power_percent

    # -- motion --------------------------------------------------------------
    def move_xy(self, x_um: float, y_um: float) -> StagePosition:
        check_xy(self._config, x_um, y_um)
        self._note_move()
        self._position = StagePosition(x_um, y_um, self._position.z_um)
        self.move_log.append(("xy", x_um, y_um))
        return self._position

    def move_z(self, z_um: float) -> StagePosition:
        check_z(self._config, self._objective_id, z_um)
        self._note_move()
        self._position = StagePosition(self._position.x_um, self._position.y_um, z_um)
        self.move_log.append(("z", z_um, 0.0))
        return self._position

    def change_objective(self, objective: Objective) -> None:
        # The real adapter moves Z to a clearance height, rotates the turret and
        # waits for the devices to settle before reporting ready.
        self._objective_id = objective.id
        low, high = self._config.safe_z_range(objective.id)
        z = min(high, max(low, self._position.z_um))
        self._position = StagePosition(self._position.x_um, self._position.y_um, z)

    # -- illumination & camera -----------------------------------------------
    def set_channel(self, channel: Channel, exposure_ms: float, power_percent: float) -> None:
        if not 0 < exposure_ms <= channel.max_exposure_ms:
            raise HardwareError(f"{channel.name}: exposure {exposure_ms} ms is out of range")
        if not 0 <= power_percent <= channel.max_power_percent:
            raise HardwareError(f"{channel.name}: power {power_percent}% is out of range")
        self._channel = channel
        self._exposure_ms = exposure_ms
        self._power_percent = power_percent

    def set_shutter_per_frame(self, enabled: bool) -> None:
        self.shutter_per_frame = enabled
        if enabled:
            self.shutter_open = False

    def set_shutter_open(self, is_open: bool) -> None:
        self.shutter_open = is_open

    def _note_move(self) -> None:
        if self.shutter_open:
            self.light_on_during_moves += 1

    def snap(self) -> np.ndarray:
        self.snap_count += 1
        if not self.shutter_per_frame and not self.shutter_open:
            self.dark_snaps += 1
        return self.render_frame(self._position, self._channel)

    def close(self) -> None:
        pass

    # -- synthetic sample ----------------------------------------------------
    def sample_focus_z(self, x_um: float, y_um: float) -> float:
        """Z of best focus for the simulated section at an XY position."""

        return 0.0004 * x_um - 0.0003 * y_um + 3.0 * math.sin(x_um / 4000.0)

    def render_frame(self, position: StagePosition, channel: Channel | None) -> np.ndarray:
        size = self.frame_px
        objective = self._config.objective(self._objective_id)
        fov_w, fov_h = self._config.fov_um(self._objective_id)
        xs = position.x_um + (np.arange(size) + 0.5 - size / 2) * (fov_w / size)
        ys = position.y_um + (np.arange(size) + 0.5 - size / 2) * (fov_h / size)
        gx, gy = np.meshgrid(xs, ys)
        if self._distorted(channel):
            gx, gy = self._distorted_sample_points(position, channel, size, fov_w, fov_h)
        signal = (self._exposure_ms * max(self._power_percent, 1.0)) * 4.0
        slide = self.calibration_slide
        if slide is not None:
            half = float(slide.get("size_um", 8000.0)) / 2
            sx, sy = float(slide.get("x_um", 30000.0)), float(slide.get("y_um", 0.0))
            if abs(position.x_um - sx) < half and abs(position.y_um - sy) < half:
                return self._render_grid(gx - sx, gy - sy, position, objective, signal, slide)

        spacing = 14.0
        cell_x = np.floor(gx / spacing)
        cell_y = np.floor(gy / spacing)
        seed = channel.id if channel else "none"
        channel_hash = sum(ord(c) for c in seed) % 97
        h = _hash2(cell_x, cell_y, 11)
        jitter_x = (h % 1000) / 1000.0
        jitter_y = (_hash2(cell_x, cell_y, 29) % 1000) / 1000.0
        center_x = (cell_x + 0.2 + 0.6 * jitter_x) * spacing
        center_y = (cell_y + 0.2 + 0.6 * jitter_y) * spacing
        distance_sq = (gx - center_x) ** 2 + (gy - center_y) ** 2
        nuclei = np.exp(-distance_sq / (2 * 3.2**2))
        if channel is not None and channel.id != "dapi":
            positive = (_hash2(cell_x, cell_y, channel_hash) % 5) == 0
            nuclei = nuclei * positive
        # Tissue mask: a large soft blob so section edges are visible.
        tissue = 0.5 + 0.5 * np.tanh((9000.0 - np.hypot(gx * 0.8, gy * 1.2)) / 800.0)

        defocus = position.z_um - self.sample_focus_z(position.x_um, position.y_um)
        depth_of_field = max(0.5, 2.0 * objective.immersion_refractive_index * 0.55 / objective.numerical_aperture**2)
        sharpness = math.exp(-((defocus / (3 * depth_of_field)) ** 2))
        image = 100.0 + tissue * (signal * (0.15 + 0.85 * sharpness) * nuclei + signal * 0.05)
        return np.clip(image, 0, 65535).astype(np.uint16)

    # -- simulated optics and calibration slide -------------------------------------
    def _distorted(self, channel: Channel | None) -> bool:
        d = self.distortion
        if not d:
            return False
        shift = (d.get("channel_shift_px") or {}).get(channel.id if channel else "", (0, 0))
        return bool(d.get("k1", 0.0) or d.get("chromatic_scale_per_nm", 0.0) or any(shift))

    def _distorted_sample_points(self, position, channel, size, fov_w, fov_h):
        """Stage points seen by each camera pixel through the simulated optics."""

        d = self.distortion
        shift_x, shift_y = (d.get("channel_shift_px") or {}).get(channel.id if channel else "", (0, 0))
        wavelength = channel.emission_wavelength_nm if channel else 550.0
        scale = 1.0 + float(d.get("chromatic_scale_per_nm", 0.0)) * (wavelength - 550.0)
        radius = math.hypot(size, size) / 2
        cols = np.arange(size) - (size - 1) / 2 - float(shift_x)
        rows = np.arange(size) - (size - 1) / 2 - float(shift_y)
        u, v = np.meshgrid(cols / radius, rows / radius)
        radial = (1.0 + float(d.get("k1", 0.0)) * (u * u + v * v)) / scale
        gx = position.x_um + u * radial * radius * (fov_w / size)
        gy = position.y_um + v * radial * radius * (fov_h / size)
        return gx, gy

    def _render_grid(self, dx_um, dy_um, position, objective, signal, slide) -> np.ndarray:
        pitch = float(slide.get("pitch_um", 50.0))
        sigma = float(slide.get("line_width_um", 5.0)) / 2
        to_line_x = np.abs((dx_um + pitch / 2) % pitch - pitch / 2)
        to_line_y = np.abs((dy_um + pitch / 2) % pitch - pitch / 2)
        lines = np.maximum(np.exp(-0.5 * (to_line_x / sigma) ** 2), np.exp(-0.5 * (to_line_y / sigma) ** 2))
        defocus = position.z_um - float(slide.get("focus_z_um", 0.0))
        depth_of_field = max(
            0.5, 2.0 * objective.immersion_refractive_index * 0.55 / objective.numerical_aperture**2
        )
        sharpness = math.exp(-((defocus / (3 * depth_of_field)) ** 2))
        image = 100.0 + signal * (0.1 + 0.9 * sharpness) * lines + signal * 0.05 * (1 - sharpness)
        return np.clip(image, 0, 65535).astype(np.uint16)


def _hash2(a: np.ndarray, b: np.ndarray, salt: int) -> np.ndarray:
    ai = a.astype(np.int64)
    bi = b.astype(np.int64)
    value = (ai * 73856093) ^ (bi * 19349663) ^ (salt * 83492791)
    value = (value ^ (value >> 13)) * 1274126177
    return np.abs(value ^ (value >> 16))
