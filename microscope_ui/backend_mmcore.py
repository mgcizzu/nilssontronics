"""Micro-Manager backend built on pymmcore-plus.

Install with ``pip install -e .[hardware]`` and set ``"backend": "mmcore"`` in
the ``hardware`` section of ``config/microscope.json``.

Safety rules implemented here:

* every XY and Z target is checked against the configured stage limits and
  the objective's safe Z range *before* a command is sent (never clamped);
* objective changes first move Z to ``objective_change_z_um``, rotate the
  turret, wait for every device, and leave Z at that clearance height so
  the operator refocuses deliberately;
* channels map to named presets of one Micro-Manager config group, so normal
  users never see raw device properties;
* device errors surface as :class:`HardwareError` with the device message.

Frames are re-oriented with ``camera.orientation`` so +column = +stage X and
+row = +stage Y for the rest of the program.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import (
    HardwareError,
    StagePosition,
    apply_orientation,
    check_xy,
    check_z,
)
from .models import Channel, MicroscopeConfig, Objective


class MMCoreBackend:
    name = "Micro-Manager"
    simulated = False

    def __init__(self, config: MicroscopeConfig, core: Any | None = None) -> None:
        self._config = config
        self._hw = config.hardware
        if core is None:
            core = _create_core(self._hw)
        self._core = core
        self._timeout_ms = int(self._hw.get("settle_timeout_ms", 10000))
        self._channel_group = self._hw.get("channel_group", "Channel")
        self._objective_device = self._hw.get("objective_device", "Objective")
        self._turret_offset = int(self._hw.get("turret_position_offset", -1))
        self._presets: dict[str, str] = dict(self._hw.get("channel_presets", {}))
        self._power: dict[str, dict] = dict(self._hw.get("power_properties", {}))
        self._objective_id = self._read_objective_id()
        missing = [channel.id for channel in config.channels if channel.id not in self._presets]
        if missing:
            raise HardwareError(
                "hardware.channel_presets has no Micro-Manager preset for: " + ", ".join(missing)
            )

    # -- state ---------------------------------------------------------------
    @property
    def core(self):
        return self._core

    @property
    def position(self) -> StagePosition:
        x, y = self._call("read XY position", self._core.getXYPosition)
        z = self._call("read Z position", self._core.getPosition)
        return StagePosition(float(x), float(y), float(z))

    @property
    def current_objective_id(self) -> str:
        return self._objective_id

    def _read_objective_id(self) -> str:
        """Identify the objective in the light path from the turret, never by guessing.

        Safe-Z checks and preflight depend on it, so an unreadable turret is a
        startup error. Only a setup without a turret device and exactly one
        configured objective may skip the read.
        """

        if not self._objective_device:
            if len(self._config.objectives) == 1:
                return self._config.objectives[0].id
            raise HardwareError(
                "hardware.objective_device is not set but several objectives are configured"
            )
        try:
            state = int(self._core.getState(self._objective_device))
        except Exception as exc:
            raise HardwareError(
                f"Could not read the objective turret '{self._objective_device}': {exc}. "
                "The objective in the light path is unknown, so safe Z limits cannot be applied."
            ) from exc
        for objective in self._config.objectives:
            if objective.turret_position + self._turret_offset == state:
                return objective.id
        raise HardwareError(
            f"Turret state {state} on '{self._objective_device}' matches no configured objective"
        )

    # -- motion --------------------------------------------------------------
    def move_xy(self, x_um: float, y_um: float) -> StagePosition:
        check_xy(self._config, x_um, y_um)
        stage = self._core.getXYStageDevice()
        self._call("move XY stage", self._core.setXYPosition, x_um, y_um)
        self._wait(stage)
        return self.position

    def move_z(self, z_um: float) -> StagePosition:
        check_z(self._config, self._objective_id, z_um)
        focus = self._core.getFocusDevice()
        self._call("move focus", self._core.setPosition, z_um)
        self._wait(focus)
        return self.position

    def change_objective(self, objective: Objective) -> None:
        clearance = float(self._hw.get("objective_change_z_um", self._config.stage_limits.z_min_um))
        if not self._config.stage_limits.contains_z(clearance):
            raise HardwareError("hardware.objective_change_z_um is outside the stage Z limits")
        focus = self._core.getFocusDevice()
        self._call("move focus to clearance", self._core.setPosition, clearance)
        self._wait(focus)
        state = objective.turret_position + self._turret_offset
        self._call("rotate objective turret", self._core.setState, self._objective_device, state)
        self._wait(self._objective_device)
        self._call("wait for system", self._core.waitForSystem)
        self._objective_id = objective.id
        pixel_config = self._hw.get("pixel_size_configs", {}).get(objective.id)
        if pixel_config:
            self._call("select pixel size", self._core.setPixelSizeConfig, pixel_config)

    # -- illumination & camera -----------------------------------------------
    def set_channel(self, channel: Channel, exposure_ms: float, power_percent: float) -> None:
        if not 0 < exposure_ms <= channel.max_exposure_ms:
            raise HardwareError(f"{channel.name}: exposure {exposure_ms} ms is out of range")
        if not 0 <= power_percent <= channel.max_power_percent:
            raise HardwareError(f"{channel.name}: power {power_percent}% is out of range")
        preset = self._presets[channel.id]
        self._call("select channel", self._core.setConfig, self._channel_group, preset)
        self._call("wait for channel", self._core.waitForConfig, self._channel_group, preset)
        self._call("set exposure", self._core.setExposure, float(exposure_ms))
        power = self._power.get(channel.id)
        if power:
            value = power_percent * float(power.get("scale", 1.0)) + float(power.get("offset", 0.0))
            self._call(
                f"set {channel.name} power",
                self._core.setProperty,
                power["device"],
                power["property"],
                value,
            )
            self._wait(power["device"])

    def set_shutter_per_frame(self, enabled: bool) -> None:
        if enabled:
            self._call("close shutter", self._core.setShutterOpen, False)
        self._call("set auto-shutter", self._core.setAutoShutter, bool(enabled))

    def set_shutter_open(self, is_open: bool) -> None:
        if self._core.getAutoShutter() and is_open:
            return  # per-frame mode: the camera opens the shutter for each exposure
        self._call("operate shutter", self._core.setShutterOpen, bool(is_open))

    def snap(self) -> np.ndarray:
        self._call("snap image", self._core.snapImage)
        frame = self._call("read image", self._core.getImage)
        return apply_orientation(np.asarray(frame), self._config.camera_orientation)

    def close(self) -> None:
        try:
            self._core.setShutterOpen(False)
        except Exception:
            pass

    # -- helpers ---------------------------------------------------------------
    def _wait(self, device: str) -> None:
        if device:
            self._call(f"wait for {device}", self._core.waitForDevice, device)

    def _call(self, action: str, function, *args):
        try:
            return function(*args)
        except HardwareError:
            raise
        except Exception as exc:
            raise HardwareError(f"Could not {action}: {exc}") from exc


def _create_core(hardware: dict):  # pragma: no cover - needs Micro-Manager installed
    try:
        from pymmcore_plus import CMMCorePlus
    except ImportError as exc:
        raise HardwareError(
            "pymmcore-plus is not installed. Install with: pip install -e .[hardware]"
        ) from exc
    adapter_path = hardware.get("mm_adapter_path")
    core = CMMCorePlus(mm_path=adapter_path) if adapter_path else CMMCorePlus()
    config_path = hardware.get("mm_config_path")
    if not config_path:
        raise HardwareError("hardware.mm_config_path is not set")
    try:
        core.loadSystemConfiguration(config_path)
    except Exception as exc:
        raise HardwareError(f"Could not load Micro-Manager configuration {config_path}: {exc}") from exc
    core.setTimeoutMs(int(hardware.get("settle_timeout_ms", 10000)))
    return core


def create_backend(config: MicroscopeConfig):
    """Return the backend selected by ``hardware.backend`` (default: simulator)."""

    kind = config.hardware.get("backend", "simulator")
    if kind == "mmcore":
        return MMCoreBackend(config)
    if kind == "simulator":
        from .backend import SimulatedMicroscope

        return SimulatedMicroscope(config)
    raise HardwareError(f"Unknown hardware.backend '{kind}'")
