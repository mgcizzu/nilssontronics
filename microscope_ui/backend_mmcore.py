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

import math
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
        preset_config = self._hw.get("channel_presets", {})
        power_config = self._hw.get("power_properties", {})
        if not isinstance(preset_config, dict):
            raise HardwareError("hardware.channel_presets must be an object")
        if not isinstance(power_config, dict):
            raise HardwareError("hardware.power_properties must be an object")
        self._presets: dict[str, str] = dict(preset_config)
        self._power: dict[str, dict] = dict(power_config)
        self._objective_id = self._read_objective_id()
        pixel_configs = self._hw.get("pixel_size_configs", {})
        if not isinstance(pixel_configs, dict):
            raise HardwareError("hardware.pixel_size_configs must be an object")
        for objective in config.objectives:
            self._check_pixel_config(objective)
        missing = [channel.id for channel in config.channels if channel.id not in self._presets]
        if missing:
            raise HardwareError(
                "hardware.channel_presets has no Micro-Manager preset for: " + ", ".join(missing)
            )
        channel_ids = {channel.id for channel in config.channels}
        unknown_power = sorted(set(self._power) - channel_ids)
        if unknown_power:
            raise HardwareError(
                "hardware.power_properties has unknown channels: " + ", ".join(unknown_power)
            )
        for channel_id, mapping in self._power.items():
            if not isinstance(mapping, dict) or not mapping.get("device") or not mapping.get("property"):
                raise HardwareError(
                    f"hardware.power_properties.{channel_id} needs device and property"
                )
            try:
                float(mapping.get("scale", 1.0))
                float(mapping.get("offset", 0.0))
            except (TypeError, ValueError) as exc:
                raise HardwareError(
                    f"hardware.power_properties.{channel_id} scale and offset must be numbers"
                ) from exc

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

    def refresh_objective_id(self) -> str:
        """Read the physical turret and refresh the cached UI value."""

        self._objective_id = self._read_objective_id()
        return self._objective_id

    def power_is_adjustable(self, channel_id: str) -> bool:
        return channel_id in self._power

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
        self.move_xy_for_acquisition(x_um, y_um)
        return self.position

    def move_xy_for_acquisition(self, x_um: float, y_um: float) -> None:
        """Move during a scan without an unnecessary position readback."""

        check_xy(self._config, x_um, y_um)
        stage = self._core.getXYStageDevice()
        self._call("move XY stage", self._core.setXYPosition, x_um, y_um)
        self._wait(stage)

    def move_z(self, z_um: float) -> StagePosition:
        # The turret can be changed from its own controls or Micro-Manager's
        # property browser, so never apply a cached objective's safety range.
        objective_id = self.refresh_objective_id()
        self.move_z_for_acquisition(objective_id, z_um)
        return self.position

    def move_z_for_acquisition(self, objective_id: str, z_um: float) -> None:
        """Move during a scan after the engine has verified the physical turret."""

        check_z(self._config, objective_id, z_um)
        focus = self._core.getFocusDevice()
        self._call("move focus", self._core.setPosition, z_um)
        self._wait(focus)

    def _check_pixel_config(self, objective: Objective) -> None:
        """Refuse a pixel-size preset that would move the turret to another objective.

        ``setPixelSizeConfig`` *applies* the preset's device settings. A preset
        that sets the objective device to a different state than the one
        mapped to ``objective`` would silently rotate the turret after the
        objective change was verified (and with Z about to leave clearance).
        """

        name = self._hw.get("pixel_size_configs", {}).get(objective.id)
        if not name or not self._objective_device:
            return
        try:
            settings = list(self._core.getPixelSizeConfigData(name))
        except Exception as exc:
            raise HardwareError(f"Pixel-size preset '{name}' for {objective.id}: {exc}") from exc
        target = objective.turret_position + self._turret_offset
        for device, prop, value in settings:
            if str(device) != self._objective_device:
                continue
            try:
                if str(prop) == "State":
                    state = int(value)
                elif str(prop) == "Label":
                    state = int(self._core.getStateFromLabel(self._objective_device, str(value)))
                else:
                    continue
            except Exception as exc:
                raise HardwareError(
                    f"Pixel-size preset '{name}' for {objective.id}: cannot resolve "
                    f"{device}-{prop} '{value}': {exc}"
                ) from exc
            if state != target:
                raise HardwareError(
                    f"Pixel-size preset '{name}' sets {device} to state {state}, but "
                    f"{objective.id} is at turret state {target} (turret_position "
                    f"{objective.turret_position} + offset {self._turret_offset}). Selecting it "
                    "would rotate the turret to another objective; fix "
                    "hardware.pixel_size_configs or the turret mapping."
                )

    def change_objective(self, objective: Objective) -> None:
        self._check_pixel_config(objective)  # before any motion
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
        actual = self.refresh_objective_id()
        if actual != objective.id:
            raise HardwareError(
                f"Objective change requested {objective.id}, but the turret reports {actual}"
            )
        pixel_config = self._hw.get("pixel_size_configs", {}).get(objective.id)
        if pixel_config:
            self._call("select pixel size", self._core.setPixelSizeConfig, pixel_config)
            self._call("wait for system", self._core.waitForSystem)

    # -- illumination & camera -----------------------------------------------
    def set_channel(self, channel: Channel, exposure_ms: float, power_percent: float) -> None:
        if not 0 < exposure_ms <= channel.max_exposure_ms:
            raise HardwareError(f"{channel.name}: exposure {exposure_ms} ms is out of range")
        if not 0 <= power_percent <= channel.max_power_percent:
            raise HardwareError(f"{channel.name}: power {power_percent}% is out of range")
        power = self._power.get(channel.id)
        if power is None and not math.isclose(
            power_percent, channel.default_power_percent, rel_tol=0.0, abs_tol=1e-6
        ):
            raise HardwareError(
                f"{channel.name}: power is fixed by its Micro-Manager preset at the "
                f"configured default {channel.default_power_percent:g}%; add "
                f"hardware.power_properties.{channel.id} to make it adjustable"
            )
        preset = self._presets[channel.id]
        self._call("select channel", self._core.setConfig, self._channel_group, preset)
        self._call("wait for channel", self._core.waitForConfig, self._channel_group, preset)
        self._call("set exposure", self._core.setExposure, float(exposure_ms))
        if power is not None:
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
        # Establish a known dark state before either auto- or manually-controlled
        # acquisition. In particular, the first XY move in manual mode must not
        # inherit an open shutter from Micro-Manager or an earlier session.
        self._call("close shutter", self._core.setShutterOpen, False)
        self._call("set auto-shutter", self._core.setAutoShutter, bool(enabled))

    def set_shutter_open(self, is_open: bool) -> None:
        if self._core.getAutoShutter() and is_open:
            return  # per-frame mode: the camera opens the shutter for each exposure
        self._call("operate shutter", self._core.setShutterOpen, bool(is_open))

    def snap(self) -> np.ndarray:
        self._call("snap image", self._core.snapImage)
        frame = self._call("read image", self._core.getImage)
        frame = apply_orientation(np.asarray(frame), self._config.camera_orientation)
        expected_width, expected_height = self._config.oriented_camera_size_px
        expected_shape = (expected_height, expected_width)
        if frame.ndim != 2:
            raise HardwareError(
                f"Camera returned shape {frame.shape}; expected one 2-D monochrome plane"
            )
        if frame.shape != expected_shape:
            raise HardwareError(
                f"Camera returned {frame.shape[1]}×{frame.shape[0]} px after orientation; "
                f"configuration expects {expected_width}×{expected_height} px. Check ROI, "
                "binning, camera dimensions and transpose."
            )
        if frame.dtype.kind != "u" or frame.dtype.itemsize != 2:
            raise HardwareError(
                f"Camera returned {frame.dtype}; acquisition requires unsigned 16-bit frames"
            )
        return np.ascontiguousarray(frame, dtype=np.uint16)

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
