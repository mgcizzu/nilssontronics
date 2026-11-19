from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

from .models import (
    Z_MODES,
    CameraOrientation,
    Channel,
    MicroscopeConfig,
    Objective,
    Recipe,
    StageLimits,
    Timing,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "microscope.json"

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


class ConfigError(ValueError):
    """Raised with every problem found in a configuration file."""


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> MicroscopeConfig:
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        config = _parse(data)
        calibration = data.get("calibration", {})
        directory = Path(calibration.get("directory", "calibration")).expanduser()
        if not directory.is_absolute():
            directory = config_path.parent / directory
        hardware = dict(config.hardware)
        for key in ("mm_config_path", "mm_adapter_path"):
            value = hardware.get(key)
            if value:
                hardware_path = Path(value).expanduser()
                if not hardware_path.is_absolute():
                    hardware_path = config_path.parent / hardware_path
                hardware[key] = str(hardware_path.resolve())
        config = replace(config, calibration_dir=directory, hardware=hardware)
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Configuration is missing or has a malformed field: {exc}") from exc
    problems = validate_config(config)
    if problems:
        raise ConfigError("Invalid configuration:\n- " + "\n- ".join(problems))
    return config


def _parse(data: dict) -> MicroscopeConfig:
    objectives = tuple(Objective(**item) for item in data["objectives"])
    channels = tuple(Channel(**item) for item in data["channels"])
    stage_limits = StageLimits(**data["stage_limits"])
    recipes = tuple(
        Recipe(
            name=item["name"],
            objective_id=item["objective_id"],
            channels=tuple(item.get("channels", ())),
            z_mode=item.get("z_mode", "single"),
        )
        for item in data.get("recipes", ())
    )
    stage = data.get("stage", {})
    camera = data["camera"]
    simulator = data.get("simulator", {})
    storage = data.get("storage", {})
    return MicroscopeConfig(
        microscope_name=data["microscope_name"],
        camera_width_px=int(camera["width_px"]),
        camera_height_px=int(camera["height_px"]),
        overlap_x_percent=float(
            data.get("overlap_x_percent", data.get("overlap_percent", 10.0))
        ),
        overlap_y_percent=float(
            data.get("overlap_y_percent", data.get("overlap_percent", 10.0))
        ),
        bytes_per_pixel=int(data.get("bytes_per_pixel", 2)),
        objectives=objectives,
        channels=channels,
        stage_limits=stage_limits,
        writer_queue_frames=int(storage.get("writer_queue_frames", 4)),
        recipes=recipes,
        timing=Timing(**data.get("timing", {})),
        camera_orientation=CameraOrientation(**camera.get("orientation", {})),
        approach_offset_um=float(stage.get("approach_offset_um", 0.0)),
        simulator_frame_px=int(simulator.get("frame_size_px", 256)),
        fsync_tiles=bool(storage.get("fsync", True)),
        hardware=dict(data.get("hardware", {})),
        simulator_options={key: value for key, value in simulator.items() if key != "frame_size_px"},
    )


def validate_config(config: MicroscopeConfig) -> list[str]:
    """Return every configuration problem rather than stopping at the first."""

    problems: list[str] = []
    limits = config.stage_limits

    if config.camera_width_px <= 0 or config.camera_height_px <= 0:
        problems.append("camera width_px and height_px must be positive")
    if config.bytes_per_pixel != 2:
        problems.append("bytes_per_pixel must be 2 (the saved-plane contract is uint16)")
    for axis in ("x", "y", "z"):
        low = getattr(limits, f"{axis}_min_um")
        high = getattr(limits, f"{axis}_max_um")
        if low >= high:
            problems.append(f"stage_limits {axis}_min_um must be below {axis}_max_um")
    for axis, value in (("X", config.overlap_x_percent), ("Y", config.overlap_y_percent)):
        if not 0 <= value < 100:
            problems.append(f"default {axis} overlap must be between 0 and 99%")
    if not 1 <= config.writer_queue_frames <= 32:
        problems.append("storage.writer_queue_frames must be between 1 and 32")
    if config.approach_offset_um < 0:
        problems.append("stage.approach_offset_um cannot be negative")
    if config.simulator_frame_px < 8:
        problems.append("simulator.frame_size_px must be at least 8")
    timing = config.timing
    if timing.stage_speed_um_per_s <= 0:
        problems.append("timing.stage_speed_um_per_s must be positive")
    for name in ("stage_settle_s", "z_move_s", "channel_switch_s", "camera_readout_ms"):
        if getattr(timing, name) < 0:
            problems.append(f"timing.{name} cannot be negative")

    if not config.objectives:
        problems.append("at least one objective must be configured")
    if len({item.id for item in config.objectives}) != len(config.objectives):
        problems.append("objective IDs must be unique")
    if len({item.turret_position for item in config.objectives}) != len(config.objectives):
        problems.append("objective turret positions must be unique")
    for objective in config.objectives:
        label = f"objective {objective.id}"
        if objective.pixel_size_um <= 0:
            problems.append(f"{label}: pixel_size_um must be positive")
        if objective.numerical_aperture <= 0:
            problems.append(f"{label}: numerical_aperture must be positive")
        if objective.immersion_refractive_index < 1:
            problems.append(f"{label}: immersion_refractive_index must be at least 1")
        elif objective.numerical_aperture > objective.immersion_refractive_index:
            problems.append(f"{label}: NA cannot exceed the immersion refractive index")
        if objective.safe_z_min_um >= objective.safe_z_max_um:
            problems.append(f"{label}: safe_z_min_um must be below safe_z_max_um")
        if (
            objective.safe_z_max_um < limits.z_min_um
            or objective.safe_z_min_um > limits.z_max_um
        ):
            problems.append(f"{label}: safe Z range lies outside the stage Z limits")

    if not config.channels:
        problems.append("at least one channel must be configured")
    if len({item.id for item in config.channels}) != len(config.channels):
        problems.append("channel IDs must be unique")
    for channel in config.channels:
        label = f"channel {channel.id}"
        if not _HEX_COLOR.match(channel.color):
            problems.append(f"{label}: color must be #RRGGBB")
        if channel.emission_wavelength_nm <= 0:
            problems.append(f"{label}: emission_wavelength_nm must be positive")
        if not 0 < channel.default_exposure_ms <= channel.max_exposure_ms:
            problems.append(
                f"{label}: default exposure must be above 0 and at most max_exposure_ms"
            )
        if not 0 <= channel.default_power_percent <= channel.max_power_percent <= 100:
            problems.append(
                f"{label}: power must satisfy 0 ≤ default ≤ max ≤ 100 %"
            )

    objective_ids = {item.id for item in config.objectives}
    channel_ids = {item.id for item in config.channels}
    if len({item.name for item in config.recipes}) != len(config.recipes):
        problems.append("recipe names must be unique")
    for recipe in config.recipes:
        label = f"recipe '{recipe.name}'"
        if recipe.name == "Custom":
            problems.append(f"{label}: the name 'Custom' is reserved")
        if recipe.objective_id not in objective_ids:
            problems.append(f"{label}: unknown objective {recipe.objective_id}")
        unknown = [item for item in recipe.channels if item not in channel_ids]
        if unknown:
            problems.append(f"{label}: unknown channels {', '.join(unknown)}")
        if recipe.z_mode not in Z_MODES:
            problems.append(f"{label}: z_mode must be one of {', '.join(Z_MODES)}")
    return problems
