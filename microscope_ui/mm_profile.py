"""Discover a Micro-Manager configuration and create an application profile.

Discovery is *not* passive: ``loadSystemConfiguration`` initializes every
device and applies the System/Startup presets, so the stage may home and
turrets/filter wheels may move.  It must never run while the application
itself drives the same hardware (a second core on the same ports), and the
Qt wizard runs it on a worker thread.  Values that Micro-Manager cannot know
reliably (travel limits, objective clearance and optical metadata) are supplied
separately by the import wizard and validated by the normal config loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

from .config import load_config


@dataclass(frozen=True)
class MMConfigDiscovery:
    config_path: Path
    adapter_path: Path | None
    camera_device: str
    camera_width_px: int
    camera_height_px: int
    xy_stage_device: str
    focus_device: str
    config_groups: dict[str, tuple[str, ...]]
    state_labels: dict[str, tuple[str, ...]]


def inspect_mm_config(
    config_path: Path | str,
    adapter_path: Path | str | None = None,
    *,
    core_factory: Callable[[Path | None], Any] | None = None,
) -> MMConfigDiscovery:
    """Load a Micro-Manager config long enough to inventory its public devices.

    This initializes the hardware exactly like Micro-Manager does (see module doc).
    """

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"Micro-Manager configuration does not exist: {config_path}")
    resolved_adapter = Path(adapter_path).expanduser().resolve() if adapter_path else None
    if resolved_adapter is not None and not resolved_adapter.is_dir():
        raise ValueError(f"Micro-Manager adapter folder does not exist: {resolved_adapter}")
    if core_factory is None:
        try:
            from pymmcore_plus import CMMCorePlus
        except ImportError as exc:  # pragma: no cover - depends on optional hardware extra
            raise RuntimeError(
                "pymmcore-plus is not installed; install the hardware extra first"
            ) from exc

        def core_factory(path: Path | None):
            return CMMCorePlus(mm_path=str(path)) if path else CMMCorePlus()

    core = core_factory(resolved_adapter)
    try:
        core.loadSystemConfiguration(str(config_path))
        groups = {
            str(group): tuple(str(item) for item in core.getAvailableConfigs(group))
            for group in core.getAvailableConfigGroups()
        }
        state_labels: dict[str, tuple[str, ...]] = {}
        for device in core.getLoadedDevices():
            try:
                device_type = str(core.getDeviceType(device)).lower()
                if "state" not in device_type:
                    continue
                labels = tuple(str(label) for label in core.getStateLabels(device))
                if labels:
                    state_labels[str(device)] = labels
            except Exception:
                continue  # one unusual adapter must not hide the rest of the inventory
        return MMConfigDiscovery(
            config_path=config_path,
            adapter_path=resolved_adapter,
            camera_device=str(core.getCameraDevice()),
            camera_width_px=int(core.getImageWidth()),
            camera_height_px=int(core.getImageHeight()),
            xy_stage_device=str(core.getXYStageDevice()),
            focus_device=str(core.getFocusDevice()),
            config_groups=groups,
            state_labels=state_labels,
        )
    finally:
        try:
            core.unloadAllDevices()
        except Exception:
            pass


def safe_id(label: str, fallback: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return value or fallback


def build_profile(
    discovery: MMConfigDiscovery,
    *,
    microscope_name: str,
    channel_group: str,
    channels: list[dict],
    objective_device: str,
    objectives: list[dict],
    stage_limits: dict[str, float],
    objective_change_z_um: float,
    output_path: Path | str,
) -> dict:
    """Build the normal application JSON document from confirmed wizard values."""

    output_path = Path(output_path).expanduser().resolve()

    def portable(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.relative_to(output_path.parent))
        except ValueError:
            return str(path)

    presets = {channel["id"]: channel["preset"] for channel in channels}
    channel_rows = [
        {key: value for key, value in channel.items() if key != "preset"}
        for channel in channels
    ]
    profile = {
        "microscope_name": microscope_name.strip(),
        "camera": {
            "width_px": discovery.camera_width_px,
            "height_px": discovery.camera_height_px,
            "orientation": {"flip_x": False, "flip_y": False, "transpose": False},
        },
        "bytes_per_pixel": 2,
        "storage": {"writer_queue_frames": 4, "fsync": True},
        "overlap_x_percent": 10,
        "overlap_y_percent": 10,
        "stage_limits": stage_limits,
        "stage": {"approach_offset_um": 0},
        "objectives": objectives,
        "channels": channel_rows,
        "recipes": [],
        "calibration": {"directory": "calibration"},
        "hardware": {
            "backend": "mmcore",
            "mm_config_path": portable(discovery.config_path),
            "mm_adapter_path": portable(discovery.adapter_path),
            "channel_group": channel_group,
            "objective_device": objective_device,
            "turret_position_offset": 0,
            "objective_change_z_um": objective_change_z_um,
            "settle_timeout_ms": 10000,
            "channel_presets": presets,
            "power_properties": {},
        },
    }
    return profile


def save_profile(profile: dict, output_path: Path | str) -> Path:
    """Validate a generated profile before atomically replacing the destination."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(profile, indent=2) + "\n")
        load_config(temporary)
        temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output_path
