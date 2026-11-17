from __future__ import annotations

import json
from pathlib import Path

from .models import Channel, MicroscopeConfig, Objective, StageLimits


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "microscope.json"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> MicroscopeConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))

    objectives = tuple(Objective(**item) for item in data["objectives"])
    channels = tuple(Channel(**item) for item in data["channels"])
    stage_limits = StageLimits(**data["stage_limits"])

    if not objectives:
        raise ValueError("At least one objective must be configured")
    if len({objective.id for objective in objectives}) != len(objectives):
        raise ValueError("Objective IDs must be unique")
    if len({objective.turret_position for objective in objectives}) != len(objectives):
        raise ValueError("Objective turret positions must be unique")
    if len({channel.id for channel in channels}) != len(channels):
        raise ValueError("Channel IDs must be unique")
    writer_queue_frames = int(data.get("storage", {}).get("writer_queue_frames", 4))
    if not 1 <= writer_queue_frames <= 32:
        raise ValueError("storage.writer_queue_frames must be between 1 and 32")

    return MicroscopeConfig(
        microscope_name=data["microscope_name"],
        camera_width_px=data["camera"]["width_px"],
        camera_height_px=data["camera"]["height_px"],
        overlap_x_percent=data.get(
            "overlap_x_percent", data.get("overlap_percent", 10.0)
        ),
        overlap_y_percent=data.get(
            "overlap_y_percent", data.get("overlap_percent", 10.0)
        ),
        bytes_per_pixel=data.get("bytes_per_pixel", 2),
        objectives=objectives,
        channels=channels,
        stage_limits=stage_limits,
        writer_queue_frames=writer_queue_frames,
        recipes=tuple(data.get("recipes", ())),
    )
