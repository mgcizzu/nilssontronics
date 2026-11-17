from __future__ import annotations

import math
import shutil

from .geometry import point_in_polygon, points_are_non_collinear, rectangle_intersects_polygon
from .models import AcquisitionEstimate, AcquisitionPlan, MicroscopeConfig, ScanRegion


def estimate_acquisition(
    config: MicroscopeConfig, plan: AcquisitionPlan
) -> AcquisitionEstimate:
    objective = config.objective(plan.objective_id)
    tiles = len(
        tile_positions_for_regions(
            config,
            plan.objective_id,
            plan.regions,
            plan.overlap_x_percent,
            plan.overlap_y_percent,
        )
    )

    z_planes = plan.z.plane_count
    frames = tiles * len(plan.channels) * z_planes
    data_bytes = (
        frames
        * config.camera_width_px
        * config.camera_height_px
        * config.bytes_per_pixel
    )

    exposure_seconds_per_tile = sum(channel.exposure_ms for channel in plan.channels) / 1000
    exposure_seconds = tiles * z_planes * exposure_seconds_per_tile
    stage_overhead = tiles * 0.18
    channel_overhead = tiles * z_planes * len(plan.channels) * 0.12
    duration_seconds = exposure_seconds + stage_overhead + channel_overhead + 2.0

    return AcquisitionEstimate(
        tiles=tiles,
        frames=frames,
        z_planes=z_planes,
        duration_seconds=duration_seconds,
        data_bytes=data_bytes,
    )


def validate_plan(config: MicroscopeConfig, plan: AcquisitionPlan) -> list[str]:
    problems: list[str] = []

    if not plan.project.strip():
        problems.append("Enter a project name.")
    if not plan.sample.strip():
        problems.append("Enter a sample name.")
    if not plan.regions:
        problems.append("Draw at least one scan region on the overview.")
    if not plan.channels:
        problems.append("Select at least one imaging channel.")
    if not 0 <= plan.overlap_x_percent < 100:
        problems.append("X overlap must be between 0% and 99%.")
    if not 0 <= plan.overlap_y_percent < 100:
        problems.append("Y overlap must be between 0% and 99%.")

    try:
        objective = config.objective(plan.objective_id)
    except KeyError:
        problems.append("Select a configured objective.")
        return problems

    limits = config.stage_limits
    for index, region in enumerate(plan.regions, start=1):
        if region.width_um <= 0 or region.height_um <= 0:
            problems.append(f"Region {index} has no area.")
        if (
            region.x_um < limits.x_min_um
            or region.x_max_um > limits.x_max_um
            or region.y_um < limits.y_min_um
            or region.y_max_um > limits.y_max_um
        ):
            problems.append(f"Region {index} exceeds the configured XY stage limits.")

    if plan.z.mode == "stack":
        if not plan.z.begin_captured:
            problems.append("Capture the Z-stack Begin position from live view.")
        if not plan.z.end_captured:
            problems.append("Capture the Z-stack End position from live view.")
        if plan.z.begin_captured and plan.z.end_captured and plan.z.step_um <= 0:
            problems.append("An automatic Z step could not be calculated.")
        z_min = min(plan.z.bottom_um, plan.z.top_um)
        z_max = max(plan.z.bottom_um, plan.z.top_um)
        safe_min = max(limits.z_min_um, objective.safe_z_min_um)
        safe_max = min(limits.z_max_um, objective.safe_z_max_um)
        if plan.z.begin_captured and plan.z.end_captured and (z_min < safe_min or z_max > safe_max):
            problems.append(
                f"Z range must stay between {safe_min:.0f} and {safe_max:.0f} µm "
                f"for {objective.name}."
            )

    safe_z_min = max(limits.z_min_um, objective.safe_z_min_um)
    safe_z_max = min(limits.z_max_um, objective.safe_z_max_um)
    for index, point in enumerate(plan.focus_points, start=1):
        if not (
            limits.x_min_um <= point.x_um <= limits.x_max_um
            and limits.y_min_um <= point.y_um <= limits.y_max_um
        ):
            problems.append(f"Focus point {index} exceeds the configured XY stage limits.")
        if not safe_z_min <= point.z_um <= safe_z_max:
            problems.append(
                f"Focus point {index} Z must be between {safe_z_min:.0f} and "
                f"{safe_z_max:.0f} µm for {objective.name}."
            )
        if plan.regions and not any(point_in_polygon((point.x_um, point.y_um), region.vertices) for region in plan.regions):
            problems.append(f"Focus point {index} is outside the selected scan regions.")

    if plan.z.mode == "focus_map":
        if len(plan.focus_points) < 3:
            problems.append("Surface following needs at least three focus points.")
        elif not points_are_non_collinear(
            tuple((point.x_um, point.y_um) for point in plan.focus_points)
        ):
            problems.append("Surface-following focus points must not all lie on one line.")

    for setting in plan.channels:
        try:
            channel = config.channel(setting.channel_id)
        except KeyError:
            problems.append(f"Unknown channel: {setting.channel_id}.")
            continue
        if setting.exposure_ms <= 0 or setting.exposure_ms > channel.max_exposure_ms:
            problems.append(
                f"{channel.name} exposure must be between 0 and "
                f"{channel.max_exposure_ms:g} ms."
            )
        if setting.power_percent < 0 or setting.power_percent > channel.max_power_percent:
            problems.append(
                f"{channel.name} power must be between 0 and "
                f"{channel.max_power_percent:g}%."
            )

    try:
        free_bytes = shutil.disk_usage(plan.output_root).free
    except OSError:
        problems.append("The output folder is not accessible.")
    else:
        estimate = estimate_acquisition(config, plan)
        safety_margin = max(2 * 1024**3, int(estimate.data_bytes * 0.10))
        if estimate.data_bytes + safety_margin > free_bytes:
            problems.append("There is not enough free disk space for this scan.")

    return problems


def tile_positions_for_regions(
    config: MicroscopeConfig,
    objective_id: str,
    regions: tuple[ScanRegion, ...],
    overlap_x_percent: float | None = None,
    overlap_y_percent: float | None = None,
) -> list[tuple[float, float]]:
    objective = config.objective(objective_id)
    fov_width_um = config.camera_width_px * objective.pixel_size_um
    fov_height_um = config.camera_height_px * objective.pixel_size_um
    overlap_x = config.overlap_x_percent if overlap_x_percent is None else overlap_x_percent
    overlap_y = config.overlap_y_percent if overlap_y_percent is None else overlap_y_percent
    step_x_um = max(1.0, fov_width_um * (1.0 - overlap_x / 100.0))
    step_y_um = max(1.0, fov_height_um * (1.0 - overlap_y / 100.0))
    positions = []
    for region in regions:
        positions.extend(
            _tile_positions_for_region(
                region,
                fov_width_um,
                fov_height_um,
                step_x_um,
                step_y_um,
            )
        )
    return positions


def spiral_preview_positions(
    config: MicroscopeConfig,
    objective_id: str,
    regions: tuple[ScanRegion, ...],
    origin: tuple[float, float],
    default_grid_size: int = 5,
    overlap_x_percent: float | None = None,
    overlap_y_percent: float | None = None,
) -> list[tuple[float, float]]:
    overlap_x = config.overlap_x_percent if overlap_x_percent is None else overlap_x_percent
    overlap_y = config.overlap_y_percent if overlap_y_percent is None else overlap_y_percent
    positions = tile_positions_for_regions(
        config,
        objective_id,
        regions,
        overlap_x,
        overlap_y,
    )
    if not positions:
        objective = config.objective(objective_id)
        step_x_um = (
            config.camera_width_px
            * objective.pixel_size_um
            * (1.0 - overlap_x / 100.0)
        )
        step_y_um = (
            config.camera_height_px
            * objective.pixel_size_um
            * (1.0 - overlap_y / 100.0)
        )
        offset = (default_grid_size - 1) / 2
        positions = [
            (
                origin[0] + (x_index - offset) * step_x_um,
                origin[1] + (y_index - offset) * step_y_um,
            )
            for y_index in range(default_grid_size)
            for x_index in range(default_grid_size)
        ]
    return sorted(
        positions,
        key=lambda point: (
            math.hypot(point[0] - origin[0], point[1] - origin[1]),
            (math.atan2(point[1] - origin[1], point[0] - origin[0]) + 2 * math.pi)
            % (2 * math.pi),
        ),
    )


def _tile_positions_for_region(
    region: ScanRegion,
    fov_width_um: float,
    fov_height_um: float,
    step_x_um: float,
    step_y_um: float,
) -> list[tuple[float, float]]:
    tiles_x = max(1, math.ceil(region.width_um / step_x_um))
    tiles_y = max(1, math.ceil(region.height_um / step_y_um))
    cell_width = region.width_um / tiles_x
    cell_height = region.height_um / tiles_y
    positions = []
    for x_index in range(tiles_x):
        center_x = region.x_um + (x_index + 0.5) * cell_width
        for y_index in range(tiles_y):
            center_y = region.y_um + (y_index + 0.5) * cell_height
            if region.shape == "rectangle" or rectangle_intersects_polygon(
                center_x - fov_width_um / 2,
                center_y - fov_height_um / 2,
                center_x + fov_width_um / 2,
                center_y + fov_height_um / 2,
                region.vertices,
            ):
                positions.append((center_x, center_y))
    if not positions:
        positions.append(
            (region.x_um + region.width_um / 2, region.y_um + region.height_um / 2)
        )
    return positions
