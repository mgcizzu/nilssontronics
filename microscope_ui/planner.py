from __future__ import annotations

import math
import shutil
from collections.abc import Iterator

from .focus import FocusSurface
from .geometry import point_in_polygon, points_are_non_collinear, rectangle_intersects_polygon
from .models import (
    TILE_ORDERS,
    AcquisitionEstimate,
    AcquisitionPlan,
    MicroscopeConfig,
    ScanRegion,
    Tile,
    TileOrder,
)


def tile_grid_steps(
    config: MicroscopeConfig,
    objective_id: str,
    overlap_x_percent: float,
    overlap_y_percent: float,
) -> tuple[float, float]:
    fov_width_um, fov_height_um = config.fov_um(objective_id)
    return (
        max(1.0, fov_width_um * (1.0 - overlap_x_percent / 100.0)),
        max(1.0, fov_height_um * (1.0 - overlap_y_percent / 100.0)),
    )


def tile_positions_for_regions(
    config: MicroscopeConfig,
    objective_id: str,
    regions: tuple[ScanRegion, ...],
    overlap_x_percent: float | None = None,
    overlap_y_percent: float | None = None,
    order: TileOrder = "raster",
) -> list[Tile]:
    """Return the experiment's tile list on one shared grid.

    All regions share a single grid anchored on their combined bounding box, so
    overlapping ROIs never produce duplicate tiles and the requested overlap is
    exact everywhere (useful for stitching). A grid cell is kept when its field
    of view overlaps any region with positive area.

    ``raster`` visits every row left→right, so X targets are always approached
    from the same side. ``serpentine`` alternates direction for shorter travel.
    Rows are always visited top→bottom (increasing stage Y).
    """

    if not regions:
        return []
    if order not in TILE_ORDERS:
        raise ValueError(f"Unknown tile order: {order}")
    overlap_x = config.overlap_x_percent if overlap_x_percent is None else overlap_x_percent
    overlap_y = config.overlap_y_percent if overlap_y_percent is None else overlap_y_percent
    fov_width_um, fov_height_um = config.fov_um(objective_id)
    step_x_um, step_y_um = tile_grid_steps(config, objective_id, overlap_x, overlap_y)

    x_min = min(region.x_um for region in regions)
    x_max = max(region.x_max_um for region in regions)
    y_min = min(region.y_um for region in regions)
    y_max = max(region.y_max_um for region in regions)
    columns = _grid_count(x_max - x_min, fov_width_um, step_x_um)
    rows = _grid_count(y_max - y_min, fov_height_um, step_y_um)
    first_x = (x_min + x_max) / 2 - (columns - 1) * step_x_um / 2
    first_y = (y_min + y_max) / 2 - (rows - 1) * step_y_um / 2

    tiles: list[Tile] = []
    for row in range(rows):
        center_y = first_y + row * step_y_um
        column_order = range(columns)
        if order == "serpentine" and row % 2 == 1:
            column_order = range(columns - 1, -1, -1)
        for col in column_order:
            center_x = first_x + col * step_x_um
            left = center_x - fov_width_um / 2
            right = center_x + fov_width_um / 2
            top = center_y - fov_height_um / 2
            bottom = center_y + fov_height_um / 2
            if any(_tile_touches_region(left, top, right, bottom, region) for region in regions):
                tiles.append(Tile(len(tiles), row, col, center_x, center_y))
    return tiles


def _grid_count(extent_um: float, fov_um: float, step_um: float) -> int:
    if extent_um <= fov_um:
        return 1
    return math.ceil((extent_um - fov_um) / step_um - 1e-9) + 1


def _tile_touches_region(
    left: float, top: float, right: float, bottom: float, region: ScanRegion
) -> bool:
    overlaps_box = (
        left < region.x_max_um
        and right > region.x_um
        and top < region.y_max_um
        and bottom > region.y_um
    )
    if not overlaps_box:
        return False
    if region.shape == "rectangle":
        return True
    return rectangle_intersects_polygon(left, top, right, bottom, region.vertices)


def tiles_for_plan(config: MicroscopeConfig, plan: AcquisitionPlan) -> list[Tile]:
    return tile_positions_for_regions(
        config,
        plan.objective_id,
        plan.regions,
        plan.overlap_x_percent,
        plan.overlap_y_percent,
        plan.options.tile_order,
    )


def estimate_acquisition(
    config: MicroscopeConfig,
    plan: AcquisitionPlan,
    tiles: list[Tile] | None = None,
) -> AcquisitionEstimate:
    if tiles is None:
        tiles = tiles_for_plan(config, plan)
    tile_count = len(tiles)
    z_planes = plan.z.plane_count
    channel_count = len(plan.channels)
    frames = tile_count * channel_count * z_planes
    data_bytes = (
        frames
        * config.camera_width_px
        * config.camera_height_px
        * config.bytes_per_pixel
    )

    timing = config.timing
    exposure_s_per_plane_set = sum(channel.exposure_ms for channel in plan.channels) / 1000
    exposure_s = tile_count * z_planes * exposure_s_per_plane_set
    readout_s = frames * timing.camera_readout_ms / 1000

    travel_um = 0.0
    for previous, current in zip(tiles, tiles[1:]):
        travel_um += max(abs(current.x_um - previous.x_um), abs(current.y_um - previous.y_um))
    approach_moves = tile_count if plan.options.consistent_approach and config.approach_offset_um > 0 else 0
    stage_s = (
        travel_um / timing.stage_speed_um_per_s
        + (tile_count + approach_moves) * timing.stage_settle_s
    )
    if z_planes > 1:
        z_moves = tile_count * channel_count * z_planes
    elif plan.z.mode == "focus_map":
        z_moves = tile_count
    else:
        z_moves = 0
    z_s = z_moves * timing.z_move_s
    channel_s = tile_count * channel_count * timing.channel_switch_s if channel_count > 1 else 0.0
    duration_seconds = exposure_s + readout_s + stage_s + z_s + channel_s + 2.0

    return AcquisitionEstimate(
        tiles=tile_count,
        frames=frames,
        z_planes=z_planes,
        duration_seconds=duration_seconds,
        data_bytes=data_bytes,
        actual_overlap_x_percent=plan.overlap_x_percent,
        actual_overlap_y_percent=plan.overlap_y_percent,
    )


def focus_surface_for_plan(plan: AcquisitionPlan) -> FocusSurface:
    return FocusSurface(plan.focus_points, plan.options.focus_method)


def validate_plan(
    config: MicroscopeConfig,
    plan: AcquisitionPlan,
    tiles: list[Tile] | None = None,
    check_disk: bool = True,
) -> list[str]:
    problems: list[str] = []

    if not plan.project.strip():
        problems.append("Enter a project name.")
    if not plan.sample.strip():
        problems.append("Enter a sample name.")
    if not plan.regions:
        problems.append("Draw at least one scan region on the overview.")
    if not plan.channels:
        problems.append("Select at least one imaging channel.")
    if len({setting.channel_id for setting in plan.channels}) != len(plan.channels):
        problems.append("Each channel can only be selected once.")
    if not 0 <= plan.overlap_x_percent < 100:
        problems.append("X overlap must be between 0% and 99%.")
    if not 0 <= plan.overlap_y_percent < 100:
        problems.append("Y overlap must be between 0% and 99%.")
    if plan.options.consistent_approach and config.approach_offset_um <= 0:
        problems.append(
            "Consistent approach needs stage.approach_offset_um > 0 in the configuration."
        )

    try:
        objective = config.objective(plan.objective_id)
    except KeyError:
        problems.append("Select a configured objective.")
        return problems

    limits = config.stage_limits
    for index, region in enumerate(plan.regions, start=1):
        if region.width_um <= 0 or region.height_um <= 0:
            problems.append(f"Region {index} has no area.")

    if tiles is None and plan.regions:
        tiles = tiles_for_plan(config, plan)
    tiles = tiles or []
    approach = config.approach_offset_um if plan.options.consistent_approach else 0.0
    outside = [
        tile
        for tile in tiles
        if not limits.contains_xy(tile.x_um, tile.y_um)
        or not limits.contains_xy(tile.x_um - approach, tile.y_um - approach)
    ]
    if outside:
        problems.append(
            f"{len(outside)} tile position{'s' if len(outside) != 1 else ''} "
            f"(first: tile {outside[0].index + 1}) "
            "lie outside the configured XY stage limits."
        )

    safe_min, safe_max = config.safe_z_range(plan.objective_id)
    if plan.z.mode == "stack":
        if not plan.z.begin_captured:
            problems.append("Capture the Z-stack Begin position from live view.")
        if not plan.z.end_captured:
            problems.append("Capture the Z-stack End position from live view.")
        if plan.z.begin_captured and plan.z.end_captured and plan.z.step_um <= 0:
            problems.append("An automatic Z step could not be calculated.")
        z_min = min(plan.z.bottom_um, plan.z.top_um)
        z_max = max(plan.z.bottom_um, plan.z.top_um)
        if plan.z.begin_captured and plan.z.end_captured and (z_min < safe_min or z_max > safe_max):
            problems.append(
                f"Z range must stay between {safe_min:.0f} and {safe_max:.0f} µm "
                f"for {objective.name}."
            )

    for index, point in enumerate(plan.focus_points, start=1):
        if not limits.contains_xy(point.x_um, point.y_um):
            problems.append(f"Focus point {index} exceeds the configured XY stage limits.")
        if not safe_min <= point.z_um <= safe_max:
            problems.append(
                f"Focus point {index} Z must be between {safe_min:.0f} and "
                f"{safe_max:.0f} µm for {objective.name}."
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
        elif tiles:
            surface = focus_surface_for_plan(plan)
            low, high = surface.z_range(tile.xy for tile in tiles)
            if low < safe_min or high > safe_max:
                problems.append(
                    f"The focus surface reaches Z {low:.1f}–{high:.1f} µm, outside the "
                    f"safe range {safe_min:.0f}–{safe_max:.0f} µm for {objective.name}."
                )

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

    if check_disk:
        try:
            free_bytes = shutil.disk_usage(_existing_parent(plan.output_root)).free
        except OSError:
            problems.append("The output folder is not accessible.")
        else:
            estimate = estimate_acquisition(config, plan, tiles)
            safety_margin = max(2 * 1024**3, int(estimate.data_bytes * 0.10))
            if estimate.data_bytes + safety_margin > free_bytes:
                problems.append("There is not enough free disk space for this scan.")

    return problems


def _existing_parent(path):
    """Check free space on the nearest existing folder (output may not exist yet)."""

    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


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
    positions = [
        tile.xy
        for tile in tile_positions_for_regions(
            config,
            objective_id,
            regions,
            overlap_x,
            overlap_y,
        )
    ]
    if not positions:
        step_x_um, step_y_um = tile_grid_steps(config, objective_id, overlap_x, overlap_y)
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


def continuous_spiral_preview_positions(
    config: MicroscopeConfig,
    objective_id: str,
    origin: tuple[float, float],
    overlap_x_percent: float | None = None,
    overlap_y_percent: float | None = None,
) -> Iterator[tuple[float, float]]:
    """Yield an outward square spiral of tile centres without a fixed size.

    The caller owns the stopping rule.  The Qt preview stops before the first
    position outside the configured stage limits, which keeps this generator
    useful without coupling planning code to a particular UI workflow.
    """

    overlap_x = config.overlap_x_percent if overlap_x_percent is None else overlap_x_percent
    overlap_y = config.overlap_y_percent if overlap_y_percent is None else overlap_y_percent
    step_x_um, step_y_um = tile_grid_steps(config, objective_id, overlap_x, overlap_y)
    grid_x = grid_y = 0
    yield origin

    run_length = 1
    while True:
        for _ in range(run_length):
            grid_x += 1
            yield origin[0] + grid_x * step_x_um, origin[1] + grid_y * step_y_um
        for _ in range(run_length):
            grid_y += 1
            yield origin[0] + grid_x * step_x_um, origin[1] + grid_y * step_y_um
        run_length += 1
        for _ in range(run_length):
            grid_x -= 1
            yield origin[0] + grid_x * step_x_um, origin[1] + grid_y * step_y_um
        for _ in range(run_length):
            grid_y -= 1
            yield origin[0] + grid_x * step_x_um, origin[1] + grid_y * step_y_um
        run_length += 1
