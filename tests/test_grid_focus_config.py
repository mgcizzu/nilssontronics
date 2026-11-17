from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

from microscope_ui.config import ConfigError, load_config, validate_config
from microscope_ui.focus import FocusSurface
from microscope_ui.models import (
    AcquisitionOptions,
    AcquisitionPlan,
    ChannelSetting,
    FocusPoint,
    ScanRegion,
    ZSettings,
    endpoint_inclusive_plane_count,
)
from microscope_ui.optics import automatic_z_stack
from microscope_ui.planner import (
    estimate_acquisition,
    tile_grid_steps,
    tile_positions_for_regions,
    validate_plan,
)


class TileGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def tiles(self, regions, order="raster", overlap=10):
        return tile_positions_for_regions(self.config, "20x_air", regions, overlap, overlap, order)

    def test_overlapping_regions_share_one_grid_without_duplicates(self) -> None:
        a = ScanRegion(0, 0, 3000, 3000)
        b = ScanRegion(1500, 1500, 3000, 3000)
        tiles = self.tiles((a, b))
        positions = [tile.xy for tile in tiles]
        self.assertEqual(len(positions), len(set(positions)))
        cells = {(tile.row, tile.col) for tile in tiles}
        self.assertEqual(len(cells), len(tiles))

    def test_grid_step_gives_exactly_the_requested_overlap(self) -> None:
        tiles = self.tiles((ScanRegion(0, 0, 6000, 2000),), overlap=15)
        step_x, step_y = tile_grid_steps(self.config, "20x_air", 15, 15)
        fov_w, _ = self.config.fov_um("20x_air")
        row0 = sorted(tile.x_um for tile in tiles if tile.row == 0)
        for left, right in zip(row0, row0[1:]):
            self.assertAlmostEqual(right - left, step_x)
        self.assertAlmostEqual(1 - step_x / fov_w, 0.15)

    def test_grid_covers_the_whole_region(self) -> None:
        region = ScanRegion(-2500, -1000, 5000, 2000)
        tiles = self.tiles((region,))
        fov_w, fov_h = self.config.fov_um("20x_air")
        self.assertLessEqual(min(t.x_um for t in tiles) - fov_w / 2, region.x_um)
        self.assertGreaterEqual(max(t.x_um for t in tiles) + fov_w / 2, region.x_max_um)
        self.assertLessEqual(min(t.y_um for t in tiles) - fov_h / 2, region.y_um)
        self.assertGreaterEqual(max(t.y_um for t in tiles) + fov_h / 2, region.y_max_um)

    def test_raster_keeps_one_x_direction_and_serpentine_alternates(self) -> None:
        region = (ScanRegion(0, 0, 5000, 4000),)
        raster = self.tiles(region, "raster")
        serpentine = self.tiles(region, "serpentine")
        self.assertEqual(sorted(t.xy for t in raster), sorted(t.xy for t in serpentine))
        for tiles, reverse_odd in ((raster, False), (serpentine, True)):
            for row in {t.row for t in tiles}:
                xs = [t.x_um for t in tiles if t.row == row]
                expected = sorted(xs, reverse=reverse_odd and row % 2 == 1)
                self.assertEqual(xs, expected)
        self.assertEqual([t.index for t in serpentine], list(range(len(serpentine))))
        # Serpentine shortens total travel.
        def travel(tiles):
            return sum(abs(b.x_um - a.x_um) + abs(b.y_um - a.y_um) for a, b in zip(tiles, tiles[1:]))
        self.assertLess(travel(serpentine), travel(raster))

    def test_tile_centres_outside_stage_limits_are_reported(self) -> None:
        limits = self.config.stage_limits
        region = ScanRegion(limits.x_max_um - 1000, 0, 9000, 400)
        plan = AcquisitionPlan(
            "P", "S", Path(tempfile.gettempdir()), "4x_air",
            (ChannelSetting("dapi", 40, 20),), (region,),
        )
        problems = validate_plan(self.config, plan, check_disk=False)
        self.assertTrue(any("outside the configured XY stage limits" in p for p in problems))

    def test_estimate_includes_z_moves_and_channel_switches(self) -> None:
        region = (ScanRegion(0, 0, 3000, 3000),)
        base = AcquisitionPlan(
            "P", "S", Path("."), "20x_air",
            (ChannelSetting("dapi", 40, 20), ChannelSetting("cy5", 40, 20)), region,
        )
        stack = replace(
            base,
            z=ZSettings("stack", 0, 10, 1, begin_captured=True, end_captured=True),
        )
        single = estimate_acquisition(self.config, base)
        stacked = estimate_acquisition(self.config, stack)
        self.assertEqual(stacked.frames, single.frames * 11)
        z_time = stacked.tiles * 2 * 11 * self.config.timing.z_move_s
        self.assertGreater(stacked.duration_seconds - single.duration_seconds, z_time)


class ZPlaneTests(unittest.TestCase):
    def test_model_and_optics_agree_on_plane_count(self) -> None:
        for distance, step in ((10, 1), (10, 3), (0.9, 0.3), (7.3, 0.35)):
            planes, actual = automatic_z_stack(0, distance, step)
            z = ZSettings("stack", 0, distance, actual, begin_captured=True, end_captured=True)
            self.assertEqual(z.plane_count, planes)
            self.assertEqual(planes, endpoint_inclusive_plane_count(distance, step))
            positions = z.stack_positions()
            self.assertAlmostEqual(positions[0], 0)
            self.assertAlmostEqual(positions[-1], distance)

    def test_default_z_settings_are_not_captured(self) -> None:
        z = ZSettings(mode="stack")
        self.assertEqual(z.plane_count, 0)


class FocusSurfaceTests(unittest.TestCase):
    points = (
        FocusPoint(0, 0, 0),
        FocusPoint(1000, 0, 10),
        FocusPoint(0, 1000, 20),
        FocusPoint(1000, 1000, 30),
        FocusPoint(500, 500, 40),  # local bump a plane cannot follow
    )

    def test_linear_surface_passes_through_every_point(self) -> None:
        surface = FocusSurface(self.points, "linear")
        for point in self.points:
            self.assertAlmostEqual(surface.z_at(point.x_um, point.y_um), point.z_um)
        self.assertEqual(len(surface.triangles), 4)

    def test_linear_surface_follows_local_warping_better_than_plane(self) -> None:
        linear = FocusSurface(self.points, "linear")
        plane = FocusSurface(self.points, "plane")
        self.assertAlmostEqual(linear.z_at(500, 500), 40)
        self.assertLess(plane.z_at(500, 500), 30)

    def test_outside_the_hull_z_is_held_not_extrapolated(self) -> None:
        surface = FocusSurface(self.points, "linear")
        far = surface.z_at(50_000, 500)
        self.assertGreaterEqual(far, 0)
        self.assertLessEqual(far, 40)
        self.assertAlmostEqual(far, 20)  # midpoint of the right edge (10 → 30)

    def test_plane_fit_recovers_a_tilted_plane(self) -> None:
        tilted = tuple(FocusPoint(x, y, 0.001 * x - 0.002 * y + 5) for x, y in ((0, 0), (900, 100), (200, 800), (700, 700)))
        surface = FocusSurface(tilted, "plane")
        self.assertAlmostEqual(surface.z_at(400, 400), 0.4 - 0.8 + 5)

    def test_collinear_points_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FocusSurface((FocusPoint(0, 0, 0), FocusPoint(1, 1, 0), FocusPoint(2, 2, 0)))

    def test_surface_outside_safe_z_fails_preflight(self) -> None:
        config = load_config()
        low, high = config.safe_z_range("20x_air")
        plan = AcquisitionPlan(
            "P", "S", Path("."), "20x_air", (ChannelSetting("dapi", 40, 20),),
            (ScanRegion(-1000, -1000, 2000, 2000),),
            z=ZSettings("focus_map"),
            focus_points=(FocusPoint(-900, -900, 0), FocusPoint(900, -900, 0), FocusPoint(0, 900, high + 5)),
        )
        problems = validate_plan(config, plan, check_disk=False)
        self.assertTrue(any("focus point 3 z" in p.lower() or "surface" in p.lower() for p in problems))


class ConfigValidationTests(unittest.TestCase):
    def test_shipped_configuration_is_valid(self) -> None:
        self.assertEqual(validate_config(load_config()), [])

    def test_every_problem_is_reported(self) -> None:
        source = json.loads(Path("config/microscope.json").read_text(encoding="utf-8"))
        source["objectives"][0]["safe_z_min_um"] = 10
        source["objectives"][0]["safe_z_max_um"] = 0
        source["objectives"][1]["numerical_aperture"] = 1.2  # air objective
        source["channels"][0]["default_exposure_ms"] = 9999
        source["channels"][1]["color"] = "blue"
        source["recipes"][0]["channels"] = ["nope"]
        source["recipes"][1]["z_mode"] = "spiral"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ConfigError) as context:
                load_config(path)
        message = str(context.exception)
        for fragment in ("safe_z_min_um", "NA cannot exceed", "default exposure", "#RRGGBB", "unknown channels", "z_mode"):
            self.assertIn(fragment, message)

    def test_consistent_approach_requires_an_offset(self) -> None:
        config = load_config()
        plan = AcquisitionPlan(
            "P", "S", Path("."), "20x_air", (ChannelSetting("dapi", 40, 20),),
            (ScanRegion(0, 0, 1000, 1000),),
            options=AcquisitionOptions(consistent_approach=True),
        )
        problems = validate_plan(config, plan, check_disk=False)
        self.assertTrue(any("approach_offset_um" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
