from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from microscope_ui.config import load_config
from microscope_ui.backend import SimulatedMicroscope, StageLimitError
from microscope_ui.models import AcquisitionPlan, ChannelSetting, FocusPoint, ScanRegion, ZSettings
from microscope_ui.optics import automatic_z_stack, recommended_z_step_um
from microscope_ui.planner import (
    estimate_acquisition,
    spiral_preview_positions,
    validate_plan,
)
from microscope_ui.roi_io import load_roi_set, save_roi_set


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def make_plan(self, output: Path, **overrides) -> AcquisitionPlan:
        values = {
            "project": "Test project",
            "sample": "Slide A",
            "output_root": output,
            "objective_id": "20x_air",
            "channels": (
                ChannelSetting("dapi", exposure_ms=40, power_percent=20),
                ChannelSetting("af488", exposure_ms=60, power_percent=20),
            ),
            "regions": (ScanRegion(-500, -400, 1000, 800),),
            "z": ZSettings(),
        }
        values.update(overrides)
        return AcquisitionPlan(**values)

    def test_configuration_supports_more_than_two_objectives(self) -> None:
        self.assertGreater(len(self.config.objectives), 2)
        self.assertEqual(len({item.turret_position for item in self.config.objectives}), 6)
        self.assertEqual(self.config.overlap_x_percent, 10)
        self.assertEqual(self.config.overlap_y_percent, 10)

    def test_estimate_counts_tiles_channels_and_z_planes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self.make_plan(
                Path(temp_dir),
                z=ZSettings(mode="stack", bottom_um=-2, top_um=2, step_um=1, begin_captured=True, end_captured=True),
            )
            estimate = estimate_acquisition(self.config, plan)
        self.assertEqual(estimate.z_planes, 5)
        self.assertEqual(estimate.frames, estimate.tiles * 2 * 5)
        self.assertGreater(estimate.data_bytes, 0)

    def test_valid_plan_passes_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            problems = validate_plan(self.config, self.make_plan(Path(temp_dir)))
        self.assertEqual(problems, [])

    def test_missing_region_and_channel_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self.make_plan(Path(temp_dir), regions=(), channels=())
            problems = validate_plan(self.config, plan)
        self.assertTrue(any("region" in item.lower() for item in problems))
        self.assertTrue(any("channel" in item.lower() for item in problems))

    def test_exposure_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self.make_plan(
                Path(temp_dir),
                channels=(ChannelSetting("dapi", exposure_ms=5000, power_percent=20),),
            )
            problems = validate_plan(self.config, plan)
        self.assertTrue(any("exposure" in item.lower() for item in problems))

    def test_objective_specific_z_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self.make_plan(
                Path(temp_dir),
                objective_id="63x_oil",
                z=ZSettings(mode="stack", bottom_um=-10, top_um=2000, step_um=1, begin_captured=True, end_captured=True),
            )
            problems = validate_plan(self.config, plan)
        self.assertTrue(any("z range" in item.lower() for item in problems))

    def test_polygon_uses_fewer_tiles_than_its_bounding_rectangle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rectangle = ScanRegion(-5000, -5000, 10000, 10000)
            triangle = ScanRegion.from_polygon(((-5000, -5000), (5000, -5000), (-5000, 5000)))
            rectangle_estimate = estimate_acquisition(
                self.config,
                self.make_plan(Path(temp_dir), regions=(rectangle,)),
            )
            triangle_estimate = estimate_acquisition(
                self.config,
                self.make_plan(Path(temp_dir), regions=(triangle,)),
            )
        self.assertLess(triangle_estimate.tiles, rectangle_estimate.tiles)

    def test_surface_following_requires_non_collinear_focus_points(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            collinear = (
                FocusPoint(-300, -200, 0),
                FocusPoint(0, 0, 0),
                FocusPoint(300, 200, 0),
            )
            plan = self.make_plan(
                Path(temp_dir),
                z=ZSettings(mode="focus_map"),
                focus_points=collinear,
            )
            problems = validate_plan(self.config, plan)
        self.assertTrue(any("one line" in item.lower() for item in problems))

    def test_valid_surface_following_points_pass_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            focus_points = (
                FocusPoint(-300, -200, -1),
                FocusPoint(300, -200, 1),
                FocusPoint(0, 250, 0),
            )
            plan = self.make_plan(
                Path(temp_dir),
                z=ZSettings(mode="focus_map"),
                focus_points=focus_points,
            )
            problems = validate_plan(self.config, plan)
        self.assertEqual(problems, [])

    def test_roi_file_round_trip_preserves_polygon_and_focus_points(self) -> None:
        polygon = ScanRegion.from_polygon(((-100, -80), (120, -70), (50, 140), (-90, 60)))
        points = (FocusPoint(-50, -20, -1.5), FocusPoint(50, 20, 2.5))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample_roi.json"
            save_roi_set(path, (polygon,), points)
            regions_loaded, points_loaded = load_roi_set(path)
        self.assertEqual(regions_loaded, (polygon,))
        self.assertEqual(points_loaded, points)

    def test_roi_and_focus_point_translate_as_one_set(self) -> None:
        polygon = ScanRegion.from_polygon(((0, 0), (100, 0), (50, 80)))
        focus = FocusPoint(25, 20, 7.5)
        moved_polygon = polygon.translated(1000, -500)
        moved_focus = focus.translated_xy(1000, -500)
        self.assertEqual(moved_polygon.vertices[0], (1000, -500))
        self.assertEqual(moved_focus, FocusPoint(1025, -480, 7.5))

    def test_default_spiral_preview_starts_at_current_xy(self) -> None:
        positions = spiral_preview_positions(
            self.config,
            "20x_air",
            (),
            origin=(1234, -5678),
        )
        self.assertEqual(len(positions), 25)
        self.assertEqual(positions[0], (1234, -5678))

    def test_spiral_preview_uses_independent_xy_overlap(self) -> None:
        positions = spiral_preview_positions(
            self.config,
            "20x_air",
            (),
            origin=(0, 0),
            overlap_x_percent=0,
            overlap_y_percent=50,
        )
        x_values = sorted({point[0] for point in positions})
        y_values = sorted({point[1] for point in positions})
        x_step = x_values[1] - x_values[0]
        y_step = y_values[1] - y_values[0]
        self.assertAlmostEqual(y_step, x_step / 2)

    def test_simulated_moves_outside_limits_are_refused_not_clamped(self) -> None:
        backend = SimulatedMicroscope(self.config)
        start = backend.position
        with self.assertRaises(StageLimitError):
            backend.move_z(self.config.stage_limits.z_max_um + 1000)
        with self.assertRaises(StageLimitError):
            backend.move_xy(self.config.stage_limits.x_max_um + 1, 0)
        self.assertEqual(backend.position, start)

    def test_na_based_z_step_matches_widefield_nyquist_example(self) -> None:
        objective = self.config.objective("63x_oil")
        channel = self.config.channel("af488")
        step_um, wavelength_nm = recommended_z_step_um(objective, (channel,))
        self.assertEqual(wavelength_nm, 520)
        self.assertAlmostEqual(step_um, 1.515 * 0.520 / 1.4**2, places=6)

    def test_higher_na_automatically_produces_more_planes(self) -> None:
        channel = self.config.channel("dapi")
        low_na_step, _ = recommended_z_step_um(
            self.config.objective("10x_air"), (channel,)
        )
        high_na_step, _ = recommended_z_step_um(
            self.config.objective("63x_oil"), (channel,)
        )
        low_planes, _ = automatic_z_stack(0, 10, low_na_step)
        high_planes, high_actual_step = automatic_z_stack(0, 10, high_na_step)
        self.assertGreater(high_planes, low_planes)
        self.assertLessEqual(high_actual_step, high_na_step)

    def test_z_stack_preflight_requires_live_begin_and_end_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self.make_plan(
                Path(temp_dir),
                z=ZSettings(
                    mode="stack",
                    bottom_um=0,
                    top_um=0,
                    step_um=-1,
                    begin_captured=False,
                    end_captured=False,
                ),
            )
            problems = validate_plan(self.config, plan)
        self.assertTrue(any("begin" in item.lower() for item in problems))
        self.assertTrue(any("end" in item.lower() for item in problems))


if __name__ == "__main__":
    unittest.main()
