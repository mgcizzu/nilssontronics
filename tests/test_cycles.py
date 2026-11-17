from __future__ import annotations

from dataclasses import replace
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from microscope_ui.backend import SimulatedMicroscope
from microscope_ui.backend_mmcore import MMCoreBackend
from microscope_ui.backend import HardwareError, StageLimitError
from microscope_ui.config import load_config
from microscope_ui.controller import ControllerError, ScanController
from microscope_ui.experiment import Experiment
from microscope_ui.models import (
    AcquisitionOptions,
    AcquisitionPlan,
    CameraOrientation,
    ChannelSetting,
    FocusPoint,
    ScanRegion,
    ZSettings,
)
from microscope_ui.mosaic import completed_tile_indices, read_tile, tile_records


class SlowSimulator(SimulatedMicroscope):
    def __init__(self, *args, delay=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.channel_log: list[tuple[str, float, float]] = []

    def snap(self):
        time.sleep(self.delay)
        return super().snap()

    def set_channel(self, channel, exposure_ms, power_percent):
        self.channel_log.append((channel.id, exposure_ms, power_percent))
        super().set_channel(channel, exposure_ms, power_percent)


class CycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def setUp(self) -> None:
        # Estimates assume full 2048×2048 frames; don't let the host's free disk
        # space decide whether these (tiny simulated) scans may start.
        free = mock.patch(
            "microscope_ui.planner.shutil.disk_usage",
            return_value=mock.Mock(free=10 * 1024**4),
        )
        free.start()
        self.addCleanup(free.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.backend = SlowSimulator(self.config, frame_px=24)
        self.backend.change_objective(self.config.objective("20x_air"))
        self.controller = ScanController(self.config, self.backend)

    def tearDown(self) -> None:
        self.controller.close()
        self.temp.cleanup()

    def plan(self, **changes) -> AcquisitionPlan:
        values = dict(
            project="Proj",
            sample="Slide",
            output_root=Path(self.temp.name),
            objective_id="20x_air",
            channels=(ChannelSetting("dapi", 40, 22), ChannelSetting("cy5", 80, 33)),
            regions=(ScanRegion(-1500, -1000, 3000, 2000),),
            focus_points=(
                FocusPoint(-1200, -800, -1.0),
                FocusPoint(1200, -800, 1.0),
                FocusPoint(0, 900, 2.0),
            ),
            z=ZSettings(mode="focus_map"),
        )
        values.update(changes)
        return AcquisitionPlan(**values)

    def run_cycle(self, plan):
        record = self.controller.start_cycle(plan)
        self.assertTrue(self.controller.wait(30))
        return record, self.controller.status()

    def test_cycle_uses_focus_surface_z_and_selected_power(self) -> None:
        record, status = self.run_cycle(self.plan())
        self.assertEqual(status.state, "complete")
        self.assertEqual(status.tiles_done, status.tiles_total)
        self.assertIn(("dapi", 40, 22), self.backend.channel_log)
        self.assertIn(("cy5", 80, 33), self.backend.channel_log)
        folder = self.controller.experiment.cycle_folder(record.number)
        z_values = set()
        for tile_id in sorted(tile_records(folder)):
            header, planes = read_tile(folder, tile_id)
            self.assertEqual(set(planes), {("dapi", 0), ("cy5", 0)})
            z_values.add(round(header["z_positions_um"][0], 3))
        self.assertGreater(len(z_values), 3, "Z must follow the surface, not stay constant")
        self.assertTrue(all(-1.0 <= z <= 2.0 for z in z_values))

    def test_z_stack_moves_through_every_plane(self) -> None:
        z = ZSettings("stack", -2, 2, 1, begin_captured=True, end_captured=True)
        _record, status = self.run_cycle(self.plan(z=z, focus_points=()))
        self.assertEqual(status.state, "complete")
        z_moves = [value for kind, value, _ in self.backend.move_log if kind == "z"]
        self.assertTrue({-2.0, -1.0, 0.0, 1.0, 2.0}.issubset(set(z_moves)))
        _header, planes = read_tile(self.controller.experiment.cycle_folder(1), 0)
        self.assertEqual(len(planes), 2 * 5)

    def test_shutter_is_closed_during_moves_in_per_tile_mode(self) -> None:
        options = AcquisitionOptions(shutter_per_frame=False)
        z = ZSettings("stack", 0, 2, 1, begin_captured=True, end_captured=True)
        _record, status = self.run_cycle(self.plan(options=options, z=z, focus_points=()))
        self.assertEqual(status.state, "complete")
        self.assertEqual(self.backend.light_on_during_moves, 0)
        self.assertFalse(self.backend.shutter_open)

    def test_consistent_approach_moves_past_each_target_first(self) -> None:
        config = replace(self.config, approach_offset_um=50.0)
        backend = SlowSimulator(config, frame_px=16)
        backend.change_objective(config.objective("20x_air"))
        controller = ScanController(config, backend)
        try:
            controller.start_cycle(self.plan(options=AcquisitionOptions(consistent_approach=True)))
            controller.wait(30)
            xy = [(x, y) for kind, x, y in backend.move_log if kind == "xy"]
            tiles = controller.experiment.tiles
            self.assertEqual(len(xy), 2 * len(tiles))
            for (ax, ay), (tx, ty) in zip(xy[0::2], xy[1::2]):
                self.assertAlmostEqual(tx - ax, 50.0)
                self.assertAlmostEqual(ty - ay, 50.0)
        finally:
            controller.close()

    def test_every_cycle_replays_the_same_tiles_with_its_offset(self) -> None:
        plan = self.plan(options=AcquisitionOptions(tile_order="serpentine"))
        self.run_cycle(plan)
        experiment = self.controller.experiment
        moved = replace(
            plan,
            regions=tuple(r.translated(120, -45) for r in plan.regions),
            focus_points=tuple(p.translated_xy(120, -45) for p in plan.focus_points),
            channels=(ChannelSetting("af488", 60, 20),),
        )
        record, status = self.run_cycle(moved)
        self.assertEqual(status.state, "complete")
        self.assertEqual((record.offset_x_um, record.offset_y_um), (120, -45))
        first = tile_records(experiment.cycle_folder(1))
        second = tile_records(experiment.cycle_folder(2))
        self.assertEqual(sorted(first), sorted(second))
        for tile_id in sorted(first):
            ha, hb = first[tile_id], second[tile_id]
            self.assertEqual((ha["region"], ha["stage"]), (hb["region"], hb["stage"]))
            self.assertEqual((ha["tile"], ha["row"], ha["col"]), (hb["tile"], hb["row"], hb["col"]))
            self.assertAlmostEqual(hb["x_um"] - ha["x_um"], 120)
            self.assertAlmostEqual(hb["y_um"] - ha["y_um"], -45)
        reloaded = Experiment.load(experiment.root)
        self.assertEqual([c.status for c in reloaded.cycles], ["complete", "complete"])
        self.assertEqual(reloaded.tiles, experiment.tiles)

    def test_fixed_parameters_are_enforced_between_cycles(self) -> None:
        plan = self.plan()
        self.run_cycle(plan)
        reshaped = replace(plan, regions=(ScanRegion(-1500, -1000, 3100, 2000),))
        self.assertTrue(any("ROI set changed" in p for p in self.controller.preflight(reshaped)))
        other_overlap = replace(plan, overlap_x_percent=20)
        self.assertTrue(any("overlap" in p.lower() for p in self.controller.preflight(other_overlap)))
        with self.assertRaises(ControllerError):
            self.controller.start_cycle(other_overlap)

    def test_pause_waits_between_tiles_and_resume_continues(self) -> None:
        self.backend.delay = 0.01
        self.controller.start_cycle(self.plan())
        time.sleep(0.05)
        self.controller.pause()
        deadline = time.time() + 5
        while self.controller.status().state != "paused" and time.time() < deadline:
            time.sleep(0.01)
        paused = self.controller.status()
        self.assertEqual(paused.state, "paused")
        snaps = self.backend.snap_count
        time.sleep(0.15)
        self.assertEqual(self.backend.snap_count, snaps, "no frames while paused")
        self.assertEqual(paused.tiles_done * 2, snaps, "pause happens on a tile boundary")
        self.controller.resume()
        self.assertTrue(self.controller.wait(30))
        self.assertEqual(self.controller.status().state, "complete")

    def test_abort_then_resume_reacquires_only_missing_tiles(self) -> None:
        self.backend.delay = 0.005
        plan = self.plan()
        self.controller.start_cycle(plan)
        time.sleep(0.06)
        self.controller.abort()
        self.assertTrue(self.controller.wait(10))
        status = self.controller.status()
        self.assertEqual(status.state, "aborted")
        experiment = self.controller.experiment
        saved = completed_tile_indices(experiment.cycle_folder(1))
        self.assertEqual(len(saved), status.tiles_done)
        self.assertLess(len(saved), len(experiment.tiles))
        self.assertFalse(list((experiment.cycle_folder(1) / "tiles").glob("*.tmp")))
        self.assertTrue(any("unfinished" in p for p in self.controller.preflight(plan)))

        # Simulate restarting the program: a new controller opens the experiment.
        controller = ScanController(self.config, self.backend)
        controller.open_experiment(experiment.root)
        record, done, total = controller.resume_info()
        self.assertEqual((record.number, done), (1, len(saved)))
        self.backend.delay = 0.0
        snaps_before = self.backend.snap_count
        controller.resume_cycle()
        self.assertTrue(controller.wait(30))
        final = controller.status()
        self.assertEqual(final.state, "complete")
        self.assertEqual(final.tiles_skipped, len(saved))
        self.assertEqual(self.backend.snap_count - snaps_before, (total - len(saved)) * 2)
        self.assertEqual(completed_tile_indices(experiment.cycle_folder(1)), set(range(total)))
        self.assertEqual(Experiment.load(experiment.root).cycles[0].status, "complete")
        controller.close()

    def test_unsafe_current_z_blocks_single_plane_scan(self) -> None:
        plan = self.plan(z=ZSettings("single"), focus_points=())
        self.assertEqual(self.controller.preflight(plan, check_disk=False), [])
        self.backend._position = replace(self.backend.position, z_um=99999)
        self.assertTrue(any("Current Z" in p for p in self.controller.preflight(plan, check_disk=False)))

    def test_flat_field_profiles_are_written_after_the_cycle(self) -> None:
        record, status = self.run_cycle(self.plan(options=AcquisitionOptions(flat_field=True)))
        self.assertIn("flat-field", status.message)
        folder = self.controller.experiment.cycle_folder(record.number) / "flatfield"
        profile = np.load(folder / "dapi.npy")
        self.assertAlmostEqual(float(profile.mean()), 1.0, places=3)
        self.assertEqual(Experiment.load(self.controller.experiment.root).cycles[0].flat_field, "cycle_01/flatfield")


class FakeCore:
    """Minimal stand-in for CMMCorePlus used to test the adapter's logic."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.xy = (0.0, 0.0)
        self.z = 0.0
        self.state = 2  # turret position 3 with offset -1 → 20x_air
        self.auto_shutter = True
        self.fail_snap = False

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))
        return record

    def getXYPosition(self):
        return self.xy

    def getPosition(self):
        return self.z

    def getState(self, _device):
        return self.state

    def setXYPosition(self, x, y):
        self.calls.append(("setXYPosition", x, y))
        self.xy = (x, y)

    def setPosition(self, z):
        self.calls.append(("setPosition", z))
        self.z = z

    def setState(self, device, state):
        self.calls.append(("setState", device, state))
        self.state = state

    def getXYStageDevice(self):
        return "XY"

    def getFocusDevice(self):
        return "Z"

    def getAutoShutter(self):
        return self.auto_shutter

    def setAutoShutter(self, value):
        self.calls.append(("setAutoShutter", value))
        self.auto_shutter = value

    def snapImage(self):
        if self.fail_snap:
            raise RuntimeError("Camera timeout")

    def getImage(self):
        return np.arange(12, dtype=np.uint16).reshape(3, 4)


class MMCoreBackendTests(unittest.TestCase):
    def make(self, **hardware):
        config = load_config()
        config = replace(config, hardware={**config.hardware, **hardware})
        core = FakeCore()
        return config, core, MMCoreBackend(config, core=core)

    def test_reads_current_objective_from_turret(self) -> None:
        _config, _core, backend = self.make()
        self.assertEqual(backend.current_objective_id, "20x_air")

    def test_out_of_range_moves_never_reach_the_stage(self) -> None:
        config, core, backend = self.make()
        with self.assertRaises(StageLimitError):
            backend.move_xy(config.stage_limits.x_max_um + 1, 0)
        with self.assertRaises(StageLimitError):
            backend.move_z(config.objective("20x_air").safe_z_max_um + 1)
        self.assertFalse([c for c in core.calls if c[0] in ("setXYPosition", "setPosition")])

    def test_objective_change_goes_to_clearance_before_rotating(self) -> None:
        config, core, backend = self.make(objective_change_z_um=-4000)
        backend.change_objective(config.objective("63x_oil"))
        names = [c[0] for c in core.calls]
        self.assertLess(names.index("setPosition"), names.index("setState"))
        self.assertIn(("setPosition", -4000.0), core.calls)
        self.assertIn(("setState", "Objective", 5), core.calls)
        self.assertEqual(backend.current_objective_id, "63x_oil")

    def test_channel_uses_preset_exposure_and_power_property(self) -> None:
        config, core, backend = self.make(
            power_properties={"dapi": {"device": "LED", "property": "Intensity", "scale": 2.0}}
        )
        backend.set_channel(config.channel("dapi"), 40, 25)
        self.assertIn(("setConfig", "Channel", "DAPI"), core.calls)
        self.assertIn(("setExposure", 40.0), core.calls)
        self.assertIn(("setProperty", "LED", "Intensity", 50.0), core.calls)
        with self.assertRaises(HardwareError):
            backend.set_channel(config.channel("dapi"), 40, 99)

    def test_frames_are_oriented_to_stage_axes(self) -> None:
        _config, _core, backend = self.make()
        backend._config = replace(backend._config, camera_orientation=CameraOrientation(flip_x=True))
        frame = backend.snap()
        self.assertEqual(frame[0].tolist(), [3, 2, 1, 0])

    def test_device_errors_become_hardware_errors(self) -> None:
        _config, core, backend = self.make()
        core.fail_snap = True
        with self.assertRaises(HardwareError) as context:
            backend.snap()
        self.assertIn("Camera timeout", str(context.exception))

    def test_missing_channel_preset_is_rejected_at_startup(self) -> None:
        config = load_config()
        presets = dict(config.hardware["channel_presets"])
        presets.pop("cy5")
        config = replace(config, hardware={**config.hardware, "channel_presets": presets})
        with self.assertRaises(HardwareError):
            MMCoreBackend(config, core=FakeCore())


if __name__ == "__main__":
    unittest.main()
