"""Regression tests for the second review round (stage, writer, calibration,
completion order, flat-field memory, focus geometry)."""

from __future__ import annotations

from dataclasses import replace
import json
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from microscope_ui import flatfield as flatfield_module
from microscope_ui.acquisition import BoundedAsyncWriter
from microscope_ui.backend import SimulatedMicroscope
from microscope_ui.config import load_config
from microscope_ui.controller import ControllerError, ScanController
from microscope_ui.engine import AcquisitionEngine, CycleRun
from microscope_ui.experiment import Experiment
from microscope_ui.focus import FocusSurface
from microscope_ui.models import (
    AcquisitionPlan,
    ChannelSetting,
    FocusPoint,
    ScanRegion,
    Tile,
    ZSettings,
)
from microscope_ui.mosaic import DiskBackedMosaic, MosaicBounds


class SlowSimulator(SimulatedMicroscope):
    delay = 0.0

    def snap(self):
        time.sleep(self.delay)
        return super().snap()


class ControllerCase(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch(
            "microscope_ui.planner.shutil.disk_usage", return_value=mock.Mock(free=10 * 1024**4)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = load_config()
        self.backend = SlowSimulator(self.config, frame_px=24)
        self.backend.change_objective(self.config.objective("20x_air"))
        self.controller = ScanController(self.config, self.backend)
        self.addCleanup(self.controller.close)

    def plan(self, **changes) -> AcquisitionPlan:
        values = dict(
            project="P", sample="S", output_root=Path(self.temp.name), objective_id="20x_air",
            channels=(ChannelSetting("dapi", 10, 5),), regions=(ScanRegion(-1500, -1000, 3000, 2000),),
            focus_points=(), z=ZSettings(mode="single"),
        )
        values.update(changes)
        return AcquisitionPlan(**values)


# -- 1. Tk stage controls during acquisition ----------------------------------
class TkStageLockTests(unittest.TestCase):
    def test_stage_moves_are_refused_while_a_cycle_runs(self) -> None:
        try:
            from microscope_ui.app import ScanConsole
        except ImportError:  # pragma: no cover - no Tk
            self.skipTest("tkinter not available")
        moves = []
        messages = []
        fake = SimpleNamespace(
            controller=SimpleNamespace(busy=True),
            progress_text_var=SimpleNamespace(set=messages.append),
            bell=lambda: None,
        )
        self.assertFalse(ScanConsole._safe_move(fake, lambda *a: moves.append(a), 500.0, 500.0))
        self.assertEqual(moves, [])
        self.assertIn("in use by the running acquisition", messages[-1])
        fake.controller.busy = False
        self.assertTrue(ScanConsole._safe_move(fake, lambda *a: moves.append(a), 500.0, 500.0))
        self.assertEqual(moves, [(500.0, 500.0)])


# -- 2. Engine must not finish while its writer still runs ---------------------
class BlockingSink:
    """A sink whose plane writes block until released (a stalled SSD)."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.writing = threading.Event()
        self.closed_while_writing = False
        self.busy = False

    def begin_tile(self, *args, **kwargs) -> None:
        pass

    def write_plane(self, *args, **kwargs) -> None:
        self.busy = True
        self.writing.set()
        self.release.wait(10)
        self.busy = False

    def finish_tile(self):
        return ()

    def abandon_tile(self) -> None:
        pass


class WriterShutdownTests(unittest.TestCase):
    def test_aborted_engine_waits_for_a_stalled_writer(self) -> None:
        config = load_config()
        backend = SimulatedMicroscope(config, frame_px=8)
        backend.change_objective(config.objective("20x_air"))
        tiles = tuple(Tile(i, 0, i, i * 500.0, 0.0) for i in range(4))
        sink = BlockingSink()
        writer = BoundedAsyncWriter(sink, queue_capacity=2)
        run = CycleRun(
            objective_id="20x_air", tiles=tiles, channels=(ChannelSetting("dapi", 10, 5),),
            z_positions={t.index: (0.0,) for t in tiles},
        )
        engine = AcquisitionEngine(config, backend, writer, run, writer_stop_timeout_s=0.1)
        engine.start()
        self.assertTrue(sink.writing.wait(5))
        engine.abort()
        time.sleep(0.6)  # well past the writer stop timeout
        self.assertFalse(engine.status().finished, "reported final while the writer is mid-write")
        self.assertFalse(writer.stopped)
        sink.release.set()
        self.assertTrue(engine.join(10))
        self.assertTrue(writer.stopped)
        self.assertEqual(engine.status().state, "aborted")


# -- 3. Calibration snapshot ------------------------------------------------------
class CalibrationTests(ControllerCase):
    def test_new_cycle_and_resume_refuse_a_changed_pixel_size(self) -> None:
        self.backend.delay = 0.02
        self.controller.start_cycle(self.plan())
        time.sleep(0.15)
        self.controller.abort()
        self.assertTrue(self.controller.wait(10))
        experiment = self.controller.experiment
        self.assertAlmostEqual(experiment.calibration["pixel_size_um"], 0.325)
        self.assertEqual(Experiment.load(experiment.root).calibration, experiment.calibration)

        doubled = replace(
            self.config,
            objectives=tuple(
                replace(o, pixel_size_um=o.pixel_size_um * 2) if o.id == "20x_air" else o
                for o in self.config.objectives
            ),
        )
        self.controller.config = doubled
        with self.assertRaises(ControllerError) as context:
            self.controller.resume_cycle()
        self.assertIn("pixel size (µm/px): 0.325 when cycle 1 started, now 0.65", str(context.exception))
        problems = self.controller.preflight(self.plan(), check_disk=False)
        self.assertTrue(any("no longer matches this experiment" in p for p in problems))

        self.controller.config = self.config  # original calibration restored → resumable
        self.controller.resume_cycle()
        self.assertTrue(self.controller.wait(30))
        self.assertEqual(self.controller.status().state, "complete")

    def test_experiment_without_calibration_record_is_not_extended(self) -> None:
        self.controller.start_cycle(self.plan())
        self.assertTrue(self.controller.wait(30))
        self.controller.experiment.calibration = {}
        problems = self.controller.preflight(self.plan(), check_disk=False)
        self.assertTrue(any("no calibration record" in p for p in problems))


# -- 4. Persist before publishing completion -------------------------------------
class CompletionOrderTests(ControllerCase):
    def test_cycle_record_is_saved_before_the_controller_becomes_idle(self) -> None:
        seen = {}
        original = self.controller._cycle_finished

        def slow_persist(experiment, number, status):
            seen["busy_during_save"] = self.controller.busy
            seen["finished_during_save"] = self.controller.status().finished
            # Even if someone switched experiments now, the record goes to its own experiment.
            seen["experiment"] = experiment
            self.controller.experiment = None
            time.sleep(0.2)
            original(experiment, number, status)

        self.controller._cycle_finished = slow_persist
        record = self.controller.start_cycle(self.plan())
        experiment = self.controller.experiment
        self.assertTrue(self.controller.wait(30))
        self.assertTrue(seen["busy_during_save"])
        self.assertFalse(seen["finished_during_save"])
        self.assertIs(seen["experiment"], experiment)
        self.assertEqual(Experiment.load(experiment.root).cycle(record.number).status, "complete")


class FailurePathTests(ControllerCase):
    def test_quit_never_closes_storage_under_a_busy_writer(self) -> None:
        release = threading.Event()
        writing = threading.Event()
        real_write = DiskBackedMosaic.write_plane

        def stalled_write(mosaic, *args, **kwargs):
            writing.set()
            release.wait(10)
            return real_write(mosaic, *args, **kwargs)

        with mock.patch.object(DiskBackedMosaic, "write_plane", stalled_write):
            self.controller.start_cycle(self.plan())
            self.assertTrue(writing.wait(5))
            mosaic = self.controller.mosaic
            self.assertFalse(self.controller.close(timeout=0.3))
            self.assertIs(self.controller.mosaic, mosaic)
            self.assertFalse(mosaic.closed, "storage closed under an in-flight write")
            release.set()
            self.assertTrue(self.controller.close(timeout=10))
        self.assertIsNone(self.controller.mosaic)
        self.assertTrue(mosaic.closed)
        self.assertEqual(self.controller.status().state, "aborted")

    def test_failed_final_save_is_reported_and_stays_resumable(self) -> None:
        self.controller.start_cycle(self.plan())
        experiment = self.controller.experiment
        real_save = experiment.save

        def failing_save():
            if any(cycle.status == "complete" for cycle in experiment.cycles):
                raise OSError("No space left on device")
            return real_save()

        experiment.save = failing_save
        self.assertTrue(self.controller.wait(30))
        status = self.controller.status()
        self.assertEqual(status.state, "failed")
        self.assertIn("could not be saved", status.message)
        self.assertIn("No space left on device", status.error)
        # Memory and disk agree, and the cycle can be resumed.
        self.assertEqual(experiment.cycle(1).status, "running")
        self.assertEqual(Experiment.load(experiment.root).cycle(1).status, "running")
        self.assertIsNotNone(self.controller.resume_info())

        experiment.save = real_save  # the disk problem is fixed
        self.controller.resume_cycle()
        self.assertTrue(self.controller.wait(30))
        self.assertEqual(self.controller.status().state, "complete")
        self.assertEqual(Experiment.load(experiment.root).cycle(1).status, "complete")


# -- 5. Flat-field memory -----------------------------------------------------------
class FlatFieldMemoryTests(unittest.TestCase):
    def test_one_middle_plane_per_tile_and_bounded_binned_stack(self) -> None:
        config = load_config()
        channels = (config.channel("dapi"), config.channel("cy5"))
        rng = np.random.default_rng(0)
        vignette = np.outer(np.hanning(256) + 0.5, np.hanning(192) + 0.5).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mosaic = DiskBackedMosaic(root, MosaicBounds(-1e4, 1e4, -1e4, 1e4), channels, max_preview_side=32)
            for tile in range(6):
                mosaic.begin_tile(tile, 0, 0, 100, 100, z_planes=5)
                for channel in channels:
                    for z in range(5):
                        frame = (1000 + 200 * rng.random((256, 192))) * vignette
                        mosaic.write_plane(channel.id, z, frame.astype(np.uint16))
                mosaic.finish_tile()
            mosaic.close()

            reads = []
            real_read = flatfield_module.read_plane

            def counting_read(cycle_root, record, channel_id, z_index):
                reads.append((record["tile"], channel_id, z_index))
                return real_read(cycle_root, record, channel_id, z_index)

            budget = 6 * 64 * 48 * 4  # forces binning of the 256×192 frames
            with mock.patch.object(flatfield_module, "read_plane", counting_read):
                out = flatfield_module.estimate_flat_fields(
                    root, ("dapi", "cy5"), memory_budget_bytes=budget
                )
            self.assertIsNotNone(out)
            # Only the middle plane (z=2) of each tile, once per channel.
            self.assertEqual(sorted(reads), sorted((t, c, 2) for t in range(6) for c in ("dapi", "cy5")))
            summary = json.loads((out / "flatfield.json").read_text())["channels"]
            for channel in ("dapi", "cy5"):
                binning = summary[channel]["binning"]
                self.assertGreater(binning, 1)
                stack_bytes = 6 * (256 // binning) * (192 // binning) * 4
                self.assertLessEqual(stack_bytes, budget)
            for channel in ("dapi", "cy5"):
                profile = np.load(out / f"{channel}.npy")
                self.assertEqual(profile.shape, (256, 192))  # full resolution again
                self.assertAlmostEqual(float(profile.mean()), 1.0, places=3)
                # Brighter centre than corners, like the simulated vignette.
                self.assertGreater(profile[128, 96], 1.5 * profile[5, 5])


# -- 6. Focus geometry ----------------------------------------------------------------
class FocusGeometryTests(ControllerCase):
    def test_nearly_collinear_points_get_an_actionable_error(self) -> None:
        points = (FocusPoint(-1000, 0, 1), FocusPoint(0, 0.1, 2), FocusPoint(1000, 0, 3))
        with self.assertRaises(ValueError) as context:
            FocusSurface(points)
        self.assertIn("one line", str(context.exception))
        self.assertIn("Add a focus point", str(context.exception))
        plan = self.plan(
            regions=(ScanRegion(-1200, -300, 2400, 600),),
            focus_points=points, z=ZSettings(mode="focus_map"),
        )
        problems = self.controller.preflight(plan, check_disk=False)  # no exception
        self.assertTrue(any("one line" in p for p in problems))

    def test_accepted_layouts_interpolate_every_point_and_never_raise(self) -> None:
        rng = random.Random(7)
        checked = 0
        for _ in range(300):
            count = rng.randint(3, 9)
            scale = rng.choice((10.0, 1000.0, 30000.0))
            thin = rng.choice((1.0, 0.05, 0.012))
            points = tuple(
                FocusPoint(rng.uniform(-scale, scale), rng.uniform(-scale, scale) * thin, rng.uniform(-5, 5))
                for _ in range(count)
            )
            try:
                surface = FocusSurface(points)
            except ValueError:
                continue
            checked += 1
            for point in surface.points:
                self.assertAlmostEqual(surface.z_at(point.x_um, point.y_um), point.z_um, places=6)
            for x in np.linspace(-2 * scale, 2 * scale, 7):
                for y in np.linspace(-2 * scale, 2 * scale, 7):
                    z = surface.z_at(float(x), float(y))
                    self.assertTrue(min(p.z_um for p in surface.points) - 1e-6 <= z
                                    <= max(p.z_um for p in surface.points) + 1e-6)
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()
