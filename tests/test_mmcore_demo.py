"""MMCoreBackend against a real CMMCorePlus loaded with MMConfig_demo.cfg.

Skipped unless a Micro-Manager installation is found (``mmcore install``, a
``/Applications/Micro-Manager*`` folder, or ``MICROMANAGER_PATH``). Unlike the
fake core in ``test_cycles.py`` this exercises the real device adapters
(DemoCamera), so it catches API mismatches with pymmcore-plus.
"""

from __future__ import annotations

from dataclasses import replace
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    from pymmcore_plus import CMMCorePlus, find_micromanager
except ImportError:  # pragma: no cover
    CMMCorePlus = None
    find_micromanager = None

from microscope_ui.backend import StageLimitError
from microscope_ui.backend_mmcore import MMCoreBackend
from microscope_ui.config import load_config
from microscope_ui.controller import ScanController
from microscope_ui.models import AcquisitionPlan, ChannelSetting, ScanRegion, ZSettings
from microscope_ui.mosaic import read_tile, tile_records


def _mm_path() -> str | None:
    if find_micromanager is None:
        return None
    return os.environ.get("MICROMANAGER_PATH") or find_micromanager()


MM_PATH = _mm_path()
DEMO_CFG = Path(MM_PATH, "MMConfig_demo.cfg") if MM_PATH else None


@unittest.skipUnless(DEMO_CFG and DEMO_CFG.exists(), "Micro-Manager with MMConfig_demo.cfg not found")
class MMCoreDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = load_config()
        hardware = dict(base.hardware)
        hardware.update(
            backend="mmcore",
            mm_config_path=str(DEMO_CFG),
            mm_adapter_path=MM_PATH,
            pixel_size_configs={"10x_air": "Res10x", "20x_air": "Res20x", "40x_air": "Res40x"},
            # The DemoCamera "Z" stage (DStage) only travels ±300 µm.
            objective_change_z_um=-250,
        )
        limits = replace(base.stage_limits, z_min_um=-300, z_max_um=300)
        objectives = tuple(
            replace(o, safe_z_min_um=-280, safe_z_max_um=200) for o in base.objectives
        )
        cls.config = replace(base, hardware=hardware, stage_limits=limits, objectives=objectives)

    def setUp(self) -> None:
        self.core = CMMCorePlus(mm_path=MM_PATH)
        self.core.loadSystemConfiguration(str(DEMO_CFG))
        # Keep frames small so the end-to-end cycle stays fast.
        self.core.setProperty("Camera", "OnCameraCCDXSize", 64)
        self.core.setProperty("Camera", "OnCameraCCDYSize", 64)
        self.backend = MMCoreBackend(self.config, core=self.core)

    def tearDown(self) -> None:
        self.backend.close()
        self.core.unloadAllDevices()

    def test_motion_and_limits(self) -> None:
        self.backend.change_objective(self.config.objective("20x_air"))
        pos = self.backend.move_xy(1234.0, -567.0)
        self.assertAlmostEqual(pos.x_um, 1234.0, places=1)
        self.assertAlmostEqual(pos.y_um, -567.0, places=1)
        pos = self.backend.move_z(12.5)
        self.assertAlmostEqual(pos.z_um, 12.5, places=1)
        before = self.backend.position
        with self.assertRaises(StageLimitError):
            self.backend.move_xy(self.config.stage_limits.x_max_um + 1, 0)
        with self.assertRaises(StageLimitError):
            self.backend.move_z(self.config.objective("20x_air").safe_z_max_um + 1)
        self.assertEqual(self.backend.position, before, "rejected targets must not move the stage")

    def test_objective_change_moves_to_clearance_and_sets_pixel_size(self) -> None:
        objective = self.config.objective("10x_air")
        self.backend.change_objective(objective)
        self.assertEqual(self.backend.current_objective_id, "10x_air")
        self.assertEqual(int(self.core.getState("Objective")), objective.turret_position - 1)
        self.assertAlmostEqual(
            self.backend.position.z_um, self.config.hardware["objective_change_z_um"], places=1
        )
        self.assertEqual(self.core.getCurrentPixelSizeConfig(), "Res10x")
        # A fresh backend reads the turret back to the same objective.
        self.assertEqual(MMCoreBackend(self.config, core=self.core).current_objective_id, "10x_air")

    def test_channels_shutter_and_snap(self) -> None:
        channel = self.config.channel("af488")
        self.backend.set_channel(channel, 25.0, 10.0)
        self.assertEqual(self.core.getCurrentConfig("Channel"), "FITC")
        self.assertAlmostEqual(self.core.getExposure(), 25.0)
        self.backend.set_shutter_per_frame(True)
        self.assertTrue(self.core.getAutoShutter())
        self.backend.set_shutter_per_frame(False)
        self.backend.set_shutter_open(True)
        self.assertTrue(self.core.getShutterOpen())
        self.backend.set_shutter_open(False)
        self.assertFalse(self.core.getShutterOpen())
        frame = self.backend.snap()
        self.assertEqual(frame.shape, (64, 64))
        self.assertEqual(frame.dtype, np.uint16)

    def test_full_cycle_through_controller(self) -> None:
        self.backend.change_objective(self.config.objective("20x_air"))
        self.backend.move_z(0.0)
        free = mock.patch(
            "microscope_ui.planner.shutil.disk_usage",
            return_value=mock.Mock(free=10 * 1024**4),
        )
        free.start()
        self.addCleanup(free.stop)
        with tempfile.TemporaryDirectory() as temp:
            controller = ScanController(self.config, self.backend)
            try:
                plan = AcquisitionPlan(
                    project="Demo",
                    sample="MMConfig_demo",
                    output_root=Path(temp),
                    objective_id="20x_air",
                    channels=(ChannelSetting("dapi", 10, 5), ChannelSetting("af488", 10, 5)),
                    regions=(ScanRegion(0, 0, 1200, 800),),
                    focus_points=(),
                    z=ZSettings(mode="single"),
                )
                record = controller.start_cycle(plan)
                self.assertTrue(controller.wait(60))
                status = controller.status()
                self.assertEqual(status.state, "complete", status)
                self.assertEqual(status.tiles_done, status.tiles_total)
                folder = controller.experiment.cycle_folder(record.number)
                self.assertEqual(len(tile_records(folder)), status.tiles_total)
                _header, planes = read_tile(folder, 0)
                self.assertEqual(set(planes), {("dapi", 0), ("af488", 0)})
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
