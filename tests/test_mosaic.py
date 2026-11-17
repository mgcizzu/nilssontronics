from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from microscope_ui.config import load_config
from microscope_ui.models import ScanRegion
from microscope_ui.mosaic import DiskBackedMosaic, mosaic_bounds_for_regions


class DiskBackedMosaicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_tile_is_closed_to_disk_and_viewport_is_bounded(self) -> None:
        region = ScanRegion(-5000, -2500, 10000, 5000)
        channels = (self.config.channel("dapi"), self.config.channel("af488"))
        bounds = mosaic_bounds_for_regions((region,), 500, 500)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mosaic = DiskBackedMosaic(root, bounds, channels, max_preview_side=64)
            tile_path = mosaic.write_tile(1, 0, 0, 500, 500, z_planes=7)
            composite = mosaic.render_viewport_ppm(
                bounds, 80, 40, DiskBackedMosaic.COMPOSITE
            )
            dapi = mosaic.render_viewport_ppm(bounds, 80, 40, "DAPI")
            expected_preview_bytes = mosaic.disk_preview_bytes
            mosaic.close()

            self.assertTrue(tile_path.exists())
            self.assertEqual(tile_path.read_bytes()[:10], b"SIMSTACK1\n")
            records = [
                json.loads(line)
                for line in (root / "tiles.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["z_planes"], 7)
            self.assertEqual(len(composite), len(dapi))
            self.assertNotEqual(composite, dapi)
            self.assertLessEqual(expected_preview_bytes, 64 * 64 * 3 * 3)


if __name__ == "__main__":
    unittest.main()
