from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from microscope_ui.config import load_config
from microscope_ui.models import ScanRegion
import numpy as np

from microscope_ui.mosaic import (
    TSTACK_MAGIC,
    DiskBackedMosaic,
    MosaicBounds,
    completed_tile_indices,
    mosaic_bounds_for_regions,
    read_tile_stack,
)


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
            mosaic.begin_tile(1, 0, 0, 500, 500, z_planes=7, row=2, col=3)
            for channel in channels:
                for z in range(7):
                    frame = np.full((16, 20), 100 + 10 * z, dtype=np.uint16)
                    frame[4:8, 4:8] = 4000 if channel.id == "dapi" else 900
                    mosaic.write_plane(channel.id, z, frame, z_um=float(z))
            tile_path = mosaic.finish_tile()
            composite = mosaic.render_viewport_ppm(bounds, 80, 40, DiskBackedMosaic.COMPOSITE)
            dapi = mosaic.render_viewport_ppm(bounds, 80, 40, "DAPI")
            expected_preview_bytes = mosaic.disk_preview_bytes
            mosaic.close()

            self.assertTrue(tile_path.exists())
            self.assertEqual(tile_path.read_bytes()[:8], TSTACK_MAGIC)
            header, planes = read_tile_stack(tile_path)
            self.assertEqual((header["row"], header["col"]), (2, 3))
            self.assertEqual(len(planes), 14)
            self.assertEqual(int(planes[("dapi", 6)][0, 0]), 160)
            records = [
                json.loads(line)
                for line in (root / "tiles.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["z_planes"], 7)
            self.assertEqual(len(composite), len(dapi))
            self.assertNotEqual(composite, dapi)
            self.assertLessEqual(expected_preview_bytes, 64 * 64 * 2 * 2)

    def test_journal_lists_only_complete_tiles_and_survives_a_torn_line(self) -> None:
        channels = (self.config.channel("dapi"),)
        bounds = MosaicBounds(-1000, 1000, -1000, 1000)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mosaic = DiskBackedMosaic(root, bounds, channels, max_preview_side=32)
            for index in range(2):
                mosaic.begin_tile(index, 0, 0, 100, 100, z_planes=1)
                mosaic.write_plane("dapi", 0, np.ones((8, 8), np.uint16))
                mosaic.finish_tile()
            # A crash in the middle of tile 2 leaves only a temporary file.
            mosaic.begin_tile(2, 0, 0, 100, 100, z_planes=1)
            mosaic.write_plane("dapi", 0, np.ones((8, 8), np.uint16))
            mosaic._index.write('{"tile": 9, "pa')  # torn append
            mosaic._index.flush()
            mosaic.close()
            self.assertEqual(completed_tile_indices(root), {0, 1})
            self.assertFalse(list((root / "tiles").glob("*.tmp")))
            # Re-opening for resume keeps the preview maps and the journal.
            reopened = DiskBackedMosaic(root, bounds, channels, max_preview_side=32)
            self.assertGreater(reopened.display_range("dapi")[1], 0)
            reopened.close()


if __name__ == "__main__":
    unittest.main()
