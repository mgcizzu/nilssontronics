from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from microscope_ui.config import load_config
from microscope_ui.models import ScanRegion
from microscope_ui.mosaic import (
    PART_SUFFIX,
    DiskBackedMosaic,
    MosaicBounds,
    completed_tile_indices,
    mosaic_bounds_for_regions,
    read_tile,
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
            paths = mosaic.finish_tile()
            composite = mosaic.render_viewport_ppm(bounds, 80, 40, DiskBackedMosaic.COMPOSITE)
            dapi = mosaic.render_viewport_ppm(bounds, 80, 40, "DAPI")
            expected_preview_bytes = mosaic.disk_preview_bytes
            mosaic.close()

            self.assertEqual(len(paths), 14)
            self.assertTrue(all(path.exists() for path in paths))
            # One plain uint16 2-D TIFF per plane, channel = position in the cycle.
            plane = tifffile.imread(root / "R01--Stage0001--Z06--C00.tif")
            self.assertEqual((plane.shape, plane.dtype), ((16, 20), np.uint16))
            self.assertEqual(int(plane[0, 0]), 160)
            self.assertTrue((root / "R01--Stage0001--Z00--C01.tif").exists())
            header, planes = read_tile(root, 1)
            self.assertEqual((header["row"], header["col"]), (2, 3))
            self.assertEqual(len(planes), 14)
            self.assertEqual(int(planes[("dapi", 6)][0, 0]), 160)
            self.assertEqual(int(planes[("af488", 0)][5, 5]), 900)
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
            # A crash in the middle of tile 2 leaves only a .part file, never a .tif.
            mosaic.begin_tile(2, 0, 0, 100, 100, z_planes=1)
            mosaic.write_plane("dapi", 0, np.ones((8, 8), np.uint16))
            self.assertTrue((root / f"R01--Stage0002--Z00--C00.tif{PART_SUFFIX}").exists())
            self.assertFalse((root / "R01--Stage0002--Z00--C00.tif").exists())
            mosaic._index.write('{"tile": 9, "pa')  # torn append
            mosaic._index.flush()
            mosaic.close()
            self.assertEqual(completed_tile_indices(root), {0, 1})
            self.assertFalse(list(root.glob(f"*{PART_SUFFIX}")))
            # A journalled tile whose file vanished is no longer complete.
            (root / "R01--Stage0001--Z00--C00.tif").unlink()
            self.assertEqual(completed_tile_indices(root), {0})
            # Re-opening for resume keeps the preview maps and the journal.
            reopened = DiskBackedMosaic(root, bounds, channels, max_preview_side=32)
            self.assertGreater(reopened.display_range("dapi")[1], 0)
            reopened.close()

    def test_stale_part_files_are_removed_on_open(self) -> None:
        channels = (self.config.channel("dapi"),)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stale = root / f"R01--Stage0005--Z00--C00.tif{PART_SUFFIX}"
            stale.write_bytes(b"half a plane")
            DiskBackedMosaic(root, MosaicBounds(0, 10, 0, 10), channels, max_preview_side=8).close()
            self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
