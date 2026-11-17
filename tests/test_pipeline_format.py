"""Cycle folders must be valid ISS_preprocessing input (mode='tif_autosaved').

The regexes and XML paths below are copied from ISS_preprocessing's
``TiffHandler`` and Leica-XML helpers (github.com/Moldia/ISS2025), so a change
on either side shows up here without installing the pipeline.
"""

from __future__ import annotations

from dataclasses import replace
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

from microscope_ui.backend import SimulatedMicroscope
from microscope_ui.config import load_config
from microscope_ui.controller import ScanController
from microscope_ui.models import AcquisitionPlan, ChannelSetting, ScanRegion, Tile, ZSettings
from microscope_ui.mosaic import pipeline_regions, plane_filename, z_digits_for, TileName

# --- ISS_preprocessing, TiffHandler(mode="tif_autosaved") -------------------
CHANNEL = re.compile(r"--C0*(\d+)(?=\D|$)", re.IGNORECASE)
TILE = re.compile(r"--Stage(\d+)--")
SAMPLE_TILE = re.compile(r"--Stage0+--")


def pipeline_region_of(filename: str) -> str:
    base = filename.rsplit(".", 1)[0]
    return base.split("--Stage", 1)[0].strip()


def pipeline_tiffs(folder: Path) -> list[Path]:
    """TiffHandler._iter_tiffs: top-level *.tif, skipping names containing 'dw'."""

    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".tif" and "dw" not in f.name
    ]


def pixel_size_from_xml(root: ET.Element) -> float:
    """tiff_px_um_from_dim + normalize_pixel_size_to_um (meters band 1e-9..1e-4)."""

    sizes = []
    for dim_id in ("1", "2"):
        node = root.find(f".//ImageDescription/Dimensions/DimensionDescription[@DimID='{dim_id}']")
        raw = float(node.attrib["Length"]) / float(node.attrib["NumberOfElements"])
        sizes.append(raw * 1e6 if 1e-9 <= raw <= 1e-4 else raw)
    return sum(sizes) / 2


def tiles(cells):
    return [Tile(i, row, col, col * 900.0, row * 900.0) for i, (row, col) in enumerate(cells)]


class NamingTests(unittest.TestCase):
    def test_file_name_round_trips_through_the_pipeline_regexes(self) -> None:
        name = plane_filename(TileName("R03", 17), z_index=4, channel_index=2, z_digits=2)
        self.assertEqual(name, "R03--Stage0017--Z04--C02.tif")
        self.assertEqual(pipeline_region_of(name), "R03")
        self.assertEqual(int(TILE.search(name).group(1)), 17)
        self.assertEqual(int(CHANNEL.search(name).group(1)), 2)
        self.assertNotIn("dw", name)
        self.assertTrue(SAMPLE_TILE.search(plane_filename(TileName("R01", 0), 0, 0)))

    def test_z_padding_keeps_name_order_equal_to_plane_order(self) -> None:
        for planes in (1, 9, 10, 11, 100, 101):
            digits = z_digits_for(planes)
            names = [plane_filename(TileName("R01", 0), z, 0, digits) for z in range(planes)]
            self.assertEqual(sorted(names), names)

    def test_separate_sections_become_separate_regions(self) -> None:
        # Two blocks far apart on the shared grid, acquired interleaved (raster).
        cells = [(0, 0), (0, 1), (0, 8), (0, 9), (1, 0), (1, 1), (1, 8), (1, 9)]
        names = pipeline_regions(tiles(cells))
        self.assertEqual({names[i].region for i in (0, 1, 4, 5)}, {"R01"})
        self.assertEqual({names[i].region for i in (2, 3, 6, 7)}, {"R02"})
        # Stage numbers are 0-based and contiguous within each region, in acquisition order.
        self.assertEqual([names[i].stage for i in (0, 1, 4, 5)], [0, 1, 2, 3])
        self.assertEqual([names[i].stage for i in (2, 3, 6, 7)], [0, 1, 2, 3])

    def test_diagonal_neighbours_stay_in_one_region(self) -> None:
        names = pipeline_regions(tiles([(0, 0), (1, 1), (2, 2)]))
        self.assertEqual({n.region for n in names.values()}, {"R01"})

    def test_region_tokens_sort_in_region_order(self) -> None:
        cells = [(0, 3 * k) for k in range(12)]  # 12 isolated tiles
        regions = [pipeline_regions(tiles(cells))[i].region for i in range(12)]
        self.assertEqual(sorted(regions), regions)  # R01..R12, not R1, R10, R11, R2
        self.assertEqual(regions[0], "R01")


class CycleFolderTests(unittest.TestCase):
    """A real simulated cycle, checked the way ISS_preprocessing reads it."""

    @classmethod
    def setUpClass(cls) -> None:
        base = load_config()
        # Simulated frames must match the configured camera so tile width × pixel
        # size equals the field of view, as on the real microscope.
        cls.config = replace(base, camera_width_px=96, camera_height_px=64)

    def setUp(self) -> None:
        patcher = mock.patch(
            "microscope_ui.planner.shutil.disk_usage", return_value=mock.Mock(free=10 * 1024**4)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        backend = SimulatedMicroscope(self.config, frame_px=96)
        backend.render_frame = self._frame  # deterministic frames of the configured size
        backend.change_objective(self.config.objective("20x_air"))
        self.controller = ScanController(self.config, backend)
        self.addCleanup(self.controller.close)

    @staticmethod
    def _frame(position, channel):
        return np.full((64, 96), 1000, dtype=np.uint16)

    def run_cycle(self, regions, channels, z=None):
        plan = AcquisitionPlan(
            project="Proj", sample="Slide", output_root=Path(self.temp.name),
            objective_id="20x_air", channels=channels, regions=regions, focus_points=(),
            z=z or ZSettings(mode="single"),
        )
        record = self.controller.start_cycle(plan)
        self.assertTrue(self.controller.wait(60))
        self.assertEqual(self.controller.status().state, "complete")
        return self.controller.experiment.cycle_folder(record.number)

    def test_cycle_folder_is_valid_pipeline_input(self) -> None:
        fov_w, fov_h = self.config.fov_um("20x_air")
        regions = (
            ScanRegion(0, 0, 2.5 * fov_w, 1.5 * fov_h),
            ScanRegion(20 * fov_w, 0, 1.2 * fov_w, 1.2 * fov_h),  # a second, separate section
        )
        channels = (ChannelSetting("dapi", 10, 5), ChannelSetting("cy3", 20, 5), ChannelSetting("cy5", 30, 5))
        z = ZSettings("stack", -1, 1, 1, begin_captured=True, end_captured=True)
        folder = self.run_cycle(regions, channels, z)

        files = pipeline_tiffs(folder)
        self.assertFalse([f for f in folder.iterdir() if f.name.endswith(".part")])
        region_tokens = sorted({pipeline_region_of(f.name) for f in files})
        self.assertEqual(region_tokens, ["R01", "R02"])

        pixel_um = self.config.objective("20x_air").pixel_size_um
        for region in region_tokens:
            own = [f for f in files if f.name.startswith(f"{region}--")]
            channel_ids = sorted({int(CHANNEL.search(f.name).group(1)) for f in own})
            tile_ids = sorted({int(TILE.search(f.name).group(1)) for f in own})
            self.assertEqual(channel_ids, [0, 1, 2])
            self.assertEqual(tile_ids, list(range(len(tile_ids))))
            sample = [f for f in own if SAMPLE_TILE.search(f.name)]
            self.assertEqual(len(sample) // len(channel_ids), 3)  # size_z from tile 0
            stack = [f for f in own if TILE.search(f.name).group(1) == "0001"
                     and int(CHANNEL.search(f.name).group(1)) == 2]
            self.assertEqual(len(stack), 3)
            self.assertEqual(tifffile.imread(sorted(stack)[0]).shape, (64, 96))

            # tiff_find_metadata_dir_case_insensitive + tiff_pick_leica_xml
            metadata_dir = next(p for p in folder.iterdir() if p.is_dir() and p.name.lower() == "metadata")
            picks = sorted(f for f in metadata_dir.iterdir() if region.lower() in f.stem.lower())
            root = ET.parse(picks[0]).getroot()
            self.assertEqual(picks[0].name, f"{region}.xml")
            self.assertAlmostEqual(pixel_size_from_xml(root), pixel_um, places=6)
            dim_x = root.find(".//ImageDescription/Dimensions/DimensionDescription[@DimID='1']")
            self.assertEqual(dim_x.attrib["Unit"], "um")
            mag = root.find(".//Instrument//Objective").attrib["Magnification"]
            self.assertEqual(float(mag), self.config.objective("20x_air").magnification)

            # tiff_collect_tiles_from_tilescaninfo: TileIndex on every tile, matching file ids
            nodes = root.findall(".//Attachment[@Name='TileScanInfo']//Tile")
            self.assertEqual(sorted(int(n.attrib["TileIndex"]) for n in nodes), tile_ids)
            att = root.find(".//Attachment[@Name='TileScanInfo']")
            self.assertEqual((att.attrib["FlipX"], att.attrib["FlipY"], att.attrib["SwapXY"]), ("0", "0", "0"))
            # Positions in µm: neighbour steps equal the planned step (FOV minus overlap).
            xs = sorted({round(float(n.attrib["PosX"]), 3) for n in nodes})
            if len(xs) > 1:
                self.assertAlmostEqual(xs[1] - xs[0], fov_w * 0.9, places=3)

        names = [c.attrib["Name"] for c in root.findall(".//Channels/ChannelDescription")]
        self.assertEqual(names, ["DAPI", "Cy3", "Cy5"])

    def test_every_cycle_uses_the_same_names(self) -> None:
        fov_w, fov_h = self.config.fov_um("20x_air")
        regions = (ScanRegion(0, 0, 2 * fov_w, 2 * fov_h), ScanRegion(15 * fov_w, 0, fov_w, fov_h))
        first = self.run_cycle(regions, (ChannelSetting("dapi", 10, 5),))
        moved = tuple(r.translated(37.5, -12.0) for r in regions)
        second = self.run_cycle(moved, (ChannelSetting("dapi", 10, 5), ChannelSetting("af488", 15, 5)))
        c00 = lambda folder: sorted(f.name for f in pipeline_tiffs(folder) if "--C00" in f.name)
        self.assertEqual(c00(first), c00(second))
        # The cycle's real stage positions are recorded (re-registration offset included).
        a = ET.parse(first / "Metadata" / "R01.xml").getroot().find(".//Tile[@TileIndex='0']")
        b = ET.parse(second / "Metadata" / "R01.xml").getroot().find(".//Tile[@TileIndex='0']")
        self.assertAlmostEqual(float(b.attrib["PosX"]) - float(a.attrib["PosX"]), 37.5, places=3)
        self.assertAlmostEqual(float(b.attrib["PosY"]) - float(a.attrib["PosY"]), -12.0, places=3)


if __name__ == "__main__":
    unittest.main()
