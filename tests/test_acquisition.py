from __future__ import annotations

from threading import Event

import numpy as np
import json
from pathlib import Path
import tempfile
import unittest

from microscope_ui.acquisition import BoundedAsyncWriter, PlaneWriteRequest
from microscope_ui.config import load_config
from microscope_ui.models import ScanRegion
from microscope_ui.mosaic import DiskBackedMosaic, mosaic_bounds_for_regions


class BlockingSink:
    def __init__(self) -> None:
        self.write_started = Event()
        self.release_write = Event()
        self.frames: list[tuple[str, int, bytes]] = []
        self.finished_tiles = 0

    def begin_tile(self, *_args) -> None:
        pass

    def write_plane(self, channel_id: str, z_index: int, frame, z_um=None) -> None:
        self.write_started.set()
        if not self.release_write.wait(2.0):
            raise TimeoutError("test did not release the simulated SSD")
        self.frames.append((channel_id, z_index, frame))

    def finish_tile(self) -> None:
        self.finished_tiles += 1


def request(z_index: int, *, last: bool = False) -> PlaneWriteRequest:
    return PlaneWriteRequest(
        tile_index=1,
        x_um=0,
        y_um=0,
        fov_width_um=100,
        fov_height_um=100,
        z_planes=2,
        channel_id="dapi",
        z_index=z_index,
        frame=np.full((4, 4), z_index, dtype=np.uint16),
        is_last_plane=last,
    )


class BoundedAsyncWriterTests(unittest.TestCase):
    def test_camera_can_run_ahead_but_payload_count_is_strictly_bounded(self) -> None:
        sink = BlockingSink()
        writer = BoundedAsyncWriter(sink, queue_capacity=2)
        try:
            self.assertTrue(writer.try_submit(request(0)))
            self.assertTrue(sink.write_started.wait(1.0))

            # One frame is in the writer and one is waiting. A third frame is
            # rejected without blocking or increasing memory use.
            self.assertTrue(writer.try_submit(request(1, last=True)))
            self.assertFalse(writer.try_submit(request(2)))
            snapshot = writer.snapshot()
            self.assertEqual(snapshot.queue_depth, 2)
            self.assertEqual(snapshot.queue_capacity, 2)
            self.assertEqual(snapshot.accepted_frames, 2)

            sink.release_write.set()
            self.assertTrue(writer.wait_until_idle(timeout=2.0))
            snapshot = writer.snapshot()
            self.assertEqual(snapshot.written_frames, 2)
            self.assertEqual(snapshot.completed_tiles, 1)
            self.assertEqual(snapshot.queue_depth, 0)
            self.assertEqual(len(sink.frames), 2)
        finally:
            sink.release_write.set()
            writer.shutdown(drain=False)

    def test_storage_failure_is_reported_and_stops_accepting_frames(self) -> None:
        class FailingSink(BlockingSink):
            def write_plane(self, *_args, **_kwargs) -> None:
                raise OSError("SSD disconnected")

        sink = FailingSink()
        writer = BoundedAsyncWriter(sink, queue_capacity=2)
        try:
            self.assertTrue(writer.try_submit(request(0)))
            self.assertFalse(writer.wait_until_idle(timeout=1.0))
            snapshot = writer.snapshot()
            self.assertIn("SSD disconnected", snapshot.failure or "")
            self.assertFalse(snapshot.accepting)
            self.assertFalse(writer.try_submit(request(1, last=True)))
        finally:
            writer.shutdown(drain=False)

    def test_async_writer_finalizes_a_real_disk_backed_tile(self) -> None:
        config = load_config()
        channels = (config.channel("dapi"),)
        region = ScanRegion(-100, -100, 200, 200)
        bounds = mosaic_bounds_for_regions((region,), 100, 100)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mosaic = DiskBackedMosaic(root, bounds, channels, max_preview_side=32)
            writer = BoundedAsyncWriter(mosaic, queue_capacity=2)
            try:
                self.assertTrue(writer.try_submit(request(0)))
                self.assertTrue(writer.try_submit(request(1, last=True)))
                self.assertTrue(writer.wait_until_idle(timeout=2.0))
                snapshot = writer.snapshot()
                self.assertEqual(snapshot.written_frames, 2)
                self.assertEqual(snapshot.completed_tiles, 1)
            finally:
                writer.shutdown(drain=False)
                mosaic.close()

            self.assertTrue((root / "tiles/tile_00001.tstack").exists())
            records = [
                json.loads(line)
                for line in (root / "tiles.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
