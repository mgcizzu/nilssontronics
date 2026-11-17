"""Per-cycle tile storage and a bounded, disk-backed preview mosaic.

Storage layout of one cycle folder::

    tiles/tile_00042.tstack   one finished tile (all channels and Z planes)
    tiles.jsonl               append-only index of finished tiles (resume journal)
    disk_preview/<ch>.u16     fixed-size memory-mapped preview per channel
    disk_preview/metadata.json

The ``.tstack`` container is an interim format (JSON header + raw planes)
until the downstream pipeline's file naming and format are agreed; only
:class:`DiskBackedMosaic` and :func:`read_tile_stack` know about it.

Durability: a tile is written to ``.tmp``, fsynced, atomically renamed and only
then appended (and fsynced) to ``tiles.jsonl``. After a crash every tile listed
in the index is complete on disk; anything else is re-acquired on resume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import mmap
import os
from pathlib import Path

import numpy as np

from .models import Channel, ScanRegion, Tile

TSTACK_MAGIC = b"TSTACK1\n"


@dataclass(frozen=True)
class MosaicBounds:
    x_min_um: float
    x_max_um: float
    y_min_um: float
    y_max_um: float

    @property
    def width_um(self) -> float:
        return self.x_max_um - self.x_min_um

    @property
    def height_um(self) -> float:
        return self.y_max_um - self.y_min_um


def mosaic_bounds_for_regions(
    regions: tuple[ScanRegion, ...],
    fov_width_um: float,
    fov_height_um: float,
) -> MosaicBounds:
    return MosaicBounds(
        min(region.x_um for region in regions) - fov_width_um / 2,
        max(region.x_max_um for region in regions) + fov_width_um / 2,
        min(region.y_um for region in regions) - fov_height_um / 2,
        max(region.y_max_um for region in regions) + fov_height_um / 2,
    )


def mosaic_bounds_for_tiles(
    tiles: list[Tile], fov_width_um: float, fov_height_um: float
) -> MosaicBounds:
    return MosaicBounds(
        min(tile.x_um for tile in tiles) - fov_width_um / 2,
        max(tile.x_um for tile in tiles) + fov_width_um / 2,
        min(tile.y_um for tile in tiles) - fov_height_um / 2,
        max(tile.y_um for tile in tiles) + fov_height_um / 2,
    )


class DiskBackedMosaic:
    """Tile sink for one cycle plus fixed-size preview maps.

    Heap use does not depend on the scan size: preview maps are memory-mapped
    files and only the current tile's downsampled running MIP is kept in RAM.
    """

    COMPOSITE = "Composite"

    def __init__(
        self,
        root: Path,
        bounds: MosaicBounds,
        channels: tuple[Channel, ...],
        max_preview_side: int = 2048,
        fsync: bool = True,
    ) -> None:
        if bounds.width_um <= 0 or bounds.height_um <= 0:
            raise ValueError("Mosaic bounds must have positive area")
        if not channels:
            raise ValueError("At least one channel is required")

        self.root = Path(root)
        self.bounds = bounds
        self.channels = channels
        self.fsync = fsync
        self.closed = False
        self.preview_dir = self.root / "disk_preview"
        self.tile_dir = self.root / "tiles"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.tile_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.tile_dir.glob("*.tmp"):
            stale.unlink()

        scale = min(
            max_preview_side / bounds.width_um,
            max_preview_side / bounds.height_um,
        )
        self.width_px = max(1, math.ceil(bounds.width_um * scale))
        self.height_px = max(1, math.ceil(bounds.height_um * scale))
        self._byte_length = self.width_px * self.height_px * 2
        self._files: dict[str, object] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._display_to_key = {self.COMPOSITE: self.COMPOSITE}
        self._channel_colors = {
            channel.id: np.array(_hex_to_rgb(channel.color), dtype=np.float32) / 255.0
            for channel in channels
        }
        self._display_range: dict[str, tuple[float, float]] = {}
        for channel in channels:
            self._open_map(channel.id)
            self._display_to_key[channel.name] = channel.id
            self._display_to_key[channel.id] = channel.id
            self._restore_display_range(channel.id)

        index_path = self.root / "tiles.jsonl"
        self._index = index_path.open("a", encoding="utf-8")
        self._current_tile: dict | None = None
        self._current_handle = None
        self._current_temporary_path: Path | None = None
        self._current_final_path: Path | None = None
        self._current_mip: dict[str, np.ndarray] = {}
        self._current_rect: tuple[int, int, int, int] | None = None
        metadata = {
            "format": "disk-backed-scan-preview",
            "version": 2,
            "bounds": asdict(bounds),
            "preview_width_px": self.width_px,
            "preview_height_px": self.height_px,
            "dtype": "uint16",
            "channels": [channel.id for channel in channels],
            "ram_policy": "one viewport plus one downsampled current tile",
        }
        (self.preview_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    # -- preview maps --------------------------------------------------------
    @property
    def display_modes(self) -> tuple[str, ...]:
        return (self.COMPOSITE, *(channel.name for channel in self.channels))

    @property
    def disk_preview_bytes(self) -> int:
        return self._byte_length * len(self._maps)

    def _open_map(self, key: str) -> None:
        path = self.preview_dir / f"{key.lower().replace(' ', '_')}.u16"
        reuse = path.exists() and path.stat().st_size == self._byte_length
        handle = path.open("r+b" if reuse else "w+b")
        if not reuse:
            handle.truncate(self._byte_length)
        self._files[key] = handle
        self._maps[key] = mmap.mmap(handle.fileno(), self._byte_length)
        self._arrays[key] = np.frombuffer(self._maps[key], dtype=np.uint16).reshape(
            self.height_px, self.width_px
        )

    def _restore_display_range(self, channel_id: str) -> None:
        data = self._arrays[channel_id]
        nonzero = data[data > 0]
        if nonzero.size:
            low, high = np.percentile(nonzero, (0.5, 99.8))
            self._display_range[channel_id] = (float(low), float(max(high, low + 1)))

    def display_range(self, channel_id: str) -> tuple[float, float]:
        return self._display_range.get(channel_id, (0.0, 1.0))

    def _update_display_range(self, channel_id: str, frame: np.ndarray) -> None:
        low, high = np.percentile(frame, (0.5, 99.8))
        previous = self._display_range.get(channel_id)
        if previous is None:
            self._display_range[channel_id] = (float(low), float(max(high, low + 1)))
        else:
            self._display_range[channel_id] = (
                float(min(previous[0], low)),
                float(max(previous[1], high)),
            )

    # -- tile sink -------------------------------------------------------------
    def begin_tile(
        self,
        tile_index: int,
        x_um: float,
        y_um: float,
        fov_width_um: float,
        fov_height_um: float,
        z_planes: int,
        row: int = 0,
        col: int = 0,
        z_positions_um: tuple[float, ...] = (),
    ) -> None:
        if self.closed:
            raise RuntimeError("Mosaic store is closed")
        if self._current_handle is not None:
            raise RuntimeError("Finish the current tile before beginning another")
        tile_path = self.tile_dir / f"tile_{tile_index:05d}.tstack"
        temporary_path = tile_path.with_suffix(".tmp")
        self._current_tile = {
            "tile": tile_index,
            "row": row,
            "col": col,
            "x_um": x_um,
            "y_um": y_um,
            "fov_width_um": fov_width_um,
            "fov_height_um": fov_height_um,
            "z_planes": z_planes,
            "z_positions_um": list(z_positions_um),
            "started_at": datetime.now().astimezone().isoformat(),
        }
        self._current_temporary_path = temporary_path
        self._current_final_path = tile_path
        self._current_handle = temporary_path.open("wb")
        self._current_handle.write(TSTACK_MAGIC)
        self._current_handle.write(
            (json.dumps(self._current_tile, separators=(",", ":")) + "\n").encode("utf-8")
        )
        self._current_mip = {}
        left, top = self._stage_to_preview(x_um - fov_width_um / 2, y_um - fov_height_um / 2)
        right, bottom = self._stage_to_preview(x_um + fov_width_um / 2, y_um + fov_height_um / 2)
        x0, x1 = max(0, min(left, right)), min(self.width_px, max(left, right) + 1)
        y0, y1 = max(0, min(top, bottom)), min(self.height_px, max(top, bottom) + 1)
        self._current_rect = (x0, x1, y0, y1)

    def write_plane(
        self,
        channel_id: str,
        z_index: int,
        frame: np.ndarray,
        z_um: float | None = None,
    ) -> None:
        if self._current_handle is None or self._current_tile is None:
            raise RuntimeError("Begin a tile before writing planes")
        if channel_id not in self._channel_colors:
            raise KeyError(f"Unknown mosaic channel: {channel_id}")
        frame = np.ascontiguousarray(frame)
        plane_header = {
            "channel": channel_id,
            "z_index": z_index,
            "z_um": z_um,
            "height": int(frame.shape[0]),
            "width": int(frame.shape[1]),
            "dtype": str(frame.dtype),
            "bytes": int(frame.nbytes),
        }
        self._current_handle.write(
            (json.dumps(plane_header, separators=(",", ":")) + "\n").encode("utf-8")
        )
        self._current_handle.write(frame.tobytes())
        self._current_handle.flush()
        self._paint_plane(channel_id, frame)

    def _paint_plane(self, channel_id: str, frame: np.ndarray) -> None:
        assert self._current_rect is not None
        x0, x1, y0, y1 = self._current_rect
        if x1 <= x0 or y1 <= y0:
            return
        small = _resize_nearest(frame, y1 - y0, x1 - x0)
        running = self._current_mip.get(channel_id)
        running = small if running is None else np.maximum(running, small)
        self._current_mip[channel_id] = running
        self._arrays[channel_id][y0:y1, x0:x1] = running
        self._update_display_range(channel_id, small)

    def finish_tile(self) -> Path:
        if (
            self._current_handle is None
            or self._current_tile is None
            or self._current_temporary_path is None
            or self._current_final_path is None
        ):
            raise RuntimeError("There is no current tile to finish")
        tile = self._current_tile
        tile_path = self._current_final_path
        handle = self._current_handle
        handle.flush()
        if self.fsync:
            os.fsync(handle.fileno())
        handle.close()
        self._current_temporary_path.replace(tile_path)
        if self.fsync:
            _fsync_directory(self.tile_dir)

        record = {
            "tile": tile["tile"],
            "row": tile["row"],
            "col": tile["col"],
            "x_um": tile["x_um"],
            "y_um": tile["y_um"],
            "z_planes": tile["z_planes"],
            "z_positions_um": tile["z_positions_um"],
            "channels": [channel.id for channel in self.channels],
            "path": str(tile_path.relative_to(self.root)),
            "written_at": datetime.now().astimezone().isoformat(),
        }
        self._index.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._index.flush()
        if self.fsync:
            os.fsync(self._index.fileno())
        for preview_map in self._maps.values():
            preview_map.flush()
        self._current_tile = None
        self._current_handle = None
        self._current_temporary_path = None
        self._current_final_path = None
        self._current_mip = {}
        self._current_rect = None
        return tile_path

    def abandon_tile(self) -> None:
        """Drop an unfinished tile (abort); it will be re-acquired on resume."""

        if self._current_handle is not None:
            self._current_handle.close()
            if self._current_temporary_path is not None and self._current_temporary_path.exists():
                self._current_temporary_path.unlink()
        self._current_tile = None
        self._current_handle = None
        self._current_temporary_path = None
        self._current_final_path = None
        self._current_mip = {}
        self._current_rect = None

    def _stage_to_preview(self, x_um: float, y_um: float) -> tuple[int, int]:
        x = round(
            (x_um - self.bounds.x_min_um) / self.bounds.width_um * (self.width_px - 1)
        )
        y = round(
            (y_um - self.bounds.y_min_um) / self.bounds.height_um * (self.height_px - 1)
        )
        return x, y

    # -- display -------------------------------------------------------------
    def render_viewport_rgb(
        self,
        view_bounds: MosaicBounds,
        output_width: int,
        output_height: int,
        display_mode: str,
    ) -> np.ndarray:
        if self.closed:
            raise RuntimeError("Mosaic store is closed")
        output_width = max(1, output_width)
        output_height = max(1, output_height)
        stage_x = view_bounds.x_min_um + (np.arange(output_width) + 0.5) / output_width * view_bounds.width_um
        stage_y = view_bounds.y_min_um + (np.arange(output_height) + 0.5) / output_height * view_bounds.height_um
        source_x = np.floor((stage_x - self.bounds.x_min_um) / self.bounds.width_um * self.width_px).astype(np.int64)
        source_y = np.floor((stage_y - self.bounds.y_min_um) / self.bounds.height_um * self.height_px).astype(np.int64)
        valid_x = (source_x >= 0) & (source_x < self.width_px)
        valid_y = (source_y >= 0) & (source_y < self.height_px)
        sx = np.clip(source_x, 0, self.width_px - 1)
        sy = np.clip(source_y, 0, self.height_px - 1)
        mask = valid_y[:, None] & valid_x[None, :]

        key = self._display_to_key.get(display_mode, self.COMPOSITE)
        keys = [channel.id for channel in self.channels] if key == self.COMPOSITE else [key]
        rgb = np.zeros((output_height, output_width, 3), dtype=np.float32)
        for channel_id in keys:
            data = self._arrays[channel_id][np.ix_(sy, sx)].astype(np.float32)
            low, high = self.display_range(channel_id)
            scaled = np.clip((data - low) / max(high - low, 1.0), 0.0, 1.0)
            scaled[data == 0] = 0.0
            rgb += scaled[:, :, None] * self._channel_colors[channel_id][None, None, :]
        rgb = np.clip(rgb, 0.0, 1.0) * mask[:, :, None]
        return (rgb * 255).astype(np.uint8)

    def render_viewport_ppm(
        self,
        view_bounds: MosaicBounds,
        output_width: int,
        output_height: int,
        display_mode: str,
    ) -> bytes:
        rgb = self.render_viewport_rgb(view_bounds, output_width, output_height, display_mode)
        header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii")
        return header + rgb.tobytes()

    def close(self) -> None:
        if self.closed:
            return
        self.abandon_tile()
        self._index.close()
        self._arrays.clear()
        for preview_map in self._maps.values():
            preview_map.flush()
            preview_map.close()
        for handle in self._files.values():
            handle.close()
        self.closed = True


def completed_tile_indices(cycle_root: Path) -> set[int]:
    """Tiles recorded as complete in a cycle's journal (tolerates a torn last line)."""

    index_path = Path(cycle_root) / "tiles.jsonl"
    done: set[int] = set()
    if not index_path.exists():
        return done
    for line in index_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = Path(cycle_root) / record.get("path", "")
        if path.is_file():
            done.add(int(record["tile"]))
    return done


def read_tile_stack(path: Path) -> tuple[dict, dict[tuple[str, int], np.ndarray]]:
    """Read a ``.tstack`` file into its header and ``{(channel, z_index): frame}``."""

    planes: dict[tuple[str, int], np.ndarray] = {}
    with Path(path).open("rb") as handle:
        if handle.readline() != TSTACK_MAGIC:
            raise ValueError(f"{path} is not a tile stack")
        header = json.loads(handle.readline())
        while True:
            line = handle.readline()
            if not line:
                break
            plane = json.loads(line)
            data = handle.read(plane["bytes"])
            frame = np.frombuffer(data, dtype=plane["dtype"]).reshape(plane["height"], plane["width"])
            planes[(plane["channel"], plane["z_index"])] = frame
    return header, planes


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = (np.arange(height) * frame.shape[0] / height).astype(np.int64)
    cols = (np.arange(width) * frame.shape[1] / width).astype(np.int64)
    return frame[np.ix_(rows, cols)].astype(np.uint16, copy=False)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color}")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
