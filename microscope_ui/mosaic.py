from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import mmap
from pathlib import Path

from .models import Channel, ScanRegion


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


class DiskBackedMosaic:
    """Bounded-resolution preview maps and immediately closed tile files.

    Full scan dimensions do not affect heap use. Preview maps are fixed-size files
    accessed through mmap, while render_viewport creates only one screen-sized PPM.
    The simulator writes a small PPM payload per tile; a hardware backend can feed
    full camera frames to an OME-Zarr tile sink at the same lifecycle boundary.
    """

    COMPOSITE = "Composite"

    def __init__(
        self,
        root: Path,
        bounds: MosaicBounds,
        channels: tuple[Channel, ...],
        max_preview_side: int = 1024,
    ) -> None:
        if bounds.width_um <= 0 or bounds.height_um <= 0:
            raise ValueError("Mosaic bounds must have positive area")
        if not channels:
            raise ValueError("At least one channel is required")

        self.root = root
        self.bounds = bounds
        self.channels = channels
        self.closed = False
        self.preview_dir = root / "disk_preview"
        self.tile_dir = root / "simulated_tiles"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.tile_dir.mkdir(parents=True, exist_ok=True)

        scale = min(
            max_preview_side / bounds.width_um,
            max_preview_side / bounds.height_um,
        )
        self.width_px = max(1, math.ceil(bounds.width_um * scale))
        self.height_px = max(1, math.ceil(bounds.height_um * scale))
        self._byte_length = self.width_px * self.height_px * 3
        self._files: dict[str, object] = {}
        self._maps: dict[str, mmap.mmap] = {}
        self._display_to_key = {self.COMPOSITE: self.COMPOSITE}
        self._channel_colors = {
            channel.id: _hex_to_rgb(channel.color) for channel in channels
        }
        self._create_map(self.COMPOSITE)
        for channel in channels:
            self._create_map(channel.id)
            self._display_to_key[channel.name] = channel.id

        self._index = (root / "tiles.jsonl").open("a", encoding="utf-8", buffering=1)
        self._current_tile: dict | None = None
        self._current_handle = None
        self._current_temporary_path: Path | None = None
        self._current_final_path: Path | None = None
        metadata = {
            "format": "disk-backed-scan-preview",
            "version": 1,
            "bounds": asdict(bounds),
            "preview_width_px": self.width_px,
            "preview_height_px": self.height_px,
            "channels": [channel.id for channel in channels],
            "ram_policy": "one viewport plus one current simulated tile",
        }
        (self.preview_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @property
    def display_modes(self) -> tuple[str, ...]:
        return (self.COMPOSITE, *(channel.name for channel in self.channels))

    @property
    def disk_preview_bytes(self) -> int:
        return self._byte_length * len(self._maps)

    def _create_map(self, key: str) -> None:
        path = self.preview_dir / f"{key.lower().replace(' ', '_')}.rgb"
        handle = path.open("w+b")
        handle.truncate(self._byte_length)
        self._files[key] = handle
        self._maps[key] = mmap.mmap(handle.fileno(), self._byte_length)

    def write_tile(
        self,
        tile_index: int,
        x_um: float,
        y_um: float,
        fov_width_um: float,
        fov_height_um: float,
        z_planes: int,
    ) -> Path:
        self.begin_tile(
            tile_index,
            x_um,
            y_um,
            fov_width_um,
            fov_height_um,
            z_planes,
        )
        for channel in self.channels:
            for z_index in range(z_planes):
                self.write_plane(channel.id, z_index)
        return self.finish_tile()

    def begin_tile(
        self,
        tile_index: int,
        x_um: float,
        y_um: float,
        fov_width_um: float,
        fov_height_um: float,
        z_planes: int,
    ) -> None:
        if self.closed:
            raise RuntimeError("Mosaic store is closed")
        if self._current_handle is not None:
            raise RuntimeError("Finish the current tile before beginning another")
        tile_path = self.tile_dir / f"tile_{tile_index:07d}.simstack"
        temporary_path = tile_path.with_suffix(".tmp")
        self._current_tile = {
            "tile": tile_index,
            "x_um": x_um,
            "y_um": y_um,
            "fov_width_um": fov_width_um,
            "fov_height_um": fov_height_um,
            "z_planes": z_planes,
        }
        self._current_temporary_path = temporary_path
        self._current_final_path = tile_path
        self._current_handle = temporary_path.open("wb")
        self._current_handle.write(b"SIMSTACK1\n")
        self._current_handle.write(
            (json.dumps(self._current_tile, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        self._current_handle.flush()

    def write_plane(
        self,
        channel_id: str,
        z_index: int,
        payload: bytes | None = None,
        width: int = 64,
        height: int = 64,
    ) -> None:
        if self._current_handle is None or self._current_tile is None:
            raise RuntimeError("Begin a tile before writing planes")
        if channel_id not in self._channel_colors:
            raise KeyError(f"Unknown mosaic channel: {channel_id}")
        tile_index = self._current_tile["tile"]
        if payload is None:
            payload = simulated_plane_payload(
                tile_index,
                channel_id,
                z_index,
                width,
                height,
            )
        plane_header = {
            "channel": channel_id,
            "z_index": z_index,
            "width": width,
            "height": height,
            "bytes": len(payload),
        }
        self._current_handle.write(
            (json.dumps(plane_header, separators=(",", ":")) + "\n").encode("utf-8")
        )
        self._current_handle.write(payload)
        self._current_handle.flush()
        self._paint_current_plane(channel_id, z_index)
        for preview_map in self._maps.values():
            preview_map.flush()

    def _paint_current_plane(self, channel_id: str, z_index: int) -> None:
        assert self._current_tile is not None
        x_um = self._current_tile["x_um"]
        y_um = self._current_tile["y_um"]
        fov_width_um = self._current_tile["fov_width_um"]
        fov_height_um = self._current_tile["fov_height_um"]
        tile_index = self._current_tile["tile"]
        left, top = self._stage_to_preview(
            x_um - fov_width_um / 2,
            y_um - fov_height_um / 2,
        )
        right, bottom = self._stage_to_preview(
            x_um + fov_width_um / 2,
            y_um + fov_height_um / 2,
        )
        x0 = max(0, min(left, right))
        x1 = min(self.width_px, max(left, right) + 1)
        y0 = max(0, min(top, bottom))
        y1 = min(self.height_px, max(top, bottom) + 1)
        color = self._channel_colors[channel_id]
        for y in range(y0, y1):
            row = bytearray()
            for x in range(x0, x1):
                intensity = _synthetic_intensity(x, y, tile_index + z_index * 7, 0)
                row.extend(intensity * component // 255 for component in color)
            byte_offset = (y * self.width_px + x0) * 3
            self._maps[channel_id][byte_offset : byte_offset + len(row)] = row
            self._maps[self.COMPOSITE][byte_offset : byte_offset + len(row)] = row

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
        self._current_handle.close()
        self._current_temporary_path.replace(tile_path)

        left, top = self._stage_to_preview(
            tile["x_um"] - tile["fov_width_um"] / 2,
            tile["y_um"] - tile["fov_height_um"] / 2,
        )
        right, bottom = self._stage_to_preview(
            tile["x_um"] + tile["fov_width_um"] / 2,
            tile["y_um"] + tile["fov_height_um"] / 2,
        )
        self._paint_preview_rect(
            tile["tile"],
            max(0, min(left, right)),
            min(self.width_px, max(left, right) + 1),
            max(0, min(top, bottom)),
            min(self.height_px, max(top, bottom) + 1),
        )

        record = {
            "tile": tile["tile"],
            "x_um": tile["x_um"],
            "y_um": tile["y_um"],
            "z_planes": tile["z_planes"],
            "channels": [channel.id for channel in self.channels],
            "simulated_payload": str(tile_path.relative_to(self.root)),
            "written_at": datetime.now().astimezone().isoformat(),
        }
        self._index.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._index.flush()
        for preview_map in self._maps.values():
            preview_map.flush()
        self._current_tile = None
        self._current_handle = None
        self._current_temporary_path = None
        self._current_final_path = None
        return tile_path

    def _stage_to_preview(self, x_um: float, y_um: float) -> tuple[int, int]:
        x = round(
            (x_um - self.bounds.x_min_um) / self.bounds.width_um * (self.width_px - 1)
        )
        y = round(
            (y_um - self.bounds.y_min_um) / self.bounds.height_um * (self.height_px - 1)
        )
        return x, y

    def _paint_preview_rect(
        self,
        tile_index: int,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> None:
        if x1 <= x0 or y1 <= y0:
            return
        for y in range(y0, y1):
            channel_rows: dict[str, bytearray] = {
                channel.id: bytearray() for channel in self.channels
            }
            composite_row = bytearray()
            for x in range(x0, x1):
                composite = [0, 0, 0]
                for channel_index, channel in enumerate(self.channels):
                    intensity = _synthetic_intensity(x, y, tile_index, channel_index)
                    color = self._channel_colors[channel.id]
                    colored = tuple(intensity * component // 255 for component in color)
                    channel_rows[channel.id].extend(colored)
                    for component_index in range(3):
                        composite[component_index] = min(
                            255,
                            composite[component_index] + colored[component_index],
                        )
                composite_row.extend(composite)
            byte_offset = (y * self.width_px + x0) * 3
            self._maps[self.COMPOSITE][byte_offset : byte_offset + len(composite_row)] = (
                composite_row
            )
            for channel in self.channels:
                row = channel_rows[channel.id]
                self._maps[channel.id][byte_offset : byte_offset + len(row)] = row

    def render_viewport_ppm(
        self,
        view_bounds: MosaicBounds,
        output_width: int,
        output_height: int,
        display_mode: str,
    ) -> bytes:
        if self.closed:
            raise RuntimeError("Mosaic store is closed")
        output_width = max(1, output_width)
        output_height = max(1, output_height)
        key = self._display_to_key.get(display_mode, self.COMPOSITE)
        source = self._maps[key]
        result = bytearray(output_width * output_height * 3)

        x_offsets = []
        for output_x in range(output_width):
            stage_x = view_bounds.x_min_um + (
                (output_x + 0.5) / output_width * view_bounds.width_um
            )
            if not self.bounds.x_min_um <= stage_x <= self.bounds.x_max_um:
                x_offsets.append(-1)
            else:
                source_x = min(
                    self.width_px - 1,
                    max(
                        0,
                        int(
                            (stage_x - self.bounds.x_min_um)
                            / self.bounds.width_um
                            * self.width_px
                        ),
                    ),
                )
                x_offsets.append(source_x * 3)

        destination = 0
        black_row = bytes(output_width * 3)
        for output_y in range(output_height):
            stage_y = view_bounds.y_min_um + (
                (output_y + 0.5) / output_height * view_bounds.height_um
            )
            if not self.bounds.y_min_um <= stage_y <= self.bounds.y_max_um:
                result[destination : destination + len(black_row)] = black_row
                destination += len(black_row)
                continue
            source_y = min(
                self.height_px - 1,
                max(
                    0,
                    int(
                        (stage_y - self.bounds.y_min_um)
                        / self.bounds.height_um
                        * self.height_px
                    ),
                ),
            )
            row_offset = source_y * self.width_px * 3
            for x_offset in x_offsets:
                if x_offset < 0:
                    destination += 3
                    continue
                source_offset = row_offset + x_offset
                result[destination : destination + 3] = source[
                    source_offset : source_offset + 3
                ]
                destination += 3
        header = f"P6\n{output_width} {output_height}\n255\n".encode("ascii")
        return header + result

    def close(self) -> None:
        if self.closed:
            return
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None
        self._index.close()
        for preview_map in self._maps.values():
            preview_map.flush()
            preview_map.close()
        for handle in self._files.values():
            handle.close()
        self.closed = True


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color}")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _synthetic_intensity(x: int, y: int, tile_index: int, channel_index: int) -> int:
    wave = (x * 17 + y * 29 + tile_index * 37 + channel_index * 61) % 211
    spots = 55 if ((x // 9 + y // 7 + tile_index + channel_index) % 11 == 0) else 0
    return min(255, 24 + wave + spots)


def simulated_plane_payload(
    tile_index: int,
    channel_id: str,
    z_index: int,
    width: int = 64,
    height: int = 64,
) -> bytes:
    """Return one immutable simulator frame, analogous to a camera snap."""

    if width < 1 or height < 1:
        raise ValueError("Simulated frame dimensions must be positive")
    pixels = bytearray()
    channel_seed = sum(ord(character) for character in channel_id)
    for y in range(height):
        for x in range(width):
            pixels.append(
                _synthetic_intensity(
                    x,
                    y,
                    tile_index + z_index * 7,
                    channel_seed % 13,
                )
            )
    return bytes(pixels)
