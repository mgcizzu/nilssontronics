"""Per-cycle tile storage and a bounded, disk-backed preview mosaic.

One cycle folder is directly usable as an input folder of the ISS_preprocessing
pipeline (github.com/Moldia/ISS2025, ``mode='tif_autosaved'``)::

    R01--Stage0000--Z00--C00.tif   one 2-D plane per file (uint16, uncompressed)
    R01--Stage0000--Z01--C00.tif
    ...
    Metadata/R01.xml               Leica-style XML per region: TileScanInfo
                                   (TileIndex = Stage number, PosX/PosY in µm),
                                   pixel size, objective, channels
    tiles.jsonl                    append-only index of finished tiles (resume journal)
    disk_preview/<ch>.u16          fixed-size memory-mapped preview per channel
    flatfield/                     optional flat-field profiles (never applied to raw data)

Naming rules (what the pipeline parses):

* ``R01`` – region: one connected group of tiles in the experiment's shared
  grid (8-neighbourhood). Separate tissue sections become separate regions, so
  each is stitched on its own. Numbered by first acquired tile; identical in
  every cycle because the tile grid is fixed by cycle 1.
* ``Stage0000`` – tile within the region, 0-based, in acquisition order. It is
  also the ``TileIndex`` in ``Metadata/R01.xml``.
* ``Z00`` – Z plane, zero-padded (the pipeline orders planes by file name).
* ``C00`` – channel, 0-based in the cycle's acquisition order. The mapping to
  channel names, exposures and powers is in the XML and in ``experiment.json``.

Stage positions are written so that +PosX = +image column and +PosY = +image
row, matching the oriented frames (``FlipX=FlipY=SwapXY=0``).

Durability: each plane is written to ``<name>.tif.part`` (ignored by the
pipeline), fsynced, and only when the whole tile is complete are its planes
renamed to ``.tif`` and the tile appended (and fsynced) to ``tiles.jsonl``.
After a crash every tile listed in the index is complete on disk; anything
else is re-acquired on resume. Only this module knows the file format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import mmap
import os
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import tifffile

from .models import Channel, ChannelSetting, Objective, ScanRegion, Tile

PIPELINE_MODE = "tif_autosaved"
PART_SUFFIX = ".part"
METADATA_DIR = "Metadata"
APPLICATION = "nilssontronics"


@dataclass(frozen=True)
class TileName:
    """Where a tile lives in the pipeline layout: region token and Stage number."""

    region: str
    stage: int


def pipeline_regions(tiles: Iterable[Tile]) -> dict[int, TileName]:
    """Map tile index → (region, stage) for the ISS_preprocessing file names.

    Regions are the 8-connected groups of tiles on the shared grid, numbered by
    their first acquired tile. Stage numbers are 0-based within each region, in
    acquisition order.
    """

    ordered = sorted(tiles, key=lambda tile: tile.index)
    by_cell = {(tile.row, tile.col): tile for tile in ordered}
    assigned: set[int] = set()
    groups: list[list[Tile]] = []
    for seed in ordered:
        if seed.index in assigned:
            continue
        assigned.add(seed.index)
        group, stack = [], [seed]
        while stack:
            tile = stack.pop()
            group.append(tile)
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    neighbour = by_cell.get((tile.row + d_row, tile.col + d_col))
                    if neighbour is not None and neighbour.index not in assigned:
                        assigned.add(neighbour.index)
                        stack.append(neighbour)
        groups.append(sorted(group, key=lambda tile: tile.index))
    width = max(2, len(str(len(groups))))
    names: dict[int, TileName] = {}
    for number, group in enumerate(groups, start=1):
        for stage, tile in enumerate(group):
            names[tile.index] = TileName(f"R{number:0{width}d}", stage)
    return names


def plane_filename(name: TileName, z_index: int, channel_index: int, z_digits: int = 2) -> str:
    return (
        f"{name.region}--Stage{name.stage:04d}--Z{z_index:0{z_digits}d}"
        f"--C{channel_index:02d}.tif"
    )


def z_digits_for(z_planes: int) -> int:
    return max(2, len(str(max(0, z_planes - 1))))


@dataclass(frozen=True)
class CycleMetadata:
    """What the per-region XML records besides tile positions."""

    cycle: int
    objective: Objective
    channel_settings: tuple[ChannelSetting, ...]
    camera_width_px: int
    camera_height_px: int
    z_planes: int = 1
    z_step_um: float | None = None
    project: str = ""
    sample: str = ""


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
        tiles: Sequence[Tile] = (),
        metadata: CycleMetadata | None = None,
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
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.root.glob(f"*{PART_SUFFIX}"):
            stale.unlink()  # planes of a tile that never finished
        self._channel_index = {channel.id: index for index, channel in enumerate(channels)}
        self._names = pipeline_regions(tiles)
        if tiles and metadata is not None:
            write_region_metadata(self.root, tiles, self._names, channels, metadata)

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
        _repair_journal_tail(index_path, fsync=fsync)
        self._index = index_path.open("a", encoding="utf-8")
        self._current_tile: dict | None = None
        self._current_planes: list[dict] = []
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
        if self._current_tile is not None:
            raise RuntimeError("Finish the current tile before beginning another")
        name = self._names.get(tile_index) or TileName("R01", tile_index)
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
            "region": name.region,
            "stage": name.stage,
            "z_digits": z_digits_for(z_planes),
            "started_at": datetime.now().astimezone().isoformat(),
        }
        self._current_planes = []
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
        if self._current_tile is None:
            raise RuntimeError("Begin a tile before writing planes")
        if channel_id not in self._channel_colors:
            raise KeyError(f"Unknown mosaic channel: {channel_id}")
        frame = np.ascontiguousarray(frame)
        if frame.ndim != 2:
            raise ValueError(f"Expected a 2-D plane, got shape {frame.shape}")
        tile = self._current_tile
        filename = plane_filename(
            TileName(tile["region"], tile["stage"]),
            z_index,
            self._channel_index[channel_id],
            tile["z_digits"],
        )
        final_path = self.root / filename
        part_path = final_path.with_name(filename + PART_SUFFIX)
        _write_plane_tiff(part_path, frame, fsync=self.fsync)
        self._current_planes.append(
            {"channel": channel_id, "z_index": z_index, "z_um": z_um, "file": filename}
        )
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
        if self._current_tile is None:
            raise RuntimeError("There is no current tile to finish")
        tile = self._current_tile
        paths = []
        for plane in self._current_planes:
            final_path = self.root / plane["file"]
            final_path.with_name(plane["file"] + PART_SUFFIX).replace(final_path)
            paths.append(final_path)
        if self.fsync:
            _fsync_directory(self.root)

        record = {
            "tile": tile["tile"],
            "row": tile["row"],
            "col": tile["col"],
            "x_um": tile["x_um"],
            "y_um": tile["y_um"],
            "z_planes": tile["z_planes"],
            "z_positions_um": tile["z_positions_um"],
            "channels": [channel.id for channel in self.channels],
            "region": tile["region"],
            "stage": tile["stage"],
            "planes": self._current_planes,
            "written_at": datetime.now().astimezone().isoformat(),
        }
        self._index.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._index.flush()
        if self.fsync:
            os.fsync(self._index.fileno())
        for preview_map in self._maps.values():
            preview_map.flush()
        self._current_tile = None
        self._current_planes = []
        self._current_mip = {}
        self._current_rect = None
        return tuple(paths)

    def abandon_tile(self) -> None:
        """Drop an unfinished tile (abort); it will be re-acquired on resume."""

        for plane in self._current_planes:
            (self.root / (plane["file"] + PART_SUFFIX)).unlink(missing_ok=True)
        self._current_tile = None
        self._current_planes = []
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


def tile_records(cycle_root: Path) -> dict[int, dict]:
    """Journal records of finished tiles whose plane files all exist (last record wins).

    Tolerates a torn last line from a crash during an append.
    """

    index_path = Path(cycle_root) / "tiles.jsonl"
    records: dict[int, dict] = {}
    if not index_path.exists():
        return records
    for line in index_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            planes = record["planes"]
            tile = int(record["tile"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if planes and all((Path(cycle_root) / plane["file"]).is_file() for plane in planes):
            records[tile] = record
    return records


def completed_tile_indices(cycle_root: Path) -> set[int]:
    """Tiles recorded as complete in a cycle's journal."""

    return set(tile_records(cycle_root))


def read_tile(cycle_root: Path, tile_index: int) -> tuple[dict, dict[tuple[str, int], np.ndarray]]:
    """A finished tile's journal record and its planes as ``{(channel_id, z_index): frame}``."""

    record = tile_records(cycle_root).get(int(tile_index))
    if record is None:
        raise KeyError(f"Tile {tile_index} is not complete in {cycle_root}")
    planes = {
        (plane["channel"], int(plane["z_index"])): tifffile.imread(Path(cycle_root) / plane["file"])
        for plane in record["planes"]
    }
    return record, planes


def middle_plane(record: dict, channel_id: str) -> int | None:
    """Z index of the middle plane a finished tile has for ``channel_id`` (None if absent)."""

    z_indices = sorted(int(p["z_index"]) for p in record.get("planes", ()) if p["channel"] == channel_id)
    return z_indices[len(z_indices) // 2] if z_indices else None


def read_plane(cycle_root: Path, record: dict, channel_id: str, z_index: int) -> np.ndarray:
    """Read one plane of a finished tile (no other planes are loaded)."""

    for plane in record["planes"]:
        if plane["channel"] == channel_id and int(plane["z_index"]) == int(z_index):
            return tifffile.imread(Path(cycle_root) / plane["file"])
    raise KeyError(f"Tile {record.get('tile')} has no plane ({channel_id}, z={z_index})")


def write_region_metadata(
    root: Path,
    tiles: Sequence[Tile],
    names: dict[int, TileName],
    channels: tuple[Channel, ...],
    metadata: CycleMetadata,
) -> list[Path]:
    """Write ``Metadata/<region>.xml`` (Leica-style) for every region of the cycle.

    The pipeline reads, per region: ``TileScanInfo/Tile`` (TileIndex, FieldX/Y,
    PosX/Y), the X/Y ``DimensionDescription`` (pixel size = Length / elements,
    unit µm) and ``Instrument/Objective@Magnification``. Everything else is
    provenance for people.
    """

    folder = Path(root) / METADATA_DIR
    folder.mkdir(parents=True, exist_ok=True)
    pixel_um = metadata.objective.pixel_size_um
    settings = {item.channel_id: item for item in metadata.channel_settings}
    regions: dict[str, list[Tile]] = {}
    for tile in tiles:
        regions.setdefault(names[tile.index].region, []).append(tile)
    written = []
    for region, region_tiles in sorted(regions.items()):
        region_tiles.sort(key=lambda tile: names[tile.index].stage)
        data = ET.Element("Data")
        image = ET.SubElement(data, "Image", Name=region)
        description = ET.SubElement(image, "ImageDescription")
        channel_list = ET.SubElement(description, "Channels")
        for index, channel in enumerate(channels):
            setting = settings.get(channel.id)
            attributes = {
                "ChannelTag": str(index),
                "Name": channel.name,
                "Id": channel.id,
                "EmissionWavelengthNm": f"{channel.emission_wavelength_nm:g}",
                "Color": channel.color,
            }
            if setting is not None:
                attributes["ExposureTimeMs"] = f"{setting.exposure_ms:g}"
                attributes["PowerPercent"] = f"{setting.power_percent:g}"
            ET.SubElement(channel_list, "ChannelDescription", attributes)
        dimensions = ET.SubElement(description, "Dimensions")
        for dim_id, count in (("1", metadata.camera_width_px), ("2", metadata.camera_height_px)):
            ET.SubElement(
                dimensions, "DimensionDescription", DimID=dim_id,
                NumberOfElements=str(count), Length=f"{count * pixel_um:.6f}", Unit="um",
            )
        if metadata.z_planes > 1 and metadata.z_step_um:
            ET.SubElement(
                dimensions, "DimensionDescription", DimID="3",
                NumberOfElements=str(metadata.z_planes),
                Length=f"{(metadata.z_planes - 1) * metadata.z_step_um:.6f}", Unit="um",
            )
        min_row = min(tile.row for tile in region_tiles)
        min_col = min(tile.col for tile in region_tiles)
        scan = ET.SubElement(
            image, "Attachment", Name="TileScanInfo", Application=APPLICATION,
            FlipX="0", FlipY="0", SwapXY="0",
        )
        for tile in region_tiles:
            ET.SubElement(
                scan, "Tile",
                TileIndex=str(names[tile.index].stage),
                FieldX=str(tile.col - min_col), FieldY=str(tile.row - min_row),
                PosX=f"{tile.x_um:.4f}", PosY=f"{tile.y_um:.4f}",
                ExperimentTile=str(tile.index),
            )
        objective = metadata.objective
        instrument = ET.SubElement(image, "Instrument")
        ET.SubElement(
            instrument, "Objective",
            Name=objective.name, Id=objective.id,
            Magnification=f"{objective.magnification:g}",
            NumericalAperture=f"{objective.numerical_aperture:g}",
            Immersion=objective.immersion,
            RefractiveIndex=f"{objective.immersion_refractive_index:g}",
        )
        ET.SubElement(
            image, "Attachment", Name="Acquisition", Application=APPLICATION,
            Cycle=str(metadata.cycle), Project=metadata.project, Sample=metadata.sample,
            PixelSizeUm=f"{pixel_um:.6f}", ZPlanes=str(metadata.z_planes),
            FileNaming="<region>--Stage<tile>--Z<plane>--C<channel>.tif",
            PipelineMode=PIPELINE_MODE,
        )
        ET.indent(data)
        path = folder / f"{region}.xml"
        temporary = path.with_name(path.name + PART_SUFFIX)
        ET.ElementTree(data).write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(path)
        written.append(path)
    return written


def _repair_journal_tail(path: Path, fsync: bool = True) -> None:
    """Make the journal end with a newline before appending again.

    A crash during an append can leave a last line without its newline; the
    next record would be glued onto it and both would be unreadable.

    * If that line is a complete record (``tile_records`` already counts it, so
      resume skips that tile), keep it and add the missing newline.
    * Otherwise it is a torn fragment that never described a finished tile:
      cut it; that tile is re-acquired on resume.
    """

    if not path.exists():
        return
    with path.open("rb+") as handle:
        size = handle.seek(0, os.SEEK_END)
        if size == 0:
            return
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return
        # Find the end of the last complete line (0 if there is none).
        position = size
        keep = 0
        while position > 0:
            step = min(65536, position)
            position -= step
            handle.seek(position)
            chunk = handle.read(step)
            newline = chunk.rfind(b"\n")
            if newline != -1:
                keep = position + newline + 1
                break
        handle.seek(keep)
        tail = handle.read()
        if _is_complete_record(tail):
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
        else:
            handle.truncate(keep)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def _is_complete_record(line: bytes) -> bool:
    """Same acceptance rule as :func:`tile_records` (minus the file-existence check)."""

    try:
        record = json.loads(line.decode("utf-8"))
        return bool(record["planes"]) and isinstance(int(record["tile"]), int)
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def _write_plane_tiff(path: Path, frame: np.ndarray, fsync: bool) -> None:
    with path.open("wb") as handle:
        tifffile.imwrite(handle, frame, photometric="minisblack", metadata=None)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


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
