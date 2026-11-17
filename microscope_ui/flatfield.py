"""Optional, non-destructive flat-field estimation after a cycle.

Raw tiles are never modified. For each channel a smooth multiplicative
illumination profile (mean 1.0) is written to ``<cycle>/flatfield/<ch>.npy``
so the downstream pipeline can decide whether to apply it.

If the ``basicpy`` package (BaSiC) is installed it is used. Otherwise a robust
fallback is applied: the per-pixel median of mean-normalised tiles, smoothed
with a wide box filter. The fallback is adequate for vignetting; BaSiC also
models a dark-field term and is preferred for quantitative work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .mosaic import middle_plane, read_plane, tile_records

# Upper bound for one channel's stack of sample frames held in memory at once.
DEFAULT_MEMORY_BUDGET_BYTES = 512 * 1024**2


def estimate_flat_fields(
    cycle_root: Path,
    channel_ids: tuple[str, ...],
    max_tiles: int = 200,
    should_stop: Callable[[], bool] = lambda: False,
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
) -> Path | None:
    """Estimate one illumination profile per channel from a sample of the cycle's tiles.

    Memory stays bounded whatever the camera size: channels are processed one at
    a time, only the middle Z plane of each sampled tile is read, and frames are
    binned so one channel's float32 stack fits ``memory_budget_bytes``. The
    profile is smooth, so it is estimated at that resolution and upsampled
    (bilinear) back to the full frame size.
    """

    cycle_root = Path(cycle_root)
    records = tile_records(cycle_root)
    tile_ids = sorted(records)
    if len(tile_ids) < 3:
        return None
    if len(tile_ids) > max_tiles:
        picks = np.linspace(0, len(tile_ids) - 1, max_tiles).round().astype(int)
        tile_ids = [tile_ids[i] for i in sorted(set(picks))]

    out_dir = cycle_root / "flatfield"
    method = "basicpy" if _has_basicpy() else "median-box"
    summary: dict[str, dict] = {}
    for channel in channel_ids:
        planes = [
            (tile_id, z)
            for tile_id in tile_ids
            if (z := middle_plane(records[tile_id], channel)) is not None
        ]
        if len(planes) < 3:
            continue
        first = read_plane(cycle_root, records[planes[0][0]], channel, planes[0][1])
        full_shape = first.shape
        binning = _binning_for(full_shape, len(planes), memory_budget_bytes)
        binned_shape = (full_shape[0] // binning, full_shape[1] // binning)
        stack = np.empty((len(planes), *binned_shape), dtype=np.float32)
        for index, (tile_id, z) in enumerate(planes):
            if should_stop():
                return None
            frame = first if index == 0 else read_plane(cycle_root, records[tile_id], channel, z)
            if frame.shape != full_shape:
                raise ValueError(f"Tile {tile_id} {channel}: frame shape {frame.shape} != {full_shape}")
            stack[index] = _bin(frame, binning)
        del first
        profile = _basic_profile(stack) if method == "basicpy" else _median_profile(stack)
        del stack
        profile = _upsample(profile, full_shape)
        profile /= profile.mean() or 1.0
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / f"{channel}.npy", profile.astype(np.float32))
        summary[channel] = {
            "tiles_used": len(planes),
            "binning": binning,
            "min": float(profile.min()),
            "max": float(profile.max()),
        }
    if not summary:
        return None
    (out_dir / "flatfield.json").write_text(
        json.dumps(
            {
                "method": method,
                "applied_to_raw_data": False,
                "definition": "corrected = raw / profile (profile mean = 1)",
                "plane": "middle Z plane of each sampled tile",
                "channels": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def _binning_for(shape: tuple[int, int], count: int, budget_bytes: int) -> int:
    """Smallest integer binning so ``count`` float32 frames fit ``budget_bytes``."""

    binning = 1
    while count * (shape[0] // binning) * (shape[1] // binning) * 4 > budget_bytes and min(shape) // (binning + 1) >= 8:
        binning += 1
    return binning


def _bin(frame: np.ndarray, binning: int) -> np.ndarray:
    data = np.asarray(frame, dtype=np.float32)
    if binning == 1:
        return data
    h, w = data.shape[0] // binning, data.shape[1] // binning
    return data[: h * binning, : w * binning].reshape(h, binning, w, binning).mean(axis=(1, 3))


def _upsample(profile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a smooth profile to ``shape`` (pixel-centre aligned)."""

    if profile.shape == tuple(shape):
        return profile.astype(np.float32)
    h, w = profile.shape
    ys = np.clip((np.arange(shape[0]) + 0.5) * h / shape[0] - 0.5, 0, h - 1)
    xs = np.clip((np.arange(shape[1]) + 0.5) * w / shape[1] - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = profile[y0][:, x0] * (1 - fx) + profile[y0][:, x1] * fx
    bottom = profile[y1][:, x0] * (1 - fx) + profile[y1][:, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def _has_basicpy() -> bool:
    try:
        import basicpy  # noqa: F401
    except Exception:
        return False
    return True


def _basic_profile(data: np.ndarray) -> np.ndarray:  # pragma: no cover - optional dependency
    from basicpy import BaSiC

    model = BaSiC(get_darkfield=False)
    model.fit(data)
    profile = np.asarray(model.flatfield, dtype=np.float32)
    return profile / profile.mean()


def _median_profile(data: np.ndarray) -> np.ndarray:
    means = data.reshape(len(data), -1).mean(axis=1)
    means[means == 0] = 1.0
    normalised = data / means[:, None, None]
    median = np.median(normalised, axis=0)
    size = max(3, (min(median.shape) // 16) | 1)
    smooth = _box_blur(median, size)
    mean = smooth.mean()
    return smooth / (mean if mean else 1.0)


def _box_blur(image: np.ndarray, size: int) -> np.ndarray:
    pad = size // 2
    padded = np.pad(image, pad, mode="reflect")
    cumulative = padded.cumsum(axis=0).cumsum(axis=1)
    cumulative = np.pad(cumulative, ((1, 0), (1, 0)))
    h, w = image.shape
    total = (
        cumulative[size : size + h, size : size + w]
        - cumulative[0:h, size : size + w]
        - cumulative[size : size + h, 0:w]
        + cumulative[0:h, 0:w]
    )
    return total / (size * size)
