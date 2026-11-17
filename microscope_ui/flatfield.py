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

from .mosaic import read_tile, tile_records


def estimate_flat_fields(
    cycle_root: Path,
    channel_ids: tuple[str, ...],
    max_tiles: int = 200,
    should_stop: Callable[[], bool] = lambda: False,
) -> Path | None:
    cycle_root = Path(cycle_root)
    tile_ids = sorted(tile_records(cycle_root))
    if len(tile_ids) < 3:
        return None
    if len(tile_ids) > max_tiles:
        picks = np.linspace(0, len(tile_ids) - 1, max_tiles).round().astype(int)
        tile_ids = [tile_ids[i] for i in sorted(set(picks))]

    stacks: dict[str, list[np.ndarray]] = {channel: [] for channel in channel_ids}
    for tile_id in tile_ids:
        if should_stop():
            return None
        _record, planes = read_tile(cycle_root, tile_id)
        for channel in channel_ids:
            z_indices = sorted(z for (c, z) in planes if c == channel)
            if not z_indices:
                continue
            frame = planes[(channel, z_indices[len(z_indices) // 2])].astype(np.float32)
            stacks[channel].append(frame)

    out_dir = cycle_root / "flatfield"
    out_dir.mkdir(exist_ok=True)
    method = "basicpy" if _has_basicpy() else "median-box"
    summary: dict[str, dict] = {}
    for channel, frames in stacks.items():
        if len(frames) < 3:
            continue
        data = np.stack(frames)
        profile = _basic_profile(data) if method == "basicpy" else _median_profile(data)
        np.save(out_dir / f"{channel}.npy", profile.astype(np.float32))
        summary[channel] = {
            "tiles_used": len(frames),
            "min": float(profile.min()),
            "max": float(profile.max()),
        }
    (out_dir / "flatfield.json").write_text(
        json.dumps(
            {
                "method": method,
                "applied_to_raw_data": False,
                "definition": "corrected = raw / profile (profile mean = 1)",
                "channels": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


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
