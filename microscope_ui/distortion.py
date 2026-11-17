"""Optional lens-distortion and channel-registration correction.

Some microscopes image a straight grid with visible barrel/pincushion
distortion, and each filter set can shift or scale its channel by a fraction
of a pixel. Leica's LAS X corrects this on the fly; this module does the same
for labs whose system needs it, so every ``cycle_NN/`` plane handed to
ISS_preprocessing is already geometrically correct. Off by default.

Calibration (once per objective, camera size and set of channels):

1. Image a grid slide (fluorescent, or reflective/brightfield for geometry
   only) with every channel to be corrected, refocusing per channel, at one
   stage position.
2. :func:`detect_grid_points` finds the grid features (dots, or the cells
   between lines) to sub-pixel accuracy, and :func:`index_lattice` gives each
   one its row/column on the slide.
3. :func:`fit_correction` fits one model for all channels against the *ideal*
   grid (known pitch): a shared grid pose plus, per channel, the true pixel
   size, radial (k1, k2) and tangential (p1, p2) distortion and a shift
   relative to the reference channel.

Model (per channel; X, Y = ideal µm from the frame centre, image axes)::

    a, b   = X / pixel_size, Y / pixel_size          (ideal pixels)
    u, v   = a / R, b / R                            (R = half the frame diagonal)
    r²     = u² + v²
    u'     = u (1 + k1 r² + k2 r⁴) + 2 p1 u v + p2 (r² + 2 u²)
    v'     = v (1 + k1 r² + k2 r⁴) + p1 (r² + 2 v²) + 2 p2 u v
    raw    = (cx + R u' + shift_x, cy + R v' + shift_y)  (cx, cy = frame centre)

Correction (:class:`FrameCorrector`) resamples each frame (bicubic) onto an
ideal grid whose pixel size is the configured one, so the pixel size written
to ``Metadata/R01.xml`` stays exact and every channel lands on the same grid.
Pixels whose source lies outside the camera sensor become 0. Rotation between
camera and stage is not corrected (that is the camera-orientation setting).

The correction file is JSON (``distortion_<objective>.json`` in the
configuration's calibration folder). An experiment copies it when cycle 1
starts and every later cycle uses that copy, so all cycles share one geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

CORRECTION_FORMAT = "nilssontronics-distortion"
CORRECTION_VERSION = 1
EXPERIMENT_COPY = "distortion_correction.json"
MIN_PITCH_PX = 6.0
MIN_POINTS_PER_CHANNEL = 30
WARN_RMS_PX = 0.35
WARN_COVERAGE = 0.7


class CalibrationError(ValueError):
    """The grid images cannot produce a trustworthy correction."""


def correction_filename(objective_id: str) -> str:
    return f"distortion_{objective_id}.json"


def correction_path(calibration_dir: Path | str, objective_id: str) -> Path:
    return Path(calibration_dir) / correction_filename(objective_id)


# -- model ------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelModel:
    pixel_size_um: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    shift_x_px: float = 0.0
    shift_y_px: float = 0.0
    # Fit quality, for people; not used by the correction.
    rms_px: float = 0.0
    max_residual_px: float = 0.0
    points: int = 0


@dataclass(frozen=True)
class DistortionCorrection:
    objective_id: str
    frame_width_px: int
    frame_height_px: int
    grid_pitch_um: float
    reference_channel: str
    channels: dict[str, ChannelModel]
    nominal_pixel_size_um: float = 0.0
    grid_rotation_deg: float = 0.0
    created_at: str = ""
    notes: tuple[str, ...] = ()

    # -- persistence --------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": CORRECTION_FORMAT,
            "version": CORRECTION_VERSION,
            "objective_id": self.objective_id,
            "frame_width_px": self.frame_width_px,
            "frame_height_px": self.frame_height_px,
            "nominal_pixel_size_um": self.nominal_pixel_size_um,
            "grid_pitch_um": self.grid_pitch_um,
            "grid_rotation_deg": self.grid_rotation_deg,
            "reference_channel": self.reference_channel,
            "created_at": self.created_at,
            "model": (
                "Brown-Conrady: radial k1,k2 and tangential p1,p2 in coordinates normalised by "
                "half the frame diagonal; maps ideal µm (relative to the frame centre) to raw pixels"
            ),
            "channels": {key: asdict(value) for key, value in sorted(self.channels.items())},
            "notes": list(self.notes),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DistortionCorrection":
        if data.get("format") != CORRECTION_FORMAT:
            raise ValueError("This is not a distortion correction file")
        if data.get("version") != CORRECTION_VERSION:
            raise ValueError(f"Unsupported distortion correction version: {data.get('version')}")
        correction = cls(
            objective_id=str(data["objective_id"]),
            frame_width_px=int(data["frame_width_px"]),
            frame_height_px=int(data["frame_height_px"]),
            grid_pitch_um=float(data["grid_pitch_um"]),
            reference_channel=str(data["reference_channel"]),
            channels={key: ChannelModel(**value) for key, value in data["channels"].items()},
            nominal_pixel_size_um=float(data.get("nominal_pixel_size_um", 0.0)),
            grid_rotation_deg=float(data.get("grid_rotation_deg", 0.0)),
            created_at=str(data.get("created_at", "")),
            notes=tuple(data.get("notes", ())),
        )
        recorded = data.get("fingerprint")
        if recorded and recorded != correction.fingerprint:
            raise ValueError("The distortion correction file was edited after it was made")
        return correction

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "DistortionCorrection":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def fingerprint(self) -> str:
        """Identity of the geometry (everything that changes corrected pixels)."""

        core = {
            "objective_id": self.objective_id,
            "frame": [self.frame_width_px, self.frame_height_px],
            "channels": {
                key: [
                    round(value.pixel_size_um, 12), round(value.k1, 12), round(value.k2, 12),
                    round(value.p1, 12), round(value.p2, 12),
                    round(value.shift_x_px, 9), round(value.shift_y_px, 9),
                ]
                for key, value in sorted(self.channels.items())
            },
        }
        digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()[:16]

    # -- geometry -------------------------------------------------------------------
    @property
    def centre(self) -> tuple[float, float]:
        return ((self.frame_width_px - 1) / 2.0, (self.frame_height_px - 1) / 2.0)

    @property
    def norm_radius(self) -> float:
        return math.hypot(self.frame_width_px, self.frame_height_px) / 2.0

    def raw_position(
        self, channel_id: str, x_um: np.ndarray, y_um: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raw pixel position seen by ``channel_id`` for ideal µm offsets from the centre."""

        cx, cy = self.centre
        return _project(self.channels[channel_id], x_um, y_um, cx, cy, self.norm_radius)

    def missing_channels(self, channel_ids: Iterable[str]) -> list[str]:
        return [item for item in channel_ids if item not in self.channels]

    def summary_lines(self) -> list[str]:
        lines = [
            f"Objective {self.objective_id} • frame {self.frame_width_px}×{self.frame_height_px} px • "
            f"grid pitch {self.grid_pitch_um:g} µm (rotated {self.grid_rotation_deg:+.2f}°)",
        ]
        for channel_id, model in sorted(self.channels.items()):
            corner = self.max_distortion_px(channel_id)
            reference = " (reference)" if channel_id == self.reference_channel else ""
            lines.append(
                f"{channel_id}{reference}: pixel {model.pixel_size_um:.5f} µm • distortion ≤ "
                f"{corner:.2f} px at the corners • shift ({model.shift_x_px:+.2f}, "
                f"{model.shift_y_px:+.2f}) px • fit RMS {model.rms_px:.3f} px on {model.points} points"
            )
        return lines

    def max_distortion_px(self, channel_id: str) -> float:
        """Largest displacement due to k/p terms alone (no scale or shift), on the frame edge."""

        model = self.channels[channel_id]
        straight = replace(model, k1=0.0, k2=0.0, p1=0.0, p2=0.0)
        cx, cy = self.centre
        w, h = self.frame_width_px, self.frame_height_px
        edge = np.linspace(0, 1, 64)
        cols = np.concatenate([edge * (w - 1), edge * (w - 1), np.zeros(64), np.full(64, w - 1.0)])
        rows = np.concatenate([np.zeros(64), np.full(64, h - 1.0), edge * (h - 1), edge * (h - 1)])
        x_um = (cols - cx) * model.pixel_size_um
        y_um = (rows - cy) * model.pixel_size_um
        bent = _project(model, x_um, y_um, cx, cy, self.norm_radius)
        flat = _project(straight, x_um, y_um, cx, cy, self.norm_radius)
        return float(np.max(np.hypot(bent[0] - flat[0], bent[1] - flat[1])))


def _project(model: ChannelModel, x_um, y_um, cx: float, cy: float, radius: float):
    u = np.asarray(x_um, dtype=np.float64) / model.pixel_size_um / radius
    v = np.asarray(y_um, dtype=np.float64) / model.pixel_size_um / radius
    r2 = u * u + v * v
    radial = 1.0 + model.k1 * r2 + model.k2 * r2 * r2
    ud = u * radial + 2.0 * model.p1 * u * v + model.p2 * (r2 + 2.0 * u * u)
    vd = v * radial + model.p1 * (r2 + 2.0 * v * v) + 2.0 * model.p2 * u * v
    return cx + radius * ud + model.shift_x_px, cy + radius * vd + model.shift_y_px


# -- applying the correction ----------------------------------------------------------


class FrameCorrector:
    """Resamples frames onto the ideal grid; one cached sampling map per channel.

    ``fov_width_um`` / ``fov_height_um`` fix the output pixel size
    (field of view / frame size), so corrected frames match the metadata.
    Thread-safe for use from the single SSD-writer thread.
    """

    def __init__(
        self,
        correction: DistortionCorrection,
        channel_ids: Sequence[str],
        fov_width_um: float,
        fov_height_um: float,
    ) -> None:
        missing = correction.missing_channels(channel_ids)
        if missing:
            raise CalibrationError(
                "The distortion correction has no calibration for channel(s) " + ", ".join(missing)
            )
        self.correction = correction
        self.fov_width_um = fov_width_um
        self.fov_height_um = fov_height_um
        self._maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _map(self, channel_id: str, shape: tuple[int, int]):
        cached = self._maps.get(channel_id)
        if cached is not None:
            return cached
        c = self.correction
        height, width = shape
        if (width, height) != (c.frame_width_px, c.frame_height_px):
            raise CalibrationError(
                f"Frames are {width}×{height} px but the distortion correction was made for "
                f"{c.frame_width_px}×{c.frame_height_px} px (binning or ROI changed?). "
                "Recalibrate for this camera setting."
            )
        out_px_x = self.fov_width_um / width
        out_px_y = self.fov_height_um / height
        cx, cy = c.centre
        cols = (np.arange(width, dtype=np.float64) - cx) * out_px_x
        rows = (np.arange(height, dtype=np.float64) - cy) * out_px_y
        x_um, y_um = np.meshgrid(cols, rows)
        sx, sy = c.raw_position(channel_id, x_um, y_um)
        # Up to half a pixel beyond the edge pixel centres is still inside the sensor.
        valid = (sx >= -0.5) & (sx <= width - 0.5) & (sy >= -0.5) & (sy <= height - 0.5)
        x0 = np.floor(sx)
        y0 = np.floor(sy)
        fx = (sx - x0).astype(np.float32)
        fy = (sy - y0).astype(np.float32)
        # Index into the frame padded by 2 px on every side (edge values): taps x0-1..x0+2.
        padded_width = width + 4
        x0 = np.clip(x0.astype(np.int64), -1, width - 1)
        y0 = np.clip(y0.astype(np.int64), -1, height - 1)
        base = ((y0 + 1) * padded_width + (x0 + 1)).astype(np.int32)
        cached = (base, fx, fy, valid)
        self._maps[channel_id] = cached
        return cached

    def outside_fraction(self, channel_id: str) -> float:
        c = self.correction
        _base, _fx, _fy, valid = self._map(channel_id, (c.frame_height_px, c.frame_width_px))
        return float(1.0 - valid.mean())

    def correct(self, channel_id: str, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 2:
            raise ValueError(f"Expected a 2-D plane, got shape {frame.shape}")
        base, fx, fy, valid = self._map(channel_id, frame.shape)
        height, width = frame.shape
        padded = np.pad(frame.astype(np.float32, copy=False), 2, mode="edge").ravel()
        stride = width + 4
        wx = _keys_weights(fx)
        wy = _keys_weights(fy)
        out = np.zeros(frame.shape, dtype=np.float32)
        for j in range(4):
            row = wx[0] * padded[base + j * stride]
            for i in range(1, 4):
                row += wx[i] * padded[base + (j * stride + i)]
            out += wy[j] * row
        out[~valid] = 0.0
        if np.issubdtype(frame.dtype, np.integer):
            info = np.iinfo(frame.dtype)
            np.rint(out, out=out)
            np.clip(out, info.min, info.max, out=out)
        return out.astype(frame.dtype, copy=False)


def _keys_weights(t: np.ndarray) -> list[np.ndarray]:
    """Bicubic convolution (Keys, a = -0.5) weights for taps at -1, 0, 1, 2."""

    t2 = t * t
    t3 = t2 * t
    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2
    return [w0, w1, w2, w3]


class CorrectingPlaneSink:
    """Wraps a plane sink: frames are corrected before they reach the disk.

    With ``keep_uncorrected`` the sink also receives the original frame
    (``uncorrected=``), which the mosaic stores beside the corrected plane.
    """

    def __init__(self, sink, corrector: FrameCorrector, keep_uncorrected: bool = False) -> None:
        self._sink = sink
        self._corrector = corrector
        self._keep = keep_uncorrected

    def begin_tile(self, *args, **kwargs) -> None:
        self._sink.begin_tile(*args, **kwargs)

    def write_plane(self, channel_id: str, z_index: int, frame: np.ndarray, z_um: float | None = None) -> None:
        corrected = self._corrector.correct(channel_id, frame)
        if self._keep:
            self._sink.write_plane(channel_id, z_index, corrected, z_um, uncorrected=frame)
        else:
            self._sink.write_plane(channel_id, z_index, corrected, z_um)

    def finish_tile(self):
        return self._sink.finish_tile()

    def abandon_tile(self) -> None:
        self._sink.abandon_tile()


# -- grid detection -----------------------------------------------------------------


def _box_1d(image: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius < 1:
        return image
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius + 1, radius)
    padded = np.pad(image, pad, mode="reflect")
    summed = np.cumsum(padded, axis=axis, dtype=np.float64)
    size = 2 * radius + 1
    upper = np.take(summed, np.arange(size, summed.shape[axis]), axis=axis)
    lower = np.take(summed, np.arange(0, summed.shape[axis] - size), axis=axis)
    return ((upper - lower) / size).astype(np.float32)


def _box(image: np.ndarray, radius: int, passes: int = 1) -> np.ndarray:
    for _ in range(passes):
        image = _box_1d(_box_1d(image, radius, 0), radius, 1)
    return image


def _max_filter(image: np.ndarray, radius: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    size = 2 * radius + 1
    out = image
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="constant", constant_values=-np.inf)
        out = sliding_window_view(padded, size, axis=axis).max(axis=-1)
    return out


def detect_grid_points(
    image: np.ndarray, pitch_px: float, polarity: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel centres of periodic grid features.

    ``polarity=+1`` finds bright features (fluorescent dots, or the crossings
    of bright lines); ``-1`` finds dark ones (the cells between bright lines,
    or dark dots). Either gives a lattice with the grid's pitch. Returns
    ``(points[N, 2] as x, y in pixels, strength[N])``. Features whose
    measuring disc (radius pitch/2) would leave the frame are dropped.
    """

    if pitch_px < MIN_PITCH_PX:
        raise CalibrationError(
            f"The grid pitch is only {pitch_px:.1f} px at this magnification; use a grid with "
            f"a pitch of at least {MIN_PITCH_PX:g} px (larger pitch or higher magnification)."
        )
    img = np.asarray(image, dtype=np.float32)
    height, width = img.shape
    background = _box(img, max(2, int(round(pitch_px))), passes=2)
    signal = polarity * (img - background)
    smooth = _box(signal, max(1, int(round(pitch_px / 8))), passes=3)
    peaks = (smooth >= _max_filter(smooth, max(1, int(pitch_px / 3)))) & (smooth > 0)
    radius = max(2, int(pitch_px / 2))
    margin = radius + 2
    peaks[:margin, :] = False
    peaks[-margin:, :] = False
    peaks[:, :margin] = False
    peaks[:, -margin:] = False
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    strength = smooth[ys, xs]
    keep = strength > 0.3 * np.percentile(strength, 90)
    xs, ys, strength = xs[keep], ys[keep], strength[keep]

    offsets = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    cx = xs.astype(np.float64)
    cy = ys.astype(np.float64)
    for _ in range(4):
        ix = np.clip(np.rint(cx).astype(np.int64), radius, width - radius - 1)
        iy = np.clip(np.rint(cy).astype(np.int64), radius, height - radius - 1)
        # Centroids use the unfiltered image: the background filter is not symmetric
        # near the frame edges and would pull features outwards (a fake k1).
        patch = polarity * img[iy[:, None, None] + dy[None], ix[:, None, None] + dx[None]]
        rel_x = dx[None] + (ix - cx)[:, None, None]
        rel_y = dy[None] + (iy - cy)[:, None, None]
        disc = (rel_x**2 + rel_y**2) <= radius**2
        low = np.where(disc, patch, np.inf).min(axis=(1, 2))
        high = np.where(disc, patch, -np.inf).max(axis=(1, 2))
        level = 0.5 * (low + high)
        weight = np.clip(patch - level[:, None, None], 0, None) * disc
        total = weight.sum(axis=(1, 2))
        ok = total > 0
        new_x = cx + np.where(ok, (weight * rel_x).sum(axis=(1, 2)) / np.where(ok, total, 1), 0)
        new_y = cy + np.where(ok, (weight * rel_y).sum(axis=(1, 2)) / np.where(ok, total, 1), 0)
        cx, cy = new_x, new_y
    inside = (cx >= margin) & (cx <= width - 1 - margin) & (cy >= margin) & (cy <= height - 1 - margin)
    points = np.column_stack([cx, cy])[inside]
    strength = strength[inside]
    return _deduplicate(points, strength, pitch_px / 3)


def _deduplicate(points: np.ndarray, strength: np.ndarray, distance: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-strength)
    bins: dict[tuple[int, int], list[int]] = {}
    kept: list[int] = []
    for index in order:
        x, y = points[index]
        bx, by = int(x // distance), int(y // distance)
        clash = False
        for nx in (bx - 1, bx, bx + 1):
            for ny in (by - 1, by, by + 1):
                for other in bins.get((nx, ny), ()):
                    if (points[other][0] - x) ** 2 + (points[other][1] - y) ** 2 < distance**2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            bins.setdefault((bx, by), []).append(index)
            kept.append(index)
    kept_sorted = np.array(sorted(kept), dtype=np.int64)
    return points[kept_sorted], strength[kept_sorted]


class _PointIndex:
    def __init__(self, points: np.ndarray, cell: float) -> None:
        self.points = points
        self.cell = cell
        self.bins: dict[tuple[int, int], list[int]] = {}
        for index, (x, y) in enumerate(points):
            self.bins.setdefault((int(x // cell), int(y // cell)), []).append(index)

    def nearest(self, x: float, y: float, within: float) -> int | None:
        bx, by = int(x // self.cell), int(y // self.cell)
        reach = int(math.ceil(within / self.cell))
        best, best_d2 = None, within * within
        for nx in range(bx - reach, bx + reach + 1):
            for ny in range(by - reach, by + reach + 1):
                for index in self.bins.get((nx, ny), ()):
                    px, py = self.points[index]
                    d2 = (px - x) ** 2 + (py - y) ** 2
                    if d2 <= best_d2:
                        best, best_d2 = index, d2
        return best


@dataclass(frozen=True)
class LatticeIndex:
    points: np.ndarray  # (N, 2) x, y px of the indexed points
    ij: np.ndarray  # (N, 2) integer lattice coordinates (i along b1 ≈ +x, j along b2 ≈ +y)
    spacing_px: float
    angle_rad: float  # direction of b1 in image coordinates


def index_lattice(points: np.ndarray, pitch_px_guess: float, centre: tuple[float, float]) -> LatticeIndex:
    """Give every detected point its (i, j) on a square lattice by growing from the centre.

    Each step predicts the neighbour from the locally observed spacing, so a
    smoothly distorted grid is followed across the whole field.
    """

    if len(points) < 9:
        raise CalibrationError(f"Only {len(points)} grid features were found")
    index = _PointIndex(points, max(pitch_px_guess, 2.0))
    nearest = []
    for k, (x, y) in enumerate(points):
        best = None
        for other in _neighbours(index, x, y, 1.6 * pitch_px_guess):
            if other == k:
                continue
            d = math.hypot(points[other][0] - x, points[other][1] - y)
            best = d if best is None or d < best else best
        if best is not None:
            nearest.append(best)
    if not nearest:
        raise CalibrationError("The grid features are too far apart for the pitch given")
    spacing = float(np.median(nearest))
    vectors = []
    for k, (x, y) in enumerate(points):
        for other in _neighbours(index, x, y, 1.3 * spacing):
            if other != k:
                d = points[other] - points[k]
                if 0.75 * spacing <= math.hypot(*d) <= 1.25 * spacing:
                    vectors.append(d)
    if len(vectors) < 8:
        raise CalibrationError("The grid features do not form a regular square lattice")
    vectors = np.asarray(vectors)
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle = float(np.angle(np.mean(np.exp(4j * phi))) / 4.0)
    axes = []
    for direction in (angle, angle + math.pi / 2):
        unit = np.array([math.cos(direction), math.sin(direction)])
        aligned = vectors[(vectors @ unit) > 0.9 * spacing]
        axes.append(np.median(aligned, axis=0) if len(aligned) else unit * spacing)
    b1, b2 = axes
    spacing = float((np.linalg.norm(b1) + np.linalg.norm(b2)) / 2)

    start = int(np.argmin(np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1])))
    ij = {start: (0, 0)}
    local = {start: np.column_stack([b1, b2])}
    queue = [start]
    tolerance = 0.3 * spacing
    head = 0
    while head < len(queue):
        current = queue[head]
        head += 1
        ci, cj = ij[current]
        basis = local[current]
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            target = points[current] + sign * basis[:, axis]
            found = index.nearest(target[0], target[1], tolerance)
            if found is None or found in ij:
                continue
            step = (ci + sign, cj) if axis == 0 else (ci, cj + sign)
            ij[found] = step
            new_basis = basis.copy()
            new_basis[:, axis] = sign * (points[found] - points[current])
            local[found] = new_basis
            queue.append(found)
    order = sorted(ij)
    return LatticeIndex(
        points=points[order],
        ij=np.array([ij[k] for k in order], dtype=np.int64),
        spacing_px=spacing,
        angle_rad=float(math.atan2(b1[1], b1[0])),
    )


def _neighbours(index: _PointIndex, x: float, y: float, within: float):
    bx, by = int(x // index.cell), int(y // index.cell)
    reach = int(math.ceil(within / index.cell))
    for nx in range(bx - reach, bx + reach + 1):
        for ny in range(by - reach, by + reach + 1):
            yield from index.bins.get((nx, ny), ())


def inspect_grid_image(image: np.ndarray, pitch_px: float) -> tuple[int, float]:
    """Quick check of one grid image: (features that can be followed, spacing in px).

    Raises :class:`CalibrationError` with the reason when no lattice is found.
    """

    height, width = image.shape
    centre = ((width - 1) / 2.0, (height - 1) / 2.0)
    best: tuple[int, float] | None = None
    error = None
    for sign in (-1, 1):
        try:
            points, _strength = detect_grid_points(image, pitch_px, sign)
            lattice = index_lattice(points, pitch_px, centre)
        except CalibrationError as exc:
            error = exc
            continue
        if best is None or len(lattice.ij) > best[0]:
            best = (len(lattice.ij), lattice.spacing_px)
    if best is None:
        raise error or CalibrationError("No grid found")
    return best


# -- calibration --------------------------------------------------------------------


@dataclass
class GridImage:
    channel_id: str
    image: np.ndarray
    stage_xy_um: tuple[float, float] | None = None


@dataclass
class CalibrationResult:
    correction: DistortionCorrection
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = self.correction.summary_lines()
        if self.warnings:
            lines += ["", "Warnings:"] + [f"• {item}" for item in self.warnings]
        return "\n".join(lines)


def fit_correction(
    images: Sequence[GridImage],
    objective_id: str,
    grid_pitch_um: float,
    nominal_pixel_size_um: float,
    reference_channel: str | None = None,
    polarity: int | None = None,
) -> CalibrationResult:
    """Fit one correction for every channel from grid images taken at one stage position."""

    if not images:
        raise CalibrationError("Add at least one grid image")
    if grid_pitch_um <= 0 or nominal_pixel_size_um <= 0:
        raise CalibrationError("Grid pitch and pixel size must be positive")
    channel_ids = [item.channel_id for item in images]
    if len(set(channel_ids)) != len(channel_ids):
        raise CalibrationError("Use one grid image per channel")
    shapes = {item.image.shape for item in images}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise CalibrationError("All grid images must be 2-D and the same size")
    positions = [item.stage_xy_um for item in images if item.stage_xy_um is not None]
    if positions and max(math.hypot(x - positions[0][0], y - positions[0][1]) for x, y in positions) > 0.5:
        raise CalibrationError(
            "The stage moved between grid images. Snap every channel at the same XY position "
            "(only Z may change), otherwise the shifts between channels cannot be measured."
        )
    reference = reference_channel or channel_ids[0]
    if reference not in channel_ids:
        raise CalibrationError(f"The reference channel {reference} has no grid image")

    height, width = images[0].image.shape
    centre = ((width - 1) / 2.0, (height - 1) / 2.0)
    pitch_guess = grid_pitch_um / nominal_pixel_size_um
    # Bright features (dots, line crossings) and dark ones (cells between lines) both
    # form the lattice; keep whichever fits best (crossings of thin lines centre poorly).
    candidates = []
    errors = []
    for sign in (polarity,) if polarity in (1, -1) else (-1, 1):
        try:
            lattices = _index_all(images, pitch_guess, centre, sign)
            _align_indices(lattices, reference)
            model = _fit(lattices, reference, grid_pitch_um, pitch_guess, width, height)
        except CalibrationError as exc:
            errors.append(str(exc))
            continue
        worst = max(channel.rms_px for channel in model["channels"].values())
        candidates.append((worst, lattices, model))
    if not candidates:
        raise CalibrationError(errors[0] if errors else "No grid found")
    _worst, lattices, model = min(candidates, key=lambda item: item[0])

    warnings = []
    for channel_id, lattice in lattices.items():
        span_x = np.ptp(lattice.points[:, 0]) / width
        span_y = np.ptp(lattice.points[:, 1]) / height
        if min(span_x, span_y) < WARN_COVERAGE:
            warnings.append(
                f"{channel_id}: grid features cover only {100 * span_x:.0f}% × {100 * span_y:.0f}% "
                "of the field; the corners are extrapolated. Fill the whole field with the grid."
            )

    correction = DistortionCorrection(
        objective_id=objective_id,
        frame_width_px=width,
        frame_height_px=height,
        grid_pitch_um=grid_pitch_um,
        reference_channel=reference,
        channels=model["channels"],
        nominal_pixel_size_um=nominal_pixel_size_um,
        grid_rotation_deg=math.degrees(model["theta"]),
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    for channel_id, channel in correction.channels.items():
        if channel.rms_px > WARN_RMS_PX:
            warnings.append(
                f"{channel_id}: fit RMS {channel.rms_px:.2f} px is high (focus, dirt on the slide "
                "or a damaged grid?). Check the image before trusting this correction."
            )
        error = channel.pixel_size_um / nominal_pixel_size_um - 1.0
        if abs(error) > 0.01:
            warnings.append(
                f"{channel_id}: measured pixel size {channel.pixel_size_um:.5f} µm differs by "
                f"{100 * error:+.1f}% from the configured {nominal_pixel_size_um:g} µm. Corrected "
                "images use the configured size: check the grid pitch, or fix pixel_size_um."
            )
    corrector = FrameCorrector(
        correction, channel_ids, width * nominal_pixel_size_um, height * nominal_pixel_size_um
    )
    for channel_id in channel_ids:
        outside = corrector.outside_fraction(channel_id)
        if outside > 0.002:
            warnings.append(
                f"{channel_id}: {100 * outside:.1f}% of each corrected frame (at the edges) lies "
                "outside the camera image and will be 0."
            )
    return CalibrationResult(replace(correction, notes=tuple(warnings)), warnings)


def _index_all(images, pitch_guess, centre, polarity: int) -> dict[str, LatticeIndex]:
    lattices = {}
    for item in images:
        points, _strength = detect_grid_points(item.image, pitch_guess, polarity)
        lattices[item.channel_id] = index_lattice(points, pitch_guess, centre)
    score = min(len(lattice.ij) for lattice in lattices.values())
    if score < MIN_POINTS_PER_CHANNEL:
        weakest = min(lattices, key=lambda key: len(lattices[key].ij))
        raise CalibrationError(
            f"{weakest}: only {score} grid features could be followed (need at least "
            f"{MIN_POINTS_PER_CHANNEL}). Check focus, exposure and the grid pitch."
        )
    spacing = np.array([lattice.spacing_px for lattice in lattices.values()])
    if np.any(np.abs(spacing / pitch_guess - 1) > 0.3):
        raise CalibrationError(
            f"The grid spacing in the images ({float(np.median(spacing)):.1f} px) does not match "
            f"the pitch given ({pitch_guess:.1f} px expected). Check the grid pitch and objective."
        )
    return lattices


def _align_indices(lattices: dict[str, LatticeIndex], reference: str) -> None:
    """Shift each channel's (i, j) so the same physical grid feature has the same index."""

    ref = lattices[reference]
    ref_index = _PointIndex(ref.points, max(ref.spacing_px, 2.0))
    for channel_id, lattice in list(lattices.items()):
        if channel_id == reference:
            continue
        differences = []
        for point, ij in zip(lattice.points, lattice.ij):
            match = ref_index.nearest(point[0], point[1], 0.35 * ref.spacing_px)
            if match is not None:
                differences.append(tuple(ref.ij[match] - ij))
        if len(differences) < MIN_POINTS_PER_CHANNEL // 2:
            raise CalibrationError(
                f"{channel_id} and {reference} disagree by more than a third of the grid pitch; "
                "were both snapped at the same stage position?"
            )
        values, counts = np.unique(np.array(differences), axis=0, return_counts=True)
        offset = values[int(np.argmax(counts))]
        lattices[channel_id] = replace(lattice, ij=lattice.ij + offset)


_CHANNEL_PARAMS = ("pixel_size_um", "k1", "k2", "p1", "p2", "shift_x_px", "shift_y_px")


def _fit(lattices, reference, pitch_um, pitch_guess, width, height) -> dict:
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius = math.hypot(width, height) / 2.0
    channel_ids = list(lattices)
    ref = lattices[reference]
    origin_row = int(np.argmin(np.abs(ref.ij).sum(axis=1)))
    pixel0 = pitch_um / ref.spacing_px
    theta0 = ref.angle_rad
    i0, j0 = ref.ij[origin_row]
    x0 = (ref.points[origin_row, 0] - cx) * pixel0
    y0 = (ref.points[origin_row, 1] - cy) * pixel0
    ox = x0 - pitch_um * (i0 * math.cos(theta0) - j0 * math.sin(theta0))
    oy = y0 - pitch_um * (i0 * math.sin(theta0) + j0 * math.cos(theta0))

    params = [ox, oy, theta0]
    scales = [1.0, 1.0, 1e-3]
    layout: dict[str, list[int | None]] = {}
    for channel_id in channel_ids:
        slots: list[int | None] = []
        for name in _CHANNEL_PARAMS:
            if channel_id == reference and name in ("shift_x_px", "shift_y_px"):
                slots.append(None)
                continue
            slots.append(len(params))
            if name == "pixel_size_um":
                params.append(pitch_um / lattices[channel_id].spacing_px)
                scales.append(1e-4)
            else:
                params.append(0.0)
                scales.append(1e-3 if name.startswith(("k", "p")) else 1e-2)
        layout[channel_id] = slots
    params = np.array(params, dtype=np.float64)
    scales = np.array(scales)

    def unpack(vector):
        models = {}
        for channel_id in channel_ids:
            values = [0.0 if slot is None else float(vector[slot]) for slot in layout[channel_id]]
            models[channel_id] = ChannelModel(**dict(zip(_CHANNEL_PARAMS, values)))
        return models

    def residuals(vector):
        o_x, o_y, theta = vector[:3]
        cos, sin = math.cos(theta), math.sin(theta)
        models = unpack(vector)
        parts = []
        for channel_id in channel_ids:
            lattice = lattices[channel_id]
            i = lattice.ij[:, 0].astype(np.float64)
            j = lattice.ij[:, 1].astype(np.float64)
            x_um = o_x + pitch_um * (i * cos - j * sin)
            y_um = o_y + pitch_um * (i * sin + j * cos)
            px, py = _project(models[channel_id], x_um, y_um, cx, cy, radius)
            parts.append(px - lattice.points[:, 0])
            parts.append(py - lattice.points[:, 1])
        return np.concatenate(parts)

    params = _levenberg_marquardt(residuals, params, scales)
    models = unpack(params)
    final = residuals(params)
    offset = 0
    channels = {}
    for channel_id in channel_ids:
        count = len(lattices[channel_id].ij)
        rx = final[offset:offset + count]
        ry = final[offset + count:offset + 2 * count]
        offset += 2 * count
        distance = np.hypot(rx, ry)
        channels[channel_id] = replace(
            models[channel_id],
            rms_px=float(np.sqrt(np.mean(distance**2))),
            max_residual_px=float(distance.max()),
            points=count,
        )
    return {"channels": channels, "theta": float(params[2])}


def _levenberg_marquardt(function, params, scales, iterations: int = 200) -> np.ndarray:
    current = function(params)
    cost = float(current @ current)
    damping = 1e-3
    for _ in range(iterations):
        jacobian = np.empty((current.size, params.size))
        for k in range(params.size):
            step = 1e-6 * (abs(params[k]) + scales[k])
            shifted = params.copy()
            shifted[k] += step
            forward = function(shifted)
            shifted[k] -= 2 * step
            jacobian[:, k] = (forward - function(shifted)) / (2 * step)
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        improved = False
        while damping < 1e12:
            system = normal + damping * np.diag(np.maximum(np.diag(normal), 1e-12))
            try:
                delta = np.linalg.solve(system, -gradient)
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            candidate = params + delta
            trial = function(candidate)
            trial_cost = float(trial @ trial)
            if trial_cost < cost:
                relative = (cost - trial_cost) / max(cost, 1e-300)
                params, current, cost = candidate, trial, trial_cost
                damping = max(damping / 3, 1e-12)
                improved = True
                break
            damping *= 4
        if not improved or relative < 1e-12:
            break
    return params


# -- command line -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Fit a correction from saved grid images: ``python -m microscope_ui.distortion``."""

    import argparse

    import tifffile

    from .config import DEFAULT_CONFIG_PATH, load_config

    parser = argparse.ArgumentParser(
        prog="python -m microscope_ui.distortion",
        description="Fit a lens-distortion/channel correction from grid-slide images "
        "(one per channel, all at the same stage position, already oriented like the app's frames).",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--objective", required=True, help="objective id from the configuration")
    parser.add_argument("--pitch", type=float, required=True, help="grid pitch in µm")
    parser.add_argument(
        "--image", action="append", required=True, metavar="CHANNEL=FILE.tif",
        help="grid image for one channel (repeat per channel)",
    )
    parser.add_argument("--reference", help="reference channel (default: the first image)")
    parser.add_argument("--output", help="where to write the file (default: the calibration folder)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    images = []
    for item in args.image:
        channel_id, _, filename = item.partition("=")
        if not filename:
            parser.error(f"--image needs CHANNEL=FILE, got {item!r}")
        config.channel(channel_id)
        image = tifffile.imread(filename)
        if image.ndim != 2:
            parser.error(f"{filename}: expected one 2-D plane")
        images.append(GridImage(channel_id, image))
    objective = config.objective(args.objective)
    try:
        result = fit_correction(images, objective.id, args.pitch, objective.pixel_size_um, args.reference)
    except CalibrationError as exc:
        print(f"Calibration failed: {exc}")
        return 1
    output = Path(args.output) if args.output else correction_path(config.calibration_dir, objective.id)
    result.correction.save(output)
    print(result.report())
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
