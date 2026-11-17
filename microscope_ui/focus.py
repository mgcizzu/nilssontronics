"""Focus surfaces built from user-captured focus points.

Two methods are available:

``linear`` (default)
    Piecewise-linear interpolation over a Delaunay triangulation of the focus
    points. Inside the convex hull of the points every tile Z is a weighted
    mix of the three surrounding points, so local tilt and warping of large
    sections is followed. Outside the hull Z is taken from the nearest point
    on the hull boundary (constant extrapolation), which never runs away the
    way extrapolating a tilted plane can.

``plane``
    Least-squares plane through all points. Robust when focus points are noisy
    and the section is flat.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .models import FocusMethod, FocusPoint

Point = tuple[float, float]

# Focus points must spread across, not only along, a line: the across-line spread
# (standard deviation) must reach this fraction of the along-line spread and at
# least MIN_ACROSS_SPREAD_UM. Thinner layouts cannot define a tilt across the line.
MIN_SPREAD_RATIO = 0.01
MIN_ACROSS_SPREAD_UM = 5.0


@dataclass(frozen=True)
class _Triangle:
    a: int
    b: int
    c: int


class FocusSurface:
    def __init__(self, points: Iterable[FocusPoint], method: FocusMethod = "linear") -> None:
        unique = _merge_duplicates(tuple(points))
        if len(unique) < 3:
            raise ValueError("A focus surface needs at least three focus points")
        _check_spread(tuple((p.x_um, p.y_um) for p in unique))
        if method not in ("linear", "plane"):
            raise ValueError(f"Unknown focus surface method: {method}")
        self.method = method
        self.points = unique
        self._xy: tuple[Point, ...] = tuple((p.x_um, p.y_um) for p in unique)
        self._z: tuple[float, ...] = tuple(p.z_um for p in unique)
        self._plane = _fit_plane(unique)
        self._triangles: tuple[_Triangle, ...] = ()
        self._hull_edges: tuple[tuple[int, int], ...] = ()
        if method == "linear":
            self._triangles = _delaunay(self._xy)
            self._hull_edges = _boundary_edges(self._triangles)
            if not self._triangles:  # cannot happen after _check_spread; never crash anyway
                self.method = "plane"

    @property
    def triangles(self) -> tuple[tuple[int, int, int], ...]:
        return tuple((t.a, t.b, t.c) for t in self._triangles)

    def z_at(self, x_um: float, y_um: float) -> float:
        if self.method == "plane":
            a, b, c = self._plane
            return a * x_um + b * y_um + c
        point = (x_um, y_um)
        for triangle in self._triangles:
            weights = _barycentric(
                point,
                self._xy[triangle.a],
                self._xy[triangle.b],
                self._xy[triangle.c],
            )
            if weights is not None and min(weights) >= -1e-9:
                return (
                    weights[0] * self._z[triangle.a]
                    + weights[1] * self._z[triangle.b]
                    + weights[2] * self._z[triangle.c]
                )
        return self._z_on_hull(point)

    def z_range(self, positions: Iterable[Point]) -> tuple[float, float]:
        values = [self.z_at(x, y) for x, y in positions]
        if not values:
            return (min(self._z), max(self._z))
        return (min(values), max(values))

    def _z_on_hull(self, point: Point) -> float:
        if not self._hull_edges:
            a, b, c = self._plane
            return a * point[0] + b * point[1] + c
        best: tuple[float, float] | None = None
        for start, end in self._hull_edges:
            (x1, y1), (x2, y2) = self._xy[start], self._xy[end]
            dx, dy = x2 - x1, y2 - y1
            length_sq = dx * dx + dy * dy
            t = 0.0 if length_sq == 0 else ((point[0] - x1) * dx + (point[1] - y1) * dy) / length_sq
            t = min(1.0, max(0.0, t))
            px, py = x1 + t * dx, y1 + t * dy
            distance = math.hypot(point[0] - px, point[1] - py)
            z = self._z[start] + t * (self._z[end] - self._z[start])
            if best is None or distance < best[0]:
                best = (distance, z)
        return best[1]


def _check_spread(xy: tuple[Point, ...]) -> None:
    """Reject layouts that are (almost) on one line, with an actionable message."""

    n = len(xy)
    mx = sum(x for x, _ in xy) / n
    my = sum(y for _, y in xy) / n
    sxx = sum((x - mx) ** 2 for x, _ in xy) / n
    syy = sum((y - my) ** 2 for _, y in xy) / n
    sxy = sum((x - mx) * (y - my) for x, y in xy) / n
    half_trace = (sxx + syy) / 2
    root = math.sqrt(max(0.0, half_trace**2 - (sxx * syy - sxy * sxy)))
    along = math.sqrt(max(0.0, half_trace + root))
    across = math.sqrt(max(0.0, half_trace - root))
    needed = max(MIN_ACROSS_SPREAD_UM, MIN_SPREAD_RATIO * along)
    if across < needed:
        raise ValueError(
            "Focus points lie (almost) on one line: they spread "
            f"{along:.0f} µm along it but only {across:.1f} µm across, so the tilt across "
            f"that line is unknown. Add a focus point at least {2 * needed:.0f} µm away from "
            "the line (or use a single Z / Z-stack instead of a focus map)."
        )


def _merge_duplicates(points: tuple[FocusPoint, ...], tolerance: float = 1e-6) -> tuple[FocusPoint, ...]:
    merged: list[list[float]] = []
    for point in points:
        for item in merged:
            if abs(item[0] - point.x_um) <= tolerance and abs(item[1] - point.y_um) <= tolerance:
                item[2] += point.z_um
                item[3] += 1
                break
        else:
            merged.append([point.x_um, point.y_um, point.z_um, 1])
    return tuple(FocusPoint(x, y, z / n) for x, y, z, n in merged)


def _fit_plane(points: tuple[FocusPoint, ...]) -> tuple[float, float, float]:
    """Least-squares z = a·x + b·y + c, solved in centred coordinates."""

    n = len(points)
    mx = sum(p.x_um for p in points) / n
    my = sum(p.y_um for p in points) / n
    mz = sum(p.z_um for p in points) / n
    sxx = sum((p.x_um - mx) ** 2 for p in points)
    syy = sum((p.y_um - my) ** 2 for p in points)
    sxy = sum((p.x_um - mx) * (p.y_um - my) for p in points)
    sxz = sum((p.x_um - mx) * (p.z_um - mz) for p in points)
    syz = sum((p.y_um - my) * (p.z_um - mz) for p in points)
    determinant = sxx * syy - sxy * sxy
    if abs(determinant) < 1e-12:
        raise ValueError("Focus points must not all lie on one line")
    a = (sxz * syy - syz * sxy) / determinant
    b = (syz * sxx - sxz * sxy) / determinant
    return a, b, mz - a * mx - b * my


def _barycentric(p: Point, a: Point, b: Point, c: Point) -> tuple[float, float, float] | None:
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denominator) < 1e-12:
        return None
    wa = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denominator
    wb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denominator
    return wa, wb, 1.0 - wa - wb


def _circumcircle_contains(points: list[Point], triangle: _Triangle, p: Point) -> bool:
    ax, ay = points[triangle.a]
    bx, by = points[triangle.b]
    cx, cy = points[triangle.c]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-18:
        return False
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    radius_sq = (ax - ux) ** 2 + (ay - uy) ** 2
    return (p[0] - ux) ** 2 + (p[1] - uy) ** 2 < radius_sq * (1 - 1e-12)


def _delaunay(xy: tuple[Point, ...]) -> tuple[_Triangle, ...]:
    """Bowyer–Watson triangulation; focus point sets are small (tens of points)."""

    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    scale = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    # Triangulate in centred, unit-scaled coordinates: a huge super-triangle in raw
    # µm loses so much precision that thin triangles vanish from the result.
    unit = [((x - cx) / scale, (y - cy) / scale) for x, y in xy]
    span = 20.0
    points: list[Point] = unit + [
        (-2 * span, -span),
        (2 * span, -span),
        (0.0, 2 * span),
    ]
    n = len(xy)
    triangles = [_Triangle(n, n + 1, n + 2)]
    for index in range(n):
        p = points[index]
        bad = [t for t in triangles if _circumcircle_contains(points, t, p)]
        edge_count: dict[tuple[int, int], int] = {}
        for t in bad:
            for edge in ((t.a, t.b), (t.b, t.c), (t.c, t.a)):
                key = (min(edge), max(edge))
                edge_count[key] = edge_count.get(key, 0) + 1
        triangles = [t for t in triangles if t not in bad]
        for (start, end), count in edge_count.items():
            if count == 1:
                triangles.append(_Triangle(start, end, index))
    return tuple(
        t
        for t in triangles
        if t.a < n and t.b < n and t.c < n and _area2(tuple(unit), t) > 1e-12
    )


def _area2(xy: tuple[Point, ...], t: _Triangle) -> float:
    (ax, ay), (bx, by), (cx, cy) = xy[t.a], xy[t.b], xy[t.c]
    return abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))


def _boundary_edges(triangles: tuple[_Triangle, ...]) -> tuple[tuple[int, int], ...]:
    counts: dict[tuple[int, int], int] = {}
    for t in triangles:
        for edge in ((t.a, t.b), (t.b, t.c), (t.c, t.a)):
            key = (min(edge), max(edge))
            counts[key] = counts.get(key, 0) + 1
    return tuple(edge for edge, count in counts.items() if count == 1)
