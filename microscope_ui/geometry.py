from __future__ import annotations


Point = tuple[float, float]


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Return True when a point is inside or on the boundary of a polygon."""
    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if _point_on_segment(point, (x1, y1), (x2, y2)):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def rectangle_intersects_polygon(
    left: float,
    top: float,
    right: float,
    bottom: float,
    polygon: tuple[Point, ...],
) -> bool:
    corners = ((left, top), (right, top), (right, bottom), (left, bottom))
    if any(point_in_polygon(corner, polygon) for corner in corners):
        return True
    if any(left <= x <= right and top <= y <= bottom for x, y in polygon):
        return True

    rectangle_edges = tuple(
        (corners[index], corners[(index + 1) % len(corners)])
        for index in range(len(corners))
    )
    polygon_edges = tuple(
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )
    return any(
        _segments_intersect(poly_start, poly_end, rect_start, rect_end)
        for poly_start, poly_end in polygon_edges
        for rect_start, rect_end in rectangle_edges
    )


def points_are_non_collinear(points: tuple[Point, ...], tolerance: float = 1e-6) -> bool:
    if len(points) < 3:
        return False
    first = points[0]
    for second_index in range(1, len(points) - 1):
        second = points[second_index]
        for third in points[second_index + 1 :]:
            twice_area = abs(
                (second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0])
            )
            if twice_area > tolerance:
                return True
    return False


def _point_on_segment(point: Point, start: Point, end: Point, tolerance: float = 1e-9) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > tolerance:
        return False
    return (
        min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    values = (orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b))
    if values[0] * values[1] < 0 and values[2] * values[3] < 0:
        return True
    return (
        _point_on_segment(c, a, b)
        or _point_on_segment(d, a, b)
        or _point_on_segment(a, c, d)
        or _point_on_segment(b, c, d)
    )
