"""Overview canvas: the stage in µm, with regions, tiles, focus points and FOV.

The scene uses stage coordinates directly (1 scene unit = 1 µm, +y down, like
the oriented camera frames), so no coordinate bookkeeping is needed. The
canvas only *reports* gestures as signals; the window decides what they mean
(and whether they are allowed).

Modes
-----
navigate   drag pans, wheel zooms, double-click moves the stage there
rectangle  drag draws a rectangular region
polygon    click adds corners, double-click (or Enter) closes the polygon
move_set   drag translates the whole ROI set (regions + focus points), XY only
focus      click near a focus point selects/goes to it; elsewhere moves XY
Esc cancels the current drawing or set move.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from qtpy.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from ..models import FocusPoint, ScanRegion, StageLimits, Tile

MODES = ("navigate", "rectangle", "polygon", "move_set", "focus")
MAX_DRAWN_TILES = 4000
FOCUS_PICK_PX = 14
MIN_RECT_PX = 8

REGION_PEN = QColor("#2bb3a3")
DRAFT_PEN = QColor("#69f0d0")
TILE_PEN = QColor(150, 200, 255, 190)
FOCUS_COLOR = QColor("#f2a541")
SELECTED_COLOR = QColor("#ff5a36")
FOV_PEN = QColor("#ffd34d")
LIMIT_PEN = QColor("#555b66")


def bounded_set_translation(
    regions: Sequence[ScanRegion],
    focus_points: Sequence[FocusPoint],
    limits: StageLimits,
    dx_um: float,
    dy_um: float,
) -> tuple[float, float]:
    """Clamp a set translation so every vertex and focus point stays inside the stage limits.

    This limits a *drag gesture* (the set stops at the edge); it is not a stage
    command, so the "never clamp stage targets" rule does not apply here.
    """

    xs = [x for region in regions for x, _ in region.vertices]
    ys = [y for region in regions for _, y in region.vertices]
    xs.extend(point.x_um for point in focus_points)
    ys.extend(point.y_um for point in focus_points)
    if not xs:
        return 0.0, 0.0
    dx_um = min(limits.x_max_um - max(xs), max(limits.x_min_um - min(xs), dx_um))
    dy_um = min(limits.y_max_um - max(ys), max(limits.y_min_um - min(ys), dy_um))
    return dx_um, dy_um


class OverviewCanvas(QGraphicsView):
    region_drawn = Signal(object)  # ScanRegion
    stage_move_requested = Signal(float, float)
    focus_clicked = Signal(float, float, int)  # x, y, index of picked point or -1
    move_set_started = Signal()
    move_set_dragged = Signal(float, float)  # raw dx, dy from drag start (µm)
    move_set_finished = Signal()
    draft_cancelled = Signal()
    hint = Signal(str)

    def __init__(self, limits: StageLimits, parent=None) -> None:
        self._scene = QGraphicsScene()
        super().__init__(self._scene, parent)
        self.limits = limits
        self.mode = "navigate"
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#15181d"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        margin = 0.05 * max(limits.x_max_um - limits.x_min_um, limits.y_max_um - limits.y_min_um)
        self._scene.setSceneRect(
            QRectF(
                limits.x_min_um - margin,
                limits.y_min_um - margin,
                limits.x_max_um - limits.x_min_um + 2 * margin,
                limits.y_max_um - limits.y_min_um + 2 * margin,
            )
        )
        limit_item = self._scene.addRect(
            QRectF(limits.x_min_um, limits.y_min_um, limits.x_max_um - limits.x_min_um,
                   limits.y_max_um - limits.y_min_um),
            _cosmetic(LIMIT_PEN, 1, Qt.PenStyle.DashLine),
        )
        limit_item.setZValue(-20)
        self._mosaic_item = QGraphicsPixmapItem()
        self._mosaic_item.setZValue(-10)
        self._mosaic_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._mosaic_item)
        self._live_item = QGraphicsPixmapItem()
        self._live_item.setZValue(-4)  # above saved previews, below regions and tile outlines
        self._live_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._live_item.hide()
        self._scene.addItem(self._live_item)
        self._dynamic: list = []
        self._preview_items: list[QGraphicsPixmapItem] = []
        self._focus_points: list[FocusPoint] = []
        self._draft_items: list = []
        self._polygon: list[tuple[float, float]] = []
        self._press: QPointF | None = None
        self._drag_item: QGraphicsRectItem | None = None
        self._moving_set = False

    # -- content -----------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(mode)
        self.cancel_draft(emit=False)
        self.mode = mode
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag if mode == "navigate" else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.OpenHandCursor if mode == "navigate"
            else Qt.CursorShape.SizeAllCursor if mode == "move_set"
            else Qt.CursorShape.CrossCursor
        )

    def set_content(
        self,
        regions: Iterable[ScanRegion],
        focus_points: Iterable[FocusPoint],
        tiles: Sequence[Tile],
        fov_um: tuple[float, float],
        stage_xy: tuple[float, float],
        selected_focus: int | None = None,
    ) -> None:
        for item in self._dynamic:
            self._scene.removeItem(item)
        self._dynamic = []
        fw, fh = fov_um
        tile_path = QPainterPath()
        for tile in tiles[:MAX_DRAWN_TILES]:
            tile_path.addRect(QRectF(tile.x_um - fw / 2, tile.y_um - fh / 2, fw, fh))
        self._add(QGraphicsPathItem(tile_path), _cosmetic(TILE_PEN, 1), z=1)
        for region in regions:
            path = QPainterPath()
            path.addPolygon(QPolygonF([QPointF(x, y) for x, y in region.vertices]))
            path.closeSubpath()
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(QColor(43, 179, 163, 40)))
            self._add(item, _cosmetic(REGION_PEN, 2), z=2)
        self._focus_points = list(focus_points)
        fixed_size = QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations
        for index, point in enumerate(self._focus_points):
            color = SELECTED_COLOR if index == selected_focus else FOCUS_COLOR
            dot = QGraphicsEllipseItem(-6, -6, 12, 12)  # screen pixels at any zoom
            dot.setFlag(fixed_size)
            dot.setPos(point.x_um, point.y_um)
            dot.setBrush(QBrush(color))
            self._add(dot, _cosmetic(QColor("black"), 1), z=4)
            label = QGraphicsSimpleTextItem(f"F{index + 1}")
            label.setBrush(QBrush(color))
            label.setFlag(fixed_size)
            label.setPos(point.x_um, point.y_um)
            label.setTransform(QTransform.fromTranslate(8, 4))  # screen-pixel offset
            self._add(label, None, z=4)
        x, y = stage_xy
        self._add(QGraphicsRectItem(x - fw / 2, y - fh / 2, fw, fh), _cosmetic(FOV_PEN, 2), z=5)

    def set_mosaic(self, rgb: np.ndarray | None, bounds=None) -> None:
        """Underlay the cycle mosaic (RGB uint8) across ``bounds`` (MosaicBounds)."""

        if rgb is None or bounds is None:
            self._mosaic_item.setPixmap(QPixmap())
            return
        rgb = np.ascontiguousarray(rgb[:, :, :3])
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888)
        self._mosaic_item.setPixmap(QPixmap.fromImage(image.copy()))
        self._mosaic_item.setPos(bounds.x_min_um, bounds.y_min_um)
        self._mosaic_item.setScale(bounds.width_um / rgb.shape[1])

    def add_preview_tile(
        self,
        frame: np.ndarray,
        center: tuple[float, float],
        fov_um: tuple[float, float],
        display_range: tuple[float, float],
        color: str,
    ) -> None:
        """Paint one spiral-preview frame (oriented, +col = +X, +row = +Y) at its stage position."""

        low, high = display_range
        scaled = np.clip((frame.astype(np.float32) - low) / max(high - low, 1.0), 0.0, 1.0)
        tint = QColor(color)
        rgb = scaled[:, :, None] * np.array([tint.red(), tint.green(), tint.blue()], np.float32)
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888)
        item = QGraphicsPixmapItem(QPixmap.fromImage(image.copy()))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        fw, fh = fov_um
        item.setPos(center[0] - fw / 2, center[1] - fh / 2)
        item.setTransform(QTransform.fromScale(fw / rgb.shape[1], fh / rgb.shape[0]))
        item.setZValue(-5)  # above the cycle mosaic, below regions and tiles
        self._scene.addItem(item)
        self._preview_items.append(item)

    def set_live_frame(
        self,
        frame: np.ndarray,
        center: tuple[float, float],
        fov_um: tuple[float, float],
        color: str,
    ) -> None:
        """Show the latest live frame at the current physical stage position."""

        low, high = np.percentile(frame, (1.0, 99.7)) if frame.size else (0.0, 1.0)
        high = max(float(high), float(low) + 1.0)
        scaled = np.clip((frame.astype(np.float32) - low) / (high - low), 0.0, 1.0)
        tint = QColor(color)
        rgb = scaled[:, :, None] * np.array(
            [tint.red(), tint.green(), tint.blue()], dtype=np.float32
        )
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.shape[1] * 3,
            QImage.Format.Format_RGB888,
        )
        fw, fh = fov_um
        self._live_item.setPixmap(QPixmap.fromImage(image.copy()))
        self._live_item.setPos(center[0] - fw / 2, center[1] - fh / 2)
        self._live_item.setTransform(QTransform.fromScale(fw / rgb.shape[1], fh / rgb.shape[0]))
        self._live_item.show()

    def clear_live_frame(self) -> None:
        self._live_item.hide()
        self._live_item.setPixmap(QPixmap())

    @property
    def live_frame_visible(self) -> bool:
        return self._live_item.isVisible() and not self._live_item.pixmap().isNull()

    def clear_preview(self) -> None:
        for item in self._preview_items:
            self._scene.removeItem(item)
        self._preview_items = []

    @property
    def preview_tile_count(self) -> int:
        return len(self._preview_items)

    def fit(self, rect: QRectF | None = None) -> None:
        target = rect if rect is not None and rect.isValid() else self._scene.sceneRect()
        self.fitInView(target, Qt.AspectRatioMode.KeepAspectRatio)

    def fit_items(self, regions: Sequence[ScanRegion], focus_points: Sequence[FocusPoint]) -> None:
        xs = [x for r in regions for x, _ in r.vertices] + [p.x_um for p in focus_points]
        ys = [y for r in regions for _, y in r.vertices] + [p.y_um for p in focus_points]
        if not xs:
            self.fit()
            return
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        pad = 0.15 * max(w, h, 500.0)
        self.fit(QRectF(min(xs) - pad, min(ys) - pad, w + 2 * pad, h + 2 * pad))

    # -- interaction -------------------------------------------------------------
    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self.mode == "navigate":
            super().mousePressEvent(event)
            return
        pos = self.mapToScene(event.position().toPoint())
        if self.mode == "rectangle":
            self._press = pos
            self._drag_item = self._scene.addRect(QRectF(pos, pos), _cosmetic(DRAFT_PEN, 2, Qt.PenStyle.DashLine))
            self._drag_item.setZValue(10)
        elif self.mode == "polygon":
            point = (pos.x(), pos.y())
            if not self._polygon or math.dist(point, self._polygon[-1]) > self._px_to_um(3):
                self._polygon.append(point)
            self._draw_polygon_draft()
        elif self.mode == "move_set":
            self._press = pos
            self._moving_set = True
            self.move_set_started.emit()
        elif self.mode == "focus":
            self.focus_clicked.emit(pos.x(), pos.y(), self._pick_focus(pos))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press is None:
            super().mouseMoveEvent(event)
            return
        pos = self.mapToScene(event.position().toPoint())
        if self.mode == "rectangle" and self._drag_item is not None:
            self._drag_item.setRect(QRectF(self._press, pos).normalized())
        elif self.mode == "move_set" and self._moving_set:
            self.move_set_dragged.emit(pos.x() - self._press.x(), pos.y() - self._press.y())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._press is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        pos = self.mapToScene(event.position().toPoint())
        press, self._press = self._press, None
        if self.mode == "rectangle":
            self._remove_drag_item()
            rect = QRectF(press, pos).normalized()
            if min(rect.width(), rect.height()) < self._px_to_um(MIN_RECT_PX):
                self.hint.emit("Drag a larger rectangle to add a region")
                return
            self.region_drawn.emit(ScanRegion(rect.left(), rect.top(), rect.width(), rect.height()))
        elif self.mode == "move_set" and self._moving_set:
            self._moving_set = False
            self.move_set_finished.emit()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        pos = self.mapToScene(event.position().toPoint())
        if self.mode == "navigate":
            self.stage_move_requested.emit(pos.x(), pos.y())
        elif self.mode == "polygon":
            self.finish_polygon()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_draft()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.mode == "polygon":
            self.finish_polygon()
        else:
            super().keyPressEvent(event)

    def finish_polygon(self) -> None:
        corners = tuple(self._polygon)
        if len(corners) >= 3:
            self._polygon = []
            self._draw_polygon_draft()
            self.region_drawn.emit(ScanRegion.from_polygon(corners))
        else:
            self.hint.emit("A polygon needs at least three corners")

    def cancel_draft(self, emit: bool = True) -> None:
        was_moving = self._moving_set
        self._polygon = []
        self._draw_polygon_draft()
        self._remove_drag_item()
        self._press = None
        self._moving_set = False
        if emit and was_moving:
            self.draft_cancelled.emit()

    # -- helpers -----------------------------------------------------------------
    def _add(self, item, pen: QPen | None, z: float) -> None:
        if pen is not None:
            item.setPen(pen)
        item.setZValue(z)
        self._scene.addItem(item)
        self._dynamic.append(item)

    def _remove_drag_item(self) -> None:
        if self._drag_item is not None:
            self._scene.removeItem(self._drag_item)
            self._drag_item = None

    def _draw_polygon_draft(self) -> None:
        for item in self._draft_items:
            self._scene.removeItem(item)
        self._draft_items = []
        if not self._polygon:
            return
        path = QPainterPath(QPointF(*self._polygon[0]))
        for point in self._polygon[1:]:
            path.lineTo(QPointF(*point))
        item = QGraphicsPathItem(path)
        item.setPen(_cosmetic(DRAFT_PEN, 2, Qt.PenStyle.DashLine))
        item.setZValue(10)
        self._scene.addItem(item)
        self._draft_items.append(item)

    def _pick_focus(self, pos: QPointF) -> int:
        limit = self._px_to_um(FOCUS_PICK_PX)
        best = (-1, limit)
        for index, point in enumerate(self._focus_points):
            distance = math.hypot(point.x_um - pos.x(), point.y_um - pos.y())
            if distance <= best[1]:
                best = (index, distance)
        return best[0]

    def _px_to_um(self, pixels: float) -> float:
        scale = self.transform().m11() or 1.0
        return pixels / abs(scale)


def _cosmetic(color: QColor, width: float, style=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color, width, style)
    pen.setCosmetic(True)  # constant on-screen width at any zoom
    return pen
