"""Overview canvas: drawing, set moves and stage navigation with real mouse events."""

from __future__ import annotations

import os
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QPointF, QRectF, Qt  # noqa: E402

from microscope_ui.backend import SimulatedMicroscope  # noqa: E402
from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.models import FocusPoint, ScanRegion, StageLimits  # noqa: E402
from microscope_ui.qt.overview import bounded_set_translation  # noqa: E402
from microscope_ui.qt.window import ScanWindow  # noqa: E402

LEFT = Qt.MouseButton.LeftButton


@pytest.fixture
def window(qtbot, tmp_path):
    config = load_config()
    backend = SimulatedMicroscope(config, frame_px=24)
    backend.change_objective(config.objective("20x_air"))
    with mock.patch(
        "microscope_ui.planner.shutil.disk_usage", return_value=mock.Mock(free=10 * 1024**4)
    ):
        win = ScanWindow(config, backend, use_ndv=False)
        qtbot.addWidget(win)
        win.confirm = lambda *_: True
        win.messages = []
        win.inform = lambda title, text: win.messages.append(("info", title, text))
        win.warn = lambda title, text: win.messages.append(("warn", title, text))
        win.output_edit.setText(str(tmp_path))
        win.show()
        qtbot.waitExposed(win)
        win.tabs.setCurrentIndex(0)
        # A 10 × 10 mm view around the origin so gestures map to sensible µm values.
        win.overview.fit(QRectF(-5000, -5000, 10000, 10000))
        yield win
        win.close()


def vp(win, x_um, y_um):
    return win.overview.mapFromScene(QPointF(x_um, y_um))


def drag(qtbot, win, start, end, steps=5):
    viewport = win.overview.viewport()
    a, b = vp(win, *start), vp(win, *end)
    qtbot.mousePress(viewport, LEFT, pos=a)
    for i in range(1, steps + 1):
        qtbot.mouseMove(viewport, a + (b - a) * (i / steps))
    qtbot.mouseRelease(viewport, LEFT, pos=b)


def test_bounded_set_translation_stops_at_stage_edge():
    limits = StageLimits(-100, 100, -50, 50, -10, 10)
    regions = [ScanRegion(0, 0, 20, 10)]
    points = [FocusPoint(-10, -10, 0)]
    assert bounded_set_translation(regions, points, limits, 500, 0) == (80, 0)
    assert bounded_set_translation(regions, points, limits, -500, -500) == (-90, -40)
    assert bounded_set_translation([], [], limits, 5, 5) == (0.0, 0.0)


def test_draw_rectangle_region(qtbot, window):
    window.set_overview_mode("rectangle")
    drag(qtbot, window, (-1000, -800), (1000, 600))
    assert len(window.region_list.items) == 1
    region = window.region_list.items[0]
    assert region.shape == "rectangle"
    unit = 10000 / min(window.overview.viewport().width(), window.overview.viewport().height())
    assert region.x_um == pytest.approx(-1000, abs=2 * unit)
    assert region.width_um == pytest.approx(2000, abs=2 * unit)
    assert "tiles" in window.summary_label.text()


def test_tiny_drag_adds_nothing(qtbot, window):
    window.set_overview_mode("rectangle")
    drag(qtbot, window, (0, 0), (5, 5))
    assert window.region_list.items == []


def test_draw_polygon_region(qtbot, window):
    window.set_overview_mode("polygon")
    viewport = window.overview.viewport()
    for corner in ((-1000, -1000), (1000, -1000), (0, 1000)):
        qtbot.mouseClick(viewport, LEFT, pos=vp(window, *corner))
    qtbot.mouseDClick(viewport, LEFT, pos=vp(window, 0, 1000))
    assert len(window.region_list.items) == 1
    region = window.region_list.items[0]
    assert region.shape == "polygon" and len(region.polygon_points_um) == 3


def test_move_set_drag_and_escape(qtbot, window):
    window.region_list.append(ScanRegion(-500, -500, 1000, 1000))
    window.focus_list.append(FocusPoint(0, 0, 1.5))
    window.set_overview_mode("move_set")
    drag(qtbot, window, (0, 0), (1000, -500))
    region = window.region_list.items[0]
    point = window.focus_list.items[0]
    assert region.x_um - (-500) == pytest.approx(point.x_um, abs=1e-6)  # moved together
    assert point.x_um == pytest.approx(1000, abs=40) and point.y_um == pytest.approx(-500, abs=40)
    assert point.z_um == 1.5  # XY only

    # Esc mid-drag restores the set as it was when the drag started.
    viewport = window.overview.viewport()
    qtbot.mousePress(viewport, LEFT, pos=vp(window, 0, 0))
    qtbot.mouseMove(viewport, vp(window, 2000, 2000))
    assert window.focus_list.items[0].x_um != pytest.approx(point.x_um)
    qtbot.keyClick(window.overview, Qt.Key.Key_Escape)
    assert window.focus_list.items[0] == point
    assert window.region_list.items[0] == region


def test_regions_locked_after_cycle_1_but_set_can_move(qtbot, window):
    window.region_list.append(ScanRegion(-300, -200, 600, 400))
    window.channel_table._checks["dapi"].setChecked(True)
    window.start_button.click()
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    assert window.controller.status().state == "complete"
    assert not window.mode_buttons["rectangle"].isEnabled()
    assert not window.mode_buttons["polygon"].isEnabled()
    assert window.mode_buttons["move_set"].isEnabled()
    window.set_overview_mode("rectangle")  # refused → falls back to navigate
    assert window.overview.mode == "navigate"
    window.set_overview_mode("move_set")
    drag(qtbot, window, (0, 0), (300, 0))
    assert window.region_list.items[0].x_um == pytest.approx(0, abs=40)
    assert "ROI offset vs cycle 1" in window.summary_label.text()


def test_double_click_moves_stage_and_focus_click_goes_to_point(qtbot, window):
    viewport = window.overview.viewport()
    qtbot.mouseDClick(viewport, LEFT, pos=vp(window, 1200, -700))
    qtbot.waitUntil(lambda: window.stage_position.x_um == pytest.approx(1200, abs=40), timeout=5000)
    assert window.stage_position.y_um == pytest.approx(-700, abs=40)

    window.focus_list.set_items([FocusPoint(-2000, 1500, 3.0)])
    window.set_overview_mode("focus")
    qtbot.mouseClick(viewport, LEFT, pos=vp(window, -2000, 1500))
    qtbot.waitUntil(lambda: window.stage_position.z_um == pytest.approx(3.0), timeout=5000)
    assert (window.stage_position.x_um, window.stage_position.y_um) == (-2000, 1500)
    assert window.focus_list.selected_index() == 0
    assert window.live_button.isChecked()
    window.live_button.setChecked(False)
