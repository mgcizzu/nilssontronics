"""Guided "Refocus all" and "Align selected F → stage" in the Qt view."""

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
from microscope_ui.models import FocusPoint, ScanRegion  # noqa: E402
from microscope_ui.qt.window import ScanWindow  # noqa: E402

POINTS = [FocusPoint(-1000, -800, 0.0), FocusPoint(1000, -800, 0.5), FocusPoint(0, 900, 1.0)]


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
        win.region_list.set_items([ScanRegion(-1500, -1000, 3000, 2000)])
        win.focus_list.set_items(POINTS)
        yield win
        win.live_button.setChecked(False)
        win.close()


def at(win, point, z=None):
    p = win.backend.position
    return (p.x_um, p.y_um) == (point.x_um, point.y_um) and (z is None or p.z_um == pytest.approx(z))


def test_refocus_all_walks_every_point(qtbot, window):
    window.refocus_button.click()
    assert window.refocus_button.text() == "Stop refocus"
    assert window.set_z_button.text() == "Accept Z ▶"
    assert window.overview.mode == "focus" and window.live_button.isChecked()
    assert not window.overview_accept_button.isHidden()

    new_z = [2.25, -1.5, 3.75]
    for index, z in enumerate(new_z):
        qtbot.waitUntil(lambda: at(window, POINTS[index], POINTS[index].z_um), timeout=5000)
        assert window.refocus_index == index
        assert f"F{index + 1} of 3" in window.refocus_label.text()
        # Focus with a Z move and accept immediately: Accept must wait for the move.
        window.device_move(window.backend.move_z, z)
        # Alternate the two Accept buttons (setup panel / under the overview).
        (window.set_z_button if index % 2 == 0 else window.overview_accept_button).click()
        qtbot.waitUntil(
            lambda: window.focus_list.items[index].z_um == pytest.approx(z), timeout=5000
        )

    qtbot.waitUntil(lambda: window.refocus_index is None, timeout=5000)
    assert [p.z_um for p in window.focus_list.items] == pytest.approx(new_z)
    assert [(p.x_um, p.y_um) for p in window.focus_list.items] == [(p.x_um, p.y_um) for p in POINTS]
    assert window.refocus_button.text() == "Refocus all"
    assert window.set_z_button.text() == "Set Z here"
    assert window.refocus_label.isHidden() and window.overview_accept_button.isHidden()
    assert "All 3 focus points refocused" in window.overview_hint.text()


def test_double_accept_only_stores_one_point(qtbot, window):
    window.refocus_button.click()
    qtbot.waitUntil(lambda: at(window, POINTS[0]), timeout=5000)
    window.device_move(window.backend.move_z, 4.0)
    window.set_z_button.click()
    window.set_z_button.click()  # impatient double click
    qtbot.waitUntil(lambda: window.refocus_index == 1, timeout=5000)
    qtbot.waitUntil(lambda: at(window, POINTS[1]), timeout=5000)
    assert window.focus_list.items[0].z_um == pytest.approx(4.0)
    assert window.focus_list.items[1].z_um == pytest.approx(POINTS[1].z_um)  # untouched


def test_failed_position_read_reenables_accept(qtbot, window):
    window.refocus_index = 0
    window.with_fresh_position = (
        lambda _callback, on_error=None: on_error(RuntimeError("encoder offline"))
    )
    window._set_focus_z_here()
    assert not window._accept_pending
    assert window.messages[-1][0:2] == ("warn", "Could not read the stage position")


def test_stopping_refocus_clears_pending_accept(window):
    window.refocus_index = 0
    window._accept_pending = True
    window.stop_refocus()
    assert not window._accept_pending


def test_stop_and_overview_click_during_refocus(qtbot, window):
    window.show()
    qtbot.waitExposed(window)
    window.tabs.setCurrentIndex(0)
    window.overview.fit(QRectF(-3000, -3000, 6000, 6000))
    window.refocus_button.click()
    qtbot.waitUntil(lambda: at(window, POINTS[0]), timeout=5000)
    # Clicking F3 on the overview jumps the walk there.
    pos = window.overview.mapFromScene(QPointF(POINTS[2].x_um, POINTS[2].y_um))
    qtbot.mouseClick(window.overview.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    assert window.refocus_index == 2
    qtbot.waitUntil(lambda: at(window, POINTS[2], POINTS[2].z_um), timeout=5000)
    window.refocus_button.click()
    assert window.refocus_index is None and window.refocus_button.text() == "Refocus all"


def test_refocus_needs_focus_points(window):
    window.focus_list.set_items([])
    window.refocus_button.click()
    assert window.refocus_index is None
    assert window.messages[-1][1] == "No focus points"


def test_align_selected_point_to_stage(qtbot, window):
    window.focus_list.list.setCurrentRow(1)
    target = (POINTS[1].x_um + 120.0, POINTS[1].y_um - 80.0)
    window.device_move(window.backend.move_xy, *target)
    window.align_button.click()  # queued after the move, so it sees the new position
    qtbot.waitUntil(lambda: window.focus_list.items[1].x_um == pytest.approx(target[0]), timeout=5000)
    moved = window.focus_list.items
    assert (moved[1].x_um, moved[1].y_um) == pytest.approx(target)
    assert [p.z_um for p in moved] == [p.z_um for p in POINTS]  # Z kept
    assert (moved[0].x_um - POINTS[0].x_um, moved[0].y_um - POINTS[0].y_um) == pytest.approx((120, -80))
    region = window.region_list.items[0]
    assert (region.x_um, region.y_um) == pytest.approx((-1380, -1080))
    assert "ΔX +120.0 ΔY -80.0" in window.overview_hint.text()


def test_align_without_selection_explains(window):
    window.focus_list.list.setCurrentRow(-1)
    window.align_button.click()
    assert window.messages[-1][1] == "Select a focus point"


def test_cycle_2_after_align_records_offset(qtbot, window):
    window.channel_table._checks["dapi"].setChecked(True)
    window.z_mode_box.setCurrentIndex(window.z_mode_box.findData("focus_map"))
    window.start_button.click()
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    assert window.controller.status().state == "complete", window.messages

    # Re-register for cycle 2: F1's landmark turned up 30 µm right, 20 µm down.
    window.focus_list.list.setCurrentRow(0)
    window.device_move(window.backend.move_xy, POINTS[0].x_um + 30, POINTS[0].y_um + 20)
    window.align_button.click()
    qtbot.waitUntil(lambda: window.focus_list.items[0].x_um == pytest.approx(-970), timeout=5000)
    window.refocus_button.click()
    window.start_button.click()  # starting a cycle ends the refocus walk
    assert window.refocus_index is None
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    assert window.controller.status().state == "complete", window.messages
    record = window.controller.experiment.cycle(2)
    assert (record.offset_x_um, record.offset_y_um) == pytest.approx((30, 20))
