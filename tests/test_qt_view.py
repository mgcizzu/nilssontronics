"""Headless tests for the Qt view (skipped when qtpy/pytest-qt are missing)."""

from __future__ import annotations

import os
import time
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import Qt, QTimer  # noqa: E402
from qtpy.QtWidgets import QScrollArea, QTabWidget  # noqa: E402

from microscope_ui.backend import SimulatedMicroscope  # noqa: E402
from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.models import ScanRegion  # noqa: E402
from microscope_ui.mosaic import read_tile  # noqa: E402
from microscope_ui.qt.window import ScanWindow  # noqa: E402


class SlowSimulator(SimulatedMicroscope):
    delay = 0.0

    def snap(self):
        time.sleep(self.delay)
        return super().snap()


@pytest.fixture
def window(qtbot, tmp_path):
    config = load_config()
    backend = SlowSimulator(config, frame_px=24)
    backend.change_objective(config.objective("20x_air"))
    with mock.patch(
        "microscope_ui.planner.shutil.disk_usage", return_value=mock.Mock(free=10 * 1024**4)
    ):
        win = ScanWindow(config, backend, use_ndv=os.environ.get("TEST_NDV") == "1")
        qtbot.addWidget(win)
        win.confirm = lambda *_: True
        win.messages = []
        win.inform = lambda title, text: win.messages.append(("info", title, text))
        win.warn = lambda title, text: win.messages.append(("warn", title, text))
        win.output_edit.setText(str(tmp_path))
        yield win
        win.close()


def run_cycle(qtbot, win):
    win.start_button.click()
    assert win.controller.status() is not None, win.messages
    qtbot.waitUntil(lambda: not win.poll_timer.isActive(), timeout=30000)
    return win.controller.status()


def test_main_panels_are_resizable_scrollable_and_resettable(qtbot, window):
    window.resize(1400, 860)
    window.show()
    qtbot.waitExposed(window)
    splitter = window.main_splitter
    assert splitter.count() == 3
    assert splitter.handleWidth() >= 8
    assert all(not splitter.isCollapsible(index) for index in range(3))
    assert isinstance(window.setup_tabs, QTabWidget)
    assert all(isinstance(page, QScrollArea) for page in window.setup_pages)
    assert isinstance(window.review_scroll, QScrollArea)

    splitter.setSizes([520, 400, 230])
    changed = splitter.sizes()
    assert changed[0] >= 500 and changed[2] <= 250 and all(size > 0 for size in changed)
    window.reset_panel_sizes()
    reset = splitter.sizes()
    assert reset[1] > reset[0] > reset[2] > 0

    reset_actions = [action for action in window.menuBar().actions() if action.text() == "View"]
    assert reset_actions
    assert any(action.text() == "Reset panel sizes" for action in reset_actions[0].menu().actions())


def test_window_fits_screen(window):
    available = window.screen().availableGeometry()
    assert window.width() <= available.width()
    assert window.height() <= available.height()


def test_setup_workflow_is_split_into_named_tabs_and_stage_is_on_right(qtbot, window):
    window.show()
    qtbot.waitExposed(window)
    assert [window.setup_tabs.tabText(index) for index in range(3)] == [
        "Project",
        "Regions",
        "Acquisition",
    ]
    assert not window.setup_tabs.usesScrollButtons()
    assert all(
        page.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        for page in window.setup_pages
    )
    for index, control in enumerate((window.project_edit, window.region_list, window.z_mode_box)):
        window.setup_tabs.setCurrentIndex(index)
        assert control.isVisibleTo(window)
    assert window.review_scroll.widget().isAncestorOf(window.jog)
    assert window.jog.isVisibleTo(window)


def test_two_cycles_with_xy_shift(qtbot, window):
    window.region_list.append(ScanRegion(-600, -400, 1200, 800))
    window.channel_table.set_settings(())
    window.channel_table._checks["dapi"].setChecked(True)
    window.channel_table._checks["cy5"].setChecked(True)
    assert "passes" in window.safety_label.text()

    status = run_cycle(qtbot, window)
    assert status.state == "complete"
    assert window.messages[-1][1] == "Cycle complete"
    experiment = window.controller.experiment
    assert experiment is not None and len(experiment.cycles) == 1
    tiles_1 = experiment.tiles_for_cycle(1)

    # Cycle 2: objective/regions locked, only an XY translation is allowed.
    assert not window.objective_box.isEnabled()
    assert window.shift_button.isEnabled()
    window.shift_roi_set(15.0, -7.5)
    status = run_cycle(qtbot, window)
    assert status.state == "complete"
    record = experiment.cycle(2)
    assert (record.offset_x_um, record.offset_y_um) == (15.0, -7.5)
    tiles_2 = experiment.tiles_for_cycle(2)
    assert [(t.index, t.row, t.col) for t in tiles_1] == [(t.index, t.row, t.col) for t in tiles_2]
    _header, planes = read_tile(experiment.cycle_folder(2), 0)
    assert set(planes) == {("dapi", 0), ("cy5", 0)}


def test_snap_runs_on_device_worker(qtbot, window):
    with qtbot.waitSignal(window.device.frame_ready, timeout=5000) as blocker:
        assert window._snap()
    assert blocker.args[0].shape == (24, 24)


def test_jog_is_limit_checked_and_reports_position(qtbot, window):
    limits = window.config_data.stage_limits
    window.device_move(window.backend.move_xy, limits.x_max_um - 10, 0)
    qtbot.waitUntil(lambda: window.stage_position.x_um == pytest.approx(limits.x_max_um - 10))
    assert f"X {limits.x_max_um - 10:.1f}" in window.position_label.text()
    window._jog(100, 0, 0)
    qtbot.waitUntil(lambda: bool(window.messages), timeout=5000)
    assert window.messages[-1][1] == "Stage move refused"
    assert window.device.stop_and_drain(5)
    assert window.backend.position.x_um == pytest.approx(limits.x_max_um - 10)


def test_gui_stays_responsive_during_slow_live(qtbot, window):
    window.backend.delay = 0.4  # long exposure
    window.live_button.setChecked(True)
    qtbot.waitUntil(lambda: window.device.live, timeout=2000)
    time.sleep(0.05)  # let the worker enter a snap
    fired = []
    start = time.monotonic()
    QTimer.singleShot(10, lambda: fired.append(time.monotonic() - start))
    qtbot.waitUntil(lambda: bool(fired), timeout=2000)
    assert fired[0] < 0.2, "GUI event loop blocked by a camera exposure"
    with qtbot.waitSignal(window.device.frame_ready, timeout=3000):
        pass
    window.live_button.setChecked(False)
    assert window.device.stop_and_drain(5)


def test_live_stays_in_overview_and_refreshes_current_fov(qtbot, window):
    window.tabs.setCurrentIndex(0)
    window.live_button.setChecked(True)
    qtbot.waitUntil(lambda: window.overview.live_frame_visible, timeout=5000)
    assert window.tabs.currentIndex() == 0

    window.live_button.setChecked(False)
    assert not window.overview.live_frame_visible
    assert window.device.stop_and_drain(5)


def test_live_stops_before_cycle_starts(qtbot, window):
    window.backend.delay = 0.05
    window.region_list.append(ScanRegion(-300, -200, 600, 400))
    window.channel_table._checks["dapi"].setChecked(True)
    window.live_button.setChecked(True)
    qtbot.waitUntil(lambda: window.device.live, timeout=2000)
    status = run_cycle(qtbot, window)
    assert status.state == "complete"
    assert not window.device.live and not window.live_button.isChecked()


def test_abort_then_resume(qtbot, window):
    window.backend.delay = 0.02
    window.region_list.append(ScanRegion(-1500, -1000, 3000, 2000))
    window.channel_table._checks["dapi"].setChecked(True)
    window.start_button.click()
    assert window.controller.status() is not None, window.messages
    qtbot.waitUntil(lambda: window.controller.status().tiles_done >= 1, timeout=10000)
    window.controller.abort()
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    assert window.controller.status().state == "aborted"
    assert window.resume_button.isEnabled()
    window.resume_button.click()
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    assert window.controller.status().state == "complete"
