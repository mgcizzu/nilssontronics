"""Headless tests for the Qt view (skipped when qtpy/pytest-qt are missing)."""

from __future__ import annotations

import os
import time
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QPoint, Qt, QTimer  # noqa: E402
from qtpy.QtWidgets import QScrollArea, QTabWidget  # noqa: E402

from microscope_ui.backend import SimulatedMicroscope, StagePosition  # noqa: E402
from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.models import FocusPoint, ScanRegion  # noqa: E402
from microscope_ui.mosaic import read_tile  # noqa: E402
from microscope_ui.qt.widgets import ChannelTable  # noqa: E402
from microscope_ui.qt.window import ScanWindow  # noqa: E402


class SlowSimulator(SimulatedMicroscope):
    delay = 0.0

    def snap(self):
        time.sleep(self.delay)
        return super().snap()


class FakeWheelEvent:
    def __init__(self, delta: int) -> None:
        self.delta = delta
        self.accepted = False

    def angleDelta(self):  # noqa: N802 - mirrors the Qt API
        return QPoint(0, self.delta)

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.accepted = False


def test_unmapped_hardware_power_is_fixed_in_the_channel_table(qtbot):
    config = load_config()
    table = ChannelTable(config.channels, adjustable_power_ids={"dapi"})
    qtbot.addWidget(table)
    assert table._power["dapi"].isEnabled()
    assert not table._power["af488"].isEnabled()
    assert "Fixed" in table._power["af488"].toolTip()


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


def test_dedicated_wheel_zone_jogs_z_without_stealing_viewer_zoom(qtbot, window):
    window.show()
    window.jog.z_step.setValue(2.5)
    start_z = window.stage_position.z_um

    first_half = FakeWheelEvent(60)
    window.z_wheel.wheelEvent(first_half)
    assert first_half.accepted
    assert window.backend.position.z_um == pytest.approx(start_z)

    second_half = FakeWheelEvent(60)
    window.z_wheel.wheelEvent(second_half)
    qtbot.waitUntil(lambda: window.stage_position.z_um == pytest.approx(start_z + 2.5))
    assert second_half.accepted

    two_notches_down = FakeWheelEvent(-240)
    window.z_wheel.wheelEvent(two_notches_down)
    qtbot.waitUntil(lambda: window.stage_position.z_um == pytest.approx(start_z - 2.5))
    assert window.z_wheel.isVisibleTo(window)
    window.tabs.setCurrentIndex(1)
    assert window.z_wheel.isVisibleTo(window)  # available beside both Overview and Camera
    assert window.review_scroll.widget().isAncestorOf(window.z_wheel)


def test_z_stack_view_highlights_the_nearest_active_plane(qtbot, window):
    window.z_mode_box.setCurrentIndex(window.z_mode_box.findData("stack"))
    window.z_bottom.setValue(-2.0)
    window.z_top.setValue(2.0)
    window.z_step.setValue(1.0)
    assert window.z_wheel.plane_count == 5
    assert window.z_wheel.active_plane_index == 2
    assert window.z_wheel.plane_label.text() == "3 / 5"

    window.jog.z_step.setValue(1.0)
    window.z_wheel.wheelEvent(FakeWheelEvent(120))
    qtbot.waitUntil(lambda: window.z_wheel.active_plane_index == 3)
    assert window.z_wheel.plane_label.text() == "4 / 5"
    assert "1.00 µm" in window.z_wheel.position_label.text()


def test_surface_stack_builds_offsets_around_focus_map(window):
    window.focus_list.set_items(
        [
            FocusPoint(-1000, -1000, -1),
            FocusPoint(1000, -1000, 1),
            FocusPoint(0, 1000, 2),
        ]
    )
    window.z_mode_box.setCurrentIndex(window.z_mode_box.findData("surface_stack"))
    window.z_below.setValue(2.0)
    window.z_above.setValue(3.0)
    window.z_step.setValue(1.0)
    plan = window.build_plan()
    assert plan.z.mode == "surface_stack"
    assert (plan.z.bottom_um, plan.z.top_um, plan.z.plane_count) == (-2.0, 3.0, 6)
    assert not window.z_below.isHidden() and not window.z_above.isHidden()
    assert not window.z_begin_button.isHidden()
    assert window.z_begin_button.text() == "Set start"
    assert window.z_end_button.text() == "Set end"
    assert window.focus_method_box.isEnabled()
    assert window.z_wheel.plane_count == 6


def test_surface_stack_capture_keeps_span_and_centres_it(window):
    window.region_list.set_items([ScanRegion(-1000, -1000, 2000, 2000)])
    window.focus_list.set_items(
        [
            FocusPoint(-1000, -1000, -1),
            FocusPoint(1000, -1000, 1),
            FocusPoint(0, 1000, 2),
        ]
    )
    window.z_mode_box.setCurrentIndex(window.z_mode_box.findData("surface_stack"))
    positions = iter((StagePosition(0, 0, 8), StagePosition(0, 0, 14)))
    window.with_fresh_position = lambda callback, on_error=None: callback(next(positions))
    window._capture_z_begin()
    window._capture_z_end()
    assert window.z_below.value() == pytest.approx(3)
    assert window.z_above.value() == pytest.approx(3)
    assert "Captured span 6" in window.surface_span_label.text()
    plan = window.build_plan()
    assert (plan.z.bottom_um, plan.z.top_um, plan.z.plane_count) == (-3, 3, 7)
    z_by_tile = window.controller.z_positions(
        plan.z,
        plan.focus_points,
        plan.options.focus_method,
        tuple(window.controller.tiles_for(plan)),
        None,
    )
    for values in z_by_tile.values():
        assert values[0] == pytest.approx(values[3] - 3)
        assert values[-1] == pytest.approx(values[3] + 3)
    window._reset_z_stack_capture()
    assert (window.z_below.value(), window.z_above.value()) == pytest.approx((3, 3))
    assert "Span 6" in window.surface_span_label.text()
    assert "Captured" not in window.surface_span_label.text()


def test_compact_button_labels_fit_at_small_window_size(qtbot, window):
    window.resize(1024, 720)
    window.show()
    window.setup_tabs.setCurrentIndex(1)
    qtbot.wait(20)
    buttons = (
        window.add_region_button,
        window.save_set_button,
        window.load_set_button,
        window.shift_button,
        window.focus_list.remove_button,
        window.set_z_button,
        window.refocus_button,
        window.align_button,
        *window.mode_buttons.values(),
        window.preview_button,
        window.clear_preview_button,
    )
    for button in buttons:
        text_width = button.fontMetrics().horizontalAdvance(button.text())
        assert button.width() >= text_width, button.text()


def test_finished_cycle_invalidates_captured_stack_bounds(qtbot, window):
    window.region_list.set_items([ScanRegion(-300, -200, 600, 400)])
    window.channel_table._checks["dapi"].setChecked(True)
    window.z_mode_box.setCurrentIndex(window.z_mode_box.findData("stack"))
    window.z_bottom.setValue(-1.0)
    window.z_top.setValue(1.0)
    window.z_step.setValue(1.0)
    window.z_begin_captured = window.z_end_captured = True
    window.start_button.click()
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    assert window.controller.status().state == "complete"
    assert not window.z_begin_captured and not window.z_end_captured


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


def test_live_restores_per_frame_shutter_after_a_cycle(qtbot, window):
    window.region_list.append(ScanRegion(-300, -200, 600, 400))
    window.channel_table._checks["dapi"].setChecked(True)
    window.shutter_per_frame.setChecked(False)  # shutter open per tile, closed during moves
    status = run_cycle(qtbot, window)
    assert status.state == "complete"
    backend = window.backend
    # The cycle leaves auto-shutter off and the shutter closed ...
    assert not backend.shutter_per_frame and not backend.shutter_open
    assert backend.dark_snaps == 0
    # ... so the next snap must turn per-frame shutter back on before exposing.
    with qtbot.waitSignal(window.device.frame_ready, timeout=5000):
        window._snap()
    assert backend.shutter_per_frame
    assert backend.dark_snaps == 0


def test_window_stays_open_while_storage_cannot_be_closed(qtbot, window):
    window.show()
    with mock.patch.object(window.controller, "close", return_value=False):
        assert not window.close()
    assert window.isVisible()
    assert window.messages[-1][1] == "Still saving"
    with mock.patch.object(window.controller, "close", return_value=True):
        assert window.close()


def test_window_stays_open_while_a_device_command_runs(qtbot, window):
    import threading

    import microscope_ui.qt.window as window_module

    window.show()
    release, started = threading.Event(), threading.Event()

    def stalled_stage_command():
        started.set()
        release.wait(10)

    window.device.submit(stalled_stage_command)
    assert started.wait(5)
    with mock.patch.object(window_module, "DEVICE_DRAIN_TIMEOUT_S", 0.2), \
            mock.patch.object(window.controller, "close", wraps=window.controller.close) as close:
        assert not window.close()
        close.assert_not_called()  # backend untouched while the command runs
    assert window.isVisible()
    assert window.messages[-1][1] == "Microscope busy"
    assert window.position_timer.isActive()

    release.set()
    assert window.device.stop_and_drain(5)
    assert window.close()


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


def test_import_wizard_refuses_while_the_app_drives_hardware(window, monkeypatch):
    monkeypatch.setattr(window.backend, "simulated", False)
    with mock.patch("microscope_ui.qt.profile_wizard.MicroManagerImportWizard") as wizard:
        window.import_mm_configuration()
    wizard.assert_not_called()
    assert window.messages and window.messages[-1][0] == "warn"
    assert "simulator" in window.messages[-1][2]


def test_surface_span_label_shows_when_the_grid_extends_the_span(window):
    window.z_mode_box.setCurrentIndex(window.z_mode_box.findData("surface_stack"))
    window.z_below.setValue(5)
    window.z_above.setValue(3)
    window.z_step.setValue(2)
    assert "-6 … +4 µm (6 planes)" in window.surface_span_label.text()
    window.z_below.setValue(4)
    window.z_above.setValue(4)
    assert "planes" not in window.surface_span_label.text()


def _startup_settings(tmp_path, profile):
    from qtpy.QtCore import QSettings

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("startup_profile", str(profile))
    return settings


def _hardware_profile(tmp_path):
    import json

    from microscope_ui.config import DEFAULT_CONFIG_PATH

    data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["hardware"] = {**data.get("hardware", {}), "backend": "mmcore"}
    profile = tmp_path / "scope.json"
    profile.write_text(json.dumps(data), encoding="utf-8")
    return profile


def test_failed_startup_keeps_the_saved_profile(tmp_path):
    from microscope_ui.backend import HardwareError
    from microscope_ui.config import DEFAULT_CONFIG_PATH
    from microscope_ui.qt import window as window_module

    profile = _hardware_profile(tmp_path)
    settings = _startup_settings(tmp_path, profile)
    real_create = window_module.create_backend
    asked = []

    def create(config):
        if config.hardware.get("backend") == "mmcore":
            raise HardwareError("microscope switched off")
        return real_create(config)

    def choose(path, error):
        asked.append((path, str(error)))
        return window_module.STARTUP_SIMULATOR_ONCE

    with mock.patch.object(window_module, "create_backend", side_effect=create):
        _config, backend, path = window_module.open_startup_backend(None, None, settings, choose)
    assert backend.simulated and path == str(DEFAULT_CONFIG_PATH)
    assert asked == [(str(profile), "microscope switched off")]  # names the failed profile
    assert settings.value("startup_profile") == str(profile)


def test_startup_retry_and_simulator_from_now_on(tmp_path):
    from microscope_ui.backend import HardwareError
    from microscope_ui.qt import window as window_module

    profile = _hardware_profile(tmp_path)
    settings = _startup_settings(tmp_path, profile)
    hardware = mock.Mock(simulated=False)
    attempts = []

    def create(_config):
        attempts.append(1)
        if len(attempts) == 1:
            raise HardwareError("not yet")
        return hardware

    with mock.patch.object(window_module, "create_backend", side_effect=create):
        _c, backend, path = window_module.open_startup_backend(
            None, None, settings, lambda *_a: window_module.STARTUP_RETRY
        )
    assert backend is hardware and path == str(profile) and len(attempts) == 2

    real_create = window_module.create_backend

    def broken(config):
        if config.hardware.get("backend") == "mmcore":
            raise HardwareError("off")
        return real_create(config)

    with mock.patch.object(window_module, "create_backend", side_effect=broken):
        window_module.open_startup_backend(
            None, None, settings, lambda *_a: window_module.STARTUP_SIMULATOR_ALWAYS
        )
    assert settings.value("startup_profile") is None


def test_nyquist_button_sets_the_na_based_z_step(window):
    from microscope_ui.optics import recommended_z_step_um

    window.z_mode_box.setCurrentIndex(window.z_mode_box.findData("stack"))
    assert not window.z_step_row.isHidden()  # Z group may sit on a hidden tab
    objective = window.config_data.objective(window.objective_box.currentData())
    selected = {item.channel_id for item in window.channel_table.settings()}
    channels = tuple(c for c in window.config_data.channels if c.id in selected)
    expected, wavelength = recommended_z_step_um(objective, channels)
    window.z_step.setValue(7)
    window.z_nyquist_button.click()
    assert window.z_step.value() == pytest.approx(expected, abs=0.006)
    assert f"{wavelength:g} nm" in window.z_nyquist_button.toolTip()
    window.z_mode_box.setCurrentIndex(window.z_mode_box.findData("single"))
    assert window.z_step_row.isHidden()
