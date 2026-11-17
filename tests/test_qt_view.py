"""Headless tests for the Qt view (skipped when qtpy/pytest-qt are missing)."""

from __future__ import annotations

import os
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from microscope_ui.backend import SimulatedMicroscope  # noqa: E402
from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.models import ScanRegion  # noqa: E402
from microscope_ui.mosaic import read_tile_stack  # noqa: E402
from microscope_ui.qt.window import ScanWindow  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path):
    config = load_config()
    backend = SimulatedMicroscope(config, frame_px=24)
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
    tile_file = next((experiment.cycle_folder(2) / "tiles").glob("*.tstack"))
    _header, planes = read_tile_stack(tile_file)
    assert set(planes) == {("dapi", 0), ("cy5", 0)}


def test_snap_and_limit_checked_jog(window):
    assert window._snap()
    limits = window.config_data.stage_limits
    window.backend.move_xy(limits.x_max_um - 10, 0)
    window.jog.xy_step.setValue(100)
    window._jog(100, 0, 0)
    assert window.messages and window.messages[-1][1] == "Stage move refused"
    assert window.backend.position.x_um == pytest.approx(limits.x_max_um - 10)


def test_abort_then_resume(qtbot, window):
    window.backend.delay = 0.0
    window.region_list.append(ScanRegion(-1500, -1000, 3000, 2000))
    window.channel_table._checks["dapi"].setChecked(True)
    window.start_button.click()
    assert window.controller.status() is not None, window.messages
    window.controller.abort()
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    if window.controller.status().state == "complete":
        pytest.skip("cycle finished before abort took effect")
    assert window.resume_button.isEnabled()
    window.resume_button.click()
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    assert window.controller.status().state == "complete"
