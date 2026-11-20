"""Qt view: recipes, ROI set files, spiral preview and the configuration dialog."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from microscope_ui.backend import SimulatedMicroscope  # noqa: E402
from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.models import FocusPoint, ScanRegion, StageLimits  # noqa: E402
from microscope_ui.qt.window import ScanWindow  # noqa: E402


class SlowSimulator(SimulatedMicroscope):
    delay = 0.0

    def snap(self):
        time.sleep(self.delay)
        return super().snap()


@pytest.fixture
def window(qtbot, tmp_path):
    config = load_config()
    backend = SlowSimulator(config, frame_px=64)
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
        yield win
        win.live_button.setChecked(False)
        win.close()


def choose_recipe(win, name):
    win.recipe_box.setCurrentText(name)
    win._apply_recipe()


# -- recipes -----------------------------------------------------------------
def test_recipe_sets_objective_channels_and_z_mode(qtbot, window):
    choose_recipe(window, "Tissue survey")
    assert [c.channel_id for c in window.channel_table.settings()] == ["dapi", "af488"]
    assert window.z_mode_box.currentData() == "single"
    qtbot.waitUntil(lambda: window.backend.current_objective_id == "10x_air", timeout=5000)
    assert window.objective_box.currentData() == "10x_air"
    assert window.recipe_box.currentText() == "Tissue survey"

    # Any manual channel change turns the recipe back into "Custom".
    window.channel_table._checks["cy5"].setChecked(True)
    assert window.recipe_box.currentText() == "Custom"


def test_recipe_confirms_before_automatic_turret_move(window):
    prompts = []
    window.confirm = lambda title, text: prompts.append((title, text)) or False
    choose_recipe(window, "Tissue survey")
    assert prompts and prompts[-1][0] == "Change objective"
    assert "safe clearance" in prompts[-1][1]
    assert window.backend.current_objective_id == "20x_air"
    assert window.objective_box.currentData() == "20x_air"


def test_recipe_keeps_exposures_already_set(window):
    window.channel_table._checks["dapi"].setChecked(True)
    window.channel_table._exposure["dapi"].setValue(123.0)
    choose_recipe(window, "Four-colour section")
    settings = {c.channel_id: c for c in window.channel_table.settings()}
    assert set(settings) == {"dapi", "af488", "cy3", "cy5"}
    assert settings["dapi"].exposure_ms == 123.0
    assert window.z_mode_box.currentData() == "focus_map"


def test_exposure_numeric_value_is_preserved_precisely(window):
    exposure = window.channel_table._exposure["dapi"]
    window.channel_table._checks["dapi"].setChecked(True)
    exposure.setValue(123.0)
    settings = {setting.channel_id: setting for setting in window.channel_table.settings()}
    assert settings["dapi"].exposure_ms == 123.0


def test_recipe_with_other_objective_is_refused_inside_experiment(qtbot, window):
    window.region_list.append(ScanRegion(-300, -200, 600, 400))
    window.channel_table._checks["dapi"].setChecked(True)
    window.start_button.click()
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    choose_recipe(window, "Tissue survey")  # 10x, but the experiment is 20x
    assert window.messages[-1][1] == "Objective is fixed"
    assert window.recipe_box.currentText() == "Custom"
    assert window.backend.current_objective_id == "20x_air"
    assert [c.channel_id for c in window.channel_table.settings()] == ["dapi"]


# -- ROI set files -----------------------------------------------------------
def test_roi_set_round_trip(window, tmp_path):
    regions = [ScanRegion(-500, -400, 1000, 800), ScanRegion.from_polygon(((0, 0), (900, 0), (450, 700)))]
    points = [FocusPoint(-100, -100, 1.25), FocusPoint(200, 300, -0.5)]
    window.region_list.set_items(regions)
    window.focus_list.set_items(points)
    path = tmp_path / "set.json"
    assert window.save_roi_set(path)
    window.region_list.set_items([])
    window.focus_list.set_items([])
    assert window.load_roi_set(path)
    assert window.region_list.items == regions
    assert window.focus_list.items == points


def test_roi_set_load_is_refused_after_cycle_1(qtbot, window, tmp_path):
    path = tmp_path / "other.json"
    window.region_list.set_items([ScanRegion(5000, 5000, 400, 400)])
    window.save_roi_set(path)
    window.region_list.set_items([ScanRegion(-300, -200, 600, 400)])
    window.channel_table._checks["dapi"].setChecked(True)
    window.start_button.click()
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    assert not window.load_set_button.isEnabled()
    assert not window.load_roi_set(path)
    assert window.region_list.items[0].x_um == -300


def test_bad_roi_file_warns(window, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"something": "else"}')
    assert not window.load_roi_set(path)
    assert window.messages[-1][1] == "Could not load ROI set"


# -- spiral preview ----------------------------------------------------------
def test_spiral_preview_continues_past_25_until_stopped_without_moving_z(qtbot, window):
    window.device_move(window.backend.move_z, 1.5)
    qtbot.waitUntil(lambda: window.stage_position.z_um == 1.5, timeout=5000)
    window.preview_button.click()
    qtbot.waitUntil(lambda: window.overview.preview_tile_count >= 30, timeout=15000)
    window.preview_button.click()
    assert not window.preview_running
    assert window.overview.preview_tile_count >= 30
    assert window.backend.position.z_um == 1.5
    z_moves = [value for kind, value, _ in window.backend.move_log if kind == "z"]
    assert z_moves == [1.5]


def test_spiral_preview_stops_before_stage_limit(qtbot, window):
    window.config_data = replace(
        window.config_data,
        stage_limits=StageLimits(-700, 700, -700, 700, -5000, 5000),
    )
    window.start_spiral_preview()
    qtbot.waitUntil(lambda: not window.preview_running, timeout=15000)
    assert window.overview.preview_tile_count == 9
    assert "stage limit" in window.overview_hint.text()


def test_new_spiral_retains_existing_preview_tiles(qtbot, window):
    window.preview_button.click()
    qtbot.waitUntil(lambda: window.overview.preview_tile_count >= 3, timeout=5000)
    window.preview_button.click()
    first_count = window.overview.preview_tile_count

    window.device_move(window.backend.move_xy, 5000, 5000)
    qtbot.waitUntil(lambda: window.stage_position.x_um == 5000, timeout=5000)
    window.preview_button.click()
    qtbot.waitUntil(lambda: window.overview.preview_tile_count >= first_count + 3, timeout=5000)
    window.preview_button.click()
    assert window.overview.preview_tile_count >= first_count + 3


def test_objective_change_retains_existing_preview_tiles(qtbot, window):
    overview_index = window.objective_box.findData("4x_air")
    window.objective_box.setCurrentIndex(overview_index)
    window._objective_selected(overview_index)
    qtbot.waitUntil(lambda: window.backend.current_objective_id == "4x_air", timeout=5000)

    window.preview_button.click()
    qtbot.waitUntil(lambda: window.overview.preview_tile_count >= 3, timeout=5000)
    window.preview_button.click()
    preview_count = window.overview.preview_tile_count

    imaging_index = window.objective_box.findData("20x_air")
    window.objective_box.setCurrentIndex(imaging_index)
    window._objective_selected(imaging_index)
    qtbot.waitUntil(lambda: window.backend.current_objective_id == "20x_air", timeout=5000)
    assert window.overview.preview_tile_count == preview_count


def test_spiral_preview_follows_regions(qtbot, window):
    window.region_list.append(ScanRegion(-400, -300, 800, 600))
    expected = len(window.controller.tiles_for(window.build_plan()))
    window.start_spiral_preview()
    qtbot.waitUntil(lambda: not window.preview_running, timeout=15000)
    assert window.overview.preview_tile_count == expected


def test_spiral_preview_stops_promptly(qtbot, window):
    window.backend.delay = 0.05
    window.preview_button.click()
    qtbot.waitUntil(lambda: window.overview.preview_tile_count >= 2, timeout=5000)
    window.preview_button.click()
    assert not window.preview_running
    assert window.device.stop_and_drain(5)
    stopped_at = window.overview.preview_tile_count
    qtbot.wait(300)
    assert window.overview.preview_tile_count == stopped_at <= 3
    assert "stopped after" in window.overview_hint.text()


def test_live_and_cycle_start_stop_the_preview(qtbot, window):
    window.backend.delay = 0.05
    window.preview_button.click()
    qtbot.waitUntil(lambda: window.overview.preview_tile_count >= 1, timeout=5000)
    window.live_button.setChecked(True)
    assert not window.preview_running and window.device.live
    window.live_button.setChecked(False)

    window.preview_button.click()
    window.region_list.append(ScanRegion(-300, -200, 600, 400))
    window.channel_table._checks["dapi"].setChecked(True)
    window.start_button.click()
    assert not window.preview_running
    qtbot.waitUntil(lambda: not window.poll_timer.isActive(), timeout=30000)
    assert window.controller.status().state == "complete"


# -- Z capture & configuration -----------------------------------------------
def test_z_stack_capture_waits_for_queued_moves(qtbot, window):
    window.z_mode_box.setCurrentIndex(window.z_mode_box.findData("stack"))
    window.device_move(window.backend.move_z, -2.0)
    window._capture_z_begin()
    window.device_move(window.backend.move_z, 3.0)
    window._capture_z_end()
    qtbot.waitUntil(lambda: window.z_end_captured, timeout=5000)
    assert (window.z_bottom.value(), window.z_top.value()) == (-2.0, 3.0)


def test_configuration_summary(window):
    html = window.configuration_html()
    for objective in window.config_data.objectives:
        assert objective.name in html
    limits = window.config_data.stage_limits
    assert f"{limits.x_min_um:g} … {limits.x_max_um:g} µm" in html
    assert "Simulated stage" in html and "MM preset" not in html
    assert str(window.config_path) in html
