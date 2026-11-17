"""Fluent-style themes and toggle switches in the current tabbed Qt view."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QSettings  # noqa: E402
from qtpy.QtWidgets import QApplication, QCheckBox  # noqa: E402

from microscope_ui.backend import SimulatedMicroscope  # noqa: E402
from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.qt.theme import DARK, LIGHT, ToggleSwitch, current_palette, stylesheet  # noqa: E402
from microscope_ui.qt.window import ScanWindow  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path):
    config = load_config()
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    win = ScanWindow(config, SimulatedMicroscope(config, frame_px=16), use_ndv=False, settings=settings)
    qtbot.addWidget(win)
    yield win
    win.close()


def test_theme_switch_applies_and_is_remembered(window, tmp_path):
    window.theme_actions["dark"].trigger()
    assert current_palette().dark
    assert DARK.window in QApplication.instance().styleSheet()
    assert QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat).value("theme") == "dark"

    window.apply_theme_mode("light", remember=True)
    assert not current_palette().dark
    assert LIGHT.window in QApplication.instance().styleSheet()
    assert window.theme_mode() == "light"


def test_unknown_stored_theme_falls_back_to_system(window):
    window.settings.setValue("theme", "neon")
    assert window.theme_mode() == "system"


def test_theme_menu_preserves_the_tabbed_resizable_layout(window):
    assert window.main_splitter.count() == 3
    assert window.setup_tabs.count() == 3
    assert set(window.theme_actions) == {"system", "light", "dark"}


def test_options_are_toggle_switches_that_still_behave_as_checkboxes(window, qtbot):
    for toggle in (
        window.consistent_approach,
        window.shutter_per_frame,
        window.flat_field,
        window.distortion,
        window.keep_uncorrected,
    ):
        assert isinstance(toggle, ToggleSwitch) and isinstance(toggle, QCheckBox)
    assert window.shutter_per_frame.isChecked()  # safe default kept
    with qtbot.waitSignal(window.flat_field.toggled):
        window.flat_field.click()
    assert window.flat_field.isChecked()
    assert window.build_plan().options.flat_field


def test_stylesheet_has_no_unfilled_placeholders():
    css = stylesheet(DARK, {"chevron-down": "a.png", "chevron-up": "b.png", "check": "c.png"})
    assert "{" in css and "{p." not in css and "None" not in css
