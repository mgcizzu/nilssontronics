from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest import mock

import pytest

os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QLabel  # noqa: E402

from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.mm_profile import MMConfigDiscovery  # noqa: E402
from microscope_ui.qt.profile_wizard import OBJECTIVE_COLUMNS, MicroManagerImportWizard  # noqa: E402


def _discovery(mm_config: Path) -> MMConfigDiscovery:
    return MMConfigDiscovery(
        config_path=mm_config,
        adapter_path=None,
        camera_device="Camera",
        camera_width_px=512,
        camera_height_px=256,
        xy_stage_device="XY",
        focus_device="Z",
        config_groups={"Channel": ("DAPI", "FITC")},
        state_labels={"Objective": ("10x", "20x")},
    )


def _inspect(qtbot, wizard, mm_config: Path) -> None:
    page = wizard.source_page
    page.config.setText(str(mm_config))
    assert not page.inspect_button.isEnabled()  # devices move: acknowledge first
    page.acknowledge.setChecked(True)
    page.inspect_button.click()
    qtbot.waitUntil(lambda: not page.busy, timeout=5000)
    assert page.isComplete()


def test_inspection_runs_off_the_gui_thread_and_blocks_cancel(qtbot, tmp_path):
    mm_config = tmp_path / "MMConfig.cfg"
    mm_config.write_text("# fake", encoding="utf-8")
    release = threading.Event()
    seen = {}

    def slow_inspector(*_args):
        seen["thread"] = threading.current_thread()
        release.wait(5)
        return _discovery(mm_config)

    wizard = MicroManagerImportWizard(inspector=slow_inspector)
    qtbot.addWidget(wizard)
    wizard.show()
    page = wizard.source_page
    page.config.setText(str(mm_config))
    page.acknowledge.setChecked(True)
    page.inspect_button.click()
    qtbot.waitUntil(lambda: "thread" in seen, timeout=5000)
    assert seen["thread"] is not threading.main_thread()
    assert page.busy and page.progress.isVisible() and not page.isComplete()
    wizard.reject()  # refused while devices are initializing
    assert wizard.isVisible()
    release.set()
    qtbot.waitUntil(lambda: not page.busy, timeout=5000)
    assert wizard.discovery is not None and page.validatePage()


def test_wizard_says_devices_are_initialized(qtbot):
    wizard = MicroManagerImportWizard(inspector=lambda *_a: None)
    qtbot.addWidget(wizard)
    texts = " ".join(label.text() for label in wizard.source_page.findChildren(QLabel))
    assert "only reads" not in texts
    assert "initialized" in texts and "home" in texts


def test_failed_inspection_keeps_the_page_incomplete(qtbot, tmp_path):
    def broken(*_args):
        raise RuntimeError("port busy")

    wizard = MicroManagerImportWizard(inspector=broken)
    qtbot.addWidget(wizard)
    page = wizard.source_page
    page.config.setText(str(tmp_path / "x.cfg"))
    page.acknowledge.setChecked(True)
    with mock.patch("microscope_ui.qt.profile_wizard.QMessageBox.warning") as warning:
        page.inspect_button.click()
        qtbot.waitUntil(lambda: not page.busy, timeout=5000)
    assert "port busy" in warning.call_args.args[2]
    assert not page.isComplete() and not page.validatePage()


def _set(page, row: int, key: str, text: str) -> None:
    column = [k for _t, k in OBJECTIVE_COLUMNS].index(key)
    page.objectives.item(row, column).setText(text)


def _fill_objective(page, row: int, **overrides) -> None:
    values = {
        "magnification": "10",
        "na": "0.3",
        "pixel": "0.65",
        "safe_min": "-4000",
        "safe_max": "3500",
        **overrides,
    }
    for key, text in values.items():
        _set(page, row, key, text)


def _safety_page(qtbot, tmp_path):
    mm_config = tmp_path / "MMConfig.cfg"
    mm_config.write_text("# fake", encoding="utf-8")
    wizard = MicroManagerImportWizard(inspector=lambda *_args: _discovery(mm_config))
    qtbot.addWidget(wizard)
    _inspect(qtbot, wizard, mm_config)
    wizard.mapping_page.initializePage()
    wizard.safety_page.initializePage()
    return wizard, wizard.safety_page


@pytest.mark.parametrize("key", ["magnification", "na", "pixel", "safe_min", "safe_max"])
def test_objective_safety_cells_start_empty_and_are_required(qtbot, tmp_path, key):
    _wizard, page = _safety_page(qtbot, tmp_path)
    column = [k for _t, k in OBJECTIVE_COLUMNS].index(key)
    assert page.objectives.item(0, column).text() == ""
    for row in range(2):
        _fill_objective(page, row)
    _set(page, 1, key, "")
    with pytest.raises(ValueError, match="20x: enter"):
        page.selected_objectives()
    page.confirmed.setChecked(True)
    with mock.patch("microscope_ui.qt.profile_wizard.QMessageBox.warning"):
        assert not page.validatePage()


def test_immersion_refractive_index_follows_the_medium(qtbot, tmp_path):
    _wizard, page = _safety_page(qtbot, tmp_path)
    for row in range(2):
        _fill_objective(page, row, na="0.8")
    expected = {"air": 1.0, "oil": 1.515, "water": 1.333, "silicone": 1.406}
    for medium, index in expected.items():
        _set(page, 0, "immersion", medium)
        assert page.selected_objectives()[0]["immersion_refractive_index"] == index
    _set(page, 0, "immersion", "other")
    with pytest.raises(ValueError, match="refractive index"):
        page.selected_objectives()
    _set(page, 0, "index", "1.45")
    assert page.selected_objectives()[0]["immersion_refractive_index"] == 1.45


def test_wizard_maps_discovery_and_writes_profile(qtbot, tmp_path):
    mm_config = tmp_path / "MMConfig.cfg"
    mm_config.write_text("# fake", encoding="utf-8")
    discovery = _discovery(mm_config)
    wizard = MicroManagerImportWizard(inspector=lambda *_args: discovery)
    qtbot.addWidget(wizard)
    _inspect(qtbot, wizard, mm_config)
    assert wizard.source_page.validatePage()

    wizard.mapping_page.initializePage()
    assert wizard.mapping_page.channel_group.currentText() == "Channel"
    assert wizard.mapping_page.channels.rowCount() == 2
    assert wizard.mapping_page.validatePage()

    wizard.safety_page.initializePage()
    assert wizard.safety_page.objectives.rowCount() == 2
    for row in range(2):
        _fill_objective(wizard.safety_page, row)
    wizard.safety_page.confirmed.setChecked(True)
    assert wizard.safety_page.validatePage()

    output = tmp_path / "profile.json"
    wizard.save_page.output.setText(str(output))
    assert wizard.save_page.validatePage()
    assert wizard.saved_path == output
    config = load_config(output)
    assert config.hardware["channel_presets"] == {"dapi": "DAPI", "fitc": "FITC"}
    assert [objective.turret_position for objective in config.objectives] == [0, 1]
