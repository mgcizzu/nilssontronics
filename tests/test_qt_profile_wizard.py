from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QLabel  # noqa: E402

from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.mm_profile import MMConfigDiscovery  # noqa: E402
from microscope_ui.qt.profile_wizard import MicroManagerImportWizard  # noqa: E402


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
    wizard.safety_page.confirmed.setChecked(True)
    assert wizard.safety_page.validatePage()

    output = tmp_path / "profile.json"
    wizard.save_page.output.setText(str(output))
    assert wizard.save_page.validatePage()
    assert wizard.saved_path == output
    config = load_config(output)
    assert config.hardware["channel_presets"] == {"dapi": "DAPI", "fitc": "FITC"}
    assert [objective.turret_position for objective in config.objectives] == [0, 1]
