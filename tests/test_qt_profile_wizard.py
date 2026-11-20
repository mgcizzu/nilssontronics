from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.mm_profile import MMConfigDiscovery  # noqa: E402
from microscope_ui.qt.profile_wizard import MicroManagerImportWizard  # noqa: E402


def test_wizard_maps_discovery_and_writes_profile(qtbot, tmp_path):
    mm_config = tmp_path / "MMConfig.cfg"
    mm_config.write_text("# fake", encoding="utf-8")
    discovery = MMConfigDiscovery(
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
    wizard = MicroManagerImportWizard(inspector=lambda *_args: discovery)
    qtbot.addWidget(wizard)
    wizard.source_page.config.setText(str(mm_config))
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
