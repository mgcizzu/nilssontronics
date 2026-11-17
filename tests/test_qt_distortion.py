"""Qt view: grid-slide calibration dialog and the distortion options."""

from __future__ import annotations

from dataclasses import replace
import os
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtCore import QSettings  # noqa: E402

from microscope_ui.backend import SimulatedMicroscope  # noqa: E402
from microscope_ui.config import load_config  # noqa: E402
from microscope_ui.distortion import DistortionCorrection  # noqa: E402
from microscope_ui.models import ChannelSetting, ScanRegion  # noqa: E402
from microscope_ui.qt.window import ScanWindow  # noqa: E402

SLIDE = {"x_um": 30000, "y_um": 0, "size_um": 8000, "pitch_um": 50, "line_width_um": 5}
OPTICS = {"k1": 0.02, "channel_shift_px": {"cy5": [1.0, 0.5]}}


@pytest.fixture
def window(qtbot, tmp_path):
    config = replace(load_config(), calibration_dir=tmp_path / "calibration")
    backend = SimulatedMicroscope(config, frame_px=128, distortion=OPTICS, calibration_slide=SLIDE)
    backend.change_objective(config.objective("20x_air"))
    backend.move_xy(30000, 0)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    with mock.patch(
        "microscope_ui.planner.shutil.disk_usage", return_value=mock.Mock(free=10 * 1024**4)
    ):
        win = ScanWindow(config, backend, use_ndv=False, settings=settings)
        qtbot.addWidget(win)
        win.confirm = lambda *_: True
        win.warn = lambda *_: None
        win.output_edit.setText(str(tmp_path / "out"))
        win.objective_box.setCurrentIndex(win.objective_box.findData("20x_air"))
        yield win
        win.close()


def snap_channel(qtbot, dialog, channel_id):
    dialog.channel_box.setCurrentIndex(dialog.channel_box.findData(channel_id))
    dialog.snap()
    qtbot.waitUntil(lambda: channel_id in dialog.images and not dialog.working, timeout=10000)
    rows = [dialog.image_list.item(i).text() for i in range(dialog.image_list.count())]
    name = dialog.config.channel(channel_id).name
    assert any(row.startswith(name) and "grid features" in row for row in rows), rows


def test_calibrate_with_grid_then_enable_correction(qtbot, window):
    assert "not calibrated" in window.distortion_label.text()
    dialog = window.open_distortion_calibration()
    assert dialog.current_label.text() == "Not calibrated yet."
    dialog.pitch.setValue(50)
    snap_channel(qtbot, dialog, "dapi")
    snap_channel(qtbot, dialog, "cy5")
    assert dialog.reference_box.count() == 2

    dialog.fit()
    qtbot.waitUntil(lambda: not dialog.working, timeout=20000)
    assert dialog.result is not None, dialog.report.toPlainText()
    assert "cy5" in dialog.report.toPlainText()
    assert dialog.result.correction.channels["cy5"].shift_x_px == pytest.approx(1.0, abs=0.15)
    dialog.save()
    saved = window.controller.correction_file("20x_air")
    assert DistortionCorrection.load(saved).fingerprint == dialog.result.correction.fingerprint
    assert "Saved calibration" in dialog.current_label.text()
    assert "calibrated" in window.distortion_label.text() and "Cy5" in window.distortion_label.text()
    assert window.settings.value("distortion/grid_pitch_um") in (50, 50.0, "50.0", "50")

    window.channel_table.set_settings((ChannelSetting("dapi", 40, 20), ChannelSetting("cy5", 40, 20)))
    window.region_list.set_items((ScanRegion(-300, -300, 600, 600),))
    window.distortion.setChecked(True)
    plan = window.build_plan()
    assert plan.options.distortion_correction and plan.options.keep_uncorrected
    assert window.controller.distortion_problems(plan) == []
    window.channel_table.set_settings((ChannelSetting("dapi", 40, 20), ChannelSetting("af488", 40, 20)))
    assert any("does not cover" in item for item in window.controller.distortion_problems(window.build_plan()))


def test_snap_refused_with_other_objective_in_light_path(qtbot, window):
    dialog = window.open_distortion_calibration()
    window.backend.change_objective(window.config_data.objective("10x_air"))
    dialog.snap()
    qtbot.waitUntil(lambda: not dialog.working, timeout=5000)
    assert "objective" in dialog.status_label.text()
    assert not dialog.images


def test_keep_uncorrected_follows_the_distortion_toggle(qtbot, window):
    window._refresh()
    assert window.distortion.isEnabled()
    assert not window.keep_uncorrected.isEnabled()
    window.distortion.setChecked(True)
    assert window.keep_uncorrected.isEnabled()


def test_import_adopts_a_saved_correction(qtbot, window, tmp_path):
    dialog = window.open_distortion_calibration()
    dialog.pitch.setValue(50)
    snap_channel(qtbot, dialog, "dapi")
    dialog.fit()
    qtbot.waitUntil(lambda: not dialog.working, timeout=20000)
    exported = dialog.result.correction.save(tmp_path / "old_session.json")
    target = window.controller.correction_file("20x_air")
    assert not target.exists()

    window.confirm = lambda *_: False
    assert not dialog.import_file(exported)
    assert not target.exists() and "cancelled" in dialog.status_label.text()

    window.confirm = lambda *_: True
    assert dialog.import_file(exported)
    assert DistortionCorrection.load(target) == DistortionCorrection.load(exported)
    assert "Imported" in dialog.status_label.text()
    assert "128×128" in dialog.report.toPlainText()  # frame-size check shown before adopting
    assert "Saved calibration" in dialog.current_label.text()

    (tmp_path / "bad.json").write_text("{}")
    assert not dialog.import_file(tmp_path / "bad.json")
    assert "Cannot import" in dialog.status_label.text()
