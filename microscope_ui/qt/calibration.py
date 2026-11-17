"""Grid-slide calibration of the optional distortion correction (Qt).

Non-modal, so the operator keeps Live, jog and focus in the main window while
snapping the grid channel by channel. Snaps run on the window's
:class:`DeviceWorker`; detection and fitting run on a short-lived background
thread. All maths lives in :mod:`microscope_ui.distortion`.
"""

from __future__ import annotations

import threading
from pathlib import Path

import tifffile
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from ..backend import HardwareError
from ..distortion import CalibrationError, CalibrationResult, GridImage, inspect_grid_image
from .widgets import spin

PITCH_SETTING = "distortion/grid_pitch_um"
INSTRUCTIONS = (
    "1. Put the grid slide on the stage and fill the field of view with the grid.<br>"
    "2. For each channel to correct: choose it here, focus it in Live (refocus per "
    "channel), then <b>Snap grid</b>. Do not move XY between channels.<br>"
    "3. <b>Fit</b>, check the report, then <b>Save</b>. New experiments with distortion "
    "correction on use the saved calibration."
)


class DistortionCalibrationDialog(QDialog):
    _finished = Signal(object, object, object)  # callback, result, error (GUI thread)

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window_ = window
        self.controller = window.controller
        self.config = window.config_data
        self.images: dict[str, GridImage] = {}
        self.result: CalibrationResult | None = None
        self.working = False
        self._finished.connect(self._deliver, Qt.ConnectionType.QueuedConnection)
        self.setWindowTitle("Distortion calibration")
        self.setModal(False)
        self.resize(560, 640)

        layout = QVBoxLayout(self)
        intro = QLabel(INSTRUCTIONS)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.objective_label = QLabel()
        self.current_label = QLabel()
        self.current_label.setWordWrap(True)
        self.current_label.setProperty("secondary", True)
        saved_pitch = float(window.settings.value(PITCH_SETTING, 10.0))
        self.pitch = spin(saved_pitch, 0.1, 10000, 2, "µm")
        self.pitch.setToolTip("Centre-to-centre spacing of the grid lines or dots on the slide")
        self.channel_box = QComboBox()
        for channel in self.config.channels:
            self.channel_box.addItem(channel.name, channel.id)
        self.exposure = spin(10, 0.01, 1e5, 2, "ms")
        self.power = spin(10, 0, 100, 1, "%")
        self.channel_box.currentIndexChanged.connect(self._channel_changed)
        form.addRow("Objective", self.objective_label)
        form.addRow("", self.current_label)
        form.addRow("Grid pitch", self.pitch)
        form.addRow("Channel", self.channel_box)
        form.addRow("Exposure", self.exposure)
        form.addRow("Power", self.power)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.snap_button = QPushButton("Snap grid")
        self.snap_button.setProperty("primary", True)
        self.load_button = QPushButton("Load TIFF…")
        self.load_button.setToolTip("Use a saved grid image for the selected channel (already oriented)")
        self.remove_button = QPushButton("Remove")
        self.snap_button.clicked.connect(self.snap)
        self.load_button.clicked.connect(self._load_file)
        self.remove_button.clicked.connect(self._remove_selected)
        for button in (self.snap_button, self.load_button, self.remove_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(130)
        layout.addWidget(self.image_list)

        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("Reference channel"))
        self.reference_box = QComboBox()
        self.reference_box.setToolTip("The other channels are registered onto this one")
        fit_row.addWidget(self.reference_box, 1)
        self.fit_button = QPushButton("Fit")
        self.save_button = QPushButton("Save")
        self.save_button.setProperty("primary", True)
        self.import_button = QPushButton("Import…")
        self.import_button.setToolTip(
            "Adopt a correction file made earlier, on another computer, or used by an experiment"
        )
        self.fit_button.clicked.connect(self.fit)
        self.save_button.clicked.connect(self.save)
        self.import_button.clicked.connect(self._import_clicked)
        fit_row.addWidget(self.fit_button)
        fit_row.addWidget(self.save_button)
        fit_row.addWidget(self.import_button)
        layout.addLayout(fit_row)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        layout.addWidget(self.report, 1)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._channel_changed()
        self.sync_objective()

    # -- state ------------------------------------------------------------------
    @property
    def objective_id(self) -> str:
        return self.window_.objective_box.currentData()

    def sync_objective(self) -> None:
        """Follow the main window's objective; images from another objective are dropped."""

        objective = self.config.objective(self.objective_id)
        if getattr(self, "_objective_shown", None) not in (None, objective.id):
            self.images.clear()
            self.result = None
            self.report.clear()
        self._objective_shown = objective.id
        self.objective_label.setText(objective.name)
        try:
            current = self.controller.load_correction(objective.id)
        except (OSError, ValueError) as exc:
            self.current_label.setText(f"Saved calibration unreadable: {exc}")
        else:
            if current is None:
                self.current_label.setText("Not calibrated yet.")
            else:
                channels = ", ".join(sorted(current.channels))
                self.current_label.setText(f"Saved calibration from {current.created_at[:16]} • {channels}")
        self._update()

    def _channel_changed(self, *_args) -> None:
        channel = self.config.channel(self.channel_box.currentData())
        setting = next(
            (item for item in self.window_.channel_table.settings() if item.channel_id == channel.id), None
        )
        self.exposure.setMaximum(channel.max_exposure_ms)
        self.power.setMaximum(channel.max_power_percent)
        self.exposure.setValue(setting.exposure_ms if setting else channel.default_exposure_ms)
        self.power.setValue(setting.power_percent if setting else channel.default_power_percent)

    def _update(self) -> None:
        busy = self.controller.busy or self.working
        self.snap_button.setEnabled(not busy)
        self.load_button.setEnabled(not busy)
        self.remove_button.setEnabled(not busy and bool(self.images))
        self.fit_button.setEnabled(not busy and bool(self.images))
        self.save_button.setEnabled(not busy and self.result is not None)
        self.import_button.setEnabled(not busy)
        reference = self.reference_box.currentData()
        self.reference_box.blockSignals(True)
        self.reference_box.clear()
        for channel_id in self.images:
            self.reference_box.addItem(self.config.channel(channel_id).name, channel_id)
        index = self.reference_box.findData(reference)
        self.reference_box.setCurrentIndex(max(0, index))
        self.reference_box.blockSignals(False)

    def _set_row(self, channel_id: str, text: str) -> None:
        name = self.config.channel(channel_id).name
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == channel_id:
                item.setText(f"{name}: {text}")
                return
        item = QListWidgetItem(f"{name}: {text}")
        item.setData(Qt.ItemDataRole.UserRole, channel_id)
        self.image_list.addItem(item)

    # -- background work -------------------------------------------------------------
    def _run(self, function, on_done, *args) -> None:
        self.working = True
        self._update()

        def work():
            try:
                result, error = function(*args), None
            except Exception as exc:  # reported in the dialog
                result, error = None, exc
            self._finished.emit(on_done, result, error)

        threading.Thread(target=work, name="distortion-calibration", daemon=True).start()

    def _deliver(self, callback, result, error) -> None:
        self.working = False
        callback(result, error)
        self._update()

    # -- images ------------------------------------------------------------------------
    def snap(self) -> None:
        if self.controller.busy:
            return
        channel = self.config.channel(self.channel_box.currentData())
        objective_id = self.objective_id
        exposure, power = self.exposure.value(), self.power.value()
        backend = self.window_.backend
        device = self.window_.device

        def job():  # device-worker thread
            if backend.current_objective_id != objective_id:
                raise HardwareError("Put the selected objective in the light path first")
            frame = device.acquire_with(channel, exposure, power)
            position = backend.position
            return frame, (position.x_um, position.y_um)

        self.working = True
        self._update()
        self.status_label.setText(f"Snapping {channel.name}…")
        device.submit(
            job,
            on_done=lambda result: self._snapped(channel.id, *result),
            on_error=self._snap_failed,
        )

    def _snap_failed(self, exc: Exception) -> None:
        self.working = False
        self.status_label.setText(f"Snap failed: {exc}")
        self._update()

    def _snapped(self, channel_id: str, frame, stage_xy) -> None:
        self.working = False
        self.window_.frame_view.show_frame(frame)
        self.add_image(GridImage(channel_id, frame, stage_xy))

    def _load_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Grid image", "", "TIFF (*.tif *.tiff)")
        if not filename:
            return
        try:
            image = tifffile.imread(filename)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not read {Path(filename).name}: {exc}")
            return
        if image.ndim != 2:
            self.status_label.setText(f"{Path(filename).name} is not a single 2-D plane")
            return
        self.add_image(GridImage(self.channel_box.currentData(), image, None))

    def add_image(self, item: GridImage) -> None:
        """Keep one grid image per channel and check it straight away."""

        self.images[item.channel_id] = item
        self.result = None
        self._set_row(item.channel_id, "checking…")
        fov_w, _fov_h = self.config.fov_um(self.objective_id)
        pitch_px = self.pitch.value() / (fov_w / item.image.shape[1])
        self.status_label.setText("")
        self._run(
            inspect_grid_image,
            lambda result, error, cid=item.channel_id: self._inspected(cid, result, error),
            item.image,
            pitch_px,
        )

    def _inspected(self, channel_id: str, result, error) -> None:
        if channel_id not in self.images:
            return
        if error is not None:
            self._set_row(channel_id, f"✗ {error}")
        else:
            count, spacing = result
            self._set_row(channel_id, f"{count} grid features, spacing {spacing:.1f} px")

    def _remove_selected(self) -> None:
        item = self.image_list.currentItem()
        if item is None:
            return
        self.images.pop(item.data(Qt.ItemDataRole.UserRole), None)
        self.image_list.takeItem(self.image_list.row(item))
        self.result = None
        self._update()

    # -- fit & save --------------------------------------------------------------------
    def fit(self) -> None:
        if not self.images or self.controller.busy:
            return
        self.window_.settings.setValue(PITCH_SETTING, self.pitch.value())
        self.report.setPlainText("Fitting…")
        self._run(
            self.controller.fit_distortion,
            self._fitted,
            list(self.images.values()),
            self.objective_id,
            self.pitch.value(),
            self.reference_box.currentData(),
        )

    def _fitted(self, result, error) -> None:
        if error is not None:
            self.result = None
            prefix = "" if isinstance(error, CalibrationError) else f"{type(error).__name__}: "
            self.report.setPlainText(f"Calibration failed: {prefix}{error}")
            return
        self.result = result
        self.report.setPlainText(result.report())

    def _import_clicked(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import distortion correction", "", "Distortion correction (*.json)"
        )
        if filename:
            self.import_file(filename)

    def import_file(self, filename: str | Path) -> bool:
        """Check a correction file, show it, and save it as its objective's calibration."""

        try:
            correction, warnings = self.controller.read_correction_for_import(filename)
        except (OSError, ValueError, KeyError) as exc:
            self.status_label.setText(f"Cannot import {Path(filename).name}: {exc}")
            return False
        objective = self.config.objective(correction.objective_id)
        summary = "\n".join(correction.summary_lines())
        if warnings:
            summary += "\n\nCheck:\n" + "\n".join(f"• {item}" for item in warnings)
        self.report.setPlainText(f"{Path(filename).name} (calibrated {correction.created_at[:16]})\n{summary}")
        replacing = self.controller.correction_file(objective.id).exists()
        question = (
            f"Use this file as the distortion calibration for {objective.name}?"
            + ("\nIt replaces the saved calibration." if replacing else "")
            + "\nExperiments already started keep the calibration they began with."
            + (f"\n\n{len(warnings)} point(s) to check are listed in the report." if warnings else "")
        )
        if not self.window_.confirm("Import calibration", question):
            self.status_label.setText("Import cancelled")
            return False
        try:
            saved = self.controller.save_correction(correction)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not save: {exc}")
            return False
        self.result = None
        elsewhere = correction.objective_id != self.objective_id
        self.status_label.setText(
            f"Imported as {saved}" + (f" (for {objective.name}, not the objective selected here)" if elsewhere else "")
        )
        self.sync_objective()
        self.window_._refresh()
        return True

    def save(self) -> None:
        if self.result is None:
            return
        correction = self.result.correction
        path = self.controller.correction_file(correction.objective_id)
        if path.exists() and not self.window_.confirm(
            "Replace calibration",
            f"Replace the saved calibration for {self.config.objective(correction.objective_id).name}?\n\n"
            "Experiments already started keep the calibration they began with.",
        ):
            return
        try:
            saved = self.controller.save_correction(correction)
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"Could not save: {exc}")
            return
        self.status_label.setText(f"Saved {saved}")
        self.sync_objective()
        self.window_._refresh()
