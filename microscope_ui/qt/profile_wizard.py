"""Qt wizard for turning an existing Micro-Manager config into an app profile."""

from __future__ import annotations

from pathlib import Path
import threading

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from ..mm_profile import MMConfigDiscovery, build_profile, inspect_mm_config, safe_id, save_profile

CHANNEL_COLORS = ("#e7e5d8", "#6c8cff", "#58d68d", "#ffb347", "#ff5b78", "#c67cff")


def _number(value: float, low: float = -1e9, high: float = 1e9) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(3)
    box.setRange(low, high)
    box.setValue(value)
    return box


INITIALIZATION_WARNING = (
    "Inspecting loads the configuration exactly as Micro-Manager does: every device "
    "is initialized and the System/Startup presets are applied. Depending on the "
    "adapters the XY stage may home, the objective turret and filter wheels may "
    "move, and shutters may open. Clear the stage and objectives first. Vendor "
    "drivers and matching Micro-Manager device adapters must already be installed."
)


class SourcePage(QWizardPage):
    """Pick the .cfg and inspect it on a background thread (driver loading can take long)."""

    _inspected = Signal(object, object)  # discovery, error -> delivered on the GUI thread

    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Micro-Manager configuration")
        self.setSubTitle("Select a configuration made and tested in Micro-Manager.")
        form = QFormLayout(self)
        self.config = QLineEdit()
        config_browse = QPushButton("Browse…")
        config_browse.clicked.connect(self._browse_config)
        config_row = QHBoxLayout()
        config_row.addWidget(self.config, 1)
        config_row.addWidget(config_browse)
        self.adapters = QLineEdit()
        self.adapters.setPlaceholderText("Optional; use pymmcore-plus default")
        adapter_browse = QPushButton("Browse…")
        adapter_browse.clicked.connect(self._browse_adapters)
        adapter_row = QHBoxLayout()
        adapter_row.addWidget(self.adapters, 1)
        adapter_row.addWidget(adapter_browse)
        form.addRow("Micro-Manager .cfg", config_row)
        form.addRow("Device-adapter folder", adapter_row)
        warning = QLabel(f"<b>Warning:</b> {INITIALIZATION_WARNING}")
        warning.setWordWrap(True)
        form.addRow(warning)
        self.acknowledge = QCheckBox(
            "I understand that the devices will be initialized and may move"
        )
        form.addRow(self.acknowledge)
        self.inspect_button = QPushButton("Inspect configuration")
        self.inspect_button.clicked.connect(self.start_inspection)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # busy indicator
        self.progress.setVisible(False)
        inspect_row = QHBoxLayout()
        inspect_row.addWidget(self.inspect_button)
        inspect_row.addWidget(self.progress, 1)
        form.addRow(inspect_row)
        self.status = QLabel("Not inspected yet.")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self.thread: threading.Thread | None = None
        self._inspected.connect(self._inspection_finished, Qt.ConnectionType.QueuedConnection)
        self.config.textChanged.connect(self._source_changed)
        self.adapters.textChanged.connect(self._source_changed)
        self.acknowledge.toggled.connect(self._update_buttons)
        self._update_buttons()

    @property
    def busy(self) -> bool:
        return self.thread is not None

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Micro-Manager configuration", "", "Micro-Manager (*.cfg);;All files (*)")
        if path:
            self.config.setText(path)

    def _browse_adapters(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Micro-Manager folder")
        if path:
            self.adapters.setText(path)

    def _source_changed(self) -> None:
        if self.busy:
            return
        wizard = self.wizard()
        if wizard is not None and wizard.discovery is not None:
            wizard.discovery = None
            self.status.setText("Not inspected yet.")
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.inspect_button.setEnabled(
            not self.busy and self.acknowledge.isChecked() and bool(self.config.text().strip())
        )
        self.config.setEnabled(not self.busy)
        self.adapters.setEnabled(not self.busy)
        self.acknowledge.setEnabled(not self.busy)
        wizard = self.wizard()
        if wizard is not None:
            cancel = wizard.button(QWizard.WizardButton.CancelButton)
            if cancel is not None:
                cancel.setEnabled(not self.busy)
        self.completeChanged.emit()

    def start_inspection(self) -> None:
        """Load the configuration on a worker thread; the window stays responsive."""

        if self.busy or not self.acknowledge.isChecked():
            return
        wizard = self.wizard()
        wizard.discovery = None
        inspector = wizard.inspector
        config, adapters = self.config.text(), self.adapters.text() or None

        def run() -> None:
            try:
                result, error = inspector(config, adapters), None
            except Exception as exc:  # reported in the GUI, never raised on the worker
                result, error = None, exc
            self._inspected.emit(result, error)

        self.thread = threading.Thread(target=run, name="mm-profile-inspect", daemon=True)
        self.progress.setVisible(True)
        self.status.setText("Initializing devices…")
        self._update_buttons()
        self.thread.start()

    def _inspection_finished(self, discovery, error) -> None:
        self.thread = None
        self.progress.setVisible(False)
        if error is not None:
            self.status.setText("Inspection failed.")
            QMessageBox.warning(self, "Could not inspect configuration", str(error))
        else:
            self.wizard().discovery = discovery
            self.status.setText(f"Inspected {discovery.config_path.name}.")
        self._update_buttons()

    def isComplete(self) -> bool:  # noqa: N802 - Qt API
        wizard = self.wizard()
        return not self.busy and wizard is not None and wizard.discovery is not None

    def validatePage(self) -> bool:  # noqa: N802 - Qt API
        return self.isComplete()


class MappingPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Detected hardware and channels")
        self.setSubTitle("Choose the preset group used for imaging and confirm its channels.")
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.channel_group = QComboBox()
        self.objective_device = QComboBox()
        self.detected = QLabel()
        self.detected.setWordWrap(True)
        form.addRow("Microscope name", self.name)
        form.addRow("Imaging channel group", self.channel_group)
        form.addRow("Objective turret", self.objective_device)
        form.addRow("Detected", self.detected)
        root.addLayout(form)
        self.channels = QTableWidget(0, 5)
        self.channels.setHorizontalHeaderLabels(
            ["Use", "Preset / channel name", "Emission (nm)", "Exposure (ms)", "Maximum (ms)"]
        )
        self.channels.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.channels)
        self.channel_group.currentTextChanged.connect(self._populate_channels)

    def initializePage(self) -> None:
        d: MMConfigDiscovery = self.wizard().discovery
        self.name.setText(d.config_path.stem)
        self.detected.setText(
            f"Camera: {d.camera_device or '—'} ({d.camera_width_px} × {d.camera_height_px})  •  "
            f"XY: {d.xy_stage_device or '—'}  •  Z: {d.focus_device or '—'}"
        )
        self.channel_group.clear()
        self.channel_group.addItems(sorted(d.config_groups))
        preferred = self.channel_group.findText("Channel")
        if preferred >= 0:
            self.channel_group.setCurrentIndex(preferred)
        self.objective_device.clear()
        self.objective_device.addItems(sorted(d.state_labels))
        preferred = self.objective_device.findText("Objective")
        if preferred >= 0:
            self.objective_device.setCurrentIndex(preferred)
        self._populate_channels(self.channel_group.currentText())

    def _populate_channels(self, group: str) -> None:
        presets = self.wizard().discovery.config_groups.get(group, ())
        self.channels.setRowCount(len(presets))
        for row, preset in enumerate(presets):
            use = QCheckBox()
            use.setChecked(True)
            use.setAccessibleName(f"Use {preset}")
            self.channels.setCellWidget(row, 0, use)
            self.channels.setItem(row, 1, QTableWidgetItem(preset))
            self.channels.setItem(row, 2, QTableWidgetItem("550"))
            self.channels.setItem(row, 3, QTableWidgetItem("50"))
            self.channels.setItem(row, 4, QTableWidgetItem("2000"))
        self.channels.resizeColumnsToContents()

    def validatePage(self) -> bool:
        if not self.name.text().strip() or not self.channel_group.currentText():
            QMessageBox.warning(self, "Incomplete mapping", "Enter a microscope name and select a channel group.")
            return False
        try:
            selected = self.selected_channels()
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid channel", str(exc))
            return False
        if not selected:
            QMessageBox.warning(self, "No channels", "Select at least one imaging preset.")
            return False
        return True

    def selected_channels(self) -> list[dict]:
        result = []
        used_ids: set[str] = set()
        for row in range(self.channels.rowCount()):
            use = self.channels.cellWidget(row, 0)
            if not use.isChecked():
                continue
            name = self.channels.item(row, 1).text().strip()
            channel_id = safe_id(name, f"channel_{row + 1}")
            while channel_id in used_ids:
                channel_id += "_2"
            used_ids.add(channel_id)
            result.append(
                {
                    "id": channel_id,
                    "name": name,
                    "preset": name,
                    "color": CHANNEL_COLORS[row % len(CHANNEL_COLORS)],
                    "emission_wavelength_nm": float(self.channels.item(row, 2).text()),
                    "default_exposure_ms": float(self.channels.item(row, 3).text()),
                    "max_exposure_ms": float(self.channels.item(row, 4).text()),
                    "default_power_percent": 0,
                    "max_power_percent": 0,
                    "manual_prompt": None,
                }
            )
        return result


class SafetyPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Travel limits and objectives")
        self.setSubTitle("These values control physical motion and must be verified on the microscope.")
        root = QVBoxLayout(self)
        limits = QGridLayout()
        self.limit_boxes = {}
        defaults = {"x_min_um": -50000, "x_max_um": 50000, "y_min_um": -30000, "y_max_um": 30000, "z_min_um": -5000, "z_max_um": 5000}
        for row, axis in enumerate("xyz"):
            limits.addWidget(QLabel(axis.upper()), row, 0)
            for column, bound in enumerate(("min", "max"), 1):
                key = f"{axis}_{bound}_um"
                box = _number(defaults[key])
                box.setSuffix(" µm")
                self.limit_boxes[key] = box
                limits.addWidget(box, row, column)
        self.clearance = _number(-4000)
        self.clearance.setSuffix(" µm")
        limits.addWidget(QLabel("Objective-change clearance"), 3, 0)
        limits.addWidget(self.clearance, 3, 1, 1, 2)
        root.addLayout(limits)
        root.addWidget(QLabel("Objectives (edit every optical and safe-Z value):"))
        self.objectives = QTableWidget(0, 8)
        self.objectives.setHorizontalHeaderLabels(
            ["Name", "Mag.", "NA", "Immersion", "Turret state", "µm/px", "Safe Z min", "Safe Z max"]
        )
        self.objectives.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.objectives)
        self.confirmed = QCheckBox("I verified the travel limits, clearance position, turret mapping, and safe Z ranges")
        root.addWidget(self.confirmed)
        self._loaded_device: str | None = None

    def initializePage(self) -> None:
        mapping: MappingPage = self.wizard().mapping_page
        device = mapping.objective_device.currentText()
        if self._loaded_device == device and self.objectives.rowCount():
            return
        self._loaded_device = device
        labels = self.wizard().discovery.state_labels.get(device, ())
        if not labels:
            labels = ("Objective 1",)
        self.objectives.setRowCount(len(labels))
        for row, label in enumerate(labels):
            values = (label, "10", "0.3", "air", str(row), "1.0", "-4000", "3500")
            for column, value in enumerate(values):
                self.objectives.setItem(row, column, QTableWidgetItem(value))
        self.objectives.resizeColumnsToContents()

    def validatePage(self) -> bool:
        limits = self.stage_limits()
        for axis in "xyz":
            if limits[f"{axis}_min_um"] >= limits[f"{axis}_max_um"]:
                QMessageBox.warning(self, "Invalid limits", f"{axis.upper()} minimum must be below its maximum.")
                return False
        if not limits["z_min_um"] <= self.clearance.value() <= limits["z_max_um"]:
            QMessageBox.warning(self, "Invalid clearance", "Objective-change clearance must lie inside the Z travel limits.")
            return False
        if not self.confirmed.isChecked():
            QMessageBox.warning(self, "Safety confirmation required", "Verify and confirm every motion-safety value before continuing.")
            return False
        try:
            self.selected_objectives()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid objective", str(exc))
            return False
        return True

    def stage_limits(self) -> dict[str, float]:
        return {key: box.value() for key, box in self.limit_boxes.items()}

    def selected_objectives(self) -> list[dict]:
        result = []
        used: set[str] = set()
        for row in range(self.objectives.rowCount()):
            values = [self.objectives.item(row, column).text().strip() for column in range(8)]
            name, magnification, na, immersion, turret, pixel, safe_min, safe_max = values
            objective_id = safe_id(name, f"objective_{row + 1}")
            if objective_id in used:
                raise ValueError(f"Objective name '{name}' is duplicated")
            used.add(objective_id)
            immersion = immersion.lower()
            if immersion not in {"air", "oil", "water", "silicone", "other"}:
                raise ValueError(f"{name}: immersion must be air, oil, water, silicone, or other")
            result.append(
                {
                    "id": objective_id,
                    "name": name,
                    "magnification": float(magnification),
                    "numerical_aperture": float(na),
                    "immersion": immersion,
                    "immersion_refractive_index": 1.0 if immersion == "air" else 1.515,
                    "turret_position": int(turret),
                    "pixel_size_um": float(pixel),
                    "safe_z_min_um": float(safe_min),
                    "safe_z_max_um": float(safe_max),
                }
            )
        return result


class SavePage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Save microscope profile")
        self.setSubTitle("The generated JSON remains separate from the Micro-Manager configuration.")
        form = QFormLayout(self)
        self.output = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.output, 1)
        row.addWidget(browse)
        form.addRow("Application profile", row)
        note = QLabel("After saving, restart the app with --config followed by this profile path.")
        note.setWordWrap(True)
        form.addRow(note)
        self.activate = QCheckBox("Use this microscope profile on the next launch")
        self.activate.setChecked(True)
        form.addRow(self.activate)

    def initializePage(self) -> None:
        if not self.output.text():
            self.output.setText(str(self.wizard().discovery.config_path.with_name("nilssontronics_profile.json")))

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save microscope profile", self.output.text(), "JSON (*.json)")
        if path:
            self.output.setText(path)

    def validatePage(self) -> bool:
        try:
            mapping = self.wizard().mapping_page
            safety = self.wizard().safety_page
            output = Path(self.output.text())
            profile = build_profile(
                self.wizard().discovery,
                microscope_name=mapping.name.text(),
                channel_group=mapping.channel_group.currentText(),
                channels=mapping.selected_channels(),
                objective_device=mapping.objective_device.currentText(),
                objectives=safety.selected_objectives(),
                stage_limits=safety.stage_limits(),
                objective_change_z_um=safety.clearance.value(),
                output_path=output,
            )
            self.wizard().saved_path = save_profile(profile, output)
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Could not save profile", str(exc))
            return False


class MicroManagerImportWizard(QWizard):
    def __init__(self, parent=None, *, inspector=inspect_mm_config) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Micro-Manager configuration")
        self.resize(920, 650)
        self.inspector = inspector
        self.discovery: MMConfigDiscovery | None = None
        self.saved_path: Path | None = None
        self.source_page = SourcePage()
        self.mapping_page = MappingPage()
        self.safety_page = SafetyPage()
        self.save_page = SavePage()
        for page in (self.source_page, self.mapping_page, self.safety_page, self.save_page):
            self.addPage(page)

    def reject(self) -> None:
        # Devices are being initialized on a worker thread; let it finish and unload.
        if self.source_page.busy:
            return
        super().reject()
