"""Qt main window for the scan console (first slice of the Qt port).

Covers the cycle workflow end to end: experiment new/open/resume, plan entry
(objective, channels, numeric regions, focus points, Z mode, options), live and
snap via ndv, preflight/estimate, start/pause/abort, progress and the cycle
mosaic. Still missing compared with the Tk view: drawing regions/polygons on
an overview canvas, drag "Move ROI set", spiral preview, recipes, flat-field
UI and the admin dialog.

Like the Tk view it only calls :class:`ScanController` for acquisition. Setup
calls (jog, objective change, go-to, live/snap) run on a :class:`DeviceWorker`
thread, never on the GUI thread, and only while no cycle is running; the
worker is drained before the engine takes over the backend. pymmcore-widgets'
StageWidget is deliberately not used: it would drive the core directly and
skip the configured stage-limit checks in the backend.
"""

from __future__ import annotations

import argparse
import json
import time
from html import escape
from dataclasses import replace
from pathlib import Path

import numpy as np
from qtpy.QtCore import QSettings, QSignalBlocker, Qt, QTimer
from qtpy.QtGui import QActionGroup
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..backend import HardwareError
from ..backend_mmcore import create_backend
from ..config import DEFAULT_CONFIG_PATH, load_config
from ..controller import ControllerError, ScanController
from ..experiment import Experiment
from ..focus import FocusSurface
from ..models import (
    AcquisitionOptions,
    AcquisitionPlan,
    ChannelSetting,
    FocusPoint,
    MicroscopeConfig,
    ZSettings,
)
from ..mosaic import DiskBackedMosaic
from ..planner import continuous_spiral_preview_positions, spiral_preview_positions
from ..roi_io import load_roi_set, save_roi_set
from .device_worker import DeviceWorker
from .overview import OverviewCanvas, bounded_set_translation
from .theme import THEME_MODES, ToggleSwitch, apply_theme, set_dark_title_bar
from .widgets import (
    ChannelTable,
    FrameView,
    ItemList,
    RectangleDialog,
    StageJog,
    ZWheelControl,
    describe_focus,
    describe_region,
    spin,
)

POSITION_REFRESH_MS = 1000
CLOSE_TIMEOUT_S = 30.0
DEVICE_DRAIN_TIMEOUT_S = 30.0
POLL_INTERVAL_MS = 150
MOSAIC_REFRESH_S = 0.3
Z_MODES = (
    ("Single plane (current Z)", "single"),
    ("Focus map", "focus_map"),
    ("Z-stack (absolute)", "stack"),
    ("Z-stack around surface", "surface_stack"),
)
LAUNCH_ERRORS = (ControllerError, HardwareError, OSError, ValueError)
MOSAIC_UNDERLAY_PX = 1024
PREVIEW_TILE_PX = 256
DEFAULT_PANEL_SIZES = (340, 780, 260)
DEFAULT_WINDOW_SIZE = (1400, 860)
OVERVIEW_MODES = (
    ("navigate", "Pan", "Drag to pan, wheel to zoom, double-click to move the stage"),
    ("rectangle", "Rect", "Drag to draw a rectangular scan region"),
    ("polygon", "Poly", "Click corners, double-click or Enter to close"),
    ("move_set", "Move", "Drag to translate all regions and focus points (XY only)"),
    ("focus", "Focus", "Click a point to go to it, elsewhere to move there"),
)
OVERVIEW_HINTS = {
    "navigate": "Drag to pan • wheel to zoom • double-click moves the stage",
    "rectangle": "Drag to draw a region • Esc cancels",
    "polygon": "Click corners • double-click or Enter closes • Esc cancels",
    "move_set": "Drag to move regions and focus points together • Esc restores",
    "focus": "Click F-point to go there, elsewhere to move XY • focus in Live, then Capture / Set Z",
}


class ScanWindow(QMainWindow):
    def __init__(
        self,
        config: MicroscopeConfig,
        backend=None,
        use_ndv: bool = True,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings if settings is not None else QSettings(
            "nilssontronics", "scan-console"
        )
        self.config_data = config
        self.config_path = Path(config_path)
        self.backend = backend if backend is not None else create_backend(config)
        self.controller = ScanController(config, self.backend)
        self.device = DeviceWorker(self.backend, self)
        self.z_begin_captured = False
        self.z_end_captured = False
        self._surface_stack_start_um: float | None = None
        self._surface_stack_end_um: float | None = None
        self._setting_surface_span = False
        self._last_mosaic_render = 0.0
        self._loading = False

        self.setWindowTitle(f"Microscope Scan Console — {self.backend.name}")
        self.fit_window_to_screen()
        self._build_ui(use_ndv)
        for signal in (
            self.z_mode_box.currentIndexChanged,
            self.z_bottom.valueChanged,
            self.z_top.valueChanged,
            self.z_below.valueChanged,
            self.z_above.valueChanged,
            self.z_step.valueChanged,
            self.focus_list.changed,
            self.focus_method_box.currentIndexChanged,
        ):
            signal.connect(self._update_z_stack_view)
        microscope_menu = self.menuBar().addMenu("Microscope")
        microscope_menu.addAction("Configuration…", self.show_configuration)
        self._build_device_docks()
        self.apply_theme_mode(self.theme_mode())
        hints = QApplication.instance().styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(self._system_scheme_changed)

        self.device.frame_ready.connect(self._show_live_frame)
        self.device.position_changed.connect(self._position_changed)
        self.device.objective_changed.connect(self._physical_objective_changed)
        self.device.live_failed.connect(self._live_failed)
        self.channel_table.changed.connect(self._push_live_settings)
        self.live_channel_box.currentIndexChanged.connect(self._push_live_settings)
        self._push_live_settings()

        # Picks up joystick/manual stage moves while nothing else polls the stage.
        self.position_timer = QTimer(self)
        self.position_timer.setInterval(POSITION_REFRESH_MS)
        self.position_timer.timeout.connect(self._refresh_position)
        self.position_timer.start()
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(POLL_INTERVAL_MS)
        self.poll_timer.timeout.connect(self._poll_scan)

        self._sync_objective_box()
        self._z_mode_changed()
        self._refresh()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self, use_ndv: bool) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.addWidget(self._build_header())
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(9)
        self.main_splitter.setToolTip("Drag the vertical dividers to resize the panels")
        self.main_splitter.addWidget(self._build_setup_panel())
        self.main_splitter.addWidget(self._build_viewer(use_ndv))
        self.main_splitter.addWidget(self._build_review_panel())
        for index in range(self.main_splitter.count()):
            self.main_splitter.setCollapsible(index, False)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.reset_panel_sizes()
        root.addWidget(self.main_splitter, 1)
        self.setCentralWidget(central)

        view_menu = self.menuBar().addMenu("View")
        self.view_menu = view_menu
        view_menu.addAction("Reset panel sizes", self.reset_panel_sizes)
        view_menu.addAction("Fit window to screen", self.fit_window_to_screen)
        theme_menu = view_menu.addMenu("Theme")
        self.theme_group = QActionGroup(self)
        self.theme_group.setExclusive(True)
        self.theme_actions = {}
        labels = {"system": "Use system setting", "light": "Light", "dark": "Dark"}
        for mode in THEME_MODES:
            action = theme_menu.addAction(labels[mode])
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, selected=mode: self.apply_theme_mode(
                    selected, remember=True
                )
            )
            self.theme_group.addAction(action)
            self.theme_actions[mode] = action

    # ---------------------------------------------------------------- theme --
    def theme_mode(self) -> str:
        mode = str(self.settings.value("theme", "dark"))
        return mode if mode in THEME_MODES else "system"

    def _system_scheme_changed(self, *_args) -> None:
        if self.theme_mode() == "system":
            self.apply_theme_mode("system")

    def apply_theme_mode(self, mode: str, remember: bool = False) -> None:
        palette = apply_theme(QApplication.instance(), mode)
        self.palette_tokens = palette
        if remember:
            self.settings.setValue("theme", mode)
            self.settings.sync()
        if hasattr(self, "theme_actions"):
            self.theme_actions[mode].setChecked(True)
        if hasattr(self, "dark_mode_toggle"):
            blocker = QSignalBlocker(self.dark_mode_toggle)
            self.dark_mode_toggle.setChecked(palette.dark)
            del blocker
        self.channel_table.refresh_swatches()
        set_dark_title_bar(self, palette.dark)

    def _toggle_dark_mode(self, checked: bool) -> None:
        self.apply_theme_mode("dark" if checked else "light", remember=True)

    def fit_window_to_screen(self) -> None:
        """Fit the window inside the current screen's usable desktop area."""

        app = QApplication.instance()
        screen = self.screen() or (app.primaryScreen() if app is not None else None)
        if screen is None:
            self.resize(*DEFAULT_WINDOW_SIZE)
            return
        available = screen.availableGeometry()
        width = min(DEFAULT_WINDOW_SIZE[0], max(640, int(available.width() * 0.94)))
        height = min(DEFAULT_WINDOW_SIZE[1], max(560, int(available.height() * 0.90)))
        width = min(width, available.width())
        height = min(height, available.height())
        self.resize(width, height)
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def reset_panel_sizes(self) -> None:
        """Restore a balanced three-panel layout after manual resizing."""

        self.main_splitter.setSizes(list(DEFAULT_PANEL_SIZES))

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("appHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(6, 2, 6, 4)
        row.setSpacing(10)
        title = QLabel(f"<b>{self.config_data.microscope_name}</b>")
        mode = "SIMULATOR CONNECTED" if self.backend.simulated else f"{self.backend.name.upper()} CONNECTED"
        self.mode_label = QLabel(mode)
        self.mode_label.setObjectName("statusPill")
        self.mode_label.setProperty("tone", "sim" if self.backend.simulated else "hw")
        self.position_label = QLabel()
        self.position_label.setProperty("secondary", True)
        self.experiment_label = QLabel()
        self.experiment_label.setProperty("secondary", True)
        self.experiment_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.experiment_label.setMinimumWidth(20)
        self.experiment_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.dark_mode_toggle = ToggleSwitch("Dark")
        self.dark_mode_toggle.setToolTip(
            "Switch between light and dark mode; use View → Theme to follow the system"
        )
        self.dark_mode_toggle.toggled.connect(self._toggle_dark_mode)
        row.addWidget(title)
        row.addWidget(self.mode_label)
        row.addSpacing(4)
        row.addWidget(self.dark_mode_toggle)
        row.addStretch(1)
        row.addWidget(self.experiment_label)
        row.addSpacing(16)
        row.addWidget(self.position_label)
        return header

    def _build_setup_panel(self) -> QWidget:
        project_page = QWidget()
        project_layout = QVBoxLayout(project_page)
        regions_page = QWidget()
        regions_page_layout = QVBoxLayout(regions_page)
        acquisition_page = QWidget()
        acquisition_layout = QVBoxLayout(acquisition_page)

        # Experiment
        experiment = QGroupBox("Experiment")
        exp_layout = QHBoxLayout(experiment)
        self.new_button = QPushButton("New")
        self.open_button = QPushButton("Open…")
        self.resume_button = QPushButton("Resume cycle")
        self.new_button.clicked.connect(self._new_experiment)
        self.open_button.clicked.connect(self._open_experiment)
        self.resume_button.clicked.connect(lambda: self._resume_cycle())
        for button in (self.new_button, self.open_button, self.resume_button):
            exp_layout.addWidget(button)
        project_layout.addWidget(experiment)

        # Sample
        sample = QGroupBox("Sample")
        form = QFormLayout(sample)
        self.project_edit = QLineEdit("Demo project")
        self.sample_edit = QLineEdit("Sample 001")
        self.output_edit = QLineEdit(str(Path.cwd()))
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._choose_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse)
        form.addRow("Project", self.project_edit)
        form.addRow("Sample", self.sample_edit)
        form.addRow("Output", out_row)
        project_layout.addWidget(sample)

        # Optics
        optics = QGroupBox("Objective && channels")
        optics_layout = QVBoxLayout(optics)
        self.recipe_box = QComboBox()
        self.recipe_box.addItem("Custom")
        for recipe in self.config_data.recipes:
            self.recipe_box.addItem(recipe.name)
        self.recipe_box.setToolTip("Recipe: objective, channels and Z mode presets from the configuration")
        self.recipe_box.activated.connect(self._apply_recipe)
        self._applying_recipe = False
        recipe_row = QHBoxLayout()
        recipe_row.addWidget(QLabel("Recipe"))
        recipe_row.addWidget(self.recipe_box, 1)
        optics_layout.addLayout(recipe_row)
        self.objective_box = QComboBox()
        for objective in self.config_data.objectives:
            self.objective_box.addItem(objective.name, objective.id)
        self.objective_box.activated.connect(self._objective_selected)
        adjustable_power_ids = (
            None
            if self.backend.simulated
            else {
                channel.id
                for channel in self.config_data.channels
                if self.backend.power_is_adjustable(channel.id)
            }
        )
        self.channel_table = ChannelTable(
            self.config_data.channels, adjustable_power_ids=adjustable_power_ids
        )
        self.channel_table.changed.connect(self._recipe_to_custom)
        optics_layout.addWidget(self.objective_box)
        optics_layout.addWidget(self.channel_table)
        project_layout.addWidget(optics)

        # Regions & focus
        regions = QGroupBox("Scan regions (stage µm)")
        regions_layout = QVBoxLayout(regions)
        self.region_list = ItemList(describe_region)
        self.add_region_button = self.region_list.add_button("Add…", self._add_region)
        self.save_set_button = QPushButton("Save…")
        self.save_set_button.setToolTip("Save regions and focus points to a JSON file")
        self.save_set_button.clicked.connect(self._save_roi_set)
        self.load_set_button = QPushButton("Load…")
        self.load_set_button.setToolTip("Replace regions and focus points from a saved set")
        self.load_set_button.clicked.connect(self._load_roi_set)
        self.shift_button = QPushButton("Shift set…")
        self.shift_button.setToolTip("Re-register for this cycle: XY translation of the whole set")
        self.shift_button.clicked.connect(self._shift_roi_set)
        set_row = QHBoxLayout()
        set_row.addWidget(self.save_set_button)
        set_row.addWidget(self.load_set_button)
        set_row.addWidget(self.shift_button)
        regions_layout.addWidget(self.region_list)
        regions_layout.addLayout(set_row)
        regions_page_layout.addWidget(regions)

        focus = QGroupBox("Focus points")
        focus_layout = QVBoxLayout(focus)
        self.focus_list = ItemList(describe_focus)
        capture_focus_button = self.focus_list.add_button("Capture", self._capture_focus_here)
        capture_focus_button.setToolTip("Capture a focus point at the current XYZ position")
        self.focus_list.add_button("Go to", self._goto_focus)
        self.set_z_button = self.focus_list.add_button("Set Z", self._set_focus_z_here)
        self.set_z_button.setToolTip("Set the selected focus point to the current Z")
        self.focus_list.remove_button.clicked.connect(lambda: self.stop_refocus("Refocus stopped"))
        self.refocus_button = QPushButton("Refocus all")
        self.refocus_button.setToolTip(
            "Visit every focus point in order: focus in Live, then Accept Z to store it and move on"
        )
        self.refocus_button.clicked.connect(self._toggle_refocus)
        self.align_button = QPushButton("Align F → stage")
        self.align_button.setToolTip(
            "Centre the selected point's landmark in Live, then press: the whole ROI set "
            "moves so that point sits at the current stage XY (Z is kept)"
        )
        self.align_button.clicked.connect(self._align_selected_focus_to_stage)
        cycle_row = QHBoxLayout()
        cycle_row.addWidget(self.refocus_button)
        cycle_row.addWidget(self.align_button)
        self.refocus_label = QLabel()
        self.refocus_label.setWordWrap(True)
        self.refocus_label.setStyleSheet("color: #b5442c; font-weight: 600;")
        self.refocus_label.hide()
        focus_layout.addWidget(self.focus_list)
        focus_layout.addLayout(cycle_row)
        focus_layout.addWidget(self.refocus_label)
        regions_page_layout.addWidget(focus)
        self.refocus_index: int | None = None
        self._accept_pending = False

        # Z
        z_group = QGroupBox("Z")
        z_form = QFormLayout(z_group)
        self.z_form = z_form
        self.z_mode_box = QComboBox()
        for label, value in Z_MODES:
            self.z_mode_box.addItem(label, value)
        self.z_mode_box.currentIndexChanged.connect(self._z_mode_changed)
        self.z_bottom = spin(0, -1e5, 1e5, 2, "µm")
        self.z_top = spin(0, -1e5, 1e5, 2, "µm")
        self.z_below = spin(2, 0, 1e4, 2, "µm")
        self.z_above = spin(2, 0, 1e4, 2, "µm")
        self.z_below.setToolTip("Distance below the interpolated focus surface at every tile")
        self.z_above.setToolTip("Distance above the interpolated focus surface at every tile")
        self.z_step = spin(1, 0.01, 1000, 2, "µm")
        self.z_begin_button = QPushButton("Set begin")
        self.z_end_button = QPushButton("Set end")
        self.z_begin_button.clicked.connect(self._capture_z_begin)
        self.z_end_button.clicked.connect(self._capture_z_end)
        stack_buttons = QHBoxLayout()
        stack_buttons.addWidget(self.z_begin_button)
        stack_buttons.addWidget(self.z_end_button)
        self.focus_method_box = QComboBox()
        self.focus_method_box.addItems(["linear", "plane"])
        z_form.addRow("Mode", self.z_mode_box)
        z_form.addRow("Focus surface", self.focus_method_box)
        z_form.addRow(stack_buttons)
        z_form.addRow("Begin", self.z_bottom)
        z_form.addRow("End", self.z_top)
        z_form.addRow("Below surface", self.z_below)
        z_form.addRow("Above surface", self.z_above)
        self.surface_span_label = QLabel()
        self.surface_span_label.setWordWrap(True)
        self.surface_span_label.setProperty("secondary", True)
        z_form.addRow(self.surface_span_label)
        self.z_step_label = QLabel("Step")
        z_form.addRow(self.z_step_label, self.z_step)
        self.z_stack_widgets = (self.z_bottom, self.z_top)
        self.z_capture_buttons = (self.z_begin_button, self.z_end_button)
        self.z_surface_stack_widgets = (self.z_below, self.z_above)
        self.z_below.valueChanged.connect(self._surface_offsets_changed)
        self.z_above.valueChanged.connect(self._surface_offsets_changed)
        self._update_surface_span_label()
        acquisition_layout.addWidget(z_group)

        # Tiling & options
        tiling = QGroupBox("Tiling && options")
        t_form = QFormLayout(tiling)
        self.overlap_x = spin(self.config_data.overlap_x_percent, 0, 50, 0, "%")
        self.overlap_y = spin(self.config_data.overlap_y_percent, 0, 50, 0, "%")
        self.tile_order_box = QComboBox()
        self.tile_order_box.addItems(["raster", "serpentine"])
        self.consistent_approach = ToggleSwitch("Consistent stage approach")
        self.shutter_per_frame = ToggleSwitch("Shutter closed during moves")
        self.shutter_per_frame.setChecked(True)
        self.flat_field = ToggleSwitch("Estimate flat-field")
        self.flat_field.setToolTip(
            "Estimate a flat-field profile for every channel after the cycle"
        )
        self.distortion = ToggleSwitch("Correct distortion / registration")
        self.distortion.setToolTip(
            "Resample every plane with the objective's grid-slide calibration before saving, "
            "so ISS_preprocessing gets geometrically corrected, channel-registered tiles. "
            "Fixed by cycle 1."
        )
        self.keep_uncorrected = ToggleSwitch("Keep uncorrected frames")
        self.keep_uncorrected.setChecked(True)
        self.keep_uncorrected.setToolTip(
            "Also save the camera frames in cycle_NN/uncorrected/ (twice the disk space)"
        )
        self.calibrate_button = QPushButton("Calibrate distortion…")
        self.calibrate_button.setToolTip("Measure the correction with a grid slide")
        self.calibrate_button.clicked.connect(self.open_distortion_calibration)
        self.distortion_label = QLabel()
        self.distortion_label.setWordWrap(True)
        self.distortion_label.setProperty("secondary", True)
        self.calibration_dialog = None
        t_form.addRow("Overlap X", self.overlap_x)
        t_form.addRow("Overlap Y", self.overlap_y)
        t_form.addRow("Tile order", self.tile_order_box)
        t_form.addRow(self.consistent_approach)
        t_form.addRow(self.shutter_per_frame)
        t_form.addRow(self.flat_field)
        t_form.addRow(self.distortion)
        t_form.addRow(self.keep_uncorrected)
        t_form.addRow(self.calibrate_button)
        t_form.addRow(self.distortion_label)
        acquisition_layout.addWidget(tiling)
        project_layout.addStretch(1)
        regions_page_layout.addStretch(1)
        acquisition_layout.addStretch(1)

        for signal in (
            self.channel_table.changed,
            self.region_list.changed,
            self.focus_list.changed,
            self.z_bottom.valueChanged,
            self.z_top.valueChanged,
            self.z_below.valueChanged,
            self.z_above.valueChanged,
            self.z_step.valueChanged,
            self.overlap_x.valueChanged,
            self.overlap_y.valueChanged,
            self.tile_order_box.currentIndexChanged,
            self.focus_method_box.currentIndexChanged,
            self.consistent_approach.toggled,
            self.shutter_per_frame.toggled,
            self.flat_field.toggled,
            self.distortion.toggled,
            self.keep_uncorrected.toggled,
            self.project_edit.editingFinished,
            self.sample_edit.editingFinished,
            self.output_edit.editingFinished,
        ):
            signal.connect(self._refresh)

        self.setup_tabs = QTabWidget()
        self.setup_tabs.setMinimumWidth(320)
        self.setup_tabs.setUsesScrollButtons(False)
        self.setup_tabs.tabBar().setExpanding(True)
        self.setup_pages: list[QScrollArea] = []
        for label, page in (
            ("Project", project_page),
            ("Regions", regions_page),
            ("Acquisition", acquisition_page),
        ):
            page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setToolTip("This page scrolls when its controls do not fit")
            self.setup_tabs.addTab(scroll, label)
            self.setup_pages.append(scroll)
        self.setup_tabs.setToolTip("Choose a workflow section; drag the right divider to resize")
        return self.setup_tabs

    def _build_viewer(self, use_ndv: bool) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.frame_view = FrameView(use_ndv=use_ndv)

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(0, 6, 0, 0)
        overview_layout.setSpacing(6)
        toolbar_widget = QWidget()
        toolbar_widget.setObjectName("viewerToolbar")
        toolbar = QVBoxLayout(toolbar_widget)
        toolbar.setContentsMargins(8, 6, 8, 6)
        toolbar.setSpacing(4)
        mode_toolbar = QHBoxLayout()
        mode_toolbar.setSpacing(6)
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QToolButton] = {}
        for mode, label, tip in OVERVIEW_MODES:
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tip)
            button.setMinimumWidth(button.sizeHint().width())
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, m=mode: self.set_overview_mode(m))
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            mode_toolbar.addWidget(button)
        mode_toolbar.addStretch(1)
        toolbar.addLayout(mode_toolbar)
        self.mode_buttons["navigate"].setChecked(True)
        view_toolbar = QHBoxLayout()
        view_toolbar.setSpacing(6)
        view_toolbar.addWidget(QLabel("View"))
        fit = QToolButton()
        fit.setText("Fit")
        fit.setMinimumWidth(fit.sizeHint().width())
        fit.clicked.connect(self._fit_overview)
        self.mosaic_mode_box = QComboBox()
        self.mosaic_mode_box.addItem(DiskBackedMosaic.COMPOSITE)
        self.mosaic_mode_box.setToolTip("Cycle mosaic display")
        self.mosaic_mode_box.currentIndexChanged.connect(lambda: self._render_mosaic(force=True))
        view_toolbar.addWidget(fit)
        view_toolbar.addStretch(1)
        view_toolbar.addWidget(self.mosaic_mode_box)
        toolbar.addLayout(view_toolbar)
        self.overview = OverviewCanvas(self.config_data.stage_limits)
        self.overview_hint = QLabel(OVERVIEW_HINTS["navigate"])
        self.overview_hint.setStyleSheet("color: #888;")
        overview_layout.addWidget(toolbar_widget)
        overview_layout.addWidget(self.overview, 1)
        # Accept button under the canvas, so a refocus walk never needs the setup panel.
        self.overview_accept_button = QPushButton("Accept Z ▶")
        self.overview_accept_button.setStyleSheet("font-weight: 600;")
        self.overview_accept_button.clicked.connect(self._set_focus_z_here)
        self.overview_accept_button.hide()
        hint_row = QHBoxLayout()
        hint_row.addWidget(self.overview_hint, 1)
        hint_row.addWidget(self.overview_accept_button)
        overview_layout.addLayout(hint_row)

        self.overview.region_drawn.connect(self._region_drawn)
        self.overview.stage_move_requested.connect(
            lambda x, y: self.device_move(self.backend.move_xy, x, y)
        )
        self.overview.focus_clicked.connect(self._overview_focus_clicked)
        self.overview.move_set_started.connect(self._move_set_started)
        self.overview.move_set_dragged.connect(self._move_set_dragged)
        self.overview.move_set_finished.connect(self._move_set_finished)
        self.overview.draft_cancelled.connect(self._move_set_cancelled)
        self.overview.hint.connect(self.overview_hint.setText)
        self.focus_list.list.currentRowChanged.connect(lambda _row: self._update_overview())

        self.tabs.addTab(overview_tab, "Overview")
        self.tabs.addTab(self.frame_view, "Camera")
        layout.addWidget(self.tabs, 1)

        capture_bar = QWidget()
        capture_bar.setObjectName("viewerToolbar")
        capture_controls = QHBoxLayout(capture_bar)
        capture_controls.setContentsMargins(8, 6, 8, 6)
        capture_controls.setSpacing(6)
        capture_title = QLabel("Capture")
        capture_title.setProperty("toolbarTitle", True)
        capture_controls.addWidget(capture_title)
        self.live_button = QPushButton("Live")
        self.live_button.setCheckable(True)
        self.live_button.toggled.connect(self._toggle_live)
        self.snap_button = QPushButton("Snap")
        self.snap_button.clicked.connect(self._snap)
        self.live_channel_box = QComboBox()
        for channel in self.config_data.channels:
            self.live_channel_box.addItem(channel.name, channel.id)
        self.preview_button = QPushButton("Spiral")
        self.preview_button.setCheckable(True)
        self.preview_button.setToolTip(
            "Snap the Live channel at each tile position, spiralling out from the stage "
            "until stopped or a stage limit is reached (over all planned tiles when regions "
            "exist). Existing preview images are retained. Current Z only, no Z moves."
        )
        self.preview_button.toggled.connect(self._toggle_spiral_preview)
        self.clear_preview_button = QPushButton("Clear")
        self.clear_preview_button.setToolTip("Clear all retained spiral preview images")
        self.clear_preview_button.clicked.connect(self.clear_spiral_previews)
        capture_controls.addWidget(self.live_button)
        capture_controls.addWidget(self.snap_button)
        capture_controls.addWidget(self.live_channel_box, 1)
        capture_controls.addWidget(self.preview_button)
        capture_controls.addWidget(self.clear_preview_button)
        layout.addWidget(capture_bar)
        wrapper.setMinimumWidth(400)
        return wrapper

    def _build_review_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 8, 8)
        layout.setSpacing(8)
        summary = QGroupBox("Cycle summary")
        s_layout = QVBoxLayout(summary)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.safety_label = QLabel()
        self.safety_label.setWordWrap(True)
        s_layout.addWidget(self.summary_label)
        s_layout.addWidget(self.safety_label)
        layout.addWidget(summary)

        acquisition = QGroupBox("Acquisition")
        acquisition_layout = QVBoxLayout(acquisition)
        acquisition_layout.setSpacing(8)
        self.start_button = QPushButton("Start cycle")
        self.start_button.setMinimumHeight(40)
        self.start_button.setProperty("accent", True)
        self.start_button.clicked.connect(self._start_cycle)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.abort_button = QPushButton("Abort")
        self.abort_button.setProperty("critical", True)
        self.abort_button.clicked.connect(self._abort)
        run_row = QHBoxLayout()
        run_row.addWidget(self.pause_button)
        run_row.addWidget(self.abort_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress_label = QLabel()
        self.progress_label.setWordWrap(True)
        acquisition_layout.addWidget(self.start_button)
        acquisition_layout.addLayout(run_row)
        acquisition_layout.addWidget(self.progress)
        acquisition_layout.addWidget(self.progress_label)
        layout.addWidget(acquisition)

        stage = QGroupBox("Stage controls")
        stage_layout = QVBoxLayout(stage)
        self.jog = StageJog()
        self.jog.move_requested.connect(self._jog)
        stage_layout.addWidget(self.jog)
        self.z_wheel = ZWheelControl()
        self.z_wheel.jog_requested.connect(self._z_wheel_jog)
        stage_layout.addWidget(self.z_wheel)
        layout.addWidget(stage)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(220)
        scroll.setToolTip("This panel scrolls; drag its left divider to resize it")
        self.review_scroll = scroll
        return scroll

    def _build_device_docks(self) -> None:
        """Micro-Manager device panels (pymmcore-widgets), only for the MMCore backend."""

        self.device_docks: list[QDockWidget] = []
        core = getattr(self.backend, "core", None)
        if core is None or self.backend.simulated:
            return
        try:
            from pymmcore_widgets import CoreLogWidget, PropertyBrowser
        except ImportError:  # pragma: no cover
            return
        menu = self.menuBar().addMenu("Devices")
        for title, widget in (
            ("Device properties (admin)", PropertyBrowser(mmcore=core)),
            ("Micro-Manager log", CoreLogWidget(mmcore=core)),
        ):
            dock = QDockWidget(title, self)
            dock.setWidget(widget)
            dock.hide()
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            menu.addAction(dock.toggleViewAction())
            self.device_docks.append(dock)

    # --------------------------------------------------------- configuration --
    def configuration_html(self) -> str:
        """Read-only summary of the configuration and hardware boundary."""

        c = self.config_data
        hw = c.hardware
        limits = c.stage_limits
        mmcore = not self.backend.simulated
        boundary = (
            f"Backend <b>{escape(self.backend.name)}</b> (pymmcore-plus). Moves outside the "
            "configured limits are refused before any command is sent."
            if mmcore
            else "Simulated stage, camera and turret. Set <code>hardware.backend</code> to "
            "<code>\"mmcore\"</code> (or start with <code>--backend mmcore</code>) for Micro-Manager."
        )
        rows = [
            ("Configuration", escape(str(self.config_path))),
            ("Microscope", escape(c.microscope_name)),
            ("Camera", f"{c.camera_width_px} × {c.camera_height_px} px, {8 * c.bytes_per_pixel}-bit"),
            ("Stage X", f"{limits.x_min_um:g} … {limits.x_max_um:g} µm"),
            ("Stage Y", f"{limits.y_min_um:g} … {limits.y_max_um:g} µm"),
            ("Stage Z", f"{limits.z_min_um:g} … {limits.z_max_um:g} µm"),
            ("Default overlap", f"X {c.overlap_x_percent:g}% • Y {c.overlap_y_percent:g}%"),
        ]
        if mmcore:
            rows += [
                ("MM config", escape(str(hw.get("mm_config_path", "")))),
                ("Channel group", escape(str(hw.get("channel_group", "Channel")))),
                ("Objective change Z", f"{hw.get('objective_change_z_um', limits.z_min_um)} µm"),
            ]
        summary = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows)
        current = self.backend.current_objective_id
        objectives = "".join(
            f"<tr><td>{'▶ ' if o.id == current else ''}{escape(o.name)}</td><td>{o.turret_position}</td>"
            f"<td>{o.numerical_aperture:g} {escape(o.immersion)}</td><td>{o.pixel_size_um:g}</td>"
            f"<td>{o.safe_z_min_um:g} … {o.safe_z_max_um:g}</td></tr>"
            for o in c.objectives
        )
        presets = hw.get("channel_presets", {})
        channels = "".join(
            f"<tr><td style='color:{ch.color}'><b>{escape(ch.name)}</b></td>"
            f"<td>{ch.max_exposure_ms:g} ms</td><td>{ch.max_power_percent:g}%</td>"
            + (f"<td>{escape(str(presets.get(ch.id, '—')))}</td>" if mmcore else "")
            + "</tr>"
            for ch in c.channels
        )
        preset_head = "<th>MM preset</th>" if mmcore else ""
        return (
            f"<h3>Hardware boundary</h3><p>{boundary}</p>"
            f"<table cellpadding=3>{summary}</table>"
            "<h3>Objectives</h3><table cellpadding=3><tr><th align=left>Name</th><th>Turret</th>"
            f"<th>NA</th><th>µm/px</th><th>Safe Z (µm)</th></tr>{objectives}</table>"
            "<h3>Channels</h3><table cellpadding=3><tr><th align=left>Name</th><th>Max exposure</th>"
            f"<th>Max power</th>{preset_head}</tr>{channels}</table>"
        )

    def show_configuration(self) -> None:
        from qtpy.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser

        dialog = QDialog(self)
        dialog.setWindowTitle("Microscope configuration")
        dialog.resize(620, 560)
        layout = QVBoxLayout(dialog)
        text = QTextBrowser()
        text.setHtml(self.configuration_html())
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(text)
        layout.addWidget(buttons)
        dialog.exec()

    # ------------------------------------------------------- dialogs (seam) --
    def confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def inform(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    # ----------------------------------------------------------------- plan --
    def build_plan(self) -> AcquisitionPlan:
        z_mode = self.z_mode_box.currentData()
        z = ZSettings(
            mode=z_mode,
            bottom_um=(
                -self.z_below.value()
                if z_mode == "surface_stack"
                else self.z_bottom.value()
            ),
            top_um=(self.z_above.value() if z_mode == "surface_stack" else self.z_top.value()),
            step_um=self.z_step.value(),
            begin_captured=self.z_begin_captured,
            end_captured=self.z_end_captured,
        )
        options = AcquisitionOptions(
            tile_order=self.tile_order_box.currentText(),
            consistent_approach=self.consistent_approach.isChecked(),
            shutter_per_frame=self.shutter_per_frame.isChecked(),
            flat_field=self.flat_field.isChecked(),
            distortion_correction=self.distortion.isChecked(),
            keep_uncorrected=self.keep_uncorrected.isChecked(),
            focus_method=self.focus_method_box.currentText(),
        )
        return AcquisitionPlan(
            project=self.project_edit.text().strip(),
            sample=self.sample_edit.text().strip(),
            output_root=Path(self.output_edit.text()).expanduser(),
            objective_id=self.objective_box.currentData(),
            channels=self.channel_table.settings(),
            regions=tuple(self.region_list.items),
            z=z,
            focus_points=tuple(self.focus_list.items),
            overlap_x_percent=self.overlap_x.value(),
            overlap_y_percent=self.overlap_y.value(),
            options=options,
        )

    def _refresh(self, *_args) -> None:
        if self._loading:
            return
        self._update_position()
        self._update_experiment_label()
        self._update_distortion_label()
        self._apply_locks()
        if self.controller.busy:
            self._update_overview()
            return
        try:
            plan = self.build_plan()
            tiles, estimate = self.controller.estimate(plan)
            problems = self.controller.preflight(
                plan, current_z_um=self.device.last_position.z_um
            )
        except (ValueError, KeyError) as exc:
            self.summary_label.setText(str(exc))
            self._set_safety([str(exc)])
            self._update_overview(())
            return
        self._update_overview(tiles)
        experiment = self.controller.experiment
        cycle = f"Cycle {experiment.next_cycle_number}" if experiment else "Cycle 1 (new experiment)"
        if plan.regions:
            names = ", ".join(self.config_data.channel(c.channel_id).name for c in plan.channels)
            offset = ""
            if experiment is not None:
                dx, dy = experiment.offset_of(plan.regions)
                offset = f"<br>ROI offset vs cycle 1: ΔX {dx:+.1f} ΔY {dy:+.1f} µm"
            self.summary_label.setText(
                f"<b>{cycle}</b><br>{self.config_data.objective(plan.objective_id).name}<br>"
                f"{names or 'No channels'}<br><br>"
                f"{estimate.tiles:,} tiles • {plan.options.tile_order}<br>"
                f"{estimate.frames:,} frames • {estimate.z_planes} Z plane(s)<br>"
                f"About {estimate.duration_label} • {estimate.data_label}{offset}"
            )
        else:
            self.summary_label.setText(f"<b>{cycle}</b><br>Add a scan region to calculate the scan.")
        self._set_safety(problems)

    def _set_safety(self, problems: list[str]) -> None:
        if problems:
            self.safety_label.setStyleSheet("color: #b5442c;")
            self.safety_label.setText("● " + "<br>● ".join(problems[:5]))
        else:
            self.safety_label.setStyleSheet("color: #2e7d32;")
            self.safety_label.setText("✓ Plan passes configured safety checks")

    def _apply_locks(self) -> None:
        busy = self.controller.busy
        locked = self.controller.experiment is not None  # objective/ROI/overlap fixed by cycle 1
        status = self.controller.status()
        for widget in (self.new_button, self.open_button, self.project_edit, self.sample_edit,
                       self.output_edit, self.channel_table, self.focus_method_box,
                       self.z_mode_box, self.consistent_approach, self.shutter_per_frame,
                       self.flat_field, self.snap_button, self.live_button, self.jog,
                       self.preview_button, self.z_wheel):
            widget.setEnabled(not busy)
        for widget in (self.objective_box, self.overlap_x, self.overlap_y, self.tile_order_box,
                       self.distortion):
            widget.setEnabled(not busy and not locked)
        self.keep_uncorrected.setEnabled(not busy and self.distortion.isChecked())
        self.calibrate_button.setEnabled(not busy)
        if self.calibration_dialog is not None:
            self.calibration_dialog._update()
        self.region_list.set_locked(busy or locked)
        self.shift_button.setEnabled(not busy and locked)
        self.recipe_box.setEnabled(not busy)
        self.save_set_button.setEnabled(not busy)
        self.load_set_button.setEnabled(not busy and not locked)
        for mode, button in self.mode_buttons.items():
            allowed = mode == "navigate" or (not busy and (mode not in ("rectangle", "polygon") or not locked))
            button.setEnabled(allowed)
        if not self.mode_buttons[self.overview.mode].isEnabled():
            self.set_overview_mode("navigate")
        self.focus_list.set_locked(busy)
        self.refocus_button.setEnabled(not busy)
        self.align_button.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.pause_button.setEnabled(busy)
        self.abort_button.setEnabled(busy)
        self.pause_button.setText(
            "Continue" if status is not None and status.state in ("pausing", "paused") else "Pause"
        )
        self.resume_button.setEnabled(not busy and self.controller.resume_info() is not None)
        for dock in getattr(self, "device_docks", ()):
            dock.widget().setEnabled(not busy)

    def _update_distortion_label(self) -> None:
        experiment = self.controller.experiment
        objective = self.config_data.objective(self.objective_box.currentData())
        if experiment is not None and experiment.options.distortion_correction:
            recorded = experiment.calibration.get("distortion", {})
            text = f"Experiment uses its own copy of the calibration ({recorded.get('calibrated_at', '')[:16]})."
        else:
            try:
                correction = self.controller.load_correction(objective.id)
            except (OSError, ValueError) as exc:
                text = f"Calibration file unreadable: {exc}"
            else:
                text = (
                    f"{objective.name}: not calibrated."
                    if correction is None
                    else f"{objective.name}: calibrated {correction.created_at[:10]} for "
                    + ", ".join(self.config_data.channel(c).name for c in sorted(correction.channels)
                                if any(ch.id == c for ch in self.config_data.channels))
                )
        self.distortion_label.setText(text)
        self.distortion_label.setVisible(self.distortion.isChecked() or experiment is None)

    def open_distortion_calibration(self):
        from .calibration import DistortionCalibrationDialog

        if self.calibration_dialog is None:
            self.calibration_dialog = DistortionCalibrationDialog(self)
            self.objective_box.currentIndexChanged.connect(self.calibration_dialog.sync_objective)
        self.calibration_dialog.sync_objective()
        self.calibration_dialog.show()
        self.calibration_dialog.raise_()
        return self.calibration_dialog

    def _update_position(self) -> None:
        if self.controller.busy:
            status = self.controller.status()
            tile = status.current_tile if status else None
            if tile is not None:
                self.position_label.setText(f"Tile {tile.index + 1}  X {tile.x_um:.1f}  Y {tile.y_um:.1f} µm")
            return
        p = self.device.last_position
        self.position_label.setText(f"X {p.x_um:.1f}  Y {p.y_um:.1f}  Z {p.z_um:.2f} µm")
        self._update_z_stack_view()

    def _position_changed(self, _position) -> None:
        self._update_position()
        self._update_overview()

    @property
    def stage_position(self):
        """Last stage position reported by the device worker (no device call)."""

        return self.device.last_position

    def _refresh_position(self) -> None:
        if not self.controller.busy and self.device.idle:
            self.device.submit(lambda: None)

    def _update_experiment_label(self) -> None:
        experiment = self.controller.experiment
        if experiment is None:
            self.experiment_label.setText("No experiment — cycle 1 creates one")
        else:
            self.experiment_label.setText(
                f"{experiment.project} / {experiment.sample} • {len(experiment.cycles)} cycle(s)"
            )

    # --------------------------------------------------------------- inputs --
    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Output folder", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)
            self._refresh()

    def _sync_objective_box(self) -> None:
        index = self.objective_box.findData(self.backend.current_objective_id)
        if index >= 0:
            self.objective_box.setCurrentIndex(index)

    def _physical_objective_changed(self, objective_id: str) -> None:
        """Reflect a turret change made outside this window without commanding it again."""

        index = self.objective_box.findData(objective_id)
        if index >= 0 and self.objective_box.currentIndex() != index:
            self.objective_box.setCurrentIndex(index)
            self._reset_z_stack_capture()
            self.progress_label.setText(
                f"Turret changed externally: {self.config_data.objective(objective_id).name}"
            )
            self._refresh()

    def _objective_selected(self, _index: int, *, confirm_automatic: bool = False) -> bool:
        self._recipe_to_custom()
        objective = self.config_data.objective(self.objective_box.currentData())
        previous = self.config_data.objective(self.backend.current_objective_id)
        if objective.id == previous.id:
            return True
        if objective.immersion != previous.immersion:
            title = "Immersion change"
            message = (
                f"Changing from {previous.name} ({previous.immersion}) to {objective.name} "
                f"({objective.immersion}). Apply/remove immersion medium now. Z moves to a safe "
                "clearance before the turret rotates. Continue?"
            )
        else:
            title = "Change objective"
            message = (
                f"Change from {previous.name} to {objective.name}? Z moves to a safe clearance "
                "before the turret rotates."
            )
        if (confirm_automatic or objective.immersion != previous.immersion) and not self.confirm(
            title, message
        ):
            self._sync_objective_box()
            return False
        self.stop_spiral_preview()
        self.objective_box.setEnabled(False)
        self.progress_label.setText(f"Changing objective to {objective.name}…")

        def done(_result) -> None:
            self.progress_label.setText(f"Objective: {objective.name}")
            if not self.backend.simulated:
                self.inform("Refocus", "Z is at the objective-change clearance. Refocus in Live.")
            self._reset_z_stack_capture()
            self._refresh()

        def failed(exc: Exception) -> None:
            self.warn("Objective change failed", str(exc))
            self._sync_objective_box()
            self._refresh()

        self.device.submit(self.backend.change_objective, objective, on_done=done, on_error=failed)
        return True

    # --------------------------------------------------------------- recipes --
    def _recipe_to_custom(self, *_args) -> None:
        if not self._applying_recipe:
            self.recipe_box.setCurrentIndex(0)

    def _apply_recipe(self, _index: int = 0) -> None:
        name = self.recipe_box.currentText()
        if name == "Custom":
            return
        recipe = self.config_data.recipe(name)
        experiment = self.controller.experiment
        if experiment is not None and recipe.objective_id != experiment.objective_id:
            self.inform(
                "Objective is fixed",
                f"This experiment uses {self.config_data.objective(experiment.objective_id).name}; "
                f"recipe '{name}' needs {self.config_data.objective(recipe.objective_id).name}. "
                "Channels and Z mode were not changed.",
            )
            self.recipe_box.setCurrentIndex(0)
            return
        self._applying_recipe = True
        try:
            current = {c.channel_id: c for c in self.channel_table.settings()}
            defaults = {c.id: c for c in self.config_data.channels}
            self.channel_table.set_settings(tuple(
                current.get(cid) or ChannelSetting(cid, defaults[cid].default_exposure_ms,
                                                   defaults[cid].default_power_percent)
                for cid in recipe.channels
            ))
            self.z_mode_box.setCurrentIndex(self.z_mode_box.findData(recipe.z_mode))
            self.objective_box.setCurrentIndex(self.objective_box.findData(recipe.objective_id))
            if recipe.objective_id != self.backend.current_objective_id:
                self._objective_selected(
                    self.objective_box.currentIndex(), confirm_automatic=True
                )
        finally:
            self._applying_recipe = False
        self.recipe_box.setCurrentText(name)
        self._refresh()

    def _z_mode_changed(self, *_args) -> None:
        mode = self.z_mode_box.currentData()
        stack = mode == "stack"
        surface_stack = mode == "surface_stack"
        uses_stack_step = stack or surface_stack
        for widget in self.z_stack_widgets:
            widget.setVisible(stack)
        for button in self.z_capture_buttons:
            button.setVisible(uses_stack_step)
        for widget in self.z_surface_stack_widgets:
            widget.setVisible(surface_stack)
        for widget in self.z_stack_widgets + self.z_surface_stack_widgets:
            field_label = self.z_form.labelForField(widget)
            if field_label is not None:
                field_label.setVisible(widget.isVisible())
        self.z_begin_button.setText("Set start" if surface_stack else "Set begin")
        self.z_end_button.setText("Set end")
        if surface_stack:
            self.z_begin_button.setToolTip("Capture one edge of the desired stack span")
            self.z_end_button.setToolTip(
                "Capture the other edge; the span is centred on every tile's surface Z"
            )
        else:
            self.z_begin_button.setToolTip("Capture the absolute stack begin at the current Z")
            self.z_end_button.setToolTip("Capture the absolute stack end at the current Z")
        self.surface_span_label.setVisible(surface_stack)
        self.z_step.setVisible(uses_stack_step)
        self.z_step_label.setVisible(uses_stack_step)
        self.focus_method_box.setEnabled(mode in ("focus_map", "surface_stack"))
        self._update_z_stack_view()
        self._refresh()

    def _update_z_stack_view(self, *_args) -> None:
        if not hasattr(self, "z_wheel"):
            return
        positions: tuple[float, ...] = ()
        mode = self.z_mode_box.currentData()
        if mode == "stack":
            positions = ZSettings(
                mode="stack",
                bottom_um=self.z_bottom.value(),
                top_um=self.z_top.value(),
                step_um=self.z_step.value(),
                begin_captured=True,
                end_captured=True,
            ).stack_positions()
        elif mode == "surface_stack" and len(self.focus_list.items) >= 3:
            try:
                surface = FocusSurface(
                    self.focus_list.items, self.focus_method_box.currentText()
                )
                p = self.stage_position
                centre = surface.z_at(p.x_um, p.y_um)
                offsets = ZSettings(
                    mode="surface_stack",
                    bottom_um=-self.z_below.value(),
                    top_um=self.z_above.value(),
                    step_um=self.z_step.value(),
                ).stack_positions()
                positions = tuple(centre + offset for offset in offsets)
            except ValueError:
                positions = ()
        self.z_wheel.set_stack(positions, self.device.last_position.z_um)

    def _capture_z_begin(self) -> None:
        def capture(p) -> None:
            if self.z_mode_box.currentData() == "surface_stack":
                self._surface_stack_start_um = p.z_um
                self._apply_surface_stack_capture()
            else:
                self.z_bottom.setValue(p.z_um)
            self.z_begin_captured = True
            self._refresh()

        self.with_fresh_position(capture)

    def _capture_z_end(self) -> None:
        def capture(p) -> None:
            if self.z_mode_box.currentData() == "surface_stack":
                self._surface_stack_end_um = p.z_um
                self._apply_surface_stack_capture()
            else:
                self.z_top.setValue(p.z_um)
            self.z_end_captured = True
            self._refresh()

        self.with_fresh_position(capture)

    def _apply_surface_stack_capture(self) -> None:
        """Keep the captured span and centre it on each tile's focus-surface Z."""

        if self._surface_stack_start_um is None or self._surface_stack_end_um is None:
            self._update_surface_span_label()
            return
        half_span = abs(self._surface_stack_end_um - self._surface_stack_start_um) / 2
        self._setting_surface_span = True
        try:
            self.z_below.setValue(half_span)
            self.z_above.setValue(half_span)
        finally:
            self._setting_surface_span = False
        self._update_surface_span_label()
        self._update_z_stack_view()

    def _surface_offsets_changed(self, *_args) -> None:
        if not self._setting_surface_span:
            self._surface_stack_start_um = None
            self._surface_stack_end_um = None
            self.z_begin_captured = self.z_end_captured = False
        self._update_surface_span_label()

    def _update_surface_span_label(self) -> None:
        if not hasattr(self, "surface_span_label"):
            return
        span = self.z_below.value() + self.z_above.value()
        if self._surface_stack_start_um is not None and self._surface_stack_end_um is not None:
            self.surface_span_label.setText(
                f"Captured span {span:g} µm → centred ±{span / 2:g} µm on the surface"
            )
        else:
            self.surface_span_label.setText(
                f"Span {span:g} µm → centred on each tile's surface Z"
            )

    def _reset_z_stack_capture(self) -> None:
        self.z_begin_captured = self.z_end_captured = False
        self._surface_stack_start_um = self._surface_stack_end_um = None
        self._update_surface_span_label()

    def _add_region(self) -> None:
        p = self.stage_position
        dialog = RectangleDialog((p.x_um, p.y_um), self)
        if dialog.exec():
            self.region_list.append(dialog.region())

    def _shift_roi_set(self) -> None:
        from qtpy.QtWidgets import QDialog, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Shift ROI set (XY translation)")
        form = QFormLayout(dialog)
        dx, dy = spin(0, -1e5, 1e5, 1, "µm"), spin(0, -1e5, 1e5, 1, "µm")
        form.addRow("ΔX", dx)
        form.addRow("ΔY", dy)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            self.shift_roi_set(dx.value(), dy.value())

    def _save_roi_set(self) -> None:
        if not self.region_list.items and not self.focus_list.items:
            self.inform("Nothing to save", "Draw at least one region or add a focus point first.")
            return
        default = Path(self.output_edit.text()).expanduser() / f"{self.sample_edit.text().strip() or 'sample'}_roi.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI and focus-point set", str(default), "ROI set (*.json)")
        if path:
            self.save_roi_set(path)

    def save_roi_set(self, path: Path | str) -> bool:
        try:
            save_roi_set(path, tuple(self.region_list.items), tuple(self.focus_list.items))
        except OSError as exc:
            self.warn("Could not save ROI set", str(exc))
            return False
        self.overview_hint.setText(f"ROI set saved • {path}")
        return True

    def _load_roi_set(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROI and focus-point set", self.output_edit.text(), "ROI set (*.json);;All files (*)"
        )
        if path:
            self.load_roi_set(path)

    def load_roi_set(self, path: Path | str) -> bool:
        if self.controller.busy or self.controller.experiment is not None:
            return False  # regions are fixed by cycle 1
        try:
            regions, focus_points = load_roi_set(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.warn("Could not load ROI set", str(exc))
            return False
        if (self.region_list.items or self.focus_list.items) and not self.confirm(
            "Replace current ROI set?",
            "Loading this file replaces the current regions and focus points. Continue?",
        ):
            return False
        self.stop_refocus()
        self.region_list.set_items(regions)
        self.focus_list.set_items(focus_points)
        self.overview.fit_items(regions, focus_points)
        self.overview_hint.setText(f"ROI set loaded • {path}")
        return True

    def shift_roi_set(self, dx_um: float, dy_um: float) -> None:
        """Translate regions and focus points together (re-registration, XY only)."""

        self.region_list.set_items([r.translated(dx_um, dy_um) for r in self.region_list.items])
        self.focus_list.set_items([f.translated_xy(dx_um, dy_um) for f in self.focus_list.items])

    # -------------------------------------------------------------- overview --
    def set_overview_mode(self, mode: str) -> None:
        button = self.mode_buttons[mode]
        if not button.isEnabled():
            mode, button = "navigate", self.mode_buttons["navigate"]
        button.setChecked(True)
        self.overview.set_mode(mode)
        self.overview_hint.setText(OVERVIEW_HINTS[mode])

    def _fit_overview(self) -> None:
        self.overview.fit_items(self.region_list.items, self.focus_list.items)

    def _update_overview(self, tiles=None) -> None:
        if tiles is not None:
            self._overview_tiles = tiles
        tiles = getattr(self, "_overview_tiles", ())
        objective_id = self.backend.current_objective_id
        p = self.stage_position
        self.overview.set_content(
            self.region_list.items,
            self.focus_list.items,
            tiles,
            self.config_data.fov_um(objective_id),
            (p.x_um, p.y_um),
            self.focus_list.selected_index(),
        )

    def _region_drawn(self, region) -> None:
        if self.controller.busy or self.controller.experiment is not None:
            self.overview_hint.setText("Regions are fixed by cycle 1 — use Move ROI set")
            return
        self.region_list.append(region)
        self.overview_hint.setText(f"{region.shape.capitalize()} region added")

    def _overview_focus_clicked(self, x_um: float, y_um: float, index: int) -> None:
        if self.refocus_index is not None and index >= 0:
            self._refocus_goto(index)
            return
        if index >= 0:
            self.focus_list.list.setCurrentRow(index)
            self._goto_focus()
            self.overview_hint.setText(f"Going to F{index + 1} — refocus in Live, then Set Z")
        else:
            self.focus_list.list.setCurrentRow(-1)
            self.device_move(self.backend.move_xy, x_um, y_um)
            self.overview_hint.setText("Moving — focus in Live, then Capture")
        if not self.live_button.isChecked():
            self.live_button.setChecked(True)
        self.tabs.setCurrentIndex(0)  # stay on the overview while live runs

    def _move_set_started(self) -> None:
        self._move_origin = (tuple(self.region_list.items), tuple(self.focus_list.items))
        self._loading = True  # skip estimates while dragging

    def _move_set_dragged(self, dx_um: float, dy_um: float) -> None:
        origin = getattr(self, "_move_origin", None)
        if origin is None:
            return
        regions, focus_points = origin
        dx_um, dy_um = bounded_set_translation(
            regions, focus_points, self.config_data.stage_limits, dx_um, dy_um
        )
        self.region_list.set_items([r.translated(dx_um, dy_um) for r in regions])
        self.focus_list.set_items([f.translated_xy(dx_um, dy_um) for f in focus_points])
        self._update_overview()
        self.overview_hint.setText(f"ROI set ΔX {dx_um:+.1f}  ΔY {dy_um:+.1f} µm")

    def _move_set_finished(self) -> None:
        self._move_origin = None
        self._loading = False
        self._refresh()

    def _move_set_cancelled(self) -> None:
        origin = getattr(self, "_move_origin", None)
        if origin is not None:
            self.region_list.set_items(origin[0])
            self.focus_list.set_items(origin[1])
        self._move_set_finished()
        self.overview_hint.setText("Move cancelled")

    def with_fresh_position(self, callback, on_error=None) -> None:
        """Run ``callback(position)`` after every queued stage move has finished.

        The cached position can lag behind a jog that is still queued; reading it
        through the worker queue guarantees the value the operator just focused at.
        """

        if self.controller.busy:
            return
        def failed(exc: Exception) -> None:
            if on_error is not None:
                on_error(exc)
            else:
                self.warn("Could not read the stage position", str(exc))

        self.device.submit(
            lambda: self.backend.position,
            on_done=callback,
            on_error=failed,
        )

    def _capture_focus_here(self) -> None:
        def capture(p) -> None:
            self.focus_list.append(FocusPoint(p.x_um, p.y_um, p.z_um))
            self.focus_list.list.setCurrentRow(len(self.focus_list.items) - 1)
            self.overview_hint.setText(f"Captured F{len(self.focus_list.items)} at Z {p.z_um:g} µm")

        self.with_fresh_position(capture)

    def _goto_focus(self) -> None:
        index = self.focus_list.selected_index()
        if index is None:
            return
        point = self.focus_list.items[index]

        def go() -> None:
            self.backend.move_xy(point.x_um, point.y_um)
            self.backend.move_z(point.z_um)

        self.device_move(go)

    def _set_focus_z_here(self) -> None:
        """Manual refocus: store the live Z on the selected point (Accept Z during Refocus all)."""

        index = self.refocus_index if self.refocus_index is not None else self.focus_list.selected_index()
        if index is None:
            self.inform("Select a focus point", "Select a focus point in the list or on the overview first.")
            return
        if self._accept_pending:  # one accept per point, even on a double click
            return
        self._accept_pending = True

        def accept(p) -> None:
            self._accept_pending = False
            if index >= len(self.focus_list.items):
                return
            items = list(self.focus_list.items)
            items[index] = replace(items[index], z_um=p.z_um)
            self.focus_list.set_items(items)
            self.focus_list.list.setCurrentRow(index)
            self.overview_hint.setText(f"F{index + 1} set to Z {p.z_um:g} µm")
            if self.refocus_index == index:
                if index + 1 < len(items):
                    self._refocus_goto(index + 1)
                else:
                    self.stop_refocus(f"All {len(items)} focus points refocused for this cycle")

        def failed(exc: Exception) -> None:
            self._accept_pending = False
            self.warn("Could not read the stage position", str(exc))

        self.with_fresh_position(accept, on_error=failed)

    # -------------------------------------------------------- spiral preview --
    def _toggle_spiral_preview(self, on: bool) -> None:
        if on:
            self.start_spiral_preview()
        else:
            self.stop_spiral_preview()

    def start_spiral_preview(self) -> bool:
        if self.controller.busy:
            self.preview_button.setChecked(False)
            return False
        objective_id = self.backend.current_objective_id
        origin = (self.stage_position.x_um, self.stage_position.y_um)
        limits = self.config_data.stage_limits
        regions = tuple(self.region_list.items)
        if regions:
            positions = [
                p
                for p in spiral_preview_positions(
                    self.config_data,
                    objective_id,
                    regions,
                    origin,
                    overlap_x_percent=self.overlap_x.value(),
                    overlap_y_percent=self.overlap_y.value(),
                )
                if limits.contains_xy(*p)
            ]
            if not positions:
                self.warn(
                    "No preview positions",
                    "The preview area has no positions inside the stage limits.",
                )
                self.preview_button.setChecked(False)
                return False
            self._preview_positions = iter(positions)
            self._preview_total: int | None = len(positions)
        else:
            self._preview_positions = continuous_spiral_preview_positions(
                self.config_data,
                objective_id,
                origin,
                overlap_x_percent=self.overlap_x.value(),
                overlap_y_percent=self.overlap_y.value(),
            )
            self._preview_total = None
        self._preview_limits = limits
        self.live_button.setChecked(False)
        self._push_live_settings()
        self._preview_generation = getattr(self, "_preview_generation", 0) + 1
        self._preview_index = 0
        self._preview_range: tuple[float, float] | None = None
        self._preview_fov = self.config_data.fov_um(objective_id)
        self._preview_active = True
        self.tabs.setCurrentIndex(0)
        if not self.preview_button.isChecked():
            self.preview_button.blockSignals(True)
            self.preview_button.setChecked(True)
            self.preview_button.blockSignals(False)
        self._preview_next(self._preview_generation)
        return True

    @property
    def preview_running(self) -> bool:
        return getattr(self, "_preview_active", False)

    def stop_spiral_preview(self, message: str | None = None) -> None:
        if not hasattr(self, "_preview_positions"):
            return
        self._preview_generation = getattr(self, "_preview_generation", 0) + 1  # drops the next job
        if not self.preview_running:
            return  # nothing running: keep the current hint
        self._preview_active = False
        self.preview_button.blockSignals(True)
        self.preview_button.setChecked(False)
        self.preview_button.blockSignals(False)
        self.overview_hint.setText(
            message or f"Spiral preview stopped after {self._preview_index} tiles"
        )

    def clear_spiral_previews(self) -> None:
        self.stop_spiral_preview()
        self.overview.clear_preview()
        self.overview_hint.setText("Spiral previews cleared")

    def _preview_next(self, generation: int) -> None:
        if generation != self._preview_generation:
            return
        try:
            x, y = next(self._preview_positions)
        except StopIteration:
            self.stop_spiral_preview(
                f"Spiral preview complete • {self._preview_index} tiles • current Z only"
            )
            return
        if not self._preview_limits.contains_xy(x, y):
            self.stop_spiral_preview(
                f"Spiral preview stopped at stage limit • {self._preview_index} tiles captured"
            )
            return
        backend, device = self.backend, self.device

        def visit():
            if generation != self._preview_generation:
                return None  # stopped while queued: no move, no exposure
            backend.move_xy(x, y)
            frame = device.acquire_frame()
            step = max(1, int(np.ceil(max(frame.shape) / PREVIEW_TILE_PX)))
            return (x, y), np.ascontiguousarray(frame[::step, ::step])

        def failed(exc: Exception) -> None:
            if generation == self._preview_generation:
                self.stop_spiral_preview(f"Spiral preview stopped: {exc}")
                self.warn("Spiral preview stopped", str(exc))

        self.device.submit(visit, on_done=lambda result: self._preview_tile_done(generation, result), on_error=failed)

    def _preview_tile_done(self, generation: int, result) -> None:
        if result is None or generation != self._preview_generation:
            return
        center, frame = result
        if self._preview_range is None:
            low, high = np.percentile(frame, (1.0, 99.7))
            self._preview_range = (float(low), float(max(high, low + 1)))
        channel = self.config_data.channel(self.live_channel_box.currentData())
        self.overview.add_preview_tile(frame, center, self._preview_fov, self._preview_range, channel.color)
        self._preview_index += 1
        progress = f"tile {self._preview_index}"
        if self._preview_total is not None:
            progress += f" of {self._preview_total}"
        self.overview_hint.setText(f"{channel.name} spiral preview • {progress} • current Z only")
        self._preview_next(generation)

    # -------------------------------------------------- refocus & alignment --
    def _toggle_refocus(self) -> None:
        if self.refocus_index is not None:
            self.stop_refocus("Refocus stopped")
            return
        if self.controller.busy:
            return
        if not self.focus_list.items:
            self.inform("No focus points", "Capture or load focus points first.")
            return
        self.refocus_button.setText("Stop refocus")
        self.set_z_button.setText("Accept Z ▶")
        self.set_overview_mode("focus")
        self._refocus_goto(0)

    def _refocus_goto(self, index: int) -> None:
        self.refocus_index = index
        self.focus_list.list.setCurrentRow(index)
        self._goto_focus()
        if not self.live_button.isChecked():
            self.live_button.setChecked(True)
        self.tabs.setCurrentIndex(0)
        text = f"Refocus F{index + 1} of {len(self.focus_list.items)}: focus in Live, then Accept Z ▶"
        self.refocus_label.setText(text)
        self.refocus_label.show()
        self.overview_accept_button.show()
        self.overview_hint.setText(text)

    def stop_refocus(self, message: str | None = None) -> None:
        self._accept_pending = False
        if self.refocus_index is None:
            return
        self.refocus_index = None
        self.refocus_button.setText("Refocus all")
        self.set_z_button.setText("Set Z")
        self.refocus_label.hide()
        self.overview_accept_button.hide()
        if message:
            self.overview_hint.setText(message)

    def _align_selected_focus_to_stage(self) -> None:
        index = self.focus_list.selected_index()
        if index is None:
            self.inform(
                "Select a focus point",
                "Select a focus point, move the stage so the same landmark is centred in Live, "
                "then press Align. The whole ROI set moves so that point sits at the current "
                "stage position (Z is kept).",
            )
            return

        def align(p) -> None:
            if index >= len(self.focus_list.items):
                return
            point = self.focus_list.items[index]
            dx, dy = p.x_um - point.x_um, p.y_um - point.y_um
            self.shift_roi_set(dx, dy)
            self.focus_list.list.setCurrentRow(index)
            self.overview_hint.setText(
                f"ROI set moved by ΔX {dx:+.1f} ΔY {dy:+.1f} µm to align F{index + 1}"
            )

        self.with_fresh_position(align)

    # ----------------------------------------------------- stage / camera --
    def device_move(self, function, *args) -> bool:
        """Queue a stage move on the device worker; limit errors come back as a warning."""

        if self.controller.busy:
            return False
        self.device.submit(
            function, *args, on_error=lambda exc: self.warn("Stage move refused", str(exc))
        )
        return True

    def _jog(self, dx: float, dy: float, dz: float) -> None:
        backend = self.backend

        def jog() -> None:  # relative to the position at execution time, not at click time
            p = backend.position
            if dz:
                backend.move_z(p.z_um + dz)
            else:
                backend.move_xy(p.x_um + dx, p.y_um + dy)

        self.device_move(jog)

    def _z_wheel_jog(self, notches: int) -> None:
        """Move Z from the dedicated viewer-edge wheel target."""

        self._jog(0.0, 0.0, notches * self.jog.z_step.value())

    def _push_live_settings(self, *_args) -> None:
        channel = self.config_data.channel(self.live_channel_box.currentData())
        settings = {c.channel_id: c for c in self.channel_table.settings()}
        exposure = settings[channel.id].exposure_ms if channel.id in settings else channel.default_exposure_ms
        power = settings[channel.id].power_percent if channel.id in settings else channel.default_power_percent
        self.device.set_live_settings(channel, exposure, power)

    def _snap(self) -> bool:
        if self.controller.busy:
            return False
        self._push_live_settings()
        self.tabs.setCurrentWidget(self.frame_view)
        self.device.snap(on_error=lambda exc: self.warn("Snap failed", str(exc)))
        return True

    def _toggle_live(self, on: bool) -> None:
        if on and self.controller.busy:
            self.live_button.setChecked(False)
            return
        if on:
            self.stop_spiral_preview()
            self._push_live_settings()
        else:
            self.overview.clear_live_frame()
        self.device.set_live(on)

    def _show_live_frame(self, frame: np.ndarray) -> None:
        """Route a frame to Camera and, when selected, the current Overview FOV."""

        self.frame_view.show_frame(frame)
        if not self.live_button.isChecked() or self.tabs.currentIndex() != 0:
            return
        step = max(1, int(np.ceil(max(frame.shape) / PREVIEW_TILE_PX)))
        preview = np.ascontiguousarray(frame[::step, ::step])
        position = self.stage_position
        objective_id = self.backend.current_objective_id
        channel = self.config_data.channel(self.live_channel_box.currentData())
        self.overview.set_live_frame(
            preview,
            (position.x_um, position.y_um),
            self.config_data.fov_um(objective_id),
            channel.color,
        )

    def _live_failed(self, message: str) -> None:
        self.live_button.setChecked(False)
        self.warn("Live stopped", message)

    def _hand_backend_to_engine(self) -> bool:
        """Stop live/preview and finish queued device jobs before the engine drives the backend."""

        self.stop_spiral_preview()
        self.live_button.setChecked(False)
        if not self.device.stop_and_drain(timeout=30):
            self.warn("Microscope busy", "A stage or camera command is still running. Try again.")
            return False
        return True

    # ----------------------------------------------------------- experiment --
    def _new_experiment(self) -> None:
        if self.controller.experiment is not None and not self.confirm(
            "New experiment", "Close the current experiment? Its data stays on disk."
        ):
            return
        self.stop_refocus()
        self.controller.new_experiment()
        self._reset_z_stack_capture()
        self._refresh()

    def _open_experiment(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open experiment", self.output_edit.text(), "experiment.json")
        if path:
            self.open_experiment(path)

    def open_experiment(self, path: Path | str) -> Experiment | None:
        try:
            experiment = self.controller.open_experiment(path)
        except (ControllerError, OSError, ValueError, KeyError) as exc:
            self.warn("Could not open experiment", str(exc))
            return None
        self._load_experiment_into_view(experiment)
        info = self.controller.resume_info()
        if info is not None:
            record, finished, total = info
            if self.confirm("Unfinished cycle", f"Cycle {record.number} has {finished}/{total} tiles. Resume now?"):
                self._resume_cycle(confirm=False)
        return experiment

    def _load_experiment_into_view(self, experiment: Experiment) -> None:
        self.stop_refocus()
        self._reset_z_stack_capture()
        self._loading = True
        try:
            last = experiment.cycles[-1] if experiment.cycles else None
            dx = last.offset_x_um if last else 0.0
            dy = last.offset_y_um if last else 0.0
            self.region_list.set_items(experiment.regions_with_offset(dx, dy))
            self.focus_list.set_items(last.focus_points if last else experiment.base_focus_points)
            self.project_edit.setText(experiment.project)
            self.sample_edit.setText(experiment.sample)
            self.output_edit.setText(str(experiment.root.parent.parent))
            self.overlap_x.setValue(experiment.overlap_x_percent)
            self.overlap_y.setValue(experiment.overlap_y_percent)
            self.tile_order_box.setCurrentText(experiment.options.tile_order)
            if last is not None:
                self.channel_table.set_settings(last.channels)
                self.z_mode_box.setCurrentIndex(self.z_mode_box.findData(last.z.mode))
                if last.z.mode == "surface_stack":
                    self.z_below.setValue(max(0.0, -last.z.bottom_um))
                    self.z_above.setValue(max(0.0, last.z.top_um))
                else:
                    self.z_bottom.setValue(last.z.bottom_um)
                    self.z_top.setValue(last.z.top_um)
                self.z_step.setValue(last.z.step_um)
                options = last.options or experiment.options
                self.focus_method_box.setCurrentText(options.focus_method)
                self.shutter_per_frame.setChecked(options.shutter_per_frame)
                self.consistent_approach.setChecked(options.consistent_approach)
                self.flat_field.setChecked(options.flat_field)
                self.keep_uncorrected.setChecked(options.keep_uncorrected)
            self.distortion.setChecked(experiment.options.distortion_correction)
            self.objective_box.setCurrentIndex(self.objective_box.findData(experiment.objective_id))
        finally:
            self._loading = False
        if self.backend.current_objective_id != experiment.objective_id:
            self._objective_selected(
                self.objective_box.currentIndex(), confirm_automatic=True
            )
        self.progress_label.setText(f"Experiment opened • {experiment.root}")
        self._z_mode_changed()

    # -------------------------------------------------------------- running --
    def _start_cycle(self) -> None:
        self.stop_refocus()
        self.live_button.setChecked(False)
        try:
            plan = self.build_plan()
            problems = self.controller.preflight(
                plan, current_z_um=self.device.last_position.z_um
            )
        except (ValueError, KeyError) as exc:
            problems = [str(exc)]
        if problems:
            self.warn("Preflight needs attention", "Fix these before scanning:\n\n• " + "\n• ".join(problems))
            return
        _tiles, estimate = self.controller.estimate(plan)
        prompts = [
            self.config_data.channel(s.channel_id).manual_prompt
            for s in plan.channels
            if self.config_data.channel(s.channel_id).manual_prompt
        ]
        experiment = self.controller.experiment
        number = experiment.next_cycle_number if experiment else 1
        text = (
            f"Cycle {number}{' (new experiment)' if experiment is None else ''}\n"
            f"Objective: {self.config_data.objective(plan.objective_id).name}\n"
            f"Tiles: {estimate.tiles:,} | Frames: {estimate.frames:,}\n"
            f"Data: {estimate.data_label} | Time: {estimate.duration_label}"
        )
        if prompts:
            text += "\n\nManual checks:\n• " + "\n• ".join(prompts)
        mode = "simulated scan" if self.backend.simulated else "scan on the microscope"
        if not self.confirm("Ready to scan", f"{text}\n\nStart the {mode}?"):
            return
        if not self._hand_backend_to_engine():
            return
        try:
            self.controller.start_cycle(plan)
        except LAUNCH_ERRORS as exc:
            self.warn("Could not start the cycle", str(exc))
            self._refresh()
            return
        self._after_launch()

    def _resume_cycle(self, confirm: bool = True) -> None:
        info = self.controller.resume_info()
        if info is None:
            return
        record, finished, total = info
        if confirm and not self.confirm(
            "Resume cycle",
            f"Resume cycle {record.number} at the first unsaved tile ({finished}/{total} saved)? "
            "The original channels, exposures, powers and focus positions are reused.",
        ):
            return
        self.stop_refocus()
        if not self._hand_backend_to_engine():
            return
        try:
            self.controller.resume_cycle()
        except LAUNCH_ERRORS as exc:
            self.warn("Could not resume", str(exc))
            return
        self._after_launch()

    def _after_launch(self) -> None:
        mosaic = self.controller.mosaic
        self.mosaic_mode_box.blockSignals(True)
        self.mosaic_mode_box.clear()
        self.mosaic_mode_box.addItems(mosaic.display_modes if mosaic else [DiskBackedMosaic.COMPOSITE])
        self.mosaic_mode_box.blockSignals(False)
        self.tabs.setCurrentIndex(0)
        self.progress.setValue(0)
        self.progress_label.setText("Starting acquisition…")
        self._apply_locks()
        self.poll_timer.start()

    def _poll_scan(self) -> None:
        status = self.controller.status()
        if status is None:
            self.poll_timer.stop()
            return
        self.progress.setValue(int(status.fraction * 1000))
        parts = [f"Cycle {self.controller.active_cycle}", f"{status.tiles_done:,}/{status.tiles_total:,} tiles saved"]
        if status.state in ("pausing", "paused", "aborting", "draining", "post-processing"):
            parts.append(status.message)
        elif status.current_tile is not None and status.current_channel_id:
            channel = self.config_data.channel(status.current_channel_id).name
            parts.append(f"tile {status.current_tile.index + 1} • {channel} • Z {status.current_z_index + 1}/{status.current_z_planes}")
        if status.backpressure:
            parts.append("SSD buffer full — camera waiting safely")
        acquired = status.tiles_done - status.tiles_skipped
        if status.started_monotonic and acquired > 0 and not status.finished:
            eta = (time.monotonic() - status.started_monotonic) / acquired * (status.tiles_total - status.tiles_done)
            parts.append(f"~{int(eta // 60)}m {int(eta % 60)}s left")
        self.progress_label.setText(" • ".join(parts))
        self._update_position()
        self._apply_locks()
        self._render_mosaic(force=status.finished)
        if status.finished:
            self.poll_timer.stop()
            self._cycle_finished(status)

    def _render_mosaic(self, force: bool = False) -> None:
        mosaic = self.controller.mosaic
        now = time.monotonic()
        if mosaic is None or mosaic.closed or (not force and now - self._last_mosaic_render < MOSAIC_REFRESH_S):
            return
        self._last_mosaic_render = now
        aspect = mosaic.bounds.width_um / max(mosaic.bounds.height_um, 1e-9)
        width = MOSAIC_UNDERLAY_PX if aspect >= 1 else max(64, int(MOSAIC_UNDERLAY_PX * aspect))
        height = max(64, int(width / aspect))
        rgb = mosaic.render_viewport_rgb(mosaic.bounds, width, height, self.mosaic_mode_box.currentText())
        self.overview.set_mosaic(rgb, mosaic.bounds)

    def _cycle_finished(self, status) -> None:
        experiment = self.controller.experiment
        cycle = self.controller.active_cycle
        folder = experiment.cycle_folder(cycle) if experiment is not None and cycle else None
        self._reset_z_stack_capture()
        self._update_z_stack_view()
        self._refresh()
        if status.state == "complete":
            self.progress_label.setText(f"{status.message} • {folder}")
            self.inform(
                "Cycle complete",
                f"Cycle {cycle} finished: {status.tiles_done:,} tiles.\n\nData:\n{folder}\n\n"
                "Strip/restain, then Align selected F → stage (or Move ROI set) and Refocus all before the next cycle.",
            )
        elif status.state == "aborted":
            self.progress_label.setText(f"Stopped • {status.tiles_done}/{status.tiles_total} tiles saved • resumable")
        else:
            self.progress_label.setText(status.error or "Acquisition failed")
            self.warn("Acquisition stopped", f"{status.error}\n\n{status.tiles_done}/{status.tiles_total} tiles saved. Fix the problem, then Resume cycle.")

    def _toggle_pause(self) -> None:
        status = self.controller.status()
        if status is None:
            return
        if status.state in ("pausing", "paused"):
            self.controller.resume()
        else:
            self.controller.pause()
        self._apply_locks()

    def _abort(self) -> None:
        if self.confirm("Abort acquisition", "Stop after the current frame? Saved tiles are kept and the cycle can be resumed."):
            self.controller.abort()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self.controller.busy and not self.confirm("Quit", "A cycle is running. Abort it and quit?"):
            event.ignore()
            return
        self.live_button.setChecked(False)
        self.stop_spiral_preview()
        self.position_timer.stop()  # no new device jobs while shutting down
        if not self.device.stop_and_drain(timeout=DEVICE_DRAIN_TIMEOUT_S):
            # A camera or stage command is still running on the device worker:
            # closing the backend under it is unsafe, so stay open.
            self.warn(
                "Microscope busy",
                "A camera or stage command is still running. The program stays open; "
                "try again when it has finished.",
            )
            self._refuse_close(event)
            return
        if not self.controller.close(timeout=CLOSE_TIMEOUT_S):
            self.warn(
                "Still saving",
                "The acquisition is stopping but the SSD writer is still saving its current "
                "frame. The program stays open so no data is lost; try again in a moment.",
            )
            self._refuse_close(event)
            return
        self.poll_timer.stop()
        self.device.shutdown()
        event.accept()

    def _refuse_close(self, event) -> None:
        self.position_timer.start()
        if self.controller.engine is not None and not self.poll_timer.isActive():
            self.poll_timer.start()
        event.ignore()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Microscope Scan Console (Qt)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="configuration JSON")
    parser.add_argument("--backend", choices=("simulator", "mmcore"), help="override hardware.backend")
    parser.add_argument("--no-ndv", action="store_true", help="plain image view instead of ndv")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication([])
    try:
        config = load_config(args.config)
        if args.backend:
            config = replace(config, hardware={**config.hardware, "backend": args.backend})
        backend = create_backend(config)
    except (OSError, ValueError, json.JSONDecodeError, HardwareError) as exc:
        QMessageBox.critical(None, "Startup error", str(exc))
        raise SystemExit(1) from exc
    window = ScanWindow(config, backend, use_ndv=not args.no_ndv, config_path=args.config)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
