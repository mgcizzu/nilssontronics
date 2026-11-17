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
from qtpy.QtCore import QSettings, Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
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
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..backend import HardwareError
from ..backend_mmcore import create_backend
from ..config import DEFAULT_CONFIG_PATH, load_config
from ..controller import ControllerError, ScanController
from ..experiment import Experiment
from ..models import (
    AcquisitionOptions,
    AcquisitionPlan,
    ChannelSetting,
    FocusPoint,
    MicroscopeConfig,
    ZSettings,
)
from ..mosaic import DiskBackedMosaic
from ..planner import spiral_preview_positions
from ..roi_io import load_roi_set, save_roi_set
from .device_worker import DeviceWorker
from .overview import OverviewCanvas, bounded_set_translation
from .theme import THEME_MODES, ToggleSwitch, apply_theme, icon, set_dark_title_bar
from .widgets import (
    ChannelTable,
    FrameView,
    ItemList,
    RectangleDialog,
    StageJog,
    describe_focus,
    describe_region,
    spin,
)

POSITION_REFRESH_MS = 1000
CLOSE_TIMEOUT_S = 30.0
DEVICE_DRAIN_TIMEOUT_S = 30.0
POLL_INTERVAL_MS = 150
MOSAIC_REFRESH_S = 0.3
Z_MODES = (("Single plane (current Z)", "single"), ("Focus map", "focus_map"), ("Z-stack", "stack"))
LAUNCH_ERRORS = (ControllerError, HardwareError, OSError, ValueError)
MOSAIC_UNDERLAY_PX = 1024
PREVIEW_TILE_PX = 256
OVERVIEW_MODES = (
    ("navigate", "Navigate", "Drag to pan, wheel to zoom, double-click to move the stage"),
    ("rectangle", "Rectangle", "Drag to draw a rectangular scan region"),
    ("polygon", "Polygon", "Click corners, double-click or Enter to close"),
    ("move_set", "Move ROI set", "Drag to translate all regions and focus points (XY only)"),
    ("focus", "Focus points", "Click a point to go to it, elsewhere to move there"),
)
OVERVIEW_HINTS = {
    "navigate": "Drag to pan • wheel to zoom • double-click moves the stage",
    "rectangle": "Drag to draw a region • Esc cancels",
    "polygon": "Click corners • double-click or Enter closes • Esc cancels",
    "move_set": "Drag to move regions and focus points together • Esc restores",
    "focus": "Click F-point to go there, elsewhere to move XY • focus in Live, then Capture here / Set Z here",
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
        # Per-user view preferences (theme); tests pass their own file.
        self.settings = settings if settings is not None else QSettings("nilssontronics", "scan-console")
        self.config_data = config
        self.config_path = Path(config_path)
        self.backend = backend if backend is not None else create_backend(config)
        self.controller = ScanController(config, self.backend)
        self.device = DeviceWorker(self.backend, self)
        self.z_begin_captured = False
        self.z_end_captured = False
        self._last_mosaic_render = 0.0
        self._loading = False

        self.setWindowTitle(f"Microscope Scan Console — {self.backend.name}")
        self.resize(1400, 860)
        self._build_ui(use_ndv)
        self._build_device_pages()
        self.apply_theme_mode(self.theme_mode())

        self.device.frame_ready.connect(self.frame_view.show_frame)
        self.device.position_changed.connect(self._position_changed)
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
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.pages = QStackedWidget()

        acquire = self._page()
        layout = acquire.layout()
        layout.addWidget(self._build_header())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_setup_panel())
        splitter.addWidget(self._build_viewer(use_ndv))
        splitter.addWidget(self._build_review_panel())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 720, 320])
        layout.addWidget(splitter, 1)

        self.devices_page = self._page("Devices")
        self.config_page = self._page("Configuration")
        self.config_text = QTextBrowser()
        self.config_text.setOpenExternalLinks(False)
        self.config_page.layout().addWidget(self.config_text, 1)
        self.settings_page = self._build_settings_page()

        self.page_names = ("acquire", "devices", "config", "settings")
        for page in (acquire, self.devices_page, self.config_page, self.settings_page):
            self.pages.addWidget(page)
        outer.addWidget(self._build_rail())
        outer.addWidget(self.pages, 1)
        self.setCentralWidget(central)

    def _page(self, title: str | None = None) -> QWidget:
        page = QWidget()
        page.setObjectName("page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(8)
        if title:
            heading = QLabel(title)
            heading.setObjectName("pageTitle")
            layout.addWidget(heading)
        return page

    def _build_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(200)
        rail.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 14, 0, 12)
        layout.setSpacing(2)
        title = QLabel("Scan Console")
        title.setObjectName("railTitle")
        layout.addWidget(title)
        self.nav_group = QButtonGroup(self)
        self.nav_buttons: dict[str, QToolButton] = {}
        entries = (
            ("acquire", "Acquire", "acquire"),
            ("devices", "Devices", "devices"),
            ("config", "Configuration", "config"),
            (None, None, None),
            ("settings", "Settings", "settings"),
        )
        for name, text, icon_name in entries:
            if name is None:
                layout.addStretch(1)
                continue
            button = QToolButton()
            button.setText(f"  {text}")
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setProperty("iconName", icon_name)
            button.clicked.connect(lambda _=False, n=name: self.show_page(n))
            self.nav_group.addButton(button)
            self.nav_buttons[name] = button
            layout.addWidget(button)
        self.nav_buttons["acquire"].setChecked(True)
        return rail

    def show_page(self, name: str) -> None:
        if name == "config":
            self.config_text.setHtml(self.configuration_html())
        self.nav_buttons[name].setChecked(True)
        self.pages.setCurrentIndex(self.page_names.index(name))

    def _build_settings_page(self) -> QWidget:
        page = self._page("Settings")
        appearance = QGroupBox("Appearance")
        form = QFormLayout(appearance)
        self.theme_box = QComboBox()
        for mode in THEME_MODES:
            self.theme_box.addItem({"system": "Use system setting", "light": "Light", "dark": "Dark"}[mode], mode)
        self.theme_box.activated.connect(
            lambda _i: self.apply_theme_mode(self.theme_box.currentData(), remember=True)
        )
        hint = QLabel("Colours follow Windows' dark/light mode and accent colour when set to the system.")
        hint.setProperty("secondary", True)
        hint.setWordWrap(True)
        form.addRow("Theme", self.theme_box)
        form.addRow(hint)
        about = QGroupBox("About")
        about_layout = QVBoxLayout(about)
        text = QLabel(
            "Microscope Scan Console — cyclic imaging front end for Micro-Manager.<br>"
            "MIT licence."
        )
        text.setProperty("secondary", True)
        about_layout.addWidget(text)
        for card in (appearance, about):
            card.setMaximumWidth(860)
        page.layout().addWidget(appearance)
        page.layout().addWidget(about)
        page.layout().addStretch(1)
        return page

    # ---------------------------------------------------------------- theme --
    def theme_mode(self) -> str:
        mode = str(self.settings.value("theme", "system"))
        return mode if mode in THEME_MODES else "system"

    def apply_theme_mode(self, mode: str, remember: bool = False) -> None:
        palette = apply_theme(QApplication.instance(), mode)
        self.palette_tokens = palette
        if remember:
            self.settings.setValue("theme", mode)
            self.settings.sync()
        self.theme_box.setCurrentIndex(self.theme_box.findData(mode))
        for button in self.nav_buttons.values():
            button.setIcon(icon(button.property("iconName"), palette.text))
        self.channel_table.refresh_swatches()
        set_dark_title_bar(self, palette.dark)

    def _build_header(self) -> QWidget:
        header = QWidget()
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 4)
        title = QLabel(self.config_data.microscope_name)
        title.setObjectName("pageTitle")
        mode = "Simulator" if self.backend.simulated else f"{self.backend.name} connected"
        self.mode_label = QLabel(mode)
        self.mode_label.setObjectName("statusPill")
        self.mode_label.setProperty("tone", "sim" if self.backend.simulated else "hw")
        self.position_label = QLabel()
        self.position_label.setProperty("secondary", True)
        self.experiment_label = QLabel()
        self.experiment_label.setProperty("secondary", True)
        row.addWidget(title)
        row.addSpacing(10)
        row.addWidget(self.mode_label)
        row.addStretch(1)
        row.addWidget(self.experiment_label)
        row.addSpacing(16)
        row.addWidget(self.position_label)
        return header

    def _build_setup_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

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
        layout.addWidget(experiment)

        # Sample
        sample = QGroupBox("Sample")
        form = QFormLayout(sample)
        self.project_edit = QLineEdit("Demo project")
        self.sample_edit = QLineEdit("Sample 001")
        self.output_edit = QLineEdit(str(Path.cwd()))
        browse = QPushButton("…")
        browse.setFixedWidth(36)
        browse.setProperty("compact", True)
        browse.setToolTip("Choose the output folder")
        browse.clicked.connect(self._choose_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse)
        form.addRow("Project", self.project_edit)
        form.addRow("Sample", self.sample_edit)
        form.addRow("Output", out_row)
        layout.addWidget(sample)

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
        self.channel_table = ChannelTable(self.config_data.channels)
        self.channel_table.changed.connect(self._recipe_to_custom)
        optics_layout.addWidget(self.objective_box)
        optics_layout.addWidget(self.channel_table)
        layout.addWidget(optics)

        # Regions & focus
        regions = QGroupBox("Scan regions (stage µm)")
        regions_layout = QVBoxLayout(regions)
        self.region_list = ItemList(describe_region)
        self.add_region_button = self.region_list.add_button("Add…", self._add_region)
        self.save_set_button = QPushButton("Save set…")
        self.save_set_button.setToolTip("Save regions and focus points to a JSON file")
        self.save_set_button.clicked.connect(self._save_roi_set)
        self.load_set_button = QPushButton("Load set…")
        self.load_set_button.setToolTip("Replace regions and focus points from a saved set")
        self.load_set_button.clicked.connect(self._load_roi_set)
        self.shift_button = QPushButton("Shift ROI set…")
        self.shift_button.setToolTip("Re-register for this cycle: XY translation of the whole set")
        self.shift_button.clicked.connect(self._shift_roi_set)
        set_row = QHBoxLayout()
        set_row.addWidget(self.save_set_button)
        set_row.addWidget(self.load_set_button)
        set_row.addWidget(self.shift_button)
        regions_layout.addWidget(self.region_list)
        regions_layout.addLayout(set_row)
        layout.addWidget(regions)

        focus = QGroupBox("Focus points")
        focus_layout = QVBoxLayout(focus)
        self.focus_list = ItemList(describe_focus)
        self.focus_list.add_button("Capture here", self._capture_focus_here)
        self.focus_list.add_button("Go to", self._goto_focus)
        self.set_z_button = self.focus_list.add_button("Set Z here", self._set_focus_z_here)
        self.focus_list.remove_button.clicked.connect(lambda: self.stop_refocus("Refocus stopped"))
        self.refocus_button = QPushButton("Refocus all")
        self.refocus_button.setToolTip(
            "Visit every focus point in order: focus in Live, then Accept Z to store it and move on"
        )
        self.refocus_button.clicked.connect(self._toggle_refocus)
        self.align_button = QPushButton("Align selected F → stage")
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
        self.refocus_label.setProperty("tone", "attention")
        self.refocus_label.hide()
        focus_layout.addWidget(self.focus_list)
        focus_layout.addLayout(cycle_row)
        focus_layout.addWidget(self.refocus_label)
        layout.addWidget(focus)
        self.refocus_index: int | None = None
        self._accept_pending = False

        # Z
        z_group = QGroupBox("Z")
        z_form = QFormLayout(z_group)
        self.z_mode_box = QComboBox()
        for label, value in Z_MODES:
            self.z_mode_box.addItem(label, value)
        self.z_mode_box.currentIndexChanged.connect(self._z_mode_changed)
        self.z_bottom = spin(0, -1e5, 1e5, 2, "µm")
        self.z_top = spin(0, -1e5, 1e5, 2, "µm")
        self.z_step = spin(1, 0.01, 1000, 2, "µm")
        self.z_begin_button = QPushButton("Capture Begin")
        self.z_end_button = QPushButton("Capture End")
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
        z_form.addRow("Step", self.z_step)
        self.z_stack_widgets = (self.z_begin_button, self.z_end_button, self.z_bottom, self.z_top, self.z_step)
        layout.addWidget(z_group)

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
        self.flat_field = ToggleSwitch("Estimate flat-field profiles")
        t_form.addRow("Overlap X", self.overlap_x)
        t_form.addRow("Overlap Y", self.overlap_y)
        t_form.addRow("Tile order", self.tile_order_box)
        t_form.addRow(self.consistent_approach)
        t_form.addRow(self.shutter_per_frame)
        t_form.addRow(self.flat_field)
        layout.addWidget(tiling)
        layout.addStretch(1)

        for signal in (
            self.channel_table.changed,
            self.region_list.changed,
            self.focus_list.changed,
            self.z_bottom.valueChanged,
            self.z_top.valueChanged,
            self.z_step.valueChanged,
            self.overlap_x.valueChanged,
            self.overlap_y.valueChanged,
            self.tile_order_box.currentIndexChanged,
            self.focus_method_box.currentIndexChanged,
            self.consistent_approach.toggled,
            self.shutter_per_frame.toggled,
            self.flat_field.toggled,
            self.project_edit.editingFinished,
            self.sample_edit.editingFinished,
            self.output_edit.editingFinished,
        ):
            signal.connect(self._refresh)

        layout.setContentsMargins(0, 0, 8, 0)
        scroll = QScrollArea()
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)
        return scroll

    def _build_viewer(self, use_ndv: bool) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.frame_view = FrameView(use_ndv=use_ndv)

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(0, 4, 0, 0)
        toolbar = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QToolButton] = {}
        for mode, label, tip in OVERVIEW_MODES:
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, m=mode: self.set_overview_mode(m))
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            toolbar.addWidget(button)
        self.mode_buttons["navigate"].setChecked(True)
        fit = QToolButton()
        fit.setText("Fit")
        fit.clicked.connect(self._fit_overview)
        self.mosaic_mode_box = QComboBox()
        self.mosaic_mode_box.addItem(DiskBackedMosaic.COMPOSITE)
        self.mosaic_mode_box.setToolTip("Cycle mosaic display")
        self.mosaic_mode_box.currentIndexChanged.connect(lambda: self._render_mosaic(force=True))
        toolbar.addWidget(fit)
        toolbar.addStretch(1)
        toolbar.addWidget(self.mosaic_mode_box)
        self.overview = OverviewCanvas(self.config_data.stage_limits)
        self.overview_hint = QLabel(OVERVIEW_HINTS["navigate"])
        self.overview_hint.setProperty("secondary", True)
        overview_layout.addLayout(toolbar)
        overview_layout.addWidget(self.overview, 1)
        # Accept button under the canvas, so a refocus walk never needs the setup panel.
        self.overview_accept_button = QPushButton("Accept Z ▶")
        self.overview_accept_button.setProperty("accent", True)
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

        controls = QHBoxLayout()
        self.live_button = QPushButton("Live")
        self.live_button.setCheckable(True)
        self.live_button.toggled.connect(self._toggle_live)
        self.snap_button = QPushButton("Snap")
        self.snap_button.clicked.connect(self._snap)
        self.live_channel_box = QComboBox()
        for channel in self.config_data.channels:
            self.live_channel_box.addItem(channel.name, channel.id)
        self.jog = StageJog()
        self.jog.move_requested.connect(self._jog)
        self.preview_button = QPushButton("Spiral preview")
        self.preview_button.setCheckable(True)
        self.preview_button.setToolTip(
            "Snap the Live channel at each tile position, spiralling out from the stage "
            "(over the regions if any, else a 5×5 grid). Current Z only, no Z moves."
        )
        self.preview_button.toggled.connect(self._toggle_spiral_preview)
        controls.addWidget(self.live_button)
        controls.addWidget(self.snap_button)
        controls.addWidget(self.live_channel_box)
        controls.addWidget(self.preview_button)
        controls.addStretch(1)
        controls.addWidget(self.jog)
        layout.addLayout(controls)
        return wrapper

    def _build_review_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        summary = QGroupBox("Cycle summary")
        s_layout = QVBoxLayout(summary)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.safety_label = QLabel()
        self.safety_label.setWordWrap(True)
        s_layout.addWidget(self.summary_label)
        s_layout.addWidget(self.safety_label)
        layout.addWidget(summary)

        self.start_button = QPushButton("Start cycle")
        self.start_button.setMinimumHeight(40)
        self.start_button.setProperty("accent", True)
        self.start_button.clicked.connect(self._start_cycle)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        self.abort_button = QPushButton("Abort")
        self.abort_button.clicked.connect(self._abort)
        run_row = QHBoxLayout()
        run_row.addWidget(self.pause_button)
        run_row.addWidget(self.abort_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress_label = QLabel()
        self.progress_label.setWordWrap(True)
        self.progress_label.setProperty("secondary", True)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.addWidget(self.start_button)
        layout.addLayout(run_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_label)
        layout.addStretch(1)
        panel.setMinimumWidth(280)
        return panel

    def _build_device_pages(self) -> None:
        """Micro-Manager device panels (pymmcore-widgets) on the Devices page, MMCore only."""

        self.device_panels: list[QWidget] = []
        layout = self.devices_page.layout()
        core = getattr(self.backend, "core", None)
        widgets = None
        if core is not None and not self.backend.simulated:
            try:
                from pymmcore_widgets import CoreLogWidget, PropertyBrowser

                widgets = (
                    ("Device properties (admin)", PropertyBrowser(mmcore=core)),
                    ("Micro-Manager log", CoreLogWidget(mmcore=core)),
                )
            except ImportError:  # pragma: no cover
                widgets = None
        if not widgets:
            note = QLabel(
                "Device panels appear here when the Micro-Manager backend is connected "
                "(start with --backend mmcore). The simulator has no devices to inspect."
            )
            note.setProperty("secondary", True)
            note.setWordWrap(True)
            layout.addWidget(note)
            layout.addStretch(1)
            return
        for title, widget in widgets:
            card = QGroupBox(title)
            QVBoxLayout(card).addWidget(widget)
            layout.addWidget(card, 1)
            self.device_panels.append(widget)

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
        self.show_page("config")

    # ------------------------------------------------------- dialogs (seam) --
    def confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def inform(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    # ----------------------------------------------------------------- plan --
    def build_plan(self) -> AcquisitionPlan:
        z = ZSettings(
            mode=self.z_mode_box.currentData(),
            bottom_um=self.z_bottom.value(),
            top_um=self.z_top.value(),
            step_um=self.z_step.value(),
            begin_captured=self.z_begin_captured,
            end_captured=self.z_end_captured,
        )
        options = AcquisitionOptions(
            tile_order=self.tile_order_box.currentText(),
            consistent_approach=self.consistent_approach.isChecked(),
            shutter_per_frame=self.shutter_per_frame.isChecked(),
            flat_field=self.flat_field.isChecked(),
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
        self._apply_locks()
        if self.controller.busy:
            self._update_overview()
            return
        try:
            plan = self.build_plan()
            tiles, estimate = self.controller.estimate(plan)
            problems = self.controller.preflight(plan)
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
            self._set_tone(self.safety_label, "bad")
            self.safety_label.setText("● " + "<br>● ".join(problems[:5]))
        else:
            self._set_tone(self.safety_label, "ok")
            self.safety_label.setText("✓ Plan passes configured safety checks")

    @staticmethod
    def _set_tone(label: QLabel, tone: str) -> None:
        if label.property("tone") != tone:
            label.setProperty("tone", tone)
            label.style().unpolish(label)
            label.style().polish(label)

    def _apply_locks(self) -> None:
        busy = self.controller.busy
        locked = self.controller.experiment is not None  # objective/ROI/overlap fixed by cycle 1
        status = self.controller.status()
        for widget in (self.new_button, self.open_button, self.project_edit, self.sample_edit,
                       self.output_edit, self.channel_table, self.focus_method_box,
                       self.z_mode_box, self.consistent_approach, self.shutter_per_frame,
                       self.flat_field, self.snap_button, self.live_button, self.jog,
                       self.preview_button):
            widget.setEnabled(not busy)
        for widget in (self.objective_box, self.overlap_x, self.overlap_y, self.tile_order_box):
            widget.setEnabled(not busy and not locked)
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
        for panel in getattr(self, "device_panels", ()):
            panel.setEnabled(not busy)

    def _update_position(self) -> None:
        if self.controller.busy:
            status = self.controller.status()
            tile = status.current_tile if status else None
            if tile is not None:
                self.position_label.setText(f"Tile {tile.index + 1}  X {tile.x_um:.1f}  Y {tile.y_um:.1f} µm")
            return
        p = self.device.last_position
        self.position_label.setText(f"X {p.x_um:.1f}  Y {p.y_um:.1f}  Z {p.z_um:.2f} µm")

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

    def _objective_selected(self, _index: int) -> None:
        self._recipe_to_custom()
        objective = self.config_data.objective(self.objective_box.currentData())
        previous = self.config_data.objective(self.backend.current_objective_id)
        if objective.id == previous.id:
            return
        if objective.immersion != previous.immersion and not self.confirm(
            "Immersion change",
            f"Changing from {previous.name} ({previous.immersion}) to {objective.name} "
            f"({objective.immersion}). Apply/remove immersion medium now. Z moves to a safe "
            "clearance before the turret rotates. Continue?",
        ):
            self._sync_objective_box()
            return
        self.stop_spiral_preview()
        self.overview.clear_preview()  # preview tiles were taken with the old objective
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
                self._objective_selected(self.objective_box.currentIndex())
        finally:
            self._applying_recipe = False
        self.recipe_box.setCurrentText(name)
        self._refresh()

    def _z_mode_changed(self, *_args) -> None:
        stack = self.z_mode_box.currentData() == "stack"
        for widget in self.z_stack_widgets:
            widget.setVisible(stack)
        self.focus_method_box.setEnabled(self.z_mode_box.currentData() == "focus_map")
        self._refresh()

    def _capture_z_begin(self) -> None:
        def capture(p) -> None:
            self.z_bottom.setValue(p.z_um)
            self.z_begin_captured = True
            self._refresh()

        self.with_fresh_position(capture)

    def _capture_z_end(self) -> None:
        def capture(p) -> None:
            self.z_top.setValue(p.z_um)
            self.z_end_captured = True
            self._refresh()

        self.with_fresh_position(capture)

    def _reset_z_stack_capture(self) -> None:
        self.z_begin_captured = self.z_end_captured = False

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
            self.overview_hint.setText(f"Going to F{index + 1} — refocus in Live, then Set Z here")
        else:
            self.focus_list.list.setCurrentRow(-1)
            self.device_move(self.backend.move_xy, x_um, y_um)
            self.overview_hint.setText("Moving — focus in Live, then Capture here")
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

    def with_fresh_position(self, callback) -> None:
        """Run ``callback(position)`` after every queued stage move has finished.

        The cached position can lag behind a jog that is still queued; reading it
        through the worker queue guarantees the value the operator just focused at.
        """

        if self.controller.busy:
            return
        self.device.submit(
            lambda: self.backend.position,
            on_done=callback,
            on_error=lambda exc: self.warn("Could not read the stage position", str(exc)),
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

        self.with_fresh_position(accept)

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
        positions = [
            p
            for p in spiral_preview_positions(
                self.config_data,
                objective_id,
                tuple(self.region_list.items),
                origin,
                overlap_x_percent=self.overlap_x.value(),
                overlap_y_percent=self.overlap_y.value(),
            )
            if limits.contains_xy(*p)
        ]
        if not positions:
            self.warn("No preview positions", "The preview area has no positions inside the stage limits.")
            self.preview_button.setChecked(False)
            return False
        self.live_button.setChecked(False)
        self._push_live_settings()
        self._preview_generation = getattr(self, "_preview_generation", 0) + 1
        self._preview_positions = positions
        self._preview_index = 0
        self._preview_range: tuple[float, float] | None = None
        self._preview_fov = self.config_data.fov_um(objective_id)
        self._preview_active = True
        self.overview.clear_preview()
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
        done = self.overview.preview_tile_count
        self.overview_hint.setText(
            message or f"Spiral preview stopped after {done} of {len(self._preview_positions)} tiles"
        )

    def _preview_next(self, generation: int) -> None:
        if generation != self._preview_generation:
            return
        if self._preview_index >= len(self._preview_positions):
            self.stop_spiral_preview(
                f"Spiral preview complete • {len(self._preview_positions)} tiles • current Z only"
            )
            return
        x, y = self._preview_positions[self._preview_index]
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
        self.overview_hint.setText(
            f"{channel.name} spiral preview • tile {self._preview_index} of "
            f"{len(self._preview_positions)} • current Z only"
        )
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
        if self.refocus_index is None:
            return
        self.refocus_index = None
        self.refocus_button.setText("Refocus all")
        self.set_z_button.setText("Set Z here")
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
            self.tabs.setCurrentWidget(self.frame_view)
        self.device.set_live(on)

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
                options = last.options or experiment.options
                self.focus_method_box.setCurrentText(options.focus_method)
                self.shutter_per_frame.setChecked(options.shutter_per_frame)
                self.consistent_approach.setChecked(options.consistent_approach)
                self.flat_field.setChecked(options.flat_field)
            self.objective_box.setCurrentIndex(self.objective_box.findData(experiment.objective_id))
        finally:
            self._loading = False
        if self.backend.current_objective_id != experiment.objective_id:
            self._objective_selected(self.objective_box.currentIndex())
        self.progress_label.setText(f"Experiment opened • {experiment.root}")
        self._z_mode_changed()

    # -------------------------------------------------------------- running --
    def _start_cycle(self) -> None:
        self.stop_refocus()
        self.live_button.setChecked(False)
        try:
            plan = self.build_plan()
            problems = self.controller.preflight(plan)
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
