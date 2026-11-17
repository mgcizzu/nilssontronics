from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import random
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .backend import HardwareError
from .backend_mmcore import create_backend
from .config import DEFAULT_CONFIG_PATH, load_config
from .controller import ControllerError, ScanController
from .experiment import Experiment
from .models import (
    AcquisitionOptions,
    AcquisitionPlan,
    ChannelSetting,
    FocusPoint,
    MicroscopeConfig,
    ScanRegion,
    ZSettings,
)
from .mosaic import DiskBackedMosaic, MosaicBounds
from .optics import automatic_z_stack, recommended_z_step_um
from .planner import spiral_preview_positions
from .roi_io import load_roi_set, save_roi_set

LIVE_INTERVAL_MS = 150


BG = "#eef3f5"
PANEL = "#ffffff"
INK = "#17252d"
MUTED = "#61727c"
ACCENT = "#087e8b"
ACCENT_DARK = "#075d66"
VIEW_BG = "#07151d"
SUCCESS = "#237a57"
WARNING = "#b66a18"


class ScanConsole(tk.Tk):
    def __init__(self, config: MicroscopeConfig, backend=None) -> None:
        super().__init__()
        self.config_data = config
        self.backend = backend if backend is not None else create_backend(config)
        self.controller = ScanController(config, self.backend)
        self.config_path = DEFAULT_CONFIG_PATH
        self.title(f"Microscope Scan Console — {self.backend.name}")
        self.geometry("1440x900")
        self.minsize(1160, 760)
        self.configure(bg=BG)

        self.regions: list[ScanRegion] = []
        self.focus_points: list[FocusPoint] = []
        self.selected_focus_index: int | None = None
        self.polygon_draft: list[tuple[float, float]] = []
        self.move_start_stage: tuple[float, float] | None = None
        self.move_original_regions: tuple[ScanRegion, ...] = ()
        self.move_original_focus_points: tuple[FocusPoint, ...] = ()
        self.overview_mode = True
        self.live = False  # Live snaps the real camera; it starts only when asked
        self.live_job: str | None = None
        self.last_frame: np.ndarray | None = None
        self.last_frame_xy: tuple[float, float] = (0.0, 0.0)
        self.live_photo: tk.PhotoImage | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_rectangle_id: int | None = None
        self.preview_job: str | None = None
        self.preview_positions: list[tuple[float, float]] = []
        self.preview_completed: list[tuple[float, float]] = []
        self.preview_index = 0
        self.preview_channel_id: str | None = None
        self.preview_view: tuple[float, float, float, float] | None = None
        self.view_zoom = 1.0
        self.view_center_override: tuple[float, float] | None = None
        self.progress_job: str | None = None
        self.mosaic_photo: tk.PhotoImage | None = None
        self.last_mosaic_render_time = 0.0
        self.refocus_index: int | None = None

        self.objective_var = tk.StringVar()
        self.recipe_var = tk.StringVar(value="Custom")
        self.z_mode_var = tk.StringVar(value="single")
        self.z_bottom_var = tk.StringVar(value="—")
        self.z_top_var = tk.StringVar(value="—")
        self.z_step_var = tk.StringVar(value="—")
        self.z_stack_summary_var = tk.StringVar(value="Set Begin and End from live Z")
        self.z_begin_captured = False
        self.z_end_captured = False
        self.overlap_x_var = tk.DoubleVar(value=config.overlap_x_percent)
        self.overlap_y_var = tk.DoubleVar(value=config.overlap_y_percent)
        self.interaction_mode_var = tk.StringVar(value="rectangle")
        self.viewer_hint_var = tk.StringVar(
            value="Drag to draw a rectangle • Double-click to move the stage"
        )
        self.project_var = tk.StringVar(value="Demo project")
        self.sample_var = tk.StringVar(value="Sample 001")
        self.output_var = tk.StringVar(value=str(Path.cwd()))
        self.idle_status = (
            "SIMULATOR CONNECTED" if self.backend.simulated else f"{self.backend.name.upper()} CONNECTED"
        )
        self.status_var = tk.StringVar(value=self.idle_status)
        self.position_var = tk.StringVar()
        self.region_var = tk.StringVar(value="No scan regions")
        self.focus_status_var = tk.StringVar(value="No focus point selected")
        self.preview_channel_var = tk.StringVar()
        self.mosaic_view_var = tk.StringVar(value=DiskBackedMosaic.COMPOSITE)
        self.summary_var = tk.StringVar(value="Draw a region to calculate the scan.")
        self.safety_var = tk.StringVar(value="Waiting for a scan region")
        self.progress_text_var = tk.StringVar(value="Ready")
        self.channel_vars: dict[str, tk.BooleanVar] = {}
        self.exposure_vars: dict[str, tk.DoubleVar] = {}
        self.power_vars: dict[str, tk.DoubleVar] = {}
        self.tile_order_var = tk.StringVar(value="raster")
        self.focus_method_var = tk.StringVar(value="linear")
        self.consistent_approach_var = tk.BooleanVar(value=False)
        self.shutter_per_frame_var = tk.BooleanVar(value=True)
        self.flat_field_var = tk.BooleanVar(value=False)
        self.experiment_var = tk.StringVar(value="No experiment — the first scan creates one")
        self.offset_var = tk.StringVar(value="")

        self._configure_styles()
        self._build_header()
        self._build_workspace()
        self._build_footer()
        self._load_initial_state()
        self.after(120, self._render_scene)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("FocusZone.TFrame", background="#e6f3f1")
        style.configure("FocusZone.TLabel", background="#e6f3f1", foreground=ACCENT_DARK)
        style.configure("Section.TLabelframe", background=PANEL, bordercolor="#dce5e8")
        style.configure(
            "Section.TLabelframe.Label",
            background=PANEL,
            foreground=INK,
            font=("TkDefaultFont", 10, "bold"),
        )
        style.configure("Panel.TLabel", background=PANEL, foreground=INK)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Header.TLabel", background=INK, foreground="white")
        style.configure(
            "Title.TLabel", background=INK, foreground="white", font=("TkDefaultFont", 18, "bold")
        )
        style.configure(
            "Primary.TButton",
            background=ACCENT,
            foreground="white",
            bordercolor=ACCENT,
            font=("TkDefaultFont", 11, "bold"),
            padding=(16, 11),
        )
        style.map("Primary.TButton", background=[("active", ACCENT_DARK), ("disabled", "#9eb4b7")])
        style.configure("Tool.TButton", padding=(9, 6))
        style.configure("Status.Horizontal.TProgressbar", troughcolor="#dce5e8", background=ACCENT)
        style.configure("TCheckbutton", background=PANEL)
        style.configure("TRadiobutton", background=PANEL)

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=INK, height=70)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        title = ttk.Label(header, text="Microscope Scan Console", style="Title.TLabel")
        title.grid(row=0, column=0, padx=(22, 12), pady=(13, 0), sticky="w")
        ttk.Label(
            header,
            text="Workflow prototype",
            style="Header.TLabel",
            font=("TkDefaultFont", 9),
        ).grid(row=1, column=0, padx=(23, 12), pady=(0, 12), sticky="w")

        status = tk.Label(
            header,
            textvariable=self.status_var,
            bg="#123b3f",
            fg="#7de2c2",
            font=("TkDefaultFont", 9, "bold"),
            padx=12,
            pady=7,
        )
        status.grid(row=0, column=2, rowspan=2, padx=8)
        ttk.Button(header, text="Simulator & configuration", command=self._show_admin).grid(
            row=0, column=3, rowspan=2, padx=(8, 20), pady=17
        )

    def _build_workspace(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        workspace = ttk.Frame(self, style="App.TFrame", padding=12)
        workspace.grid(row=1, column=0, sticky="nsew")
        workspace.grid_rowconfigure(0, weight=1)
        workspace.grid_columnconfigure(1, weight=1)

        # The setup column scrolls so every section fits on smaller screens.
        left_container = ttk.Frame(workspace, style="Panel.TFrame", width=345)
        left_container.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        left_container.grid_propagate(False)
        left_container.grid_rowconfigure(0, weight=1)
        left_canvas = tk.Canvas(left_container, bg=PANEL, highlightthickness=0, width=330)
        left_scroll = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.grid(row=0, column=0, sticky="ns")
        left_scroll.grid(row=0, column=1, sticky="ns")
        left = ttk.Frame(left_canvas, style="Panel.TFrame", padding=12, width=330)
        left_window = left_canvas.create_window(0, 0, window=left, anchor="nw")
        left.bind(
            "<Configure>",
            lambda _e: left_canvas.configure(scrollregion=left_canvas.bbox("all")),
        )
        left_canvas.bind(
            "<Configure>", lambda e: left_canvas.itemconfigure(left_window, width=e.width)
        )
        self.left_canvas = left_canvas

        center = ttk.Frame(workspace, style="Panel.TFrame", padding=10)
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_rowconfigure(3, weight=1)
        center.grid_columnconfigure(0, weight=1)

        right = ttk.Frame(workspace, style="Panel.TFrame", padding=14, width=310)
        right.grid(row=0, column=2, sticky="nse", padx=(10, 0))
        right.grid_propagate(False)
        right.pack_propagate(False)

        self._build_setup_panel(left)
        self._build_viewer(center)
        self._build_review_panel(right)

    def _section(self, parent: tk.Misc, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=9)
        frame.pack(fill="x", pady=(0, 8))
        return frame

    def _build_setup_panel(self, parent: ttk.Frame) -> None:
        recipe = self._section(parent, "Recipe")
        recipe_names = ["Custom", *(item.name for item in self.config_data.recipes)]
        recipe_box = ttk.Combobox(
            recipe,
            textvariable=self.recipe_var,
            values=recipe_names,
            state="readonly",
        )
        recipe_box.pack(fill="x")
        recipe_box.bind("<<ComboboxSelected>>", self._apply_recipe)

        objective = self._section(parent, "1  Objective")
        objective_names = [item.name for item in self.config_data.objectives]
        self.objective_box = ttk.Combobox(
            objective,
            textvariable=self.objective_var,
            values=objective_names,
            state="readonly",
        )
        self.objective_box.pack(fill="x")
        self.objective_box.bind("<<ComboboxSelected>>", self._objective_changed)
        self.objective_detail_label = ttk.Label(objective, style="Muted.TLabel", wraplength=285)
        self.objective_detail_label.pack(fill="x", pady=(5, 0))
        self.objective_lock_widgets: list[tk.Misc] = [self.objective_box, recipe_box]
        overlap_row = ttk.Frame(objective, style="Panel.TFrame")
        overlap_row.pack(fill="x", pady=(7, 0))
        ttk.Label(
            overlap_row,
            text="Tile overlap",
            style="Muted.TLabel",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(side="left")
        for axis, variable in (("X", self.overlap_x_var), ("Y", self.overlap_y_var)):
            ttk.Label(overlap_row, text=axis, style="Muted.TLabel").pack(
                side="left", padx=(12, 2)
            )
            overlap_entry = ttk.Spinbox(
                overlap_row,
                from_=0,
                to=50,
                increment=1,
                textvariable=variable,
                width=4,
                justify="right",
                format="%.0f",
                command=self._input_changed,
            )
            overlap_entry.pack(side="left")
            self.objective_lock_widgets.append(overlap_entry)
            overlap_entry.bind("<FocusOut>", self._input_changed)
            overlap_entry.bind("<Return>", self._input_changed)
            ttk.Label(overlap_row, text="%", style="Muted.TLabel").pack(side="left")

        channels = self._section(parent, "2  Channels")
        header = ttk.Frame(channels, style="Panel.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Channel", style="Muted.TLabel").pack(side="left")
        ttk.Label(header, text="Exposure ms · Power %", style="Muted.TLabel").pack(side="right")
        for channel in self.config_data.channels:
            row = ttk.Frame(channels, style="Panel.TFrame")
            row.pack(fill="x", pady=1)
            selected = tk.BooleanVar(value=channel.id in {"dapi", "af488"})
            exposure = tk.DoubleVar(value=channel.default_exposure_ms)
            exposure_label = tk.StringVar(value=f"{channel.default_exposure_ms:g}")
            power = tk.DoubleVar(value=channel.default_power_percent)
            self.channel_vars[channel.id] = selected
            self.exposure_vars[channel.id] = exposure
            self.power_vars[channel.id] = power
            check = ttk.Checkbutton(
                row,
                text=channel.name,
                variable=selected,
                command=self._channel_selection_changed,
            )
            check.pack(side="left")
            swatch = tk.Label(row, bg=channel.color, width=2, height=1)
            swatch.pack(side="left", padx=(2, 5))
            power_box = ttk.Spinbox(
                row,
                from_=0,
                to=channel.max_power_percent,
                increment=1,
                textvariable=power,
                width=3,
                justify="right",
                format="%.0f",
                command=self._input_changed,
            )
            power_box.pack(side="right", padx=(3, 0))
            power_box.bind("<FocusOut>", self._input_changed)
            power_box.bind("<Return>", self._input_changed)
            value_label = ttk.Label(
                row,
                textvariable=exposure_label,
                style="Panel.TLabel",
                width=6,
                anchor="e",
            )
            value_label.pack(side="right")
            slider = ttk.Scale(
                row,
                from_=1,
                to=channel.max_exposure_ms,
                variable=exposure,
                length=96,
                command=lambda value, display=exposure_label: self._exposure_changed(
                    value, display
                ),
            )
            slider.pack(side="right", fill="x", expand=True, padx=(2, 3))

        z_section = self._section(parent, "3  Z acquisition")
        mode_row = ttk.Frame(z_section, style="Panel.TFrame")
        mode_row.pack(fill="x")
        for value, label in (
            ("single", "Single"),
            ("stack", "Z-stack"),
            ("focus_map", "Surface map"),
        ):
            ttk.Radiobutton(
                mode_row,
                text=label,
                value=value,
                variable=self.z_mode_var,
                command=self._z_mode_changed,
            ).pack(side="left", padx=(0, 8))
        self.z_fields = ttk.Frame(z_section, style="Panel.TFrame")
        self.z_fields.pack(fill="x", pady=(5, 0))
        self.z_capture_buttons: list[ttk.Button] = []
        for column, (label, variable, command) in enumerate(
            (
                ("Begin", self.z_bottom_var, self._capture_z_begin),
                ("End", self.z_top_var, self._capture_z_end),
            )
        ):
            field = ttk.Frame(self.z_fields, style="Panel.TFrame")
            field.grid(row=0, column=column, padx=(0 if column == 0 else 5, 0), sticky="ew")
            ttk.Label(field, text=f"{label} (µm)", style="Muted.TLabel").pack(anchor="w")
            value_row = ttk.Frame(field, style="Panel.TFrame")
            value_row.pack(fill="x")
            ttk.Entry(value_row, textvariable=variable, width=7, state="readonly").pack(
                side="left", fill="x", expand=True
            )
            button = ttk.Button(value_row, text="Set", width=4, command=command)
            button.pack(side="left", padx=(3, 0))
            self.z_capture_buttons.append(button)
            self.z_fields.grid_columnconfigure(column, weight=1)
        ttk.Label(
            z_section,
            textvariable=self.z_stack_summary_var,
            style="Muted.TLabel",
            wraplength=285,
        ).pack(fill="x", pady=(5, 0))
        options = self._section(parent, "4  Options")
        order_row = ttk.Frame(options, style="Panel.TFrame")
        order_row.pack(fill="x")
        ttk.Label(order_row, text="Tile order", style="Muted.TLabel").pack(side="left")
        for value, label in (("raster", "Raster"), ("serpentine", "Serpentine")):
            button = ttk.Radiobutton(
                order_row,
                text=label,
                value=value,
                variable=self.tile_order_var,
                command=self._input_changed,
            )
            button.pack(side="left", padx=(6, 0))
            self.objective_lock_widgets.append(button)
        surface_row = ttk.Frame(options, style="Panel.TFrame")
        surface_row.pack(fill="x", pady=(2, 0))
        ttk.Label(surface_row, text="Surface", style="Muted.TLabel").pack(side="left")
        for value, label in (("linear", "Triangulated"), ("plane", "Plane fit")):
            ttk.Radiobutton(
                surface_row,
                text=label,
                value=value,
                variable=self.focus_method_var,
                command=self._input_changed,
            ).pack(side="left", padx=(6, 0))
        for text, variable in (
            ("Close shutter during moves", self.shutter_per_frame_var),
            ("Consistent stage approach (backlash)", self.consistent_approach_var),
            ("Flat-field profile after each cycle", self.flat_field_var),
        ):
            ttk.Checkbutton(
                options, text=text, variable=variable, command=self._input_changed
            ).pack(anchor="w")
        save = self._section(parent, "5  Save")
        self._labeled_entry(save, "Project", self.project_var)
        self._labeled_entry(save, "Sample", self.sample_var)
        folder = ttk.Frame(save, style="Panel.TFrame")
        folder.pack(fill="x", pady=(4, 0))
        ttk.Entry(folder, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(folder, text="Choose…", command=self._choose_output).pack(side="left", padx=(5, 0))

    def _labeled_entry(self, parent: tk.Misc, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Muted.TLabel", width=8).pack(side="left")
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<FocusOut>", self._input_changed)

    def _build_viewer(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent, style="Panel.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.live_button = ttk.Button(toolbar, text="● Live", style="Tool.TButton", command=self._toggle_live)
        self.live_button.pack(side="left")
        ttk.Button(toolbar, text="Snap", style="Tool.TButton", command=self._snap).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Overview", style="Tool.TButton", command=self._overview).pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(toolbar, text="←", width=3, command=lambda: self._nudge_stage(-1, 0)).pack(side="left")
        ttk.Button(toolbar, text="↑", width=3, command=lambda: self._nudge_stage(0, -1)).pack(side="left")
        ttk.Button(toolbar, text="↓", width=3, command=lambda: self._nudge_stage(0, 1)).pack(side="left")
        ttk.Button(toolbar, text="→", width=3, command=lambda: self._nudge_stage(1, 0)).pack(side="left")
        ttk.Label(toolbar, textvariable=self.position_var, style="Muted.TLabel").pack(side="right")

        selection_bar = ttk.Frame(parent, style="Panel.TFrame")
        selection_bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(selection_bar, text="Draw:", style="Muted.TLabel").pack(side="left", padx=(0, 4))
        self.draw_shape_buttons: list[ttk.Radiobutton] = []
        for value, label in (
            ("rectangle", "Rectangle"),
            ("polygon", "Irregular polygon"),
            ("focus", "Focus points"),
            ("move_set", "Move ROI set"),
        ):
            button = ttk.Radiobutton(
                selection_bar,
                text=label,
                value=value,
                variable=self.interaction_mode_var,
                command=self._interaction_mode_changed,
            )
            button.pack(side="left", padx=(0, 6))
            if value in ("rectangle", "polygon"):
                self.draw_shape_buttons.append(button)
        self.preview_button = ttk.Button(
            selection_bar,
            text="Spiral preview",
            command=self._toggle_spiral_preview,
        )
        self.preview_button.pack(side="right")
        self.preview_channel_box = ttk.Combobox(
            selection_bar,
            textvariable=self.preview_channel_var,
            state="readonly",
            width=13,
        )
        self.preview_channel_box.pack(side="right", padx=(0, 5))

        focus_bar = ttk.Frame(parent, style="FocusZone.TFrame", padding=(7, 4))
        focus_bar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(focus_bar, text="Live Z (wheel):", style="FocusZone.TLabel").pack(side="left")
        for label, delta in (("−10", -10), ("−1", -1), ("+1", 1), ("+10", 10)):
            ttk.Button(
                focus_bar,
                text=label,
                width=4,
                command=lambda amount=delta: self._jog_z(amount),
            ).pack(side="left", padx=(4, 0))
        ttk.Label(focus_bar, textvariable=self.focus_status_var, style="FocusZone.TLabel").pack(
            side="left", padx=10
        )
        ttk.Button(focus_bar, text="Delete", width=6, command=self._delete_selected_focus).pack(side="right")
        self.update_z_button = ttk.Button(
            focus_bar,
            text="Update Z",
            width=9,
            command=self._update_selected_focus_z,
        )
        self.update_z_button.pack(side="right", padx=5)
        self.refocus_button = ttk.Button(
            focus_bar,
            text="Refocus all",
            width=10,
            command=self._toggle_refocus,
        )
        self.refocus_button.pack(side="right", padx=(0, 5))
        ttk.Button(
            focus_bar,
            text="Capture XYZ",
            width=10,
            command=self._capture_focus_here,
        ).pack(side="right")
        self.focus_wheel_zone = focus_bar
        self._bind_z_wheel(focus_bar)

        viewer_frame = tk.Frame(parent, bg=VIEW_BG, highlightthickness=0)
        viewer_frame.grid(row=3, column=0, sticky="nsew")
        viewer_frame.grid_rowconfigure(0, weight=1)
        viewer_frame.grid_columnconfigure(0, weight=1)
        self.viewer = tk.Canvas(
            viewer_frame,
            bg=VIEW_BG,
            cursor="crosshair",
            highlightthickness=0,
        )
        self.viewer.grid(row=0, column=0, sticky="nsew")
        self.viewer.bind("<ButtonPress-1>", self._start_region)
        self.viewer.bind("<B1-Motion>", self._drag_region)
        self.viewer.bind("<ButtonRelease-1>", self._finish_region)
        self.viewer.bind("<Double-Button-1>", self._viewer_double_click)
        self.viewer.bind("<Button-3>", self._finish_polygon)
        self.viewer.bind("<MouseWheel>", self._mousewheel_zoom)
        self.viewer.bind("<Button-4>", self._mousewheel_zoom)
        self.viewer.bind("<Button-5>", self._mousewheel_zoom)
        self.viewer.bind("<Configure>", lambda _event: self._render_scene())
        self.bind("<Escape>", self._cancel_draft)

        hint = tk.Label(
            viewer_frame,
            textvariable=self.viewer_hint_var,
            bg="#102832",
            fg="#c9d9de",
            padx=12,
            pady=7,
        )
        hint.grid(row=0, column=0, sticky="sw", padx=12, pady=12)

        region_bar = ttk.Frame(parent, style="Panel.TFrame")
        region_bar.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        info_row = ttk.Frame(region_bar, style="Panel.TFrame")
        info_row.pack(side="top", fill="x", pady=(0, 4))
        ttk.Label(info_row, textvariable=self.region_var, style="Panel.TLabel").pack(side="left")
        ttk.Label(info_row, textvariable=self.offset_var, style="Muted.TLabel").pack(
            side="right"
        )
        self.clear_button = ttk.Button(region_bar, text="Clear all", command=self._clear_regions)
        self.clear_button.pack(side="right")
        ttk.Button(region_bar, text="Save ROI set…", command=self._save_roi_set).pack(
            side="right", padx=5
        )
        self.load_roi_button = ttk.Button(
            region_bar, text="Load ROI set…", command=self._load_roi_set
        )
        self.load_roi_button.pack(side="right")
        ttk.Button(
            region_bar,
            text="Align selected F → stage",
            command=self._align_selected_focus_to_stage,
        ).pack(side="left")

    def _build_review_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Review & scan",
            style="Panel.TLabel",
            font=("TkDefaultFont", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            parent,
            text="The plan is checked before any hardware moves.",
            style="Muted.TLabel",
            wraplength=265,
        ).pack(anchor="w", pady=(2, 6))

        experiment = ttk.LabelFrame(parent, text="Experiment", style="Section.TLabelframe", padding=8)
        experiment.pack(fill="x", pady=(0, 10))
        ttk.Label(
            experiment,
            textvariable=self.experiment_var,
            style="Panel.TLabel",
            wraplength=250,
            justify="left",
        ).pack(anchor="w")
        experiment_buttons = ttk.Frame(experiment, style="Panel.TFrame")
        experiment_buttons.pack(fill="x", pady=(6, 0))
        self.new_experiment_button = ttk.Button(
            experiment_buttons, text="New", width=6, command=self._new_experiment
        )
        self.new_experiment_button.pack(side="left")
        self.open_experiment_button = ttk.Button(
            experiment_buttons, text="Open…", width=7, command=self._open_experiment
        )
        self.open_experiment_button.pack(side="left", padx=4)
        self.resume_button = ttk.Button(
            experiment_buttons, text="Resume cycle", command=self._resume_cycle
        )
        self.resume_button.pack(side="left")

        summary = tk.Frame(parent, bg="#f3f7f8", padx=12, pady=12)
        summary.pack(fill="x")
        tk.Label(
            summary,
            textvariable=self.summary_var,
            bg="#f3f7f8",
            fg=INK,
            justify="left",
            anchor="nw",
            wraplength=245,
            font=("TkDefaultFont", 11),
        ).pack(fill="x")

        ttk.Separator(parent).pack(fill="x", pady=10)
        ttk.Label(parent, text="Preflight", style="Panel.TLabel", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.safety_label = tk.Label(
            parent,
            textvariable=self.safety_var,
            bg=PANEL,
            fg=WARNING,
            justify="left",
            anchor="nw",
            wraplength=265,
        )
        self.safety_label.pack(fill="x", pady=(5, 12))

        self.progress = ttk.Progressbar(
            parent,
            mode="determinate",
            maximum=100,
            style="Status.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(4, 5))
        ttk.Label(
            parent,
            textvariable=self.progress_text_var,
            style="Muted.TLabel",
            wraplength=275,
            justify="left",
        ).pack(anchor="w", fill="x")

        display_frame = ttk.Frame(parent, style="Panel.TFrame")
        display_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(
            display_frame,
            text="Scan mosaic display",
            style="Panel.TLabel",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w")
        self.mosaic_view_box = ttk.Combobox(
            display_frame,
            textvariable=self.mosaic_view_var,
            values=(DiskBackedMosaic.COMPOSITE,),
            state="readonly",
        )
        self.mosaic_view_box.pack(fill="x", pady=(5, 0))
        self.mosaic_view_box.bind("<<ComboboxSelected>>", self._mosaic_view_changed)

        spacer = ttk.Frame(parent, style="Panel.TFrame")
        spacer.pack(fill="both", expand=True)
        run_controls = ttk.Frame(parent, style="Panel.TFrame")
        run_controls.pack(fill="x", pady=(12, 0))
        self.pause_button = ttk.Button(
            run_controls, text="Pause", command=self._toggle_pause, state="disabled"
        )
        self.pause_button.pack(side="left", fill="x", expand=True)
        self.abort_button = ttk.Button(
            run_controls, text="Abort", command=self._abort_scan_clicked, state="disabled"
        )
        self.abort_button.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.scan_button = ttk.Button(
            parent,
            text="Review and start cycle 1",
            style="Primary.TButton",
            command=self._start_scan,
        )
        self.scan_button.pack(fill="x", pady=(8, 0))

    def _build_footer(self) -> None:
        footer = tk.Frame(self, bg="#dce7e9", height=30)
        footer.grid(row=2, column=0, sticky="ew")
        tk.Label(
            footer,
            text=(
                "SIMULATION MODE — no microscope hardware commands are sent"
                if self.backend.simulated
                else f"HARDWARE MODE — {self.backend.name} is controlling the microscope"
            ),
            bg="#dce7e9",
            fg="#536a73",
            font=("TkDefaultFont", 9, "bold"),
        ).pack(pady=7)

    def _load_initial_state(self) -> None:
        objective = self.config_data.objectives[0]
        self.objective_var.set(objective.name)
        self._set_objective_detail(objective)
        self._z_mode_changed()
        self._sync_preview_channels()
        self._update_position_label()
        self._update_focus_status()
        self._input_changed()

    def _selected_objective(self):
        selected_name = self.objective_var.get()
        return next(item for item in self.config_data.objectives if item.name == selected_name)

    def _set_objective_detail(self, objective) -> None:
        self.objective_detail_label.configure(
            text=(
                f"{objective.immersion.title()} • NA {objective.numerical_aperture:g} • "
                f"turret {objective.turret_position} • {objective.pixel_size_um:g} µm/px"
            )
        )

    def _objective_changed(self, _event=None) -> None:
        self._stop_spiral_preview(clear_overlay=True)
        objective = self._selected_objective()
        previous = self.config_data.objective(self.backend.current_objective_id)
        if self.controller.busy:
            self.objective_var.set(previous.name)
            self.progress_text_var.set("The objective cannot change while an acquisition runs")
            return
        if objective.id == previous.id:
            return
        if objective.immersion != previous.immersion:
            action = "apply" if objective.immersion != "air" else "remove"
            prompt = (
                f"Changing from {previous.name} ({previous.immersion}) to "
                f"{objective.name} ({objective.immersion}).\n\n"
                f"Please {action} the correct immersion medium. The real system will "
                "move Z to a safe clearance before rotating the turret.\n\nContinue?"
            )
            if not messagebox.askokcancel("Immersion change", prompt, parent=self):
                self.objective_var.set(previous.name)
                return
        try:
            self.backend.change_objective(objective)
        except HardwareError as exc:
            messagebox.showerror("Objective change failed", str(exc), parent=self)
            self.objective_var.set(previous.name)
            return
        if not self.backend.simulated:
            messagebox.showinfo(
                "Refocus",
                "Z was moved to the clearance height for the objective change.\n"
                "Refocus in Live before capturing positions.",
                parent=self,
            )
        self._set_objective_detail(objective)
        self._reset_z_stack_capture()
        self.recipe_var.set("Custom")
        self._render_scene()
        self._input_changed()

    def _apply_recipe(self, _event=None) -> None:
        name = self.recipe_var.get()
        if name == "Custom":
            return
        recipe = self.config_data.recipe(name)
        if self.controller.experiment is not None and recipe.objective_id != self.controller.experiment.objective_id:
            messagebox.showinfo(
                "Objective is fixed",
                "This experiment's objective is fixed; the recipe's objective was not applied.",
                parent=self,
            )
            self.recipe_var.set("Custom")
            return
        objective = self.config_data.objective(recipe.objective_id)
        self.objective_var.set(objective.name)
        self._objective_changed()
        self.recipe_var.set(name)
        selected = set(recipe.channels)
        for channel_id, variable in self.channel_vars.items():
            variable.set(channel_id in selected)
        self._sync_preview_channels()
        self.z_mode_var.set(recipe.z_mode)
        self._z_mode_changed()
        self._input_changed()

    def _z_mode_changed(self) -> None:
        state = "normal" if self.z_mode_var.get() == "stack" else "disabled"
        for button in self.z_capture_buttons:
            button.configure(state=state)
        self._recalculate_z_stack()
        self.recipe_var.set("Custom")
        self._input_changed()

    def _capture_z_begin(self) -> None:
        if not self.live:
            messagebox.showinfo(
                "Start live view",
                "Turn on Live, move to the first Z position, then press Set Begin.",
                parent=self,
            )
            return
        self.z_begin_captured = True
        self.z_bottom_var.set(f"{self.backend.position.z_um:.3f}")
        self.progress_text_var.set(
            f"Z-stack Begin captured at {self.backend.position.z_um:g} µm"
        )
        self._recalculate_z_stack()
        self._input_changed()

    def _capture_z_end(self) -> None:
        if not self.live:
            messagebox.showinfo(
                "Start live view",
                "Turn on Live, move to the last Z position, then press Set End.",
                parent=self,
            )
            return
        self.z_end_captured = True
        self.z_top_var.set(f"{self.backend.position.z_um:.3f}")
        self.progress_text_var.set(
            f"Z-stack End captured at {self.backend.position.z_um:g} µm"
        )
        self._recalculate_z_stack()
        self._input_changed()

    def _selected_channels(self) -> tuple:
        return tuple(
            channel
            for channel in self.config_data.channels
            if self.channel_vars[channel.id].get()
        )

    def _recalculate_z_stack(self) -> None:
        if not self.z_begin_captured or not self.z_end_captured:
            missing = []
            if not self.z_begin_captured:
                missing.append("Begin")
            if not self.z_end_captured:
                missing.append("End")
            self.z_step_var.set("—")
            self.z_stack_summary_var.set(f"Set {' and '.join(missing)} from live Z")
            return
        objective = self._selected_objective()
        recommended_step, wavelength_nm = recommended_z_step_um(
            objective,
            self._selected_channels(),
        )
        begin_um = float(self.z_bottom_var.get())
        end_um = float(self.z_top_var.get())
        planes, actual_step = automatic_z_stack(begin_um, end_um, recommended_step)
        self.z_step_var.set(f"{actual_step:.4f}")
        self.z_stack_summary_var.set(
            f"Auto: {planes} plane{'s' if planes != 1 else ''} • {actual_step:.3f} µm step "
            f"• NA {objective.numerical_aperture:g} • {wavelength_nm:.0f} nm"
        )

    def _reset_z_stack_capture(self) -> None:
        self.z_begin_captured = False
        self.z_end_captured = False
        self.z_bottom_var.set("—")
        self.z_top_var.set("—")
        self.z_step_var.set("—")
        self.z_stack_summary_var.set("Set Begin and End from live Z")

    def _choose_output(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.output_var.get(), parent=self)
        if folder:
            self.output_var.set(folder)
            self._input_changed()

    def _exposure_changed(self, value: str, display: tk.StringVar) -> None:
        display.set(f"{float(value):.0f}")
        self._input_changed()

    def _bind_z_wheel(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", self._mousewheel_z)
        widget.bind("<Button-4>", self._mousewheel_z)
        widget.bind("<Button-5>", self._mousewheel_z)
        for child in widget.winfo_children():
            self._bind_z_wheel(child)

    def _wheel_steps(self, event) -> float:
        if getattr(event, "num", None) == 4:
            return 1.0
        if getattr(event, "num", None) == 5:
            return -1.0
        delta = float(getattr(event, "delta", 0))
        if not delta:
            return 0.0
        if abs(delta) >= 120:
            return max(-3.0, min(3.0, delta / 120.0))
        return max(-3.0, min(3.0, delta))

    def _mousewheel_z(self, event) -> str:
        steps = self._wheel_steps(event)
        if not steps:
            return "break"
        multiplier = 10.0 if getattr(event, "state", 0) & 0x1 else 1.0
        self._jog_z(steps * multiplier)
        self.progress_text_var.set(
            f"Live Z {self.backend.position.z_um:g} µm • wheel step {multiplier:g} µm"
        )
        return "break"

    def _mousewheel_zoom(self, event) -> str:
        steps = self._wheel_steps(event)
        if not steps:
            return "break"
        stage_before = self._canvas_to_stage(event.x, event.y)
        new_zoom = max(0.25, min(8.0, self.view_zoom * (1.15**steps)))
        if math.isclose(new_zoom, self.view_zoom):
            return "break"
        self.view_zoom = new_zoom
        stage_after = self._canvas_to_stage(event.x, event.y)
        center_x, center_y, _width, _height = self._view_geometry()
        self.view_center_override = (
            center_x + stage_before[0] - stage_after[0],
            center_y + stage_before[1] - stage_after[1],
        )
        self.progress_text_var.set(f"Preview zoom {self.view_zoom * 100:.0f}%")
        self._render_scene()
        return "break"

    def _channel_selection_changed(self) -> None:
        self._sync_preview_channels()
        self._recalculate_z_stack()
        self._input_changed()

    def _sync_preview_channels(self) -> None:
        names = [
            channel.name
            for channel in self.config_data.channels
            if self.channel_vars[channel.id].get()
        ]
        self.preview_channel_box.configure(values=names)
        if not names:
            self.preview_channel_var.set("")
            self.preview_channel_box.configure(state="disabled")
        else:
            self.preview_channel_box.configure(state="readonly")
            if self.preview_channel_var.get() not in names:
                self.preview_channel_var.set(names[0])

    def _mosaic_view_changed(self, _event=None) -> None:
        self._render_scene()

    def _on_close(self) -> None:
        if self.controller.busy and not messagebox.askyesno(
            "Acquisition running",
            "An acquisition is running. Stop it safely and close?\n\n"
            "Finished tiles are kept and the cycle can be resumed later.",
            parent=self,
        ):
            return
        if self.progress_job is not None:
            self.after_cancel(self.progress_job)
            self.progress_job = None
        self._stop_spiral_preview(clear_overlay=True)
        self._set_live(False)
        self.controller.close()
        self.destroy()

    def _safe_move(self, action, *args) -> bool:
        """Run a stage command, reporting refused/failed moves instead of raising.

        Refused while a cycle runs: the engine owns the stage, and a move during
        an exposure would put the image at a different place than its tile record.
        """

        if self.controller.busy:
            self.progress_text_var.set("The stage is in use by the running acquisition")
            self.bell()
            return False
        try:
            action(*args)
        except HardwareError as exc:
            self.progress_text_var.set(f"Move refused: {exc}")
            self.bell()
            return False
        return True

    def _interaction_mode_changed(self) -> None:
        self.polygon_draft.clear()
        hints = {
            "rectangle": "Drag a rectangle • Double-click moves stage • Wheel zooms",
            "polygon": "Click polygon corners • Double/right-click finishes • Wheel zooms",
            "focus": "Click location, focus live, then Capture • Wheel zooms image",
            "move_set": "Drag to move all ROIs and focus points • Wheel zooms",
        }
        self.move_start_stage = None
        self.viewer_hint_var.set(hints[self.interaction_mode_var.get()])
        self._render_scene()

    def _toggle_live(self) -> None:
        self._set_live(not self.live)
        self.progress_text_var.set("Live view running" if self.live else "Live view paused")
        self._update_focus_status()
        self._render_scene()

    def _set_live(self, on: bool) -> None:
        """Start/stop real camera live view (snaps on the GUI thread; the Qt view uses a worker)."""

        if on and self.controller.busy:
            on = False
        self.live = on
        if hasattr(self, "live_button"):
            self.live_button.configure(text="● Live" if on else "○ Live")
        if on and self.live_job is None:
            self.live_job = self.after(1, self._live_tick)
        elif not on and self.live_job is not None:
            self.after_cancel(self.live_job)
            self.live_job = None

    def _live_tick(self) -> None:
        self.live_job = None
        if not self.live or self.controller.busy:
            self._set_live(False)
            return
        if self._acquire_frame():
            self.overview_mode = False
            self._update_position_label()
            self._render_scene()
            self.live_job = self.after(LIVE_INTERVAL_MS, self._live_tick)
        else:
            self._set_live(False)

    def _live_channel(self):
        name = self.preview_channel_var.get()
        for channel in self.config_data.channels:
            if channel.name == name:
                return channel
        selected = [c for c in self.config_data.channels if self.channel_vars[c.id].get()]
        return (selected or list(self.config_data.channels))[0]

    def _acquire_frame(self) -> bool:
        """Snap one real frame with the preview channel's exposure and power."""

        if self.controller.busy:
            return False
        channel = self._live_channel()
        exposure = self._float_or(self.exposure_vars[channel.id], channel.default_exposure_ms)
        power = self._float_or(self.power_vars[channel.id], channel.default_power_percent)
        try:
            # A cycle may have left auto-shutter off with the shutter closed.
            self.backend.set_shutter_per_frame(True)
            self.backend.set_channel(channel, exposure, power)
            frame = self.backend.snap()
        except HardwareError as exc:
            self.progress_text_var.set(f"Camera failed: {exc}")
            return False
        position = self.backend.position
        self.last_frame = np.asarray(frame)
        self.last_frame_xy = (position.x_um, position.y_um)
        return True

    def _snap(self) -> None:
        if self.controller.busy:
            return
        self._set_live(False)
        if not self._acquire_frame():
            return
        frame = self.last_frame
        self.overview_mode = False
        self.progress_text_var.set(
            f"Snapshot {frame.shape[1]}×{frame.shape[0]} captured • max {int(frame.max())}"
        )
        self._render_scene()

    def _draw_last_frame(self, width: int, height: int) -> bool:
        """Draw the last camera frame at its field of view (oriented: +col = +X, +row = +Y)."""

        if self.last_frame is None:
            return False
        fov_w, fov_h = self.config_data.fov_um(self.backend.current_objective_id)
        x, y = self.last_frame_xy
        x0, y0 = self._stage_to_canvas(x - fov_w / 2, y - fov_h / 2)
        x1, y1 = self._stage_to_canvas(x + fov_w / 2, y + fov_h / 2)
        out_w = int(min(abs(x1 - x0), 4 * width))
        out_h = int(min(abs(y1 - y0), 4 * height))
        if out_w < 2 or out_h < 2:
            return False
        self.live_photo = tk.PhotoImage(
            data=_frame_to_pgm(self.last_frame, out_w, out_h), format="PPM"
        )
        self.viewer.create_image(
            (x0 + x1) / 2, (y0 + y1) / 2, image=self.live_photo, anchor="center", tags="scene"
        )
        return True

    def _overview(self) -> None:
        self.overview_mode = True
        self.preview_view = None
        self.view_zoom = 1.0
        self.view_center_override = None
        self.progress_text_var.set("Overview mode — define ROIs and focus points")
        self._render_scene()

    def _nudge_stage(self, dx: int, dy: int) -> None:
        objective = self._selected_objective()
        step = self.config_data.camera_width_px * objective.pixel_size_um * 0.65
        position = self.backend.position
        if not self._safe_move(self.backend.move_xy, position.x_um + dx * step, position.y_um + dy * step):
            return
        self.overview_mode = False
        self.preview_view = None
        self.view_center_override = None
        self._update_position_label()
        self._render_scene()

    def _update_position_label(self) -> None:
        position = self.backend.position
        self.position_var.set(f"X {position.x_um:,.0f}  Y {position.y_um:,.0f}  Z {position.z_um:,.0f} µm")

    def _view_geometry(self) -> tuple[float, float, float, float]:
        width = max(1, self.viewer.winfo_width())
        height = max(1, self.viewer.winfo_height())
        position = self.backend.position
        if self.preview_view is not None:
            center_x, center_y, view_width, minimum_height = self.preview_view
            view_height = max(minimum_height, view_width * height / width)
        elif self.overview_mode:
            x_span = self.config_data.stage_limits.x_max_um - self.config_data.stage_limits.x_min_um
            y_span = self.config_data.stage_limits.y_max_um - self.config_data.stage_limits.y_min_um
            view_width = x_span * 1.04
            view_height = max(y_span * 1.08, view_width * height / width)
            center_x = (self.config_data.stage_limits.x_min_um + self.config_data.stage_limits.x_max_um) / 2
            center_y = (self.config_data.stage_limits.y_min_um + self.config_data.stage_limits.y_max_um) / 2
        else:
            objective = self._selected_objective()
            view_width = self.config_data.camera_width_px * objective.pixel_size_um
            view_height = view_width * height / width
            center_x, center_y = position.x_um, position.y_um
        if self.view_center_override is not None:
            center_x, center_y = self.view_center_override
        view_width /= self.view_zoom
        view_height /= self.view_zoom
        return center_x, center_y, view_width, view_height

    def _canvas_to_stage(self, x_px: float, y_px: float) -> tuple[float, float]:
        width = max(1, self.viewer.winfo_width())
        height = max(1, self.viewer.winfo_height())
        center_x, center_y, view_width, view_height = self._view_geometry()
        x_um = center_x + (x_px / width - 0.5) * view_width
        y_um = center_y + (y_px / height - 0.5) * view_height
        return x_um, y_um

    def _stage_to_canvas(self, x_um: float, y_um: float) -> tuple[float, float]:
        width = max(1, self.viewer.winfo_width())
        height = max(1, self.viewer.winfo_height())
        center_x, center_y, view_width, view_height = self._view_geometry()
        x_px = (0.5 + (x_um - center_x) / view_width) * width
        y_px = (0.5 + (y_um - center_y) / view_height) * height
        return x_px, y_px

    def _start_region(self, event) -> None:
        mode = self.interaction_mode_var.get()
        if mode in ("rectangle", "polygon", "move_set", "focus") and self.controller.busy:
            return
        if mode in ("rectangle", "polygon") and self._rois_locked():
            self.progress_text_var.set("ROIs are fixed for this experiment — use Move ROI set")
            return
        if mode == "polygon":
            point = self._canvas_to_stage(event.x, event.y)
            if not self.polygon_draft or math.dist(point, self.polygon_draft[-1]) > 1:
                self.polygon_draft.append(point)
            self._render_scene()
            return
        if mode == "focus":
            x_um, y_um = self._canvas_to_stage(event.x, event.y)
            selected = self._focus_point_at_canvas(event.x, event.y)
            if selected is not None:
                self.selected_focus_index = selected
                point = self.focus_points[selected]
                if self._safe_move(self.backend.move_xy, point.x_um, point.y_um) and self._safe_move(
                    self.backend.move_z, point.z_um
                ):
                    self.progress_text_var.set(
                        f"Focus point F{selected + 1} loaded at Z {point.z_um:g} µm"
                    )
            else:
                self.selected_focus_index = None
                if self._safe_move(self.backend.move_xy, x_um, y_um):
                    self.progress_text_var.set(
                        "Position loaded — focus live, then press Capture XYZ"
                    )
            self.overview_mode = False
            self.preview_view = None
            self.view_center_override = None
            self._set_live(True)
            self._update_focus_status()
            self._update_position_label()
            self._render_scene()
            return
        if mode == "move_set":
            if not self.regions and not self.focus_points:
                self.progress_text_var.set("Load or draw an ROI set before moving it")
                return
            self.move_start_stage = self._canvas_to_stage(event.x, event.y)
            self.move_original_regions = tuple(self.regions)
            self.move_original_focus_points = tuple(self.focus_points)
            return
        self.drag_start = (event.x, event.y)
        self.drag_rectangle_id = self.viewer.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#69f0d0",
            width=2,
            dash=(5, 3),
            tags="drag_region",
        )

    def _drag_region(self, event) -> None:
        if self.interaction_mode_var.get() == "move_set" and self.move_start_stage:
            current_x, current_y = self._canvas_to_stage(event.x, event.y)
            dx_um = current_x - self.move_start_stage[0]
            dy_um = current_y - self.move_start_stage[1]
            dx_um, dy_um = self._bounded_set_translation(dx_um, dy_um)
            self.regions = [
                region.translated(dx_um, dy_um) for region in self.move_original_regions
            ]
            self.focus_points = [
                point.translated_xy(dx_um, dy_um)
                for point in self.move_original_focus_points
            ]
            self._render_scene()
            return
        if self.interaction_mode_var.get() == "rectangle" and self.drag_start and self.drag_rectangle_id:
            x0, y0 = self.drag_start
            self.viewer.coords(self.drag_rectangle_id, x0, y0, event.x, event.y)

    def _finish_region(self, event) -> None:
        if self.interaction_mode_var.get() == "move_set":
            if self.move_start_stage is not None:
                self.move_start_stage = None
                self.move_original_regions = ()
                self.move_original_focus_points = ()
                self.progress_text_var.set("ROI and focus points moved together")
                self._input_changed()
            return
        if self.interaction_mode_var.get() != "rectangle":
            return
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        self.drag_start = None
        if self.drag_rectangle_id:
            self.viewer.delete(self.drag_rectangle_id)
            self.drag_rectangle_id = None
        if abs(event.x - x0) < 12 or abs(event.y - y0) < 12:
            return
        sx0, sy0 = self._canvas_to_stage(min(x0, event.x), min(y0, event.y))
        sx1, sy1 = self._canvas_to_stage(max(x0, event.x), max(y0, event.y))
        self.regions.append(ScanRegion(sx0, sy0, sx1 - sx0, sy1 - sy0))
        self._render_scene()
        self._input_changed()

    def _bounded_set_translation(self, dx_um: float, dy_um: float) -> tuple[float, float]:
        x_values = [coordinate for region in self.move_original_regions for coordinate, _ in region.vertices]
        y_values = [coordinate for region in self.move_original_regions for _, coordinate in region.vertices]
        x_values.extend(point.x_um for point in self.move_original_focus_points)
        y_values.extend(point.y_um for point in self.move_original_focus_points)
        if not x_values or not y_values:
            return 0.0, 0.0
        limits = self.config_data.stage_limits
        dx_um = min(limits.x_max_um - max(x_values), max(limits.x_min_um - min(x_values), dx_um))
        dy_um = min(limits.y_max_um - max(y_values), max(limits.y_min_um - min(y_values), dy_um))
        return dx_um, dy_um

    def _focus_point_at_canvas(self, x_px: float, y_px: float) -> int | None:
        closest: tuple[float, int] | None = None
        for index, point in enumerate(self.focus_points):
            point_x, point_y = self._stage_to_canvas(point.x_um, point.y_um)
            distance = math.hypot(point_x - x_px, point_y - y_px)
            if distance <= 16 and (closest is None or distance < closest[0]):
                closest = (distance, index)
        return closest[1] if closest else None

    def _jog_z(self, delta_um: float) -> None:
        position = self.backend.position
        if not self._safe_move(self.backend.move_z, position.z_um + delta_um):
            return
        self._set_live(True)
        self._update_position_label()
        self._update_focus_status()
        self._render_scene()

    def _capture_focus_here(self) -> None:
        if not self.live:
            messagebox.showinfo(
                "Start live view",
                "Turn on Live and focus the sample before capturing a focus point.",
                parent=self,
            )
            return
        position = self.backend.position
        self.focus_points.append(FocusPoint(position.x_um, position.y_um, position.z_um))
        self.selected_focus_index = len(self.focus_points) - 1
        self.progress_text_var.set(
            f"Captured F{len(self.focus_points)} at Z {position.z_um:g} µm"
        )
        self._update_focus_status()
        self._render_scene()
        self._input_changed()

    def _update_selected_focus_z(self) -> None:
        if self.selected_focus_index is None:
            messagebox.showinfo(
                "Select a focus point",
                "Choose Focus points, then click an existing numbered marker first.",
                parent=self,
            )
            return
        index = self.selected_focus_index
        previous = self.focus_points[index]
        current = self.backend.position
        self.focus_points[index] = FocusPoint(previous.x_um, previous.y_um, current.z_um)
        self.progress_text_var.set(f"Updated F{index + 1} to Z {current.z_um:g} µm")
        if self.refocus_index is not None:
            if index + 1 < len(self.focus_points):
                self._refocus_goto(index + 1)
            else:
                self._stop_refocus()
                self.progress_text_var.set(
                    f"All {len(self.focus_points)} focus points refocused for this cycle"
                )
        self._update_focus_status()
        self._render_scene()
        self._input_changed()

    def _delete_selected_focus(self) -> None:
        if self.selected_focus_index is None:
            return
        self._stop_refocus()
        deleted = self.selected_focus_index + 1
        del self.focus_points[self.selected_focus_index]
        self.selected_focus_index = None
        self.progress_text_var.set(f"Deleted focus point F{deleted}")
        self._update_focus_status()
        self._render_scene()
        self._input_changed()

    def _update_focus_status(self) -> None:
        position = self.backend.position
        if self.selected_focus_index is None:
            self.focus_status_var.set(f"Z {position.z_um:g} • none")
        else:
            self.focus_status_var.set(
                f"Z {position.z_um:g} • F{self.selected_focus_index + 1} selected"
            )

    def _viewer_double_click(self, event) -> str | None:
        if self.interaction_mode_var.get() == "polygon":
            return self._finish_polygon(event)
        if self.interaction_mode_var.get() == "focus":
            return "break"
        if self.interaction_mode_var.get() == "move_set":
            return "break"
        self._move_stage_to_click(event)
        return None

    def _finish_polygon(self, _event=None) -> str:
        if self.interaction_mode_var.get() != "polygon":
            return "break"
        if len(self.polygon_draft) >= 3:
            self.regions.append(ScanRegion.from_polygon(tuple(self.polygon_draft)))
            self.progress_text_var.set(
                f"Polygon ROI added with {len(self.polygon_draft)} vertices"
            )
            self.polygon_draft.clear()
            self._input_changed()
        else:
            self.progress_text_var.set("A polygon needs at least three different corners")
        self._render_scene()
        return "break"

    def _cancel_draft(self, _event=None) -> None:
        self.polygon_draft.clear()
        if self.move_original_regions or self.move_original_focus_points:
            self.regions = list(self.move_original_regions)
            self.focus_points = list(self.move_original_focus_points)
        self.move_start_stage = None
        self.move_original_regions = ()
        self.move_original_focus_points = ()
        if self.drag_rectangle_id:
            self.viewer.delete(self.drag_rectangle_id)
            self.drag_rectangle_id = None
        self.drag_start = None
        self.progress_text_var.set("Current drawing cancelled")
        self._render_scene()

    def _move_stage_to_click(self, event) -> None:
        x_um, y_um = self._canvas_to_stage(event.x, event.y)
        if not self._safe_move(self.backend.move_xy, x_um, y_um):
            return
        self.overview_mode = False
        self.preview_view = None
        self.view_zoom = 1.0
        self.view_center_override = None
        self._update_position_label()
        self._update_focus_status()
        self._render_scene()

    def _clear_regions(self) -> None:
        if self._rois_locked() or self.controller.busy:
            return
        self._stop_refocus()
        self._stop_spiral_preview(clear_overlay=True)
        self.regions.clear()
        self.focus_points.clear()
        self.selected_focus_index = None
        self.polygon_draft.clear()
        self._update_focus_status()
        self._render_scene()
        self._input_changed()

    def _save_roi_set(self) -> None:
        if not self.regions and not self.focus_points:
            messagebox.showinfo(
                "Nothing to save",
                "Draw at least one ROI or add a focus point first.",
                parent=self,
            )
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save ROI and focus-point set",
            initialdir=self.output_var.get(),
            initialfile=f"{self.sample_var.get().strip() or 'sample'}_roi.json",
            defaultextension=".json",
            filetypes=(("ROI set", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            save_roi_set(path, tuple(self.regions), tuple(self.focus_points))
        except OSError as exc:
            messagebox.showerror("Could not save ROI set", str(exc), parent=self)
            return
        self.progress_text_var.set(f"ROI set saved • {path}")

    def _load_roi_set(self) -> None:
        if self._rois_locked() or self.controller.busy:
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="Load ROI and focus-point set",
            initialdir=self.output_var.get(),
            filetypes=(("ROI set", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        self._stop_spiral_preview(clear_overlay=True)
        try:
            regions, focus_points = load_roi_set(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Could not load ROI set", str(exc), parent=self)
            return
        if (self.regions or self.focus_points) and not messagebox.askyesno(
            "Replace current ROI set?",
            "Loading this file will replace the current ROIs and focus points. Continue?",
            parent=self,
        ):
            return
        self.regions = list(regions)
        self.focus_points = list(focus_points)
        self.selected_focus_index = None
        self.polygon_draft.clear()
        self.overview_mode = True
        self.preview_view = None
        self.view_zoom = 1.0
        self.view_center_override = None
        self._update_focus_status()
        self.progress_text_var.set(f"ROI set loaded • {path}")
        self._render_scene()
        self._input_changed()

    def _toggle_spiral_preview(self) -> None:
        if self.preview_job is not None:
            self._stop_spiral_preview(clear_overlay=False)
            self.progress_text_var.set(
                f"Spiral preview stopped after {len(self.preview_completed)} tiles"
            )
            self._render_scene()
            return

        channel_name = self.preview_channel_var.get()
        if not channel_name:
            messagebox.showinfo(
                "Select a preview channel",
                "Select at least one channel, then choose the channel for the preview.",
                parent=self,
            )
            return
        channel = next(
            item for item in self.config_data.channels if item.name == channel_name
        )
        objective = self._selected_objective()
        origin = (self.backend.position.x_um, self.backend.position.y_um)
        positions = spiral_preview_positions(
            self.config_data,
            objective.id,
            tuple(self.regions),
            origin,
            overlap_x_percent=self.overlap_x_var.get(),
            overlap_y_percent=self.overlap_y_var.get(),
        )
        limits = self.config_data.stage_limits
        self.preview_positions = [
            point
            for point in positions
            if limits.x_min_um <= point[0] <= limits.x_max_um
            and limits.y_min_um <= point[1] <= limits.y_max_um
        ]
        if not self.preview_positions:
            messagebox.showwarning(
                "No preview positions",
                "The preview area does not contain any positions inside the stage limits.",
                parent=self,
            )
            return

        fov_width = self.config_data.camera_width_px * objective.pixel_size_um
        fov_height = self.config_data.camera_height_px * objective.pixel_size_um
        x_values = [point[0] for point in self.preview_positions]
        y_values = [point[1] for point in self.preview_positions]
        self.preview_view = (
            (min(x_values) + max(x_values)) / 2,
            (min(y_values) + max(y_values)) / 2,
            max(2 * fov_width, max(x_values) - min(x_values) + 1.4 * fov_width),
            max(2 * fov_height, max(y_values) - min(y_values) + 1.4 * fov_height),
        )
        self.preview_channel_id = channel.id
        self.preview_completed = []
        self.preview_index = 0
        self.overview_mode = True
        self.view_zoom = 1.0
        self.view_center_override = None
        self.preview_button.configure(text="Stop preview")
        self.status_var.set("SPIRAL PREVIEW RUNNING")
        self.progress_text_var.set(
            f"Starting {channel.name} spiral preview • one channel • current Z only"
        )
        self.preview_job = self.after(1, self._tick_spiral_preview)

    def _tick_spiral_preview(self) -> None:
        if self.preview_index >= len(self.preview_positions):
            self.preview_job = None
            self.preview_button.configure(text="Spiral preview")
            self.status_var.set(self.idle_status)
            self.progress_text_var.set(
                f"Spiral preview complete • {len(self.preview_completed)} tiles • no Z movement"
            )
            self._render_scene()
            return

        position = self.preview_positions[self.preview_index]
        if not self._safe_move(self.backend.move_xy, *position):
            self._stop_spiral_preview(clear_overlay=False)
            self._render_scene()
            return
        self.preview_completed.append(position)
        self.preview_index += 1
        self._update_position_label()
        channel = self.config_data.channel(self.preview_channel_id)
        self.progress_text_var.set(
            f"{channel.name} spiral preview • tile {self.preview_index} of "
            f"{len(self.preview_positions)} • current Z only"
        )
        self._render_scene()
        exposure_ms = self.exposure_vars[channel.id].get()
        delay_ms = int(max(80, min(450, exposure_ms * 2)))
        self.preview_job = self.after(delay_ms, self._tick_spiral_preview)

    def _stop_spiral_preview(self, clear_overlay: bool) -> None:
        if self.preview_job is not None:
            self.after_cancel(self.preview_job)
            self.preview_job = None
        self.preview_button.configure(text="Spiral preview")
        if self.status_var.get() == "SPIRAL PREVIEW RUNNING":
            self.status_var.set(self.idle_status)
        if clear_overlay:
            self.preview_positions = []
            self.preview_completed = []
            self.preview_index = 0
            self.preview_channel_id = None
            self.preview_view = None

    def _render_scene(self) -> None:
        if not hasattr(self, "viewer"):
            return
        self.viewer.delete("scene")
        width = max(10, self.viewer.winfo_width())
        height = max(10, self.viewer.winfo_height())
        rng_seed = int(self.backend.position.x_um / 100) * 31 + int(self.backend.position.y_um / 100)
        rng = random.Random(rng_seed)

        mosaic_rendered = False
        mosaic = self.controller.mosaic
        engine_status = self.controller.status()
        if mosaic is not None and not mosaic.closed:
            center_x, center_y, view_width, view_height = self._view_geometry()
            view_bounds = MosaicBounds(
                center_x - view_width / 2,
                center_x + view_width / 2,
                center_y - view_height / 2,
                center_y + view_height / 2,
            )
            ppm = mosaic.render_viewport_ppm(
                view_bounds,
                width,
                height,
                self.mosaic_view_var.get(),
            )
            self.mosaic_photo = tk.PhotoImage(data=ppm, format="PPM")
            self.viewer.create_image(
                0,
                0,
                image=self.mosaic_photo,
                anchor="nw",
                tags="scene",
            )
            mosaic_rendered = True

        if not mosaic_rendered and self.overview_mode and self.backend.simulated:
            # Simulated slide and tissue section (simulator only, never on hardware).
            x0, y0 = self._stage_to_canvas(-42000, -26000)
            x1, y1 = self._stage_to_canvas(42000, 26000)
            self.viewer.create_rectangle(x0, y0, x1, y1, fill="#d8dde0", outline="#f4f7f8", width=2, tags="scene")
            for index in range(34):
                cx = width * (0.20 + 0.60 * rng.random())
                cy = height * (0.22 + 0.56 * rng.random())
                rx = rng.randint(18, 75)
                ry = rng.randint(10, 48)
                color = "#7a8f8d" if index % 3 else "#637d79"
                self.viewer.create_oval(cx - rx, cy - ry, cx + rx, cy + ry, fill=color, outline="", tags="scene")
        if not self.overview_mode and not self.controller.busy:
            # Live/Snap: the real camera frame goes on top, an earlier cycle's mosaic
            # is only background (refocusing between cycles needs the live image).
            if not self._draw_last_frame(width, height) and not mosaic_rendered:
                self.viewer.create_text(
                    width / 2, height / 2, text="Press Live or Snap to see the camera",
                    fill="#9aa5ad", tags="scene",
                )

        for index, region in enumerate(self.regions, start=1):
            canvas_points = [self._stage_to_canvas(x, y) for x, y in region.vertices]
            flat_points = [coordinate for point in canvas_points for coordinate in point]
            self.viewer.create_polygon(
                *flat_points,
                outline="#69f0d0",
                width=3,
                fill="" if mosaic is not None else "#0a6d73",
                stipple="gray12",
                tags="scene",
            )
            label_x, label_y = canvas_points[0]
            self.viewer.create_text(
                label_x + 8,
                label_y + 8,
                text=f"R{index}",
                anchor="nw",
                fill="white",
                font=("TkDefaultFont", 10, "bold"),
                tags="scene",
            )

        if self.polygon_draft:
            draft_points = [self._stage_to_canvas(x, y) for x, y in self.polygon_draft]
            flat_draft = [coordinate for point in draft_points for coordinate in point]
            if len(draft_points) >= 2:
                self.viewer.create_line(
                    *flat_draft,
                    fill="#ffe082",
                    width=3,
                    dash=(5, 3),
                    tags="scene",
                )
            for x, y in draft_points:
                self.viewer.create_oval(
                    x - 5,
                    y - 5,
                    x + 5,
                    y + 5,
                    fill="#ffe082",
                    outline=VIEW_BG,
                    width=2,
                    tags="scene",
                )

        for index, point in enumerate(self.focus_points, start=1):
            x, y = self._stage_to_canvas(point.x_um, point.y_um)
            selected = index - 1 == self.selected_focus_index
            self.viewer.create_oval(
                x - (13 if selected else 10),
                y - (13 if selected else 10),
                x + (13 if selected else 10),
                y + (13 if selected else 10),
                fill="#fff2a8" if selected else "#ffca3a",
                outline="#ffffff" if selected else "#3b2b00",
                width=3 if selected else 2,
                tags="scene",
            )
            self.viewer.create_text(
                x,
                y,
                text=str(index),
                fill="#241900",
                font=("TkDefaultFont", 9, "bold"),
                tags="scene",
            )
            self.viewer.create_text(
                x + 13,
                y - 13,
                text=f"F{index}  Z {point.z_um:g}",
                anchor="sw",
                fill="#ffe082",
                font=("TkDefaultFont", 9, "bold"),
                tags="scene",
            )

        if self.preview_completed:
            objective = self._selected_objective()
            fov_width = self.config_data.camera_width_px * objective.pixel_size_um
            fov_height = self.config_data.camera_height_px * objective.pixel_size_um
            channel_color = (
                self.config_data.channel(self.preview_channel_id).color
                if self.preview_channel_id
                else "#6c8cff"
            )
            for tile_number, (tile_x, tile_y) in enumerate(self.preview_completed, start=1):
                x0, y0 = self._stage_to_canvas(
                    tile_x - fov_width / 2,
                    tile_y - fov_height / 2,
                )
                x1, y1 = self._stage_to_canvas(
                    tile_x + fov_width / 2,
                    tile_y + fov_height / 2,
                )
                self.viewer.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    outline=channel_color,
                    fill=channel_color,
                    stipple="gray25",
                    width=2,
                    tags="scene",
                )
                if tile_number == len(self.preview_completed):
                    self.viewer.create_text(
                        (x0 + x1) / 2,
                        (y0 + y1) / 2,
                        text=str(tile_number),
                        fill="white",
                        font=("TkDefaultFont", 9, "bold"),
                        tags="scene",
                    )
        current_tile = engine_status.current_tile if engine_status else None
        if current_tile is not None and mosaic is not None:
            objective = self._selected_objective()
            tile_width = self.config_data.camera_width_px * objective.pixel_size_um
            tile_height = self.config_data.camera_height_px * objective.pixel_size_um
            tile_x, tile_y = current_tile.x_um, current_tile.y_um
            x0, y0 = self._stage_to_canvas(
                tile_x - tile_width / 2,
                tile_y - tile_height / 2,
            )
            x1, y1 = self._stage_to_canvas(
                tile_x + tile_width / 2,
                tile_y + tile_height / 2,
            )
            self.viewer.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline="#ffffff",
                width=3,
                dash=(6, 3),
                tags="scene",
            )
        position = self.backend.position
        px, py = self._stage_to_canvas(position.x_um, position.y_um)
        self.viewer.create_line(px - 10, py, px + 10, py, fill="#ffffff", width=1, tags="scene")
        self.viewer.create_line(px, py - 10, px, py + 10, fill="#ffffff", width=1, tags="scene")
        if mosaic is not None and engine_status is not None:
            cycle = self.controller.active_cycle or 0
            mode = (
                f"CYCLE {cycle} MOSAIC • {self.mosaic_view_var.get()} • MIP • "
                f"{engine_status.tiles_done}/{engine_status.tiles_total} TILES • "
                f"{self.view_zoom * 100:.0f}%"
            )
        elif self.preview_job is not None or self.preview_completed:
            channel_name = (
                self.config_data.channel(self.preview_channel_id).name
                if self.preview_channel_id
                else ""
            )
            mode = f"SPIRAL PREVIEW • {channel_name} • XY ONLY • {self.view_zoom * 100:.0f}%"
        else:
            base_mode = "OVERVIEW" if self.overview_mode else "LIVE FIELD"
            mode = f"{base_mode} • {self.view_zoom * 100:.0f}%"
        self.viewer.create_text(
            width - 14,
            14,
            text=mode,
            anchor="ne",
            fill="#b9cbd0",
            font=("TkDefaultFont", 9, "bold"),
            tags="scene",
        )

    def _float_or(self, variable: tk.Variable, default: float) -> float:
        try:
            return float(variable.get())
        except (ValueError, tk.TclError):
            return default

    # -- plan ------------------------------------------------------------------
    def _build_plan(self) -> AcquisitionPlan:
        objective = self._selected_objective()
        channels = []
        for channel in self.config_data.channels:
            if self.channel_vars[channel.id].get():
                channels.append(
                    ChannelSetting(
                        channel_id=channel.id,
                        exposure_ms=self._float_or(self.exposure_vars[channel.id], -1),
                        power_percent=self._float_or(self.power_vars[channel.id], -1),
                    )
                )
        z = ZSettings(
            mode=self.z_mode_var.get(),
            bottom_um=self._float_or(self.z_bottom_var, 0),
            top_um=self._float_or(self.z_top_var, 0),
            step_um=self._float_or(self.z_step_var, -1),
            begin_captured=self.z_begin_captured,
            end_captured=self.z_end_captured,
        )
        options = AcquisitionOptions(
            tile_order=self.tile_order_var.get(),
            consistent_approach=self.consistent_approach_var.get(),
            shutter_per_frame=self.shutter_per_frame_var.get(),
            flat_field=self.flat_field_var.get(),
            focus_method=self.focus_method_var.get(),
        )
        return AcquisitionPlan(
            project=self.project_var.get().strip(),
            sample=self.sample_var.get().strip(),
            output_root=Path(self.output_var.get()).expanduser(),
            objective_id=objective.id,
            channels=tuple(channels),
            regions=tuple(self.regions),
            z=z,
            focus_points=tuple(self.focus_points),
            overlap_x_percent=self._float_or(self.overlap_x_var, -1),
            overlap_y_percent=self._float_or(self.overlap_y_var, -1),
            options=options,
        )

    def _input_changed(self, _event=None) -> None:
        self._update_offset_label()
        self._apply_locks()
        if self.controller.busy:
            return
        try:
            plan = self._build_plan()
            tiles, estimate = self.controller.estimate(plan)
        except (ValueError, KeyError) as exc:
            self.summary_var.set(str(exc))
            self.safety_var.set("● " + str(exc))
            self.safety_label.configure(fg=WARNING)
            return
        channel_names = [self.config_data.channel(item.channel_id).name for item in plan.channels]
        experiment = self.controller.experiment
        cycle_text = (
            f"Cycle {experiment.next_cycle_number} of experiment"
            if experiment is not None
            else "Cycle 1 (new experiment)"
        )
        if plan.regions:
            self.region_var.set(
                f"{len(plan.regions)} region{'s' if len(plan.regions) != 1 else ''} • "
                f"{len(plan.focus_points)} focus point{'s' if len(plan.focus_points) != 1 else ''} • "
                f"{estimate.tiles:,} tiles"
            )
            surface_text = ""
            if plan.z.mode == "focus_map" and len(plan.focus_points) >= 3:
                try:
                    from .focus import FocusSurface

                    low, high = FocusSurface(plan.focus_points, plan.options.focus_method).z_range(
                        tile.xy for tile in tiles
                    )
                    surface_text = f"\nSurface Z {low:.1f} … {high:.1f} µm"
                except ValueError:
                    surface_text = ""
            self.summary_var.set(
                f"{cycle_text}\n"
                f"{self.config_data.objective(plan.objective_id).name}\n"
                f"{', '.join(channel_names) or 'No channels'}\n\n"
                f"{estimate.tiles:,} tiles • {plan.options.tile_order}\n"
                f"Overlap X {plan.overlap_x_percent:.0f}%  •  Y {plan.overlap_y_percent:.0f}% (exact)\n"
                f"{estimate.frames:,} frames  •  {estimate.z_planes} Z plane"
                f"{'s' if estimate.z_planes != 1 else ''}{surface_text}\n"
                f"About {estimate.duration_label}  •  {estimate.data_label}"
            )
        else:
            self.region_var.set(
                f"No scan regions • {len(plan.focus_points)} focus point"
                f"{'s' if len(plan.focus_points) != 1 else ''}"
            )
            self.summary_var.set("Draw a region on the overview to calculate the scan.")
        problems = self.controller.preflight(plan)
        if problems:
            self.safety_var.set("● " + "\n● ".join(problems[:4]))
            self.safety_label.configure(fg=WARNING)
        else:
            self.safety_var.set("✓ Plan passes configured safety checks")
            self.safety_label.configure(fg=SUCCESS)

    # -- locks -----------------------------------------------------------------
    def _rois_locked(self) -> bool:
        return self.controller.experiment is not None

    def _apply_locks(self) -> None:
        if not hasattr(self, "scan_button"):
            return
        busy = self.controller.busy
        locked = self._rois_locked()
        lock_state = "disabled" if locked or busy else "normal"
        for widget in self.objective_lock_widgets:
            try:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="disabled" if locked or busy else "readonly")
                else:
                    widget.configure(state=lock_state)
            except tk.TclError:
                pass
        for button in self.draw_shape_buttons:
            button.configure(state=lock_state)
        if (locked or busy) and self.interaction_mode_var.get() in ("rectangle", "polygon"):
            self.interaction_mode_var.set("move_set")
            self._interaction_mode_changed()
        self.clear_button.configure(state=lock_state)
        self.load_roi_button.configure(state=lock_state)
        idle = "disabled" if busy else "normal"
        self.scan_button.configure(state=idle)
        self.new_experiment_button.configure(state=idle)
        self.open_experiment_button.configure(state=idle)
        resumable = (not busy) and self.controller.resume_info() is not None
        self.resume_button.configure(state="normal" if resumable else "disabled")
        self.pause_button.configure(state="normal" if busy else "disabled")
        self.abort_button.configure(state="normal" if busy else "disabled")
        experiment = self.controller.experiment
        next_cycle = experiment.next_cycle_number if experiment else 1
        self.scan_button.configure(text=f"Review and start cycle {next_cycle}")
        self._update_experiment_label()

    def _update_experiment_label(self) -> None:
        experiment = self.controller.experiment
        if experiment is None:
            self.experiment_var.set("No experiment — the first cycle creates one")
            return
        done = sum(1 for cycle in experiment.cycles if cycle.status == "complete")
        lines = [
            f"{experiment.project} / {experiment.sample}",
            f"{len(experiment.tiles):,} tiles • {done} cycle{'s' if done != 1 else ''} complete",
        ]
        info = self.controller.resume_info()
        if info is not None and not self.controller.busy:
            record, finished, total = info
            lines.append(f"Cycle {record.number} {record.status}: {finished}/{total} tiles saved")
        self.experiment_var.set("\n".join(lines))

    def _update_offset_label(self) -> None:
        experiment = self.controller.experiment
        if experiment is None:
            self.offset_var.set("")
            return
        try:
            dx, dy = experiment.offset_of(tuple(self.regions))
        except ValueError:
            self.offset_var.set("ROI set differs from the experiment")
            return
        self.offset_var.set(f"Offset vs cycle 1: ΔX {dx:+.1f} ΔY {dy:+.1f} µm")

    # -- experiment actions ----------------------------------------------------
    def _new_experiment(self) -> None:
        if self.controller.experiment is not None and not messagebox.askyesno(
            "New experiment",
            "Close the current experiment? Its data stays on disk and can be reopened.\n\n"
            "The next scan will create a new experiment with a new tile grid.",
            parent=self,
        ):
            return
        try:
            self.controller.new_experiment()
        except ControllerError as exc:
            messagebox.showerror("Cannot start a new experiment", str(exc), parent=self)
            return
        self.progress_text_var.set("New experiment — ROIs and objective can be edited")
        self._render_scene()
        self._input_changed()

    def _open_experiment(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Open experiment",
            initialdir=self.output_var.get(),
            filetypes=(("Experiment", "experiment.json"), ("JSON", "*.json")),
        )
        if not path:
            return
        try:
            experiment = self.controller.open_experiment(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, ControllerError) as exc:
            messagebox.showerror("Could not open experiment", str(exc), parent=self)
            return
        self._load_experiment_into_view(experiment)
        info = self.controller.resume_info()
        if info is not None:
            record, finished, total = info
            if messagebox.askyesno(
                "Resume unfinished cycle?",
                f"Cycle {record.number} stopped ({record.status}) with {finished} of {total} "
                "tiles saved.\n\nResume it now with its original settings?",
                parent=self,
            ):
                self._resume_cycle(confirm=False)

    def _load_experiment_into_view(self, experiment: Experiment) -> None:
        objective = self.config_data.objective(experiment.objective_id)
        last = experiment.cycles[-1] if experiment.cycles else None
        dx = last.offset_x_um if last else 0.0
        dy = last.offset_y_um if last else 0.0
        self.regions = list(experiment.regions_with_offset(dx, dy))
        self.focus_points = list(last.focus_points if last else experiment.base_focus_points)
        self.selected_focus_index = None
        self.project_var.set(experiment.project)
        self.sample_var.set(experiment.sample)
        self.output_var.set(str(experiment.root.parent.parent))
        self.overlap_x_var.set(experiment.overlap_x_percent)
        self.overlap_y_var.set(experiment.overlap_y_percent)
        self.tile_order_var.set(experiment.options.tile_order)
        if last is not None:
            selected = {item.channel_id: item for item in last.channels}
            for channel in self.config_data.channels:
                self.channel_vars[channel.id].set(channel.id in selected)
                if channel.id in selected:
                    self.exposure_vars[channel.id].set(selected[channel.id].exposure_ms)
                    self.power_vars[channel.id].set(selected[channel.id].power_percent)
            self.z_mode_var.set(last.z.mode)
            options = last.options or experiment.options
            self.focus_method_var.set(options.focus_method)
            self.shutter_per_frame_var.set(options.shutter_per_frame)
            self.consistent_approach_var.set(options.consistent_approach)
            self.flat_field_var.set(options.flat_field)
        self.objective_var.set(objective.name)
        if self.backend.current_objective_id != objective.id:
            self._objective_changed()
        self._sync_preview_channels()
        self._z_mode_changed()
        self.overview_mode = True
        self.preview_view = None
        self.progress_text_var.set(f"Experiment opened • {experiment.root}")
        self._render_scene()
        self._input_changed()

    def _resume_cycle(self, confirm: bool = True) -> None:
        info = self.controller.resume_info()
        if info is None:
            return
        record, finished, total = info
        if confirm and not messagebox.askokcancel(
            "Resume cycle",
            f"Resume cycle {record.number} at the first unsaved tile "
            f"({finished} of {total} already saved)?\n\n"
            "The original channels, exposures, powers and focus positions are reused.",
            parent=self,
        ):
            return
        try:
            self.controller.resume_cycle()
        except (ControllerError, HardwareError, OSError, ValueError) as exc:
            messagebox.showerror("Could not resume", str(exc), parent=self)
            return
        self._after_launch()

    # -- scanning --------------------------------------------------------------
    def _start_scan(self) -> None:
        self._set_live(False)
        self._stop_spiral_preview(clear_overlay=True)
        self._stop_refocus()
        try:
            plan = self._build_plan()
            problems = self.controller.preflight(plan)
        except (ValueError, KeyError) as exc:
            problems = [str(exc)]
        if problems:
            messagebox.showwarning(
                "Preflight needs attention",
                "Please fix these items before scanning:\n\n• " + "\n• ".join(problems),
                parent=self,
            )
            return
        tiles, estimate = self.controller.estimate(plan)
        prompts = [
            self.config_data.channel(setting.channel_id).manual_prompt
            for setting in plan.channels
            if self.config_data.channel(setting.channel_id).manual_prompt
        ]
        prompt_text = ""
        if prompts:
            prompt_text = "\n\nManual checks:\n• " + "\n• ".join(prompts)
        experiment = self.controller.experiment
        cycle_number = experiment.next_cycle_number if experiment else 1
        offset_text = ""
        if experiment is not None:
            dx, dy = experiment.offset_of(plan.regions)
            offset_text = f"\nROI offset vs cycle 1: ΔX {dx:+.1f}  ΔY {dy:+.1f} µm"
        mode = "simulated scan" if self.backend.simulated else "scan on the microscope"
        confirmation = (
            f"Cycle {cycle_number}{' (new experiment)' if experiment is None else ''}\n"
            f"Objective: {self.config_data.objective(plan.objective_id).name}\n"
            f"Tiles: {estimate.tiles:,} ({plan.options.tile_order})  |  Frames: {estimate.frames:,}\n"
            f"Estimated data: {estimate.data_label}  |  Time: {estimate.duration_label}"
            f"{offset_text}{prompt_text}\n\nStart the {mode}?"
        )
        if not messagebox.askokcancel("Ready to scan", confirmation, parent=self):
            return
        try:
            self.controller.start_cycle(plan)
        except (ControllerError, HardwareError, OSError, ValueError) as exc:
            messagebox.showerror("Could not start the cycle", str(exc), parent=self)
            self._input_changed()
            return
        self._after_launch()

    def _after_launch(self) -> None:
        mosaic = self.controller.mosaic
        assert mosaic is not None
        self.mosaic_view_box.configure(values=mosaic.display_modes)
        self.mosaic_view_var.set(DiskBackedMosaic.COMPOSITE)
        self.preview_view = (
            (mosaic.bounds.x_min_um + mosaic.bounds.x_max_um) / 2,
            (mosaic.bounds.y_min_um + mosaic.bounds.y_max_um) / 2,
            mosaic.bounds.width_um * 1.05,
            mosaic.bounds.height_um * 1.05,
        )
        self.view_zoom = 1.0
        self.view_center_override = None
        self.overview_mode = True
        self.last_mosaic_render_time = 0.0
        self.progress["value"] = 0
        self.status_var.set(
            "SIMULATED ACQUISITION RUNNING" if self.backend.simulated else "ACQUISITION RUNNING"
        )
        self.progress_text_var.set("Starting acquisition…")
        self._apply_locks()
        self._poll_scan()

    def _poll_scan(self) -> None:
        status = self.controller.status()
        if status is None:
            return
        self.progress["value"] = status.fraction * 100
        channel = (
            self.config_data.channel(status.current_channel_id).name
            if status.current_channel_id
            else ""
        )
        parts = [
            f"Cycle {self.controller.active_cycle}",
            f"{status.tiles_done:,}/{status.tiles_total:,} tiles saved",
        ]
        if status.state in ("pausing", "paused", "aborting", "draining", "post-processing"):
            parts.append(status.message)
        elif status.current_tile is not None:
            parts.append(
                f"tile {status.current_tile.index + 1} • {channel} • Z {status.current_z_index + 1}/"
                f"{status.current_z_planes}"
            )
        if status.backpressure:
            parts.append("SSD buffer full — camera waiting safely")
        acquired_since = status.tiles_done - status.tiles_skipped
        if status.started_monotonic and acquired_since > 0 and not status.finished:
            elapsed = time.monotonic() - status.started_monotonic
            remaining_tiles = status.tiles_total - status.tiles_done
            eta = elapsed / acquired_since * remaining_tiles
            parts.append(f"~{int(eta // 60)}m {int(eta % 60)}s left")
        self.progress_text_var.set(" • ".join(parts))
        self._update_position_label()
        now = time.monotonic()
        if now - self.last_mosaic_render_time >= 0.3 or status.finished:
            self._render_scene()
            self.last_mosaic_render_time = now
        if status.finished:
            self.progress_job = None
            self._scan_finished_ui(status)
            return
        self.progress_job = self.after(150, self._poll_scan)

    def _scan_finished_ui(self, status) -> None:
        self.status_var.set(self.idle_status if status.state != "failed" else "ACQUISITION FAILED")
        self.pause_button.configure(text="Pause")
        experiment = self.controller.experiment
        folder = (
            experiment.cycle_folder(self.controller.active_cycle)
            if experiment is not None and self.controller.active_cycle
            else None
        )
        self._apply_locks()
        self._input_changed()
        if status.state == "complete":
            self.progress_text_var.set(f"{status.message} • {folder}")
            messagebox.showinfo(
                "Cycle complete",
                f"Cycle {self.controller.active_cycle} finished: {status.tiles_done:,} tiles.\n\n"
                f"{status.message}\n\nData:\n{folder}\n\n"
                "Strip/restain, then re-align the ROI set and refocus before the next cycle.",
                parent=self,
            )
        elif status.state == "aborted":
            self.progress_text_var.set(
                f"Stopped • {status.tiles_done}/{status.tiles_total} tiles saved • resumable"
            )
        else:
            self.progress_text_var.set(status.error or "Acquisition failed")
            messagebox.showerror(
                "Acquisition stopped",
                f"{status.error}\n\n{status.tiles_done} of {status.tiles_total} tiles are saved. "
                "Fix the problem, then press Resume cycle.",
                parent=self,
            )

    def _toggle_pause(self) -> None:
        status = self.controller.status()
        if status is None:
            return
        if status.state in ("pausing", "paused"):
            self.controller.resume()
            self.pause_button.configure(text="Pause")
        else:
            self.controller.pause()
            self.pause_button.configure(text="Continue")

    def _abort_scan_clicked(self) -> None:
        if not messagebox.askyesno(
            "Abort acquisition",
            "Stop after the current frame? The unfinished tile is discarded; every saved "
            "tile is kept and the cycle can be resumed later.",
            parent=self,
        ):
            return
        self.controller.abort()

    # -- refocus & alignment ---------------------------------------------------
    def _toggle_refocus(self) -> None:
        if self.refocus_index is not None:
            self._stop_refocus()
            self.progress_text_var.set("Refocus stopped")
            return
        if not self.focus_points:
            messagebox.showinfo("No focus points", "Capture or load focus points first.", parent=self)
            return
        if self.controller.busy:
            return
        self.interaction_mode_var.set("focus")
        self._interaction_mode_changed()
        self.refocus_button.configure(text="Stop refocus")
        self.update_z_button.configure(text="Accept Z ▶")
        self._refocus_goto(0)

    def _refocus_goto(self, index: int) -> None:
        self.refocus_index = index
        self.selected_focus_index = index
        point = self.focus_points[index]
        moved = self._safe_move(self.backend.move_xy, point.x_um, point.y_um) and self._safe_move(
            self.backend.move_z, point.z_um
        )
        self.overview_mode = False
        self.preview_view = None
        self.view_center_override = None
        self._set_live(True)
        if moved:
            self.progress_text_var.set(
                f"Refocus F{index + 1} of {len(self.focus_points)}: focus live, then Accept Z"
            )
        self._update_position_label()
        self._update_focus_status()
        self._render_scene()

    def _stop_refocus(self) -> None:
        self.refocus_index = None
        if hasattr(self, "refocus_button"):
            self.refocus_button.configure(text="Refocus all")
            self.update_z_button.configure(text="Update Z")

    def _align_selected_focus_to_stage(self) -> None:
        if self.selected_focus_index is None:
            messagebox.showinfo(
                "Select a focus point",
                "Click a focus point (Focus points mode), navigate the stage so the same "
                "landmark is centred in Live, then press this button. The whole ROI set "
                "moves so that point sits at the current stage position (Z is kept).",
                parent=self,
            )
            return
        point = self.focus_points[self.selected_focus_index]
        position = self.backend.position
        dx, dy = position.x_um - point.x_um, position.y_um - point.y_um
        self.regions = [region.translated(dx, dy) for region in self.regions]
        self.focus_points = [p.translated_xy(dx, dy) for p in self.focus_points]
        self.progress_text_var.set(
            f"ROI set moved by ΔX {dx:+.1f} ΔY {dy:+.1f} µm to align F{self.selected_focus_index + 1}"
        )
        self._render_scene()
        self._input_changed()

    def _show_admin(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Simulator & configuration")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.configure(bg=PANEL)
        body = ttk.Frame(dialog, style="Panel.TFrame", padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text="Hardware boundary",
            style="Panel.TLabel",
            font=("TkDefaultFont", 15, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            body,
            text=(
                "This build uses simulated stage, camera, turret, and acquisition calls. "
                "Set hardware.backend to \"mmcore\" in the configuration to use the "
                "pymmcore-plus adapter."
                if self.backend.simulated
                else f"Backend: {self.backend.name} (pymmcore-plus). Moves outside the "
                "configured limits are refused before any command is sent."
            ),
            style="Muted.TLabel",
            wraplength=480,
            justify="left",
        ).pack(anchor="w", pady=(4, 16))
        details = (
            f"Configuration: {self.config_path}\n"
            f"Backend: {self.backend.name}\n"
            f"Microscope: {self.config_data.microscope_name}\n"
            f"Objectives: {len(self.config_data.objectives)}\n"
            f"Channels: {len(self.config_data.channels)}\n"
            f"Stage X: {self.config_data.stage_limits.x_min_um:g} to "
            f"{self.config_data.stage_limits.x_max_um:g} µm\n"
            f"Stage Y: {self.config_data.stage_limits.y_min_um:g} to "
            f"{self.config_data.stage_limits.y_max_um:g} µm"
        )
        ttk.Label(body, text=details, style="Panel.TLabel", justify="left").pack(anchor="w")
        ttk.Button(body, text="Close", command=dialog.destroy).pack(anchor="e", pady=(20, 0))


def _frame_to_pgm(frame: np.ndarray, width: int, height: int) -> bytes:
    """Contrast-stretched 8-bit PGM of a camera frame, resized (nearest) for Tk."""

    data = np.asarray(frame, dtype=np.float32)
    low, high = np.percentile(data, (0.5, 99.5)) if data.size else (0.0, 1.0)
    rows = (np.arange(height) * data.shape[0] / height).astype(np.int64)
    cols = (np.arange(width) * data.shape[1] / width).astype(np.int64)
    small = data[np.ix_(rows, cols)]
    gray = (np.clip((small - low) / max(high - low, 1.0), 0, 1) * 255).astype(np.uint8)
    return f"P5\n{width} {height}\n255\n".encode("ascii") + gray.tobytes()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Microscope Scan Console")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="configuration JSON")
    parser.add_argument(
        "--backend",
        choices=("simulator", "mmcore"),
        help="override hardware.backend from the configuration",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.backend:
            config = replace(config, hardware={**config.hardware, "backend": args.backend})
        backend = create_backend(config)
    except (OSError, ValueError, json.JSONDecodeError, HardwareError) as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Startup error", str(exc))
        root.destroy()
        raise SystemExit(1) from exc
    app = ScanConsole(config, backend)
    app.config_path = Path(args.config)
    app.mainloop()


if __name__ == "__main__":
    main()
