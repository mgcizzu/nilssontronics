"""Reusable Qt widgets for the scan window. No acquisition logic lives here."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import Channel, ChannelSetting, FocusPoint, ScanRegion

EXPOSURE_MIN_MS = 0.1
EXPOSURE_SLIDER_STEPS = 1000


def _exposure_from_slider(position: int, maximum_ms: float) -> float:
    """Map a slider position logarithmically onto the channel exposure range."""

    fraction = max(0.0, min(1.0, position / EXPOSURE_SLIDER_STEPS))
    return math.exp(
        math.log(EXPOSURE_MIN_MS)
        + fraction * (math.log(maximum_ms) - math.log(EXPOSURE_MIN_MS))
    )


def _slider_from_exposure(exposure_ms: float, maximum_ms: float) -> int:
    exposure_ms = max(EXPOSURE_MIN_MS, min(maximum_ms, exposure_ms))
    fraction = (math.log(exposure_ms) - math.log(EXPOSURE_MIN_MS)) / (
        math.log(maximum_ms) - math.log(EXPOSURE_MIN_MS)
    )
    return round(fraction * EXPOSURE_SLIDER_STEPS)


def _set_slider_without_feedback(slider: QSlider, position: int) -> None:
    blocker = QSignalBlocker(slider)
    slider.setValue(position)
    del blocker


def spin(value: float, low: float, high: float, decimals: int = 1, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setValue(value)
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(f" {suffix}")
    return box


class ChannelTable(QTableWidget):
    """Channel selection with free exposure/power per cycle (no marker mapping)."""

    changed = Signal()

    def __init__(self, channels: tuple[Channel, ...], parent: QWidget | None = None) -> None:
        super().__init__(len(channels), 3, parent)
        self._channels = channels
        self.setHorizontalHeaderLabels(["Channel", "Exposure", "Power"])
        self.verticalHeader().hide()
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._checks: dict[str, QCheckBox] = {}
        self._exposure: dict[str, QDoubleSpinBox] = {}
        self._exposure_sliders: dict[str, QSlider] = {}
        self._power: dict[str, QDoubleSpinBox] = {}
        for row, channel in enumerate(channels):
            check = QCheckBox(channel.name)
            check.setToolTip(f"{channel.name} • emission {channel.emission_wavelength_nm:g} nm")
            check.toggled.connect(self.changed)
            exposure = spin(
                channel.default_exposure_ms,
                EXPOSURE_MIN_MS,
                channel.max_exposure_ms,
                1,
                "ms",
            )
            exposure_slider = QSlider(Qt.Orientation.Horizontal)
            exposure_slider.setRange(0, EXPOSURE_SLIDER_STEPS)
            exposure_slider.setPageStep(25)
            exposure_slider.setMinimumWidth(72)
            exposure_slider.setValue(
                _slider_from_exposure(channel.default_exposure_ms, channel.max_exposure_ms)
            )
            exposure_slider.setAccessibleName(f"{channel.name} exposure")
            exposure_slider.setToolTip(
                f"{channel.name} exposure, logarithmic {EXPOSURE_MIN_MS:g}–"
                f"{channel.max_exposure_ms:g} ms"
            )
            exposure_cell = QWidget()
            exposure_layout = QHBoxLayout(exposure_cell)
            exposure_layout.setContentsMargins(0, 0, 0, 0)
            exposure_layout.setSpacing(4)
            exposure_layout.addWidget(exposure_slider, 1)
            exposure_layout.addWidget(exposure)
            power = spin(channel.default_power_percent, 0, channel.max_power_percent, 0, "%")
            exposure_slider.valueChanged.connect(
                lambda position, box=exposure, maximum=channel.max_exposure_ms: box.setValue(
                    _exposure_from_slider(position, maximum)
                )
            )
            exposure.valueChanged.connect(
                lambda value, slider=exposure_slider, maximum=channel.max_exposure_ms: (
                    _set_slider_without_feedback(
                        slider, _slider_from_exposure(value, maximum)
                    )
                )
            )
            exposure.valueChanged.connect(self.changed)
            power.valueChanged.connect(self.changed)
            self.setCellWidget(row, 0, check)
            self.setCellWidget(row, 1, exposure_cell)
            self.setCellWidget(row, 2, power)
            self._checks[channel.id] = check
            self._exposure[channel.id] = exposure
            self._exposure_sliders[channel.id] = exposure_slider
            self._power[channel.id] = power
        self.resizeColumnsToContents()
        for column in (1, 2):
            self.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.setMinimumHeight(self.verticalHeader().length() + self.horizontalHeader().height() + 4)
        self.refresh_swatches()

    def refresh_swatches(self) -> None:
        """Keep channel colours readable on both light and dark themes."""

        for channel in self._channels:
            self._checks[channel.id].setIcon(_swatch(channel.color))

    def settings(self) -> tuple[ChannelSetting, ...]:
        return tuple(
            ChannelSetting(channel.id, self._exposure[channel.id].value(), self._power[channel.id].value())
            for channel in self._channels
            if self._checks[channel.id].isChecked()
        )

    def set_settings(self, settings: tuple[ChannelSetting, ...]) -> None:
        selected = {item.channel_id: item for item in settings}
        for channel in self._channels:
            self._checks[channel.id].setChecked(channel.id in selected)
            if channel.id in selected:
                self._exposure[channel.id].setValue(selected[channel.id].exposure_ms)
                self._power[channel.id].setValue(selected[channel.id].power_percent)


class RectangleDialog(QDialog):
    """Numeric rectangle entry (stage µm). Drawing on the overview comes later."""

    def __init__(self, center: tuple[float, float], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add scan region")
        form = QFormLayout(self)
        self.width_box = spin(2000, 1, 200000, 1, "µm")
        self.height_box = spin(2000, 1, 200000, 1, "µm")
        self.x_box = spin(center[0], -1e6, 1e6, 1, "µm")
        self.y_box = spin(center[1], -1e6, 1e6, 1, "µm")
        form.addRow("Centre X", self.x_box)
        form.addRow("Centre Y", self.y_box)
        form.addRow("Width", self.width_box)
        form.addRow("Height", self.height_box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def region(self) -> ScanRegion:
        w, h = self.width_box.value(), self.height_box.value()
        return ScanRegion(self.x_box.value() - w / 2, self.y_box.value() - h / 2, w, h)


class ItemList(QWidget):
    """A list with add/remove buttons, used for regions and focus points."""

    changed = Signal()

    def __init__(self, describe: Callable[[object], str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._describe = describe
        self.items: list = []
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        self.buttons = QHBoxLayout()
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list)
        layout.addLayout(self.buttons)
        self.buttons.addStretch(1)
        self.buttons.addWidget(self.remove_button)

    def add_button(self, text: str, callback: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        self.buttons.insertWidget(self.buttons.count() - 2, button)
        return button

    def set_items(self, items) -> None:
        self.items = list(items)
        self.list.clear()
        for item in self.items:
            self.list.addItem(self._describe(item))
        self.changed.emit()

    def append(self, item) -> None:
        self.set_items([*self.items, item])

    def remove_selected(self) -> None:
        row = self.list.currentRow()
        if row >= 0:
            del self.items[row]
            self.set_items(self.items)

    def selected_index(self) -> int | None:
        row = self.list.currentRow()
        return row if 0 <= row < len(self.items) else None

    def set_locked(self, locked: bool) -> None:
        for index in range(self.buttons.count()):
            widget = self.buttons.itemAt(index).widget()
            if widget is not None:
                widget.setEnabled(not locked)


def describe_region(region: ScanRegion) -> str:
    return (
        f"{region.shape}: X {region.x_um:.0f}…{region.x_um + region.width_um:.0f}  "
        f"Y {region.y_um:.0f}…{region.y_um + region.height_um:.0f} µm"
    )


def describe_focus(point: FocusPoint) -> str:
    return f"X {point.x_um:.0f}  Y {point.y_um:.0f}  Z {point.z_um:.2f} µm"


class StageJog(QWidget):
    """XY/Z jog pad. Every move goes through the backend, so limits are checked first."""

    move_requested = Signal(float, float, float)  # dx, dy, dz in µm

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.xy_step = spin(100, 0.1, 20000, 1, "µm")
        self.z_step = spin(1, 0.01, 500, 2, "µm")
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        for text, row, col, d in (
            ("▲", 0, 1, (0, -1, 0)),
            ("◀", 1, 0, (-1, 0, 0)),
            ("▶", 1, 2, (1, 0, 0)),
            ("▼", 2, 1, (0, 1, 0)),
        ):
            button = QPushButton(text)
            button.setFixedWidth(36)
            button.clicked.connect(lambda _=False, d=d: self._emit_xy(*d))
            grid.addWidget(button, row, col)
        z_up = QPushButton("Z +")
        z_down = QPushButton("Z −")
        z_up.clicked.connect(lambda: self.move_requested.emit(0, 0, self.z_step.value()))
        z_down.clicked.connect(lambda: self.move_requested.emit(0, 0, -self.z_step.value()))
        grid.addWidget(z_up, 0, 3)
        grid.addWidget(z_down, 2, 3)
        grid.addWidget(QLabel("XY step"), 3, 0, 1, 2)
        grid.addWidget(self.xy_step, 3, 2, 1, 2)
        grid.addWidget(QLabel("Z step"), 4, 0, 1, 2)
        grid.addWidget(self.z_step, 4, 2, 1, 2)

    def _emit_xy(self, sx: int, sy: int, _sz: int) -> None:
        step = self.xy_step.value()
        self.move_requested.emit(sx * step, sy * step, 0.0)


class FrameView(QWidget):
    """Camera frame display. Uses ndv when it can start, else a plain image label.

    ndv gives proper LUT/contrast for 16-bit frames; the fallback keeps the view
    usable on machines without an OpenGL canvas (CI, remote desktops).
    """

    def __init__(self, parent: QWidget | None = None, use_ndv: bool = True) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._viewer = None
        self._label: QLabel | None = None
        self._shape: tuple[int, ...] | None = None
        if use_ndv:
            try:
                import ndv

                self._viewer = ndv.ArrayViewer(np.zeros((64, 64), dtype=np.uint16))
                layout.addWidget(self._viewer.widget())
            except Exception:  # pragma: no cover - depends on the GL stack
                self._viewer = None
        if self._viewer is None:
            self._label = QLabel("No image")
            self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            self._label.setStyleSheet("background: #111; color: #888;")
            layout.addWidget(self._label)

    @property
    def uses_ndv(self) -> bool:
        return self._viewer is not None

    def show_frame(self, frame: np.ndarray) -> None:
        if self._viewer is not None:
            self._viewer.data = frame
            return
        assert self._label is not None
        self._label.setPixmap(_pixmap_from_array(frame, self._label.size()))


def _swatch(color: str, size: int = 10) -> QIcon:
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(0, 0, 0, 60))
    painter.setBrush(QColor(color))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return QIcon(pixmap)


def _pixmap_from_array(array: np.ndarray, size) -> QPixmap:
    if array.ndim == 2:
        data = array.astype(np.float32)
        low, high = np.percentile(data, (0.5, 99.5)) if data.size else (0, 1)
        scaled = np.clip((data - low) / max(high - low, 1.0), 0, 1)
        gray = np.ascontiguousarray((scaled * 255).astype(np.uint8))
        image = QImage(gray.data, gray.shape[1], gray.shape[0], gray.shape[1], QImage.Format.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(array[:, :, :3].astype(np.uint8))
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy()).scaled(
        size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
