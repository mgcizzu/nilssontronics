"""Reusable Qt widgets for the scan window. No acquisition logic lives here."""

from __future__ import annotations

from typing import Callable

import numpy as np
from qtpy.QtCore import Qt, Signal
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import Channel, ChannelSetting, FocusPoint, ScanRegion

EXPOSURE_MIN_MS = 0.1


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

    def __init__(
        self,
        channels: tuple[Channel, ...],
        parent: QWidget | None = None,
        adjustable_power_ids: set[str] | None = None,
    ) -> None:
        super().__init__(len(channels), 3, parent)
        self._channels = channels
        self.setHorizontalHeaderLabels(["Channel", "Exposure", "Power"])
        self.verticalHeader().hide()
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._checks: dict[str, QCheckBox] = {}
        self._exposure: dict[str, QDoubleSpinBox] = {}
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
            exposure.setFixedWidth(82)
            exposure.setAccessibleName(f"{channel.name} exposure")
            exposure.setToolTip(
                f"{channel.name} exposure ({EXPOSURE_MIN_MS:g}–"
                f"{channel.max_exposure_ms:g} ms)"
            )
            power = spin(channel.default_power_percent, 0, channel.max_power_percent, 0, "%")
            power.setFixedWidth(60)
            if adjustable_power_ids is not None and channel.id not in adjustable_power_ids:
                power.setEnabled(False)
                power.setToolTip(
                    f"Fixed at {channel.default_power_percent:g}% by the Micro-Manager preset. "
                    f"Add hardware.power_properties.{channel.id} to make it adjustable."
                )
            exposure.valueChanged.connect(self.changed)
            power.valueChanged.connect(self.changed)
            self.setCellWidget(row, 0, check)
            self.setCellWidget(row, 1, exposure)
            self.setCellWidget(row, 2, power)
            self._checks[channel.id] = check
            self._exposure[channel.id] = exposure
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
        self.buttons = QGridLayout()
        self.buttons.setHorizontalSpacing(6)
        self.buttons.setVerticalSpacing(6)
        self._action_buttons: list[QPushButton] = []
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.list)
        layout.addLayout(self.buttons)
        self._relayout_buttons()

    def add_button(self, text: str, callback: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        self._action_buttons.append(button)
        self._relayout_buttons()
        return button

    def _relayout_buttons(self) -> None:
        """Lay actions out in pairs so descriptive labels survive a narrow panel."""

        while self.buttons.count():
            self.buttons.takeAt(0)
        all_buttons = [*self._action_buttons, self.remove_button]
        for index, button in enumerate(all_buttons):
            self.buttons.addWidget(button, index // 2, index % 2)
        self.buttons.setColumnStretch(0, 1)
        self.buttons.setColumnStretch(1, 1)

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
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        pads = QHBoxLayout()
        pads.setContentsMargins(0, 0, 0, 0)
        pads.setSpacing(14)
        pads.addStretch(1)

        xy_pad = QWidget()
        xy_pad.setObjectName("jogPad")
        grid = QGridLayout(xy_pad)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        for text, row, col, d in (
            ("▲", 0, 1, (0, -1, 0)),
            ("◀", 1, 0, (-1, 0, 0)),
            ("▶", 1, 2, (1, 0, 0)),
            ("▼", 2, 1, (0, 1, 0)),
        ):
            button = QPushButton(text)
            button.setProperty("jog", True)
            button.setFixedSize(38, 34)
            button.clicked.connect(lambda _=False, d=d: self._emit_xy(*d))
            grid.addWidget(button, row, col)

        pads.addWidget(xy_pad)
        z_pad = QVBoxLayout()
        z_pad.setSpacing(4)
        z_up = QPushButton("Z +")
        z_down = QPushButton("Z −")
        for button in (z_up, z_down):
            button.setProperty("jog", True)
            button.setFixedSize(64, 34)
        z_up.clicked.connect(lambda: self.move_requested.emit(0, 0, self.z_step.value()))
        z_down.clicked.connect(lambda: self.move_requested.emit(0, 0, -self.z_step.value()))
        z_pad.addWidget(z_up)
        z_pad.addStretch(1)
        z_pad.addWidget(z_down)
        pads.addLayout(z_pad)
        pads.addStretch(1)
        root.addLayout(pads)

        steps = QFormLayout()
        steps.setContentsMargins(0, 0, 0, 0)
        steps.setHorizontalSpacing(10)
        steps.setVerticalSpacing(6)
        steps.addRow("XY step", self.xy_step)
        steps.addRow("Z step", self.z_step)
        root.addLayout(steps)

    def _emit_xy(self, sx: int, sy: int, _sz: int) -> None:
        step = self.xy_step.value()
        self.move_requested.emit(sx * step, sy * step, 0.0)


class ZWheelControl(QWidget):
    """A deliberate mouse-wheel target plus a live Z-stack plane navigator."""

    jog_requested = Signal(int)  # signed number of wheel notches

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._wheel_delta = 0
        self.setObjectName("zWheelZone")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(128)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Z mouse wheel control and stack viewer")
        self.setToolTip(
            "Scroll here to move focus Z. Wheel up: +Z; wheel down: −Z. "
            "Distance per notch comes from Stage controls → Z step."
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(4)
        title = QLabel("Z")
        title.setProperty("wheelTitle", True)
        self.position_label = QLabel("0.00 µm")
        self.position_label.setProperty("wheelValue", True)
        self.ruler = _ZStackRuler()
        self.ruler.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.plane_label = QLabel("single")
        self.plane_label.setProperty("wheelValue", True)
        wheel_label = QLabel("↕ wheel")
        for label in (title, self.position_label, self.plane_label, wheel_label):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.position_label)
        layout.addLayout(header)
        layout.addWidget(self.ruler, 1)
        footer = QHBoxLayout()
        footer.addWidget(self.plane_label)
        footer.addStretch(1)
        footer.addWidget(wheel_label)
        layout.addLayout(footer)

    @property
    def plane_count(self) -> int:
        return len(self.ruler.positions)

    @property
    def active_plane_index(self) -> int | None:
        return self.ruler.active_index

    def set_stack(self, positions: tuple[float, ...], active_z_um: float) -> None:
        self.position_label.setText(f"{active_z_um:.2f} µm")
        self.ruler.set_stack(positions, active_z_um)
        index = self.ruler.active_index
        self.plane_label.setText(
            f"{index + 1} / {len(positions)}" if index is not None else "single"
        )

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        delta = event.angleDelta().y()
        if not delta:
            event.ignore()
            return
        self._wheel_delta += delta
        notches = int(self._wheel_delta / 120)
        if notches:
            self._wheel_delta -= notches * 120
            self.jog_requested.emit(notches)
        event.accept()


class _ZStackRuler(QWidget):
    """Paint planned planes as a ladder and highlight the nearest active one."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.positions: tuple[float, ...] = ()
        self.active_z_um = 0.0
        self.active_index: int | None = None
        self.setMinimumHeight(46)

    def set_stack(self, positions: tuple[float, ...], active_z_um: float) -> None:
        self.positions = tuple(positions)
        self.active_z_um = active_z_um
        self.active_index = (
            min(range(len(self.positions)), key=lambda i: abs(self.positions[i] - active_z_um))
            if self.positions else None
        )
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        from .theme import current_palette

        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        x = self.width() // 2
        top, bottom = 6, max(7, self.height() - 7)
        painter.setPen(QColor(palette.control_border))
        painter.drawLine(x, top, x, bottom)

        count = len(self.positions)
        if count:
            visible = set(range(count))
            if count > 48:
                stride = math.ceil(count / 48)
                visible = set(range(0, count, stride)) | {count - 1}
            for index in visible:
                y = top if count == 1 else round(top + index * (bottom - top) / (count - 1))
                width = 9 if index in (0, count - 1) else 6
                painter.setPen(QColor(palette.text_secondary))
                painter.drawLine(x - width, y, x + width, y)

            if self.active_index is not None:
                y = (
                    top
                    if count == 1
                    else round(top + self.active_index * (bottom - top) / (count - 1))
                )
                painter.setPen(QColor(palette.accent))
                painter.setBrush(QColor(palette.accent))
                painter.drawRoundedRect(x - 14, y - 3, 28, 7, 3, 3)
        else:
            painter.setPen(QColor(palette.text_disabled))
            painter.drawEllipse(x - 3, (top + bottom) // 2 - 3, 6, 6)
        painter.end()


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
