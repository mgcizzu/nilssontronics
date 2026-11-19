"""A Windows 11 / Fluent-style look for the Qt view, without extra dependencies.

Everything here is our own: colour tokens in the spirit of Microsoft's public
Fluent design guidance, a stylesheet generated from them, and a toggle switch.
Dark and light themes follow the system by default; the system accent colour is
used when Qt can read it. The application keeps its existing tabbed layout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import tempfile

from qtpy.QtCore import QByteArray, QEasingCurve, QPointF, QRectF, QSize, Qt, QVariantAnimation
from qtpy.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap
from qtpy.QtWidgets import QApplication, QCheckBox, QWidget

THEME_MODES = ("system", "light", "dark")
FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Arial"]


@dataclass(frozen=True)
class Palette:
    dark: bool
    window: str
    rail: str
    card: str
    card_border: str
    control: str
    control_hover: str
    control_pressed: str
    control_border: str
    control_bottom: str
    input: str
    text: str
    text_secondary: str
    text_disabled: str
    subtle_hover: str
    subtle_pressed: str
    accent: str
    accent_hover: str
    accent_pressed: str
    on_accent: str
    success: str
    critical: str
    caution: str


DARK = Palette(
    dark=True,
    window="#202020", rail="#1c1c1c", card="#2b2b2b", card_border="#1f1f1f",
    control="#2d2d2d", control_hover="#323232", control_pressed="#272727",
    control_border="#3a3a3a", control_bottom="#454545", input="#1f1f1f",
    text="#ffffff", text_secondary="#c5c5c5", text_disabled="#787878",
    subtle_hover="rgba(255, 255, 255, 0.06)", subtle_pressed="rgba(255, 255, 255, 0.04)",
    accent="#60cdff", accent_hover="#5abce9", accent_pressed="#52a9d1", on_accent="#000000",
    success="#6ccb5f", critical="#ff99a4", caution="#fce100",
)

LIGHT = Palette(
    dark=False,
    window="#f3f3f3", rail="#ebebeb", card="#fbfbfb", card_border="#e5e5e5",
    control="#ffffff", control_hover="#f9f9f9", control_pressed="#f5f5f5",
    control_border="#e5e5e5", control_bottom="#cccccc", input="#ffffff",
    text="#1a1a1a", text_secondary="#5d5d5d", text_disabled="#a0a0a0",
    subtle_hover="rgba(0, 0, 0, 0.04)", subtle_pressed="rgba(0, 0, 0, 0.03)",
    accent="#005fb8", accent_hover="#1975c5", accent_pressed="#3183ca", on_accent="#ffffff",
    success="#0f7b0f", critical="#c42b1c", caution="#9d5d00",
)


def system_prefers_dark(app: QApplication | None = None) -> bool:
    app = app or QApplication.instance()
    try:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:  # Qt < 6.5: fall back to the palette's lightness
        return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def palette_for(mode: str, app: QApplication | None = None, use_system_accent: bool = True) -> Palette:
    app = app or QApplication.instance()
    dark = system_prefers_dark(app) if mode == "system" else mode == "dark"
    base = DARK if dark else LIGHT
    accent = _system_accent(app) if use_system_accent else None
    if accent is None:
        return base
    # Fluent uses a lighter accent shade on dark backgrounds, a darker one on light.
    shade = accent.lighter(150) if dark and accent.lightness() < 140 else accent
    if not dark and shade.lightness() > 120:
        shade = shade.darker(140)
    on_accent = "#000000" if dark else "#ffffff"
    return replace(
        base,
        accent=shade.name(),
        accent_hover=(shade.darker(110) if dark else shade.lighter(115)).name(),
        accent_pressed=(shade.darker(120) if dark else shade.lighter(125)).name(),
        on_accent=on_accent,
    )


def _system_accent(app: QApplication) -> QColor | None:
    """The OS accent colour (Windows/macOS), or None if Qt only has its default."""

    if sys.platform not in ("win32", "darwin"):
        return None
    try:
        color = app.palette().color(QPalette.ColorRole.Accent)
    except AttributeError:
        return None
    return color if color.isValid() and color.saturation() > 40 else None


def apply_theme(app: QApplication, mode: str = "system") -> Palette:
    """Apply font, palette and stylesheet to the whole application."""

    if mode not in THEME_MODES:
        raise ValueError(f"Unknown theme mode: {mode}")
    p = palette_for(mode, app)
    font = QFont()
    installed = set(QFontDatabase.families())
    families = [family for family in FONT_FAMILIES if family in installed]
    if families:  # only installed ones: avoids Qt's slow missing-font alias search
        font.setFamilies(families)
    font.setPointSizeF(10.0 if sys.platform == "win32" else 13.0)
    app.setFont(font)
    qpal = app.palette()
    for role, value in (
        (QPalette.ColorRole.Window, p.window), (QPalette.ColorRole.Base, p.input),
        (QPalette.ColorRole.Text, p.text), (QPalette.ColorRole.WindowText, p.text),
        (QPalette.ColorRole.Button, p.control), (QPalette.ColorRole.ButtonText, p.text),
        (QPalette.ColorRole.Highlight, p.accent), (QPalette.ColorRole.HighlightedText, p.on_accent),
        (QPalette.ColorRole.ToolTipBase, p.card), (QPalette.ColorRole.ToolTipText, p.text),
        (QPalette.ColorRole.PlaceholderText, p.text_disabled),
    ):
        qpal.setColor(role, QColor(value))
    app.setPalette(qpal)
    app.setStyleSheet(stylesheet(p, _glyph_files(p)))
    app.setProperty("fluentPalette", p)
    return p


def current_palette() -> Palette:
    app = QApplication.instance()
    p = app.property("fluentPalette") if app is not None else None
    return p if isinstance(p, Palette) else LIGHT


_GLYPHS = {
    "chevron-down": '<path d="M6 9l6 6 6-6"/>',
    "chevron-up": '<path d="M6 15l6-6 6 6"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
}
_GLYPH_DIR: Path | None = None


def _glyph_files(p: Palette) -> dict[str, str]:
    """Small PNG glyphs in the palette's colours, for stylesheet ``image: url(...)``."""

    global _GLYPH_DIR
    if _GLYPH_DIR is None:
        _GLYPH_DIR = Path(tempfile.mkdtemp(prefix="scan-console-theme-"))
    colors = {"chevron-down": p.text_secondary, "chevron-up": p.text_secondary, "check": p.on_accent}
    files = {}
    for name, color in colors.items():
        path = _GLYPH_DIR / f"{name}-{'dark' if p.dark else 'light'}-{color.strip('#')}.png"
        if not path.exists():
            _render_svg(_GLYPHS[name], color, 32, stroke=2.4).save(str(path))
        files[name] = path.as_posix()
    return files


def _render_svg(body: str, color: str, size: int, stroke: float = 1.6):
    from qtpy.QtGui import QImage
    from qtpy.QtSvg import QSvgRenderer

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    )
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter)
    painter.end()
    return image


def stylesheet(p: Palette, glyphs: dict[str, str] | None = None) -> str:
    g = glyphs or {}
    return f"""
* {{ outline: none; }}
QWidget {{ color: {p.text}; }}
QMainWindow, QDialog, QMessageBox, #page {{ background: {p.window}; }}
QToolTip {{ background: {p.card}; color: {p.text}; border: 1px solid {p.control_border};
            border-radius: 4px; padding: 4px 8px; }}

/* navigation rail */
#rail {{ background: {p.rail}; border: none; }}
#railTitle {{ font-weight: 600; padding: 4px 12px 12px 14px; }}
#rail QToolButton {{ background: transparent; border: none; border-radius: 5px; text-align: left;
                    padding: 8px 10px; margin: 1px 6px; color: {p.text}; }}
#rail QToolButton:hover {{ background: {p.subtle_hover}; }}
#rail QToolButton:checked {{ background: {p.subtle_hover}; border-left: 3px solid {p.accent};
                            padding-left: 7px; font-weight: 600; }}

QToolBar {{ background: transparent; border: none; spacing: 4px; padding: 0; }}
QToolBar QToolButton#qt_toolbar_ext_button {{ background: transparent; border: none; }}

/* page header */
#pageTitle {{ font-size: 20pt; font-weight: 600; }}
#statusPill {{ border-radius: 11px; padding: 3px 12px; font-weight: 600; }}
#statusPill[tone="sim"] {{ background: {p.subtle_hover}; color: {p.text_secondary};
                          border: 1px solid {p.control_border}; }}
#statusPill[tone="hw"] {{ background: {p.accent}; color: {p.on_accent}; }}
QLabel[secondary="true"] {{ color: {p.text_secondary}; }}
QLabel[tone="ok"] {{ color: {p.success}; }}
QLabel[tone="bad"] {{ color: {p.critical}; }}
QLabel[tone="attention"] {{ color: {p.critical}; font-weight: 600; }}

/* cards: settings groups */
QGroupBox {{ background: {p.card}; border: 1px solid {p.card_border}; border-radius: 10px;
            margin-top: 24px; padding: 16px 14px 12px 14px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 2px; top: 2px;
                   padding: 0 2px; font-weight: 600; color: {p.text}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QSplitter::handle {{ background: transparent; width: 6px; }}
#viewerToolbar {{ background: {p.card}; border: 1px solid {p.card_border}; border-radius: 8px; }}
#viewerToolbar QLabel[toolbarTitle="true"] {{ color: {p.text_secondary}; font-weight: 600; }}
#zWheelZone {{ background: {p.card}; border: 1px solid {p.card_border}; border-radius: 8px; }}
#zWheelZone:hover {{ border-color: {p.accent}; background: {p.control_hover}; }}
#zWheelZone QLabel {{ color: {p.text_secondary}; }}
#zWheelZone QLabel[wheelTitle="true"] {{ color: {p.text}; font-weight: 700; }}
#zWheelZone QLabel[wheelValue="true"] {{ color: {p.text_secondary}; font-size: 9pt; }}
#zWheelZone:disabled {{ background: {p.control_pressed}; border-color: {p.card_border}; }}

/* buttons */
QPushButton {{ background: {p.control}; border: 1px solid {p.control_border};
              border-bottom: 1px solid {p.control_bottom}; border-radius: 4px;
              padding: 5px 8px; min-height: 20px; }}
QPushButton:hover {{ background: {p.control_hover}; }}
QPushButton:pressed {{ background: {p.control_pressed}; color: {p.text_secondary};
                      border-bottom: 1px solid {p.control_border}; }}
QPushButton:checked {{ background: {p.accent}; color: {p.on_accent}; border: 1px solid {p.accent}; }}
QPushButton:disabled {{ color: {p.text_disabled}; background: {p.control_pressed}; }}
QPushButton[accent="true"] {{ background: {p.accent}; color: {p.on_accent}; border: 1px solid {p.accent};
                             font-weight: 600; }}
QPushButton[accent="true"]:hover {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton[accent="true"]:pressed {{ background: {p.accent_pressed}; }}
QPushButton[accent="true"]:disabled {{ background: {p.control_pressed}; color: {p.text_disabled};
                                      border: 1px solid {p.control_border}; }}
QPushButton[critical="true"] {{ color: {p.critical}; }}
QPushButton[critical="true"]:disabled {{ color: {p.text_disabled}; }}
QPushButton[jog="true"] {{ padding: 3px; font-weight: 600; }}
QToolButton {{ background: {p.control}; border: 1px solid {p.control_border}; border-radius: 4px;
              padding: 4px 7px; }}
QToolButton:hover {{ background: {p.control_hover}; }}
QToolButton:checked {{ background: {p.accent}; color: {p.on_accent}; border-color: {p.accent}; }}
QToolButton:disabled {{ color: {p.text_disabled}; }}

/* inputs */
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background: {p.control}; border: 1px solid {p.control_border};
    border-bottom: 1px solid {p.control_bottom}; border-radius: 4px; padding: 4px 8px;
    min-height: 20px; selection-background-color: {p.accent}; selection-color: {p.on_accent}; }}
QLineEdit:hover, QDoubleSpinBox:hover, QComboBox:hover {{ background: {p.control_hover}; }}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ background: {p.input};
    border-bottom: 2px solid {p.accent}; }}
QLineEdit:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {p.text_disabled}; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 18px; border: none; background: transparent; }}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{ background: {p.subtle_hover}; border-radius: 3px; }}
QDoubleSpinBox::up-arrow {{ image: url("{g.get('chevron-up', '')}"); width: 9px; height: 9px; }}
QDoubleSpinBox::down-arrow {{ image: url("{g.get('chevron-down', '')}"); width: 9px; height: 9px; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url("{g.get('chevron-down', '')}"); width: 11px; height: 11px; }}
QPushButton[compact="true"] {{ padding: 4px 0; }}
QComboBox QAbstractItemView {{ background: {p.card}; border: 1px solid {p.control_border};
    border-radius: 6px; padding: 4px; selection-background-color: {p.subtle_hover};
    selection-color: {p.text}; }}

/* lists and tables */
QListWidget, QTableWidget, QTextBrowser {{ background: {p.input}; border: 1px solid {p.control_border};
    border-radius: 6px; padding: 2px; }}
QListWidget::item {{ border-radius: 4px; padding: 4px 6px; }}
QListWidget::item:hover {{ background: {p.subtle_hover}; }}
QListWidget::item:selected {{ background: {p.subtle_hover}; color: {p.text};
    border-left: 3px solid {p.accent}; }}
QTableWidget {{ gridline-color: transparent; }}
QHeaderView::section {{ background: transparent; border: none; color: {p.text_secondary};
    padding: 4px 6px; font-weight: 600; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {p.control_bottom}; background: {p.control}; }}
QCheckBox::indicator:checked {{ background: {p.accent}; border-color: {p.accent};
    image: url("{g.get('check', '')}"); }}

/* progress, tabs, scrollbars */
QProgressBar {{ background: {p.control_border}; border: none; border-radius: 2px;
    max-height: 4px; min-height: 4px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 2px; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; border: none; padding: 6px 14px; margin-right: 4px;
    border-radius: 5px; color: {p.text_secondary}; }}
QTabBar::tab:hover {{ background: {p.subtle_hover}; color: {p.text}; }}
QTabBar::tab:selected {{ background: {p.card}; color: {p.text}; font-weight: 600;
    border-bottom: 2px solid {p.accent}; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle {{ background: {p.control_bottom}; border-radius: 3px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; border: none; width: 0; height: 0; }}
"""


class ToggleSwitch(QCheckBox):
    """A Fluent toggle switch. Still a QCheckBox: same signals and API."""

    TRACK_W, TRACK_H, GAP = 40, 20, 10

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._offset = 0.0
        self._animation = QVariantAnimation(self, duration=120)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._set_offset)
        self.toggled.connect(self._animate)

    def _set_offset(self, value) -> None:
        self._offset = float(value)
        self.update()

    def _animate(self, checked: bool) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._offset)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def setChecked(self, checked: bool) -> None:  # noqa: N802 - Qt API
        super().setChecked(checked)
        if not self._animation.state() == QVariantAnimation.State.Running:
            self._offset = 1.0 if checked else 0.0

    def sizeHint(self) -> QSize:  # noqa: N802
        text_w = self.fontMetrics().horizontalAdvance(self.text()) if self.text() else 0
        return QSize(self.TRACK_W + (self.GAP + text_w if text_w else 0) + 2, max(self.TRACK_H + 4, self.fontMetrics().height() + 6))

    def hitButton(self, pos) -> bool:  # noqa: N802
        return self.rect().contains(pos)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = (self.height() - self.TRACK_H) / 2
        track = QRectF(1, y, self.TRACK_W, self.TRACK_H)
        enabled = self.isEnabled()
        on = self._offset
        if on > 0.5:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(p.accent if enabled else p.text_disabled))
        else:
            painter.setPen(QColor(p.text_secondary if enabled else p.text_disabled))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)
        knob = 12 if not self.underMouse() else 14
        x = track.left() + 4 + knob / 2 + on * (self.TRACK_W - 8 - knob)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(p.on_accent if on > 0.5 else (p.text_secondary if enabled else p.text_disabled)))
        painter.drawEllipse(QPointF(x, track.center().y()), knob / 2, knob / 2)
        if self.text():
            painter.setPen(QColor(p.text if enabled else p.text_disabled))
            text_rect = self.rect().adjusted(self.TRACK_W + self.GAP + 2, 0, 0, 0)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        painter.end()


# Simple line icons (24×24 viewBox, stroke = currentColor), drawn for this app.
_ICONS = {
    "acquire": '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2.5"/>'
               '<path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "devices": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4'
               'M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "config": '<path d="M7 3h7l4 4v14H7z"/><path d="M14 3v4h4M10 12h5M10 15h5M10 18h3"/>',
    "settings": '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
}


def icon(name: str, color: str | None = None, size: int = 20) -> QIcon:
    from qtpy.QtSvg import QSvgRenderer

    color = color or current_palette().text
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        f"{_ICONS[name]}</svg>"
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    result = QIcon()
    for scale in (1, 2):
        pixmap = QPixmap(size * scale, size * scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        result.addPixmap(pixmap)
    return result


def set_dark_title_bar(widget: QWidget, dark: bool) -> None:
    """Windows 10 20H1+/11: match the native title bar to the theme. No-op elsewhere."""

    if sys.platform != "win32":
        return
    try:  # pragma: no cover - Windows only
        import ctypes

        value = ctypes.c_int(1 if dark else 0)
        for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (and its pre-20H1 id)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                int(widget.winId()), attribute, ctypes.byref(value), ctypes.sizeof(value)
            ) == 0:
                break
    except Exception:
        pass
