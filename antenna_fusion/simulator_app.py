"""
COTP Antenna Pointing System: 3D Kinematics & Confidence-Score Sensor Fusion
=============================================================================
Styled to the Palatine design system (docs/design_system.md): dark-only grayscale chrome,
1px hairlines, square corners, Geologica (human) / JetBrains Mono (machine) type,
and hue reserved for live state and chart series.

Features preserved 100%:
- Page 1: Live 3D Simulation & Sensor Noise Control "Game"
    * Top Shelf: Pipeline & Solver Trajectory Ribbon (stage progress tracker)
    * Central Viewport: Exploded Isometric Digital Twin (axonometric projection,
      hairline alignment axis, per-joint angle callouts, readout strip)
    * Lower Console: 2x2 Telemetry & Diagnostic Quad ('P', 'T', 'R', 'D')
    * Parameter & Execution Docks: 1px hairline panels, grayscale controls
    * Real CAD STL mesh kinematics (Azimuth, Elevation, Polarization)
    * Real-time noise sliders (σ_enc, σ_ahrs) & interactive fault injection (Drift ramp, Glitch step)
    * Real-time causal confidence-score quaternion sensor fusion
    * Target chasing (Cascade PID plant) vs Brownian random motion
    * Live telemetry plots & real-time CSV recording
- Page 2: Post-Run Analysis & Data Inspector
    * 1-click inspection of recorded simulation runs or any exported CSV
    * Multi-axis switching (Azimuth, Elevation, Polarization)
    * Synchronized multi-panel graphs with draggable overview time window
    * Live mouse hover crosshair with exact numeric readout
    * Comprehensive quality score card (RMSE, MAE, Max Error, SNR, % Improvement)

Usage:
    python -m antenna_fusion.simulator_app      (or run_simulator.bat)
"""

import os
import sys
import glob
import ctypes
from collections import deque
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QButtonGroup, QCheckBox,
    QComboBox, QFileDialog, QGroupBox, QFrame, QSplitter,
    QTabWidget, QStatusBar, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QEvent
from PyQt5.QtGui import (
    QFont, QFontDatabase, QIcon, QPixmap, QColor, QPainter, QPainterPath, QPen, QBrush
)
import pyqtgraph as pg
import pyvista as pv
from pyvistaqt import QtInteractor
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D, vtkTextActor

from .core import (
    AntennaSimulationManager, compute_fk_matrices, apply_transform,
    AXES, AX_LIMITS, LINKS_CONFIG, STL_FILES, BASE_STALL
)
from .paths import (
    APP_ICON_PATH, FONTS_DIR, LOGS_DIR, SIM_LOG_PATTERN, TIMESERIES_PATTERN, find_latest_log
)

# ─────────────────────────────────────────────────────────────────────────────
# PALATINE DESIGN TOKENS (docs/design_system.md §3) — chrome is grayscale only
# ─────────────────────────────────────────────────────────────────────────────
PAL = {
    # Surfaces & chrome
    'void':        '#050505',  # Page backdrop outside the app frame
    'bg':          '#0A0A0A',  # App canvas / schematic board
    'surface_1':   '#111113',  # Sidebar, chart & readout panels
    'surface_2':   '#17171A',  # Cards, inputs, icon buttons
    'surface_3':   '#232327',  # Hover, active segmented chip
    'line':        '#1F1F22',  # Hairline dividers, panel borders, plot grid
    'line_strong': '#303034',  # Control borders, closed / idle rings
    # Text
    'text_1':      '#EDEDED',  # Primary text, live values
    'text_2':      '#9C9CA1',  # Labels, secondary text
    'text_3':      '#626266',  # Disabled, units, tags, axes
    'text_4':      '#3C3C40',  # Faint ticks, placeholder glyphs
    # Process & state (the only permitted hues)
    'state_live':  '#00D26A',  # Running / nominal / open
    'alarm':       '#D6001F',  # Alarm red
    'warn':        '#FFB020',  # Warning state
}

# Chart series, in the prescribed order: green -> crimson -> violet -> blue -> amber
SERIES = {
    'fused':  '#00D26A',   # series 1 — fused estimate (the nominal series)
    'enc':    '#D6001F',   # series 2 — encoder
    'ahrs':   '#7C4DFF',   # series 3 — AHRS
    'true':   PAL['text_1'],  # ground truth: neutral reference
    'target': PAL['text_3'],  # setpoint: faint dashed reference
}

# Post-run plot themes: 'dark' is the app canvas, 'paper' is white for figures in documents
P2_PLOT_THEMES = {
    'dark': {
        'bg': PAL['bg'], 'axis': PAL['line_strong'], 'tick': PAL['text_3'], 'label': PAL['text_3'],
        'title': PAL['text_2'], 'legend_text': PAL['text_2'], 'legend_bg': (17, 17, 19, 210),
        'legend_line': PAL['line'], 'crosshair': PAL['text_2'], 'zero': PAL['line_strong'],
        'handle': PAL['line'], 'true': SERIES['true'], 'target': SERIES['target'],
    },
    'paper': {
        'bg': '#FFFFFF', 'axis': '#3C3C40', 'tick': '#3C3C40', 'label': '#3C3C40',
        'title': '#111113', 'legend_text': '#111113', 'legend_bg': (255, 255, 255, 230),
        'legend_line': '#C8C8CC', 'crosshair': '#626266', 'zero': '#9C9CA1',
        'handle': '#FFFFFF', 'true': '#111113', 'target': '#8A8A8F',  # light references flipped dark
    },
}

# Post-run analysis: panels, signals, and which curve lives where (legend order)
P2_PANELS = [('angle', 'Angle'), ('confidence', 'Confidence'), ('error', 'Error')]
P2_SIGNALS = ['true', 'target', 'enc', 'ahrs', 'fused']
P2_CURVES = [  # (curve key, legend name, panel, signal)
    ('target', 'target', 'angle', 'target'), ('true', 'true', 'angle', 'true'),
    ('enc', 'enc', 'angle', 'enc'), ('ahrs', 'ahrs', 'angle', 'ahrs'), ('fused', 'fused', 'angle', 'fused'),
    ('conf_enc', 'c_enc', 'confidence', 'enc'), ('conf_ahrs', 'c_ahrs', 'confidence', 'ahrs'),
    ('err_enc', 'err_enc', 'error', 'enc'), ('err_ahrs', 'err_ahrs', 'error', 'ahrs'),
    ('err_fused', 'err_fused', 'error', 'fused'),
]

# Typography (design_system.md §4): two voices, never blended
SANS_FAMILIES = ['Geologica', 'Helvetica Neue', 'Arial']
MONO_FAMILIES = ['JetBrains Mono', 'IBM Plex Mono', 'Consolas', 'Menlo']
SANS_CSS = "'Geologica', 'Helvetica Neue', Arial, sans-serif"
MONO_CSS = "'JetBrains Mono', 'IBM Plex Mono', Consolas, Menlo, monospace"
# design_system.md asks for a −7% width axis, but Geologica ships no width axis and Qt can't
# stretch it (it silently swaps in another face). Tightening the letter advance gives
# the compressed footprint instead; 93% makes glyphs touch at 13px, so 96% is used.
SANS_ADVANCE = 96


def sans_font(px: int, weight: int = QFont.Normal) -> QFont:
    f = QFont()
    f.setFamilies(SANS_FAMILIES)
    f.setStyleHint(QFont.SansSerif)
    f.setPixelSize(px)
    f.setWeight(weight)
    f.setLetterSpacing(QFont.PercentageSpacing, SANS_ADVANCE)
    return f


def mono_font(px: int, weight: int = QFont.Normal, tracking_em: float = 0.0) -> QFont:
    f = QFont()
    f.setFamilies(MONO_FAMILIES)
    f.setStyleHint(QFont.Monospace)
    f.setPixelSize(px)
    f.setWeight(weight)
    if tracking_em:
        f.setLetterSpacing(QFont.PercentageSpacing, 100.0 + tracking_em * 100.0)
    return f


def value_html(value: str, unit: str = "", unit_px: int = 9) -> str:
    """Mono value with its unit as a separate, smaller gray token (design_system.md §4.3)."""
    v = value.replace(' ', '&nbsp;')  # keep tabular width in rich text
    if not unit:
        return v
    return f"{v}<span style=\"color:{PAL['text_3']}; font-size:{unit_px}px;\">&nbsp;{unit}</span>"


def micro_qss(color: str = PAL['text_3']) -> str:
    """Micro label: 10px mono uppercase (tracking is applied via mono_font)."""
    return f"color: {color}; font-family: {MONO_CSS}; font-size: 10px; background: transparent;"


def badge_qss(state: str) -> str:
    """State badge: mono word in the state hue with a hairline ring — no colored fill."""
    color = {'live': PAL['state_live'], 'warn': PAL['warn'], 'alarm': PAL['alarm']}.get(state, PAL['text_2'])
    ring = color if state in ('warn', 'alarm') else PAL['line_strong']
    return (f"color: {color}; background: transparent; border: 1px solid {ring}; border-radius: 2px; "
            f"padding: 1px 6px; font-family: {MONO_CSS}; font-size: 10px;")


def badge_text(state: str, word: str) -> str:
    """Redundant non-color cue: glyph changes with state (● nominal, ▲ warning)."""
    return f"{'▲' if state in ('warn', 'alarm') else '●'} {word}"


def load_bundled_fonts():
    """
    Register the design-system fonts with Qt.
    - assets/fonts/*.ttf|otf if they are shipped with the repo.
    - Static Geologica / JetBrains Mono files from the Windows font folders: when the
      Geologica *variable* font is installed, Qt only sees its default instance
      (Thin, slanted), so the static Regular / Medium cuts are registered explicitly.
    """
    font_dirs = [FONTS_DIR]
    if os.environ.get('LOCALAPPDATA'):
        font_dirs.append(os.path.join(os.environ['LOCALAPPDATA'], 'Microsoft', 'Windows', 'Fonts'))
    font_dirs.append(os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'))
    patterns = ['Geologica-Regular.ttf', 'Geologica-Medium.ttf', 'JetBrainsMono-*.ttf']
    for i, d in enumerate(font_dirs):
        for pat in (['*.[ot]tf'] if i == 0 else patterns):
            for path in glob.glob(os.path.join(d, pat)):
                QFontDatabase.addApplicationFont(path)


APP_ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)  # title bar, taskbar (100–200% DPI), Alt-Tab


def load_app_icon() -> QIcon:
    """
    Window / taskbar icon built from assets/img/app_icon.png at every size Windows asks for.
    The source is 1080px; pre-scaling with smooth filtering keeps the small
    title-bar and taskbar renditions crisp instead of relying on one huge bitmap.
    """
    icon = QIcon()
    src = QPixmap(APP_ICON_PATH)
    if src.isNull():
        return icon  # missing file: fall back to the default icon
    for size in APP_ICON_SIZES:
        icon.addPixmap(src.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    return icon


def set_native_title_bar(hwnd: int, background: str = '#000000', text: str = PAL['text_2']):
    """
    Paint the native Windows title bar black (Windows 11, build 22000+).
    DWM draws the caption, so Qt stylesheets can't reach it. Older Windows and
    non-Windows platforms fall back to the default title bar, which is harmless.
    """
    if sys.platform != 'win32':
        return

    def colorref(hex_color: str) -> ctypes.c_uint:
        c = QColor(hex_color)
        return ctypes.c_uint(c.red() | (c.green() << 8) | (c.blue() << 16))  # 0x00BBGGRR

    DWMWA_USE_IMMERSIVE_DARK_MODE = 20  # dark caption buttons / fallback on Windows 10
    DWMWA_BORDER_COLOR = 34
    DWMWA_CAPTION_COLOR = 35
    DWMWA_TEXT_COLOR = 36
    dwm = ctypes.windll.dwmapi
    attrs = [
        (DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.c_int(1)),
        (DWMWA_CAPTION_COLOR, colorref(background)),
        (DWMWA_BORDER_COLOR, colorref(background)),
        (DWMWA_TEXT_COLOR, colorref(text)),
    ]
    for attr, value in attrs:
        dwm.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), attr, ctypes.byref(value), ctypes.sizeof(value))


# Configure PyQtGraph global options
pg.setConfigOption('background', PAL['bg'])
pg.setConfigOption('foreground', PAL['text_3'])
pg.setConfigOptions(antialias=True)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM BENTO COMPONENTS: TOP SHELF PIPELINE RIBBON
# ─────────────────────────────────────────────────────────────────────────────
class PipelineRibbonWidget(QWidget):
    """
    Top Shelf: Pipeline & Solver Trajectory Ribbon.
    A full-width horizontal ribbon tracking the multi-stage simulation loop:
    Kinematic Pass -> State Estimation -> Numerical Optimization -> Hardware Command.
    Palatine styling:
    - Grayscale chrome: done / active / pending stages differ by surface and text
      level only; the active stage is the brightest card on the strip.
    - The only hue is the live-state ring on the active node while the solver is
      running (redundant cue: filled node + RUN / HOLD word).
    - Mono micro tags (STEP 01) over Title Case sans headings.
    """
    STAGES = [
        ("STEP 01", "Kinematic Pass", "FK Transform & Actuator Resolvers"),
        ("STEP 02", "State Estimation", "Rolling Variance & Detrend Filter"),
        ("STEP 03", "Numerical Optimization", "Confidence Quaternion SLERP"),
        ("STEP 04", "Hardware Command", "Cascade PID & Stall Feedforward"),
    ]
    TICKS_PER_STEP = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.active_step = 2  # Current solver step (0..3)
        self.tick_counter = 0
        self.running = False

    def set_active_step(self, step: int):
        if self.active_step != step:
            self.active_step = step
            self.update()

    def set_running(self, running: bool):
        if self.running != running:
            self.running = running
            self.update()

    def advance_step(self):
        self.tick_counter += 1
        # Advance active stage in rhythmic lockstep with simulation
        step = (self.tick_counter // self.TICKS_PER_STEP) % 4
        self.set_active_step(step)
        self.update()  # Repaint every tick so the in-step progress bar animates

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            w = self.width()
            h = self.height()
            n = len(self.STAGES)
            col_w = w / n
            c = {k: QColor(v) for k, v in PAL.items()}

            painter.fillRect(0, 0, w, h, c['bg'])

            font_tag = mono_font(10, tracking_em=0.08)
            font_title = sans_font(13, QFont.Medium)
            font_sub = sans_font(11)

            # Fraction of the active step already elapsed (drives its progress bar)
            step_frac = (self.tick_counter % self.TICKS_PER_STEP + 1) / self.TICKS_PER_STEP

            for i, (tag, title, desc) in enumerate(self.STAGES):
                x0 = i * col_w
                is_active = (i == self.active_step)
                is_done = (i < self.active_step)
                card = QRectF(x0 + 2.5, 2.5, col_w - 5, h - 5)

                # Card body — set the brush explicitly every time; drawRoundedRect fills
                # with the current brush, so a leftover brush would bleed into the card.
                painter.setBrush(QBrush(c['surface_3'] if is_active else c['surface_1']))
                painter.setPen(QPen(c['line_strong'] if is_active else c['line'], 1))
                painter.drawRoundedRect(card, 2, 2)

                # Bottom progress track: full for done, animated for active, empty for pending
                track = QRectF(card.left() + 9, card.bottom() - 4, card.width() - 18, 2)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(c['line']))
                painter.drawRect(track)
                fill_frac = 1.0 if is_done else (step_frac if is_active else 0.0)
                if fill_frac > 0:
                    painter.setBrush(QBrush(c['text_1'] if is_active else c['text_3']))
                    painter.drawRect(QRectF(track.left(), track.top(), track.width() * fill_frac, track.height()))

                # Status node (top-right): ring = state. Live ring only while running.
                node = QPointF(card.right() - 13, 13)
                if is_active:
                    ring = c['state_live'] if self.running else c['line_strong']
                    painter.setBrush(QBrush(ring))
                    painter.drawEllipse(node, 4.0, 4.0)
                elif is_done:
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(c['text_2'], 1.3))
                    check = QPainterPath(QPointF(node.x() - 3.0, node.y()))
                    check.lineTo(node.x() - 0.8, node.y() + 2.2)
                    check.lineTo(node.x() + 3.2, node.y() - 2.2)
                    painter.drawPath(check)
                else:
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(c['line_strong'], 1.2))
                    painter.drawEllipse(node, 3.5, 3.5)

                # Chevron connector into the next stage
                if i < n - 1:
                    cx, cy = (i + 1) * col_w, h / 2
                    chev = QPainterPath(QPointF(cx - 2, cy - 4))
                    chev.lineTo(cx + 1.5, cy)
                    chev.lineTo(cx - 2, cy + 4)
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(c['text_2'] if i < self.active_step else c['line_strong'], 1.2))
                    painter.drawPath(chev)

                # Typography: mono tag (+ state word on active) / sans heading / sans caption
                tx = int(card.left() + 9)
                painter.setFont(font_tag)
                painter.setPen(c['text_2'] if is_active else c['text_3'])
                tag_txt = f"{tag}  {'RUN' if self.running else 'HOLD'}" if is_active else tag
                painter.drawText(tx, 16, tag_txt)

                painter.setFont(font_title)
                painter.setPen(c['text_1'] if is_active else (c['text_2'] if is_done else c['text_3']))
                painter.drawText(tx, 32, title)

                painter.setFont(font_sub)
                painter.setPen(c['text_2'] if is_active else c['text_3'])
                painter.drawText(tx, 45, desc)
        finally:
            painter.end()


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM BENTO COMPONENTS: LOWER CONSOLE 2x2 TELEMETRY & DIAGNOSTIC QUAD
# ─────────────────────────────────────────────────────────────────────────────
class DiagnosticCard(QFrame):
    """
    Readout card for the 2x2 Telemetry & Diagnostic Quad.
    Palatine styling:
    - surface-2 card on a 1px hairline, square corners, no watermark decoration.
    - Title Case sans heading, mono state badge (hue only for live / warning state).
    - Value MD mono readout (card size) with its unit as a separate gray token.
    """
    def __init__(self, code: str, parent=None):
        super().__init__(parent)
        self.setObjectName("diagCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"QFrame#diagCard {{ background: {PAL['surface_2']}; border: 1px solid {PAL['line']}; border-radius: 2px; }}"
            f"QLabel {{ background: transparent; border: none; }}"
        )

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(9, 5, 9, 5)  # 10px nominal × 0.93
        self.layout.setSpacing(2)

        # Header Row
        h_row = QHBoxLayout()
        h_row.setSpacing(6)
        self.lbl_code = QLabel(code)
        self.lbl_code.setFont(mono_font(10, tracking_em=0.08))
        self.lbl_code.setStyleSheet(f"color: {PAL['text_3']};")
        h_row.addWidget(self.lbl_code)

        self.lbl_title = QLabel("Diagnostic")
        self.lbl_title.setFont(sans_font(12, QFont.Medium))
        self.lbl_title.setStyleSheet(f"color: {PAL['text_2']};")
        h_row.addWidget(self.lbl_title)
        h_row.addStretch()

        self.lbl_badge = QLabel(badge_text('live', "NOMINAL"))
        self.lbl_badge.setStyleSheet(badge_qss('live'))
        h_row.addWidget(self.lbl_badge)
        self.layout.addLayout(h_row)

        # Body: primary readout (Value MD) left, secondary readouts stacked right
        body = QHBoxLayout()
        body.setSpacing(15)
        self.lbl_metric_main = QLabel("--")
        self.lbl_metric_main.setStyleSheet(
            f"color: {PAL['text_1']}; font-family: {MONO_CSS}; font-size: 16px; font-weight: 500;"
        )
        body.addWidget(self.lbl_metric_main, alignment=Qt.AlignVCenter)

        subs = QVBoxLayout()
        subs.setSpacing(0)
        self.lbl_sub1 = QLabel("--")
        self.lbl_sub2 = QLabel("--")
        for lbl in (self.lbl_sub1, self.lbl_sub2):
            lbl.setStyleSheet(f"color: {PAL['text_1']}; font-family: {MONO_CSS}; font-size: 10px;")
            subs.addWidget(lbl)
        body.addLayout(subs)
        body.addStretch()
        self.layout.addLayout(body)

        self.layout.addStretch()

    @staticmethod
    def _row_html(label: str, value: str, unit: str = "") -> str:
        return f"<span style=\"color:{PAL['text_3']};\">{label}</span>&nbsp;&nbsp;{value_html(value, unit)}"

    def set_content(self, title: str, main_val: tuple, sub1: tuple, sub2: tuple,
                    badge_word: str = "NOMINAL", is_alert: bool = False):
        """main_val = (value, unit); sub rows = (label, value, unit)."""
        state = 'warn' if is_alert else 'live'
        self.lbl_title.setText(title)
        self.lbl_metric_main.setText(value_html(*main_val, unit_px=10))
        self.lbl_sub1.setText(self._row_html(*sub1))
        self.lbl_sub2.setText(self._row_html(*sub2))
        self.lbl_badge.setText(badge_text(state, badge_word))
        self.lbl_badge.setStyleSheet(badge_qss(state))
        value_color = PAL['warn'] if is_alert else PAL['text_1']
        self.lbl_metric_main.setStyleSheet(
            f"color: {value_color}; font-family: {MONO_CSS}; font-size: 16px; font-weight: 500;"
        )


class DiagnosticQuadWidget(QWidget):
    """
    Lower Console: 2x2 Telemetry & Diagnostic Quad.
    Divided into a four-quadrant diagnostic array:
    - [P]: Phase Margin & Plant Slew
    - [T]: Actuator Torques & PWM
    - [R]: Convergence Residuals & Variance
    - [D]: Dynamic Tracking Errors & SNR
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        self.card_p = DiagnosticCard("P", self)
        self.card_t = DiagnosticCard("T", self)
        self.card_r = DiagnosticCard("R", self)
        self.card_d = DiagnosticCard("D", self)

        grid.addWidget(self.card_p, 0, 0)
        grid.addWidget(self.card_t, 0, 1)
        grid.addWidget(self.card_r, 1, 0)
        grid.addWidget(self.card_d, 1, 1)

    def update_telemetry(self, sim: AntennaSimulationManager, live_axis: str):
        ax = sim.axes[live_axis]
        fusion = ax.fusion

        # [P] Phase Margin & Plant Slew
        vel = ax.true_vel
        phase_margin = 48.0 - min(abs(vel) * 0.4, 25.0)
        self.card_p.set_content(
            title="Phase Margin & Slew",
            main_val=(f"{phase_margin:5.1f}", "deg PM"),
            sub1=("plant_vel ", f"{vel:+7.2f}", "deg/s"),
            sub2=("motion    ", ax.motion_mode),
            badge_word="LOCKED" if abs(ax.fused_angle - ax.target) < 1.0 else "SLEWING",
            is_alert=False
        )

        # [T] Actuator Torques & PWM
        stall_detected = ax.stall.detected
        self.card_t.set_content(
            title="Actuator Torque & PWM",
            main_val=(f"{ax.pwm:5.1f}", "% PWM"),
            sub1=("vel_demand", f"{ax.vel_demand:+7.2f}", "deg/s"),
            sub2=("stall_ff  ", 'BOOST ACTIVE' if stall_detected else 'NOMINAL'),
            badge_word="STALL WARNING" if stall_detected else "NOMINAL",
            is_alert=stall_detected
        )

        # [R] Convergence Residuals & Variance
        has_drift = (ax.drift_rate > 0.0 or abs(ax.drift_offset) > 0.5)
        self.card_r.set_content(
            title="Convergence Residuals",
            main_val=(f"{fusion.var_enc:6.3f}", "var_enc"),
            sub1=("var_ahrs  ", f"{fusion.var_ahrs:6.3f}", "deg²"),
            sub2=("drift_pen ", f"{fusion.drift_penalty:6.2f}", "pt"),
            badge_word="DRIFT FAULT" if has_drift else "CONVERGED",
            is_alert=has_drift
        )

        # [D] Dynamic Tracking Errors & SNR
        err_fused = ax.fused_angle - ax.true_angle
        err_enc = ax.enc_reading - ax.true_angle
        snr_gain = 20.0 * np.log10(max(abs(err_enc) / max(abs(err_fused), 1e-4), 1.0))
        has_glitch = abs(ax.glitch_offset) > 0.1
        self.card_d.set_content(
            title="Dynamic Tracking Error",
            main_val=(f"{err_fused:+6.2f}", "deg err_fused"),
            sub1=("err_enc   ", f"{err_enc:+7.2f}", "deg"),
            sub2=("snr_gain  ", f"{snr_gain:+7.1f}", "dB"),
            badge_word="SPIKE GLITCH" if has_glitch else "SUB-DEGREE",
            is_alert=has_glitch
        )


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM BENTO COMPONENTS: CALLOUT OVERLAY & FLOATING METADATA BADGE
# ─────────────────────────────────────────────────────────────────────────────
class CalloutOverlayWidget(QFrame):
    """
    Central Viewport Callout Overlay: a hairline readout strip above the 3D twin
    showing the selected axis, its fused estimate, bus rate, sensor confidence
    and lock state.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("callout")
        self.setFixedHeight(46)
        self.setStyleSheet(f"""
            QFrame#callout {{
                background-color: {PAL['surface_1']};
                border: none;
                border-bottom: 1px solid {PAL['line']};
            }}
            QLabel {{
                border: none;
                background: transparent;
                font-family: {MONO_CSS};
                font-size: 10px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 4, 9, 4)
        layout.setSpacing(2)

        # Top row: hardware tag + status badge
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        self.lbl_head = QLabel("HARDWARE TWIN")
        self.lbl_head.setStyleSheet(f"color: {PAL['text_2']};")
        row1.addWidget(self.lbl_head)
        row1.addStretch()

        self.lbl_status_badge = QLabel(badge_text('live', "LOCK NOMINAL"))
        self.lbl_status_badge.setStyleSheet(badge_qss('live'))
        row1.addWidget(self.lbl_status_badge)
        layout.addLayout(row1)

        # Bottom row: estimation | bus | confidence
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(15)  # 16px nominal × 0.93

        self.lbl_est = QLabel()
        self.lbl_bus = QLabel()
        self.lbl_conf = QLabel()
        for lbl in (self.lbl_est, self.lbl_bus, self.lbl_conf):
            lbl.setStyleSheet(f"color: {PAL['text_1']};")
            row2.addWidget(lbl)
        row2.addStretch()
        layout.addLayout(row2)

        self.lbl_bus.setText(self._kv("bus_tx", "50", "Hz") + "&nbsp;&nbsp;" + value_html("115.2", "kbps"))

    @staticmethod
    def _kv(key: str, value: str, unit: str = "") -> str:
        return f"<span style=\"color:{PAL['text_3']};\">{key}</span>&nbsp;{value_html(value, unit)}"

    def update_badge(self, axis_name: str, fused_angle: float, c_enc: float, c_ahrs: float, status_text: str = "NOMINAL"):
        # Axis identifier rendered verbatim from config
        self.lbl_head.setText(
            f"HARDWARE TWIN&nbsp;&nbsp;<span style=\"color:{PAL['text_1']};\">{axis_name}</span>"
            f"&nbsp;&nbsp;GIMBAL ACTUATOR"
        )
        self.lbl_est.setText(self._kv("fused", f"{fused_angle:+7.2f}", "deg") +
                             f"<span style=\"color:{PAL['text_3']};\">&nbsp;SLERP</span>")
        self.lbl_conf.setText(self._kv("c_enc", f"{c_enc * 100:5.1f}", "%") + "&nbsp;&nbsp;" +
                              self._kv("c_ahrs", f"{c_ahrs * 100:5.1f}", "%"))
        state = 'live' if status_text == "NOMINAL" else 'warn'
        self.lbl_status_badge.setText(badge_text(state, f"LOCK {status_text}"))
        self.lbl_status_badge.setStyleSheet(badge_qss(state))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION GUI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class AntennaFusionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COTP Antenna Pointing System — Ultra-Dense Engineering Canvas")
        self.app_icon = load_app_icon()
        self.setWindowIcon(self.app_icon)

        # Startup Window Size: fit the screen's usable area (excludes the taskbar),
        # so a 1366 x 768 display gets ~1366 x 728 instead of overflowing it.
        avail = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(1024, 600)
        self.resize(min(1366, avail.width()), min(768, avail.height()) - 32)  # 32 ≈ title bar
        self.move(avail.x() + (avail.width() - self.width()) // 2, avail.y())

        # Simulation Manager
        self.sim = AntennaSimulationManager(dt=0.02)
        self.live_view_ax = 'azimuth'
        self.fault_target_ax = 'azimuth'
        self.max_pts = 350
        self.t_buf = deque(maxlen=self.max_pts)
        self.buf = {
            ax: {
                'target': deque(maxlen=self.max_pts),
                'true':   deque(maxlen=self.max_pts),
                'enc':    deque(maxlen=self.max_pts),
                'ahrs':   deque(maxlen=self.max_pts),
                'fused':  deque(maxlen=self.max_pts),
                'c_enc':  deque(maxlen=self.max_pts),
                'c_ahrs': deque(maxlen=self.max_pts),
                'err_enc': deque(maxlen=self.max_pts),
                'err_ahrs': deque(maxlen=self.max_pts),
                'err_fused': deque(maxlen=self.max_pts),
            } for ax in AXES
        }

        # Page 2 Inspector state
        self.insp_df = None
        self.insp_filepath = None
        self.insp_axis = 'azimuth'
        self.insp_curves = {}
        self.insp_crosshairs = []
        self.insp_updating_region = False

        # Exploded digital twin state
        self.is_exploded_view = True

        self._apply_global_stylesheet()
        self._build_main_ui()

        # Initial 3D antenna pose and telemetry sync (render initial home position)
        self._update_3d_antenna()
        self._update_live_status()
        self.diag_quad.update_telemetry(self.sim, self.live_view_ax)

        # Simulation Timer (50 Hz / 20ms)
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self._on_sim_tick)
        self.sim_timer.start(20)

    # ─────────────────────────────────────────────────────────────────────────
    # GLOBAL APPLICATION STYLESHEET (NO CSS FILTER / OPACITY WARNINGS)
    # ─────────────────────────────────────────────────────────────────────────
    def showEvent(self, event):
        super().showEvent(event)
        # The native handle exists once the window is shown; apply the black caption then
        set_native_title_bar(int(self.winId()))
        # Re-send the icon once the taskbar button exists so it never keeps a stale default
        QTimer.singleShot(0, lambda: self.setWindowIcon(self.app_icon))

    def _apply_global_stylesheet(self):
        # Human voice: Geologica condensed to 93% width, 13px UI size
        QApplication.setFont(sans_font(13))

        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {PAL['void']};
            }}
            QWidget {{
                background-color: {PAL['bg']};
                color: {PAL['text_1']};
            }}
            QToolTip {{
                background: {PAL['surface_2']};
                color: {PAL['text_1']};
                border: 1px solid {PAL['line_strong']};
                padding: 3px 6px;
            }}
            QTabWidget::pane {{
                border: 1px solid {PAL['line']};
                background: {PAL['bg']};
                top: -1px;
            }}
            QTabBar::tab {{
                background: {PAL['surface_1']};
                color: {PAL['text_2']};
                font-size: 13px;
                padding: 6px 15px;
                border: 1px solid {PAL['line']};
                border-bottom: none;
                margin-right: 1px;
            }}
            QTabBar::tab:selected {{
                background: {PAL['bg']};
                color: {PAL['text_1']};
                border-top: 1px solid {PAL['text_2']};
            }}
            QTabBar::tab:hover:!selected {{
                background: {PAL['surface_3']};
                color: {PAL['text_1']};
            }}
            QScrollArea {{
                border: 1px solid {PAL['line']};
                background: {PAL['surface_1']};
            }}
            QScrollBar:vertical {{
                background: {PAL['bg']};
                width: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {PAL['line_strong']};
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {PAL['text_3']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QStatusBar {{
                background: {PAL['surface_1']};
                color: {PAL['text_2']};
                font-size: 12px;
                border-top: 1px solid {PAL['line']};
            }}
            QStatusBar QLabel {{
                background: transparent;
            }}
            QSlider {{
                background: transparent;
            }}
            QSlider::groove:horizontal {{
                height: 2px;
                background: {PAL['line_strong']};
            }}
            QSlider::sub-page:horizontal {{
                background: {PAL['text_3']};
            }}
            QSlider::handle:horizontal {{
                background: {PAL['text_2']};
                border: 1px solid {PAL['text_2']};
                width: 8px;
                height: 12px;
                margin: -6px 0;
                border-radius: 1px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {PAL['text_1']};
                border-color: {PAL['text_1']};
            }}
            QComboBox {{
                background: {PAL['surface_2']};
                color: {PAL['text_1']};
                border: 1px solid {PAL['line_strong']};
                border-radius: 2px;
                padding: 3px 7px;
                font-family: {MONO_CSS};
                font-size: 11px;
            }}
            QComboBox:hover {{
                background: {PAL['surface_3']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 16px;
            }}
            QComboBox QAbstractItemView {{
                background: {PAL['surface_2']};
                color: {PAL['text_1']};
                border: 1px solid {PAL['line_strong']};
                selection-background-color: {PAL['surface_3']};
                selection-color: {PAL['text_1']};
                font-family: {MONO_CSS};
            }}
            QSplitter::handle {{
                background-color: {PAL['line']};
            }}
            QSplitter::handle:hover, QSplitter::handle:pressed {{
                background-color: {PAL['line_strong']};
            }}
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN APPLICATION UI
    # ─────────────────────────────────────────────────────────────────────────
    def _build_main_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tabs = QTabWidget()

        # Page 1: Live 3D Simulation & Noise Game
        self.page1 = QWidget()
        self._build_page1_ui(self.page1)
        self.tabs.addTab(self.page1, "Live Simulation && Noise Control")

        # Page 2: History & Post-Run Analysis
        self.page2 = QWidget()
        self._build_page2_ui(self.page2)
        self.tabs.addTab(self.page2, "Post-Run Analysis && CSV Inspector")

        layout.addWidget(self.tabs)

        # Global Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Simulation ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1: LIVE SIMULATION & NOISE CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    def _build_page1_ui(self, parent):
        p1_root = QVBoxLayout(parent)
        p1_root.setContentsMargins(2, 2, 2, 2)
        p1_root.setSpacing(4)

        # ── TOP SHELF: Pipeline & Solver Trajectory Ribbon ─────────────────
        self.pipeline_ribbon = PipelineRibbonWidget()
        p1_root.addWidget(self.pipeline_ribbon)

        # ── MAIN WORKSPACE SPLITTER (FREELY RESIZABLE DOCKS) ────────────────
        p1_split = QSplitter(Qt.Horizontal)
        p1_split.setHandleWidth(4)
        p1_split.setChildrenCollapsible(False)

        # 1. Left Dock: Parameter & Execution Docks (Side Rail)
        sidebar = self._build_p1_sidebar()
        p1_split.addWidget(sidebar)

        # 2. Central Column: 3D Viewport (Exploded Isometric) + Lower 2x2 Telemetry Quad
        center_split = QSplitter(Qt.Vertical)
        center_split.setHandleWidth(4)
        center_split.setChildrenCollapsible(False)

        self.viewport_container = self._build_p1_3d_viewport()
        center_split.addWidget(self.viewport_container)

        self.diag_quad = DiagnosticQuadWidget()
        center_split.addWidget(self.diag_quad)

        center_split.setStretchFactor(0, 3)
        center_split.setStretchFactor(1, 1)
        center_split.setSizes([450, 160])

        p1_split.addWidget(center_split)

        # 3. Right Dock: Real-Time Telemetry Plots
        plots_container = self._build_p1_live_plots()
        p1_split.addWidget(plots_container)

        p1_split.setStretchFactor(0, 2)  # Sidebar
        p1_split.setStretchFactor(1, 5)  # Center
        p1_split.setStretchFactor(2, 4)  # Plots
        p1_split.setSizes([300, 560, 480])

        p1_root.addWidget(p1_split)
        self.p1_split = p1_split
        self.p1_sidebar = sidebar
        # Styled widget sizes are only known once shown, so fit the sidebar after the first show
        QTimer.singleShot(0, self._fit_p1_sidebar_width)

    def _fit_p1_sidebar_width(self):
        """Give the left dock enough width for its content so nothing is clipped at the right edge."""
        scroll = self.p1_sidebar
        content = scroll.widget()
        need = (content.minimumSizeHint().width()
                + scroll.verticalScrollBar().sizeHint().width()  # reserve room if it appears
                + 2 * scroll.frameWidth())
        scroll.setMinimumWidth(need)

        sizes = self.p1_split.sizes()
        if sizes[0] < need:
            deficit = need - sizes[0]
            sizes[0] = need
            sizes[1] -= deficit  # take the width from the 3D viewport column
            self.p1_split.setSizes(sizes)

    def _build_p1_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)  # Refined to the content's real width in _fit_p1_sidebar_width
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        container = QWidget()
        container.setObjectName("p1SidebarContent")
        container.setStyleSheet(f"QWidget#p1SidebarContent {{ background: {PAL['surface_1']}; }}")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(7, 6, 7, 6)  # 8px nominal × 0.93 horizontally
        layout.setSpacing(6)

        # ── Group 1: Motion & Target ───────────────────────────────────────
        grp_tgt = QGroupBox("Motion && Target")
        grp_tgt.setStyleSheet(self._grp_style())
        gt_l = QVBoxLayout(grp_tgt)
        gt_l.setSpacing(5)

        # Motion mode: segmented control (state words from the sim, mono)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        self.rb_mode_target = QPushButton("TARGET")
        self.rb_mode_brownian = QPushButton("BROWNIAN")
        self.rb_mode_target.setToolTip("Cascade PID tracks the target sliders")
        self.rb_mode_brownian.setToolTip("Brownian random motion")
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for b in (self.rb_mode_target, self.rb_mode_brownian):
            b.setCheckable(True)
            b.setStyleSheet(self._segment_style())
            self.mode_group.addButton(b)
            mode_row.addWidget(b)
        self.rb_mode_target.setChecked(True)
        self.rb_mode_target.toggled.connect(self._on_motion_mode_changed)
        gt_l.addLayout(mode_row)

        # Target sliders: axis identifiers verbatim, value + separate unit token
        self.tgt_sliders = {}
        self.tgt_labels = {}
        target_defs = [
            ('azimuth',      0, 360, 90),
            ('elevation',    0, 90,  45),
            ('polarization', 0, 180, 45),
        ]
        for ax, mn, mx, dflt in target_defs:
            row_l = QHBoxLayout()
            lbl_name = QLabel(ax)
            lbl_name.setStyleSheet(self._mono_qss(PAL['text_2']))
            lbl_name.setFixedWidth(84)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(mn, mx)
            sl.setValue(dflt)
            lbl_val = QLabel(value_html(f"{dflt:3d}", "deg"))
            lbl_val.setStyleSheet(self._mono_qss())
            lbl_val.setFixedWidth(52)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            sl.valueChanged.connect(lambda v, a=ax, l=lbl_val: self._on_target_slider(a, v, l))
            row_l.addWidget(lbl_name)
            row_l.addWidget(sl)
            row_l.addWidget(lbl_val)
            gt_l.addLayout(row_l)
            self.tgt_sliders[ax] = sl
            self.tgt_labels[ax] = lbl_val
            self.sim.set_target(ax, dflt)

        # Run / Stop / Reset
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start")
        self.btn_stop = QPushButton("■ Stop")
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_start.setCheckable(True)  # checked = solver running (live ring)
        self.btn_start.setStyleSheet(self._action_btn_style(state_ring=True))
        self.btn_stop.setStyleSheet(self._action_btn_style())
        self.btn_reset.setStyleSheet(self._action_btn_style())
        self.btn_start.clicked.connect(self._on_start_sim)
        self.btn_stop.clicked.connect(self._on_stop_sim)
        self.btn_reset.clicked.connect(self._on_reset_sim)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_reset)
        gt_l.addLayout(btn_row)

        layout.addWidget(grp_tgt)

        # ── Group 2: Telemetry Logging ─────────────────────────────────────
        grp_rec = QGroupBox("Telemetry Logging")
        grp_rec.setStyleSheet(self._grp_style())
        gr_l = QVBoxLayout(grp_rec)
        gr_l.setSpacing(5)

        rec_btn_row = QHBoxLayout()
        self.btn_rec_toggle = QPushButton("● Record CSV")
        self.btn_rec_toggle.setCheckable(True)  # checked = recording (live ring)
        self.btn_rec_toggle.setStyleSheet(self._action_btn_style(state_ring=True))
        self.btn_rec_toggle.clicked.connect(self._on_toggle_recording)
        rec_btn_row.addWidget(self.btn_rec_toggle)

        self.btn_export_inspect = QPushButton("Export && Inspect →")
        self.btn_export_inspect.setStyleSheet(self._action_btn_style())
        self.btn_export_inspect.clicked.connect(self._on_export_and_inspect)
        rec_btn_row.addWidget(self.btn_export_inspect)
        gr_l.addLayout(rec_btn_row)

        self.lbl_rec_status = QLabel()
        self.lbl_rec_status.setStyleSheet(self._mono_qss(PAL['text_2'], 10))
        self._set_rec_status("IDLE", 0)
        gr_l.addWidget(self.lbl_rec_status)

        layout.addWidget(grp_rec)

        # ── Group 3: Sensor Noise & Fault Injection ────────────────────────
        grp_game = QGroupBox("Sensor Noise && Fault Injection")
        grp_game.setStyleSheet(self._grp_style())
        gg_l = QVBoxLayout(grp_game)
        gg_l.setSpacing(5)

        f_axis_row = QHBoxLayout()
        f_axis_lbl = QLabel("Inject target")
        f_axis_lbl.setStyleSheet(f"color: {PAL['text_2']}; font-size: 12px; background: transparent;")
        f_axis_row.addWidget(f_axis_lbl)
        self.cb_fault_axis = QComboBox()
        self.cb_fault_axis.addItems(AXES + ["all"])
        self.cb_fault_axis.currentTextChanged.connect(self._on_fault_axis_changed)
        f_axis_row.addWidget(self.cb_fault_axis, stretch=1)
        gg_l.addLayout(f_axis_row)

        # Noise sliders (identifiers match AntennaSimulationManager.set_noise kwargs)
        def noise_row(name, dflt):
            row = QHBoxLayout()
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet(self._mono_qss(PAL['text_2']))
            lbl_name.setFixedWidth(84)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 50)
            sl.setValue(dflt)
            lbl_val = QLabel(value_html(f"{dflt / 10:.2f}", "deg"))
            lbl_val.setStyleSheet(self._mono_qss())
            lbl_val.setFixedWidth(52)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(lbl_name)
            row.addWidget(sl)
            row.addWidget(lbl_val)
            gg_l.addLayout(row)
            return sl, lbl_val

        self.sl_noise_enc, self.lbl_noise_enc = noise_row("sigma_enc", 5)
        self.sl_noise_enc.valueChanged.connect(self._on_enc_noise_slider)
        self.sl_noise_ahrs, self.lbl_noise_ahrs = noise_row("sigma_ahrs", 3)
        self.sl_noise_ahrs.valueChanged.connect(self._on_ahrs_noise_slider)

        fault_btn_row = QHBoxLayout()
        self.btn_fault_drift = QPushButton("+ Drift 1.5°/s")
        self.btn_fault_drift.setStyleSheet(self._action_btn_style())
        self.btn_fault_drift.setToolTip("Inject a 1.5°/s encoder drift ramp")
        self.btn_fault_drift.clicked.connect(self._on_inject_drift)
        fault_btn_row.addWidget(self.btn_fault_drift)

        self.btn_fault_glitch = QPushButton("+ Glitch +12°")
        self.btn_fault_glitch.setStyleSheet(self._action_btn_style())
        self.btn_fault_glitch.setToolTip("Inject a +12° encoder glitch spike")
        self.btn_fault_glitch.clicked.connect(self._on_inject_glitch)
        fault_btn_row.addWidget(self.btn_fault_glitch)
        gg_l.addLayout(fault_btn_row)

        self.btn_fault_clear = QPushButton("↺ Clear Faults / Recover Sensor")
        self.btn_fault_clear.setStyleSheet(self._action_btn_style())
        self.btn_fault_clear.clicked.connect(self._on_clear_faults)
        gg_l.addWidget(self.btn_fault_clear)

        layout.addWidget(grp_game)

        # ── Group 4: Kinematic Bus Readout ─────────────────────────────────
        grp_stat = QGroupBox("Kinematic Bus Readout")
        grp_stat.setStyleSheet(self._grp_style())
        gs_l = QVBoxLayout(grp_stat)
        gs_l.setSpacing(3)

        hdr = QLabel(self._bus_row("axis", "true", "fused", "err") +
                     f"<span style=\"color:{PAL['text_3']};\">&nbsp;&nbsp;deg</span>")
        hdr.setStyleSheet(self._mono_qss(PAL['text_3'], 10))
        gs_l.addWidget(hdr)

        self.live_status_lbls = {}
        for ax in AXES:
            l = QLabel(ax)
            l.setStyleSheet(self._mono_qss(size=10))
            gs_l.addWidget(l)
            self.live_status_lbls[ax] = l

        self.live_conf_lbl = QLabel()
        self.live_conf_lbl.setStyleSheet(self._mono_qss(size=10))
        gs_l.addWidget(self.live_conf_lbl)

        layout.addWidget(grp_stat)
        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _panel_header(self, title: str, tag: str = ""):
        """30px panel header: Title Case sans heading + optional mono micro tag."""
        bar = QWidget()
        bar.setObjectName("panelHeader")
        bar.setFixedHeight(30)
        bar.setStyleSheet(
            f"QWidget#panelHeader {{ background: {PAL['surface_1']}; border-bottom: 1px solid {PAL['line']}; }}"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(9, 0, 7, 0)
        row.setSpacing(6)
        lbl = QLabel(title)
        lbl.setFont(sans_font(14, QFont.Medium))
        lbl.setStyleSheet(f"color: {PAL['text_1']}; background: transparent;")
        row.addWidget(lbl)
        if tag:
            t = QLabel(tag)
            t.setFont(mono_font(10, tracking_em=0.08))
            t.setStyleSheet(f"color: {PAL['text_3']}; background: transparent;")
            row.addWidget(t)
        row.addStretch()
        return bar, row

    def _build_p1_3d_viewport(self):
        vp_widget = QWidget()
        vp_widget.setObjectName("viewportPanel")
        vp_widget.setStyleSheet(f"QWidget#viewportPanel {{ background: {PAL['bg']}; border: 1px solid {PAL['line']}; }}")
        layout = QVBoxLayout(vp_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header: title + view controls
        tb, tb_layout = self._panel_header("Digital Twin", "EXPLODED ISOMETRIC")

        self.btn_explode = QPushButton("Exploded")
        self.btn_explode.setCheckable(True)
        self.btn_explode.setChecked(True)
        self.btn_explode.setToolTip("Toggle exploded / assembled view")
        self.btn_explode.setStyleSheet(self._segment_style())
        self.btn_explode.clicked.connect(self._toggle_exploded_view)
        tb_layout.addWidget(self.btn_explode)
        tb_layout.addSpacing(6)

        view_row = QHBoxLayout()
        view_row.setSpacing(0)
        for view_name in ['Iso', 'Top', 'Front', 'Side']:
            b = QPushButton(view_name)
            b.setStyleSheet(self._segment_style())
            b.clicked.connect(lambda _, vn=view_name.lower(): self._set_3d_camera(vn))
            view_row.addWidget(b)
        tb_layout.addLayout(view_row)

        layout.addWidget(tb)

        # Readout strip for the selected axis
        self.callout_overlay = CalloutOverlayWidget()
        layout.addWidget(self.callout_overlay)

        # PyVista QtInteractor on the app canvas
        self.plotter = QtInteractor(vp_widget)
        self.plotter.set_background(PAL['bg'])
        layout.addWidget(self.plotter, stretch=1)

        # Load STL meshes with kinematics configuration
        self.orig_pts = {}
        self.meshes = {}
        self.layer_offsets = {
            'azimuth':      np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'elevation':    np.array([0.0, 0.0, 45.0], dtype=np.float32),
            'polarization': np.array([0.0, 0.0, 90.0], dtype=np.float32),
        }

        for name, cfg in LINKS_CONFIG.items():
            stl_path = STL_FILES[name]
            if os.path.exists(stl_path):
                mesh = pv.read(stl_path)
                if mesh.n_cells > 10000:
                    mesh = mesh.decimate(0.5)
            else:
                c = cfg['pivot']
                if name == 'azimuth':
                    mesh = pv.Cylinder(center=c, direction=[0, 0, 1], radius=40, height=20, resolution=32)
                elif name == 'elevation':
                    mesh = pv.Box(bounds=(c[0]-30, c[0]+30, c[1]-10, c[1]+10, c[2]-40, c[2]+40))
                else:
                    mesh = pv.Cylinder(center=c, direction=[0, 1, 0], radius=20, height=60, resolution=20)

            mesh.points = mesh.points.astype(np.float32)
            self.orig_pts[name] = mesh.points.copy()
            self.meshes[name] = mesh

        # Hairline projection line along the assembly axis
        self._add_3d_projection_lines()

        # Small leader-line callouts showing each joint's live angle
        self._add_3d_angle_labels()

        # Selected actuator solid, the rest as hairline wireframes
        self._refresh_mesh_materials()

        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()
        self.plotter.camera.zoom(1.15)

        return vp_widget

    def _add_3d_projection_lines(self):
        """Hairline projection line along the assembly alignment axis."""
        p_start = np.array([0.0, 0.0, -30.0])
        p_end   = np.array([0.0, 0.0, 260.0])
        line = pv.Line(p_start, p_end, resolution=30)
        self.plotter.add_mesh(
            line, color=PAL['line_strong'], line_width=1.0,
            name='axis_projection_line'
        )

    # Callout offset (display px, y up) from each joint pivot to its angle label
    ANGLE_LABEL_OFFSETS = {
        'azimuth':      (70, -35),
        'elevation':    (-70, 30),
        'polarization': (70, 40),
    }
    ANGLE_LABEL_TAGS = {'azimuth': 'AZ', 'elevation': 'EL', 'polarization': 'POL'}

    @staticmethod
    def _rgb(hex_color: str):
        c = QColor(hex_color)
        return c.redF(), c.greenF(), c.blueF()

    def _add_3d_angle_labels(self):
        """
        Small 2D callouts inside the 3D viewer: a hairline elbow leader from each
        joint pivot to a compact mono readout — tag (gray), value (primary), unit
        (gray, separate token). Screen positions are recomputed on every render so
        the labels track both the kinematics and camera moves.
        """
        mono_files = glob.glob(os.path.join(FONTS_DIR, 'JetBrainsMono*.ttf'))
        self.angle_labels = {}
        renderer = self.plotter.renderer
        for ax in AXES:
            pieces = []
            for role, color in (('tag', PAL['text_3']), ('value', PAL['text_1']), ('unit', PAL['text_3'])):
                actor = vtkTextActor()
                tp = actor.GetTextProperty()
                if mono_files:
                    tp.SetFontFamilyAsString('File')
                    tp.SetFontFile(mono_files[0])
                else:
                    tp.SetFontFamilyToCourier()
                tp.SetFontSize(11)
                tp.SetColor(*self._rgb(color))
                tp.SetJustificationToLeft()
                tp.SetVerticalJustificationToCentered()
                tp.SetBackgroundColor(*self._rgb(PAL['bg']))
                tp.SetBackgroundOpacity(0.8)
                renderer.AddActor2D(actor)
                pieces.append(actor)
            pieces[0].SetInput(f"{self.ANGLE_LABEL_TAGS[ax]} ")
            pieces[1].SetInput("--")
            pieces[2].SetInput(" deg")

            # Leader: anchor dot (vertex) + elbow polyline (anchor -> elbow -> label)
            leader_pts = vtkPoints()
            leader_pts.SetNumberOfPoints(3)
            verts = vtkCellArray()
            verts.InsertNextCell(1, [0])
            lines = vtkCellArray()
            lines.InsertNextCell(3, [0, 1, 2])
            leader_pd = vtkPolyData()
            leader_pd.SetPoints(leader_pts)
            leader_pd.SetVerts(verts)
            leader_pd.SetLines(lines)
            mapper = vtkPolyDataMapper2D()
            mapper.SetInputData(leader_pd)
            leader = vtkActor2D()
            leader.SetMapper(mapper)
            leader.GetProperty().SetColor(*self._rgb(PAL['text_3']))
            leader.GetProperty().SetLineWidth(1.0)
            leader.GetProperty().SetPointSize(4.0)
            renderer.AddActor2D(leader)

            self.angle_labels[ax] = {
                'pieces': pieces, 'value': pieces[1], 'pts': leader_pts, 'pd': leader_pd,
                'anchor': LINKS_CONFIG[ax]['pivot'].astype(float),
            }

        renderer.AddObserver('StartEvent', self._layout_3d_angle_labels)

    def _layout_3d_angle_labels(self, renderer, _event=None):
        """Project each joint pivot to screen space and place its leader + label pieces."""
        bbox = [0.0, 0.0, 0.0, 0.0]
        for ax, lbl in self.angle_labels.items():
            renderer.SetWorldPoint(*lbl['anchor'], 1.0)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            dx, dy = self.ANGLE_LABEL_OFFSETS[ax]
            end = (x + dx, y + dy)
            lbl['pts'].SetPoint(0, x, y, 0)
            lbl['pts'].SetPoint(1, x + dx * 0.6, y + dy, 0)
            lbl['pts'].SetPoint(2, *end, 0)
            lbl['pts'].Modified()
            lbl['pd'].Modified()

            widths = []
            for actor in lbl['pieces']:
                actor.GetBoundingBox(renderer, bbox)
                widths.append(bbox[1] - bbox[0])
            # Left of the leader end for leftward callouts, right of it otherwise
            cursor = end[0] + 3 if dx >= 0 else end[0] - 3 - sum(widths)
            for actor, w in zip(lbl['pieces'], widths):
                actor.SetDisplayPosition(int(cursor), int(end[1]))
                cursor += w

    def _refresh_mesh_materials(self):
        """Selected actuator as a solid gray body; passive links as hairline wireframes."""
        for name, mesh in self.meshes.items():
            if name == self.live_view_ax:
                self.plotter.add_mesh(
                    mesh,
                    color=PAL['text_2'],
                    specular=0.2,
                    specular_power=15,
                    ambient=0.25,
                    diffuse=0.8,
                    smooth_shading=True,
                    name=name
                )
            else:
                self.plotter.add_mesh(
                    mesh,
                    color=PAL['surface_2'],
                    edge_color=PAL['line_strong'],
                    show_edges=True,
                    line_width=1.0,
                    opacity=0.55,
                    name=name
                )

    def _toggle_exploded_view(self, checked):
        self.is_exploded_view = checked
        self.btn_explode.setText("Exploded" if checked else "Assembled")
        self._update_3d_antenna()

    # ── Plot styling helpers (design_system.md: hairline grid, mono ticks, series hues) ──
    @staticmethod
    def _series_pen(key: str, width: float = 1.2, alpha: int = 255, dash: bool = False):
        c = QColor(SERIES[key])
        c.setAlpha(alpha)
        return pg.mkPen(c, width=width, style=Qt.DashLine if dash else Qt.SolidLine)

    @staticmethod
    def _style_plot(plot, title: str, y_label: str, x_label: str = None):
        plot.setTitle(title, color=PAL['text_2'], size='12px')
        plot.titleLabel.item.setFont(sans_font(12, QFont.Medium))
        for side in ('left', 'bottom'):
            axis = plot.getAxis(side)
            axis.setTickFont(mono_font(9))
            axis.enableAutoSIPrefix(False)  # plain "0.5 s", not "500 (x0.001)"
            axis.setPen(pg.mkPen(PAL['line_strong']))
            axis.setTextPen(pg.mkPen(PAL['text_3']))
        label_style = {'color': PAL['text_3'], 'font-size': '9px'}
        plot.setLabel('left', y_label, **label_style)
        if x_label:
            plot.setLabel('bottom', x_label, **label_style)
        plot.showGrid(x=True, y=True, alpha=0.12)
        legend = plot.addLegend(offset=(-8, 8), labelTextColor=PAL['text_2'], labelTextSize='9px',
                                brush=pg.mkBrush(17, 17, 19, 210), pen=pg.mkPen(PAL['line']))
        return legend

    @staticmethod
    def _mono_legend(legend):
        """Legend entries are signal identifiers -> machine voice; laid out in one row."""
        for _sample, label in legend.items:
            label.item.setFont(mono_font(10))
        legend.setColumnCount(max(1, len(legend.items)))

    def _build_p1_live_plots(self):
        container = QWidget()
        container.setObjectName("plotsPanel")
        container.setMinimumWidth(280)
        container.setStyleSheet(f"QWidget#plotsPanel {{ background: {PAL['bg']}; border: 1px solid {PAL['line']}; }}")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with axis selector (segmented chips, identifiers verbatim)
        sel_bar, sel_layout = self._panel_header("Telemetry")
        sel_row = QHBoxLayout()
        sel_row.setSpacing(0)
        self.live_ax_btns = {}
        for ax in AXES:
            b = QPushButton(ax)
            b.setCheckable(True)
            b.setChecked(ax == 'azimuth')
            b.setStyleSheet(self._segment_style(mono=True))
            b.clicked.connect(lambda _, a=ax: self._set_live_axis(a))
            sel_row.addWidget(b)
            self.live_ax_btns[ax] = b
        sel_layout.addLayout(sel_row)
        layout.addWidget(sel_bar)

        # PyQtGraph GraphicsLayoutWidget
        self.live_gw = pg.GraphicsLayoutWidget()
        self.live_gw.ci.layout.setContentsMargins(4, 4, 4, 4)
        self.live_gw.ci.layout.setSpacing(4)
        layout.addWidget(self.live_gw, stretch=1)

        self.plt_live_ang = self.live_gw.addPlot(row=0, col=0)
        lg_ang = self._style_plot(self.plt_live_ang, "Angle Response", "deg")

        self.plt_live_conf = self.live_gw.addPlot(row=1, col=0)
        lg_conf = self._style_plot(self.plt_live_conf, "Confidence Weights", "weight")
        self.plt_live_conf.setYRange(-0.05, 1.05, padding=0)

        self.plt_live_err = self.live_gw.addPlot(row=2, col=0)
        lg_err = self._style_plot(self.plt_live_err, "Tracking Error (Sensor − Truth)", "deg", "s")
        self.plt_live_err.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(PAL['line_strong'], style=Qt.DashLine, width=1)))

        # Synchronize X axes
        self.plt_live_conf.setXLink(self.plt_live_ang)
        self.plt_live_err.setXLink(self.plt_live_ang)

        # Curves (legend names are the signal identifiers)
        sp = self._series_pen
        self.crv_live = {
            'target': self.plt_live_ang.plot(pen=sp('target', 1.2, dash=True), name='target'),
            'true':   self.plt_live_ang.plot(pen=sp('true', 1.4), name='true'),
            'enc':    self.plt_live_ang.plot(pen=sp('enc', 1.0, 170), name='enc'),
            'ahrs':   self.plt_live_ang.plot(pen=sp('ahrs', 1.0, 170), name='ahrs'),
            'fused':  self.plt_live_ang.plot(pen=sp('fused', 2.0), name='fused'),

            'c_enc':  self.plt_live_conf.plot(pen=sp('enc', 1.6), name='c_enc'),
            'c_ahrs': self.plt_live_conf.plot(pen=sp('ahrs', 1.6), name='c_ahrs'),

            'err_enc':   self.plt_live_err.plot(pen=sp('enc', 1.0, 170), name='err_enc'),
            'err_ahrs':  self.plt_live_err.plot(pen=sp('ahrs', 1.0, 170), name='err_ahrs'),
            'err_fused': self.plt_live_err.plot(pen=sp('fused', 2.0), name='err_fused'),
        }
        for lg in (lg_ang, lg_conf, lg_err):
            self._mono_legend(lg)

        return container

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2: POST-RUN ANALYSIS & CSV INSPECTOR
    # ─────────────────────────────────────────────────────────────────────────
    def _build_page2_ui(self, parent):
        p2_layout = QHBoxLayout(parent)
        p2_layout.setContentsMargins(2, 2, 2, 2)
        p2_layout.setSpacing(4)

        p2_split = QSplitter(Qt.Horizontal)
        p2_split.setHandleWidth(4)
        p2_split.setChildrenCollapsible(False)

        # ── Left / Main Plots Area ──────────────────────────────────────────
        plot_col = QWidget()
        pl_layout = QVBoxLayout(plot_col)
        pl_layout.setContentsMargins(0, 0, 0, 0)
        pl_layout.setSpacing(4)

        # Top bar: file & axis selection
        top_bar = QWidget()
        top_bar.setObjectName("p2TopBar")
        top_bar.setFixedHeight(34)
        top_bar.setStyleSheet(
            f"QWidget#p2TopBar {{ background: {PAL['surface_1']}; border: 1px solid {PAL['line']}; }}"
        )
        tb_l = QHBoxLayout(top_bar)
        tb_l.setContentsMargins(7, 0, 7, 0)
        tb_l.setSpacing(6)

        self.btn_p2_inspect_last = QPushButton("Inspect Last Recording")
        self.btn_p2_inspect_last.setStyleSheet(self._action_btn_style())
        self.btn_p2_inspect_last.clicked.connect(self._on_inspect_last_recording)
        tb_l.addWidget(self.btn_p2_inspect_last)

        self.btn_p2_open = QPushButton("Open CSV…")
        self.btn_p2_open.setStyleSheet(self._action_btn_style())
        self.btn_p2_open.clicked.connect(self._on_p2_open_file)
        tb_l.addWidget(self.btn_p2_open)

        self.btn_p2_reload = QPushButton("↺ Reload")
        self.btn_p2_reload.setStyleSheet(self._action_btn_style())
        self.btn_p2_reload.clicked.connect(self._on_p2_reload)
        tb_l.addWidget(self.btn_p2_reload)

        tb_l.addSpacing(9)
        ax_lbl = QLabel("AXIS")
        ax_lbl.setFont(mono_font(10, tracking_em=0.08))
        ax_lbl.setStyleSheet(f"color: {PAL['text_3']}; background: transparent;")
        tb_l.addWidget(ax_lbl)

        self.cb_p2_axis = QComboBox()
        self.cb_p2_axis.addItems(AXES)
        self.cb_p2_axis.currentTextChanged.connect(self._on_p2_axis_changed)
        tb_l.addWidget(self.cb_p2_axis)

        self.lbl_p2_file_info = QLabel("No CSV loaded")
        self.lbl_p2_file_info.setStyleSheet(self._mono_qss(PAL['text_3'], 10) + " margin-left: 7px;")
        tb_l.addWidget(self.lbl_p2_file_info, stretch=1)
        pl_layout.addWidget(top_bar)

        # Plot controls + three resizable, independently toggleable panels
        pl_layout.addWidget(self._build_p2_controls())
        self.p2_plot_split = QSplitter(Qt.Vertical)
        self.p2_plot_split.setHandleWidth(3)
        self.p2_plot_split.setChildrenCollapsible(False)
        pl_layout.addWidget(self.p2_plot_split, stretch=1)

        # Overview timeline strip with draggable region
        ov_row = QHBoxLayout()
        ov_row.setContentsMargins(4, 2, 4, 0)
        ov_row.setSpacing(9)
        ov_title = QLabel("TIMELINE OVERVIEW")
        ov_title.setFont(mono_font(10, tracking_em=0.08))
        ov_title.setStyleSheet(f"color: {PAL['text_3']};")
        ov_row.addWidget(ov_title)
        ov_hint = QLabel("Drag the window or its edges to pan and zoom")
        ov_hint.setStyleSheet(f"color: {PAL['text_3']}; font-size: 12px;")
        ov_row.addWidget(ov_hint)
        ov_row.addStretch()
        pl_layout.addLayout(ov_row)

        self.p2_overview = pg.PlotWidget(name="P2Overview")
        self.p2_overview.setFixedHeight(68)
        self.p2_overview.showGrid(x=True, y=True, alpha=0.12)
        self.p2_overview.setMouseEnabled(x=False, y=False)
        self.p2_overview.getPlotItem().hideAxis('left')
        self.p2_overview.getPlotItem().hideAxis('bottom')

        region_fill = QColor(PAL['text_1'])
        region_fill.setAlpha(22)
        self.p2_region = pg.LinearRegionItem([0, 5], brush=pg.mkBrush(region_fill))
        for edge in self.p2_region.lines:
            edge.setPen(pg.mkPen(PAL['text_3'], width=1))
            edge.setHoverPen(pg.mkPen(PAL['text_1'], width=1))
        self.p2_region.setZValue(10)
        self.p2_overview.addItem(self.p2_region)
        self.p2_region.sigRegionChanged.connect(self._on_p2_region_changed)
        pl_layout.addWidget(self.p2_overview)

        self._setup_p2_plots()
        p2_split.addWidget(plot_col)

        # ── Right / Inspector Sidebar ───────────────────────────────────────
        p2_sidebar = self._build_p2_sidebar()
        p2_split.addWidget(p2_sidebar)

        p2_split.setStretchFactor(0, 4)
        p2_split.setStretchFactor(1, 1)
        p2_split.setSizes([950, 320])
        p2_layout.addWidget(p2_split)

    def _micro_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(mono_font(10, tracking_em=0.08))
        lbl.setStyleSheet(f"color: {PAL['text_3']}; background: transparent;")
        return lbl

    def _build_p2_controls(self):
        """Plot control bar: panel visibility, Y-axis behaviour, and view options."""
        bar = QWidget()
        bar.setObjectName("p2Controls")
        bar.setFixedHeight(30)
        bar.setStyleSheet(
            f"QWidget#p2Controls {{ background: {PAL['surface_1']}; border: 1px solid {PAL['line']}; }}"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(7, 0, 7, 0)
        row.setSpacing(6)

        # Panels: any combination of the three plots; one visible = full height
        row.addWidget(self._micro_label("PANELS"))
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.btn_p2_all = QPushButton("All")
        self.btn_p2_all.setStyleSheet(self._segment_style())
        self.btn_p2_all.setToolTip("Show all three panels stacked for comparison")
        self.btn_p2_all.clicked.connect(lambda: self._set_p2_panels(set(self.p2_plots)))
        seg.addWidget(self.btn_p2_all)
        self.p2_panel_btns = {}
        for key, title in P2_PANELS:
            b = QPushButton(title)
            b.setCheckable(True)
            b.setChecked(True)
            b.setStyleSheet(self._segment_style())
            b.setToolTip(f"Show / hide the {title.lower()} panel. Double-click a plot to focus it.")
            b.toggled.connect(self._on_p2_panel_btn_toggled)
            seg.addWidget(b)
            self.p2_panel_btns[key] = b
        row.addLayout(seg)

        row.addSpacing(9)
        row.addWidget(self._micro_label("Y AXIS"))
        yrow = QHBoxLayout()
        yrow.setSpacing(0)
        self.btn_p2_auto_y = QPushButton("Auto Y")
        self.btn_p2_auto_y.setCheckable(True)
        self.btn_p2_auto_y.setChecked(True)
        self.btn_p2_auto_y.setToolTip("Fit the Y axis to the data inside the visible time window")
        self.btn_p2_auto_y.toggled.connect(self._apply_p2_auto_y)
        self.btn_p2_lock_y = QPushButton("Lock Y")
        self.btn_p2_lock_y.setCheckable(True)
        self.btn_p2_lock_y.setToolTip("Mouse wheel and drag only move the time axis")
        self.btn_p2_lock_y.toggled.connect(self._apply_p2_lock_y)
        self.btn_p2_reset_y = QPushButton("Reset Y")
        self.btn_p2_reset_y.setToolTip("Re-fit every Y axis")
        self.btn_p2_reset_y.clicked.connect(lambda: self._reset_p2_y())
        for b in (self.btn_p2_auto_y, self.btn_p2_lock_y, self.btn_p2_reset_y):
            b.setStyleSheet(self._segment_style())
            yrow.addWidget(b)
        row.addLayout(yrow)

        row.addStretch()

        # View options
        opts = QHBoxLayout()
        opts.setSpacing(0)
        self.btn_p2_grid = QPushButton("Grid")
        self.btn_p2_grid.setCheckable(True)
        self.btn_p2_grid.setChecked(True)
        self.btn_p2_grid.toggled.connect(self._apply_p2_grid)
        self.btn_p2_legend = QPushButton("Legend")
        self.btn_p2_legend.setCheckable(True)
        self.btn_p2_legend.setChecked(True)
        self.btn_p2_legend.toggled.connect(self._apply_p2_legend)
        self.btn_p2_white = QPushButton("White BG")
        self.btn_p2_white.setCheckable(True)
        self.btn_p2_white.setToolTip("White plot background for figures in papers and reports")
        self.btn_p2_white.toggled.connect(lambda on: self._apply_p2_theme('paper' if on else 'dark'))
        self.btn_p2_save = QPushButton("Save PNG…")
        self.btn_p2_save.setToolTip("Save the visible panels as an image")
        self.btn_p2_save.clicked.connect(self._on_p2_save_png)
        for b in (self.btn_p2_grid, self.btn_p2_legend, self.btn_p2_white, self.btn_p2_save):
            b.setStyleSheet(self._segment_style())
            opts.addWidget(b)
        row.addLayout(opts)
        return bar

    def _setup_p2_plots(self):
        """Three independent plot panels in a vertical splitter (drag the handles to resize)."""
        self.p2_plots, self.p2_widgets = {}, {}
        self.p2_theme = P2_PLOT_THEMES['dark']
        self.p2_specs = specs = [
            ('angle',      "Angle Tracking",                "deg"),
            ('confidence', "Confidence Weights",            "weight"),
            ('error',      "Tracking Error (Sensor − Truth)", "deg"),
        ]
        for key, title, ylabel in specs:
            pw = pg.PlotWidget()
            pw.setFrameShape(QFrame.NoFrame)
            plot = pw.getPlotItem()
            plot.hideButtons()
            self._style_plot(plot, title, ylabel)
            self.p2_plot_split.addWidget(pw)
            self.p2_widgets[key], self.p2_plots[key] = pw, plot

        self.p2_ang = self.p2_plots['angle']
        self.p2_conf = self.p2_plots['confidence']
        self.p2_err = self.p2_plots['error']
        self.p2_conf.setYRange(-0.05, 1.05, padding=0)
        self.p2_zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(PAL['line_strong'], style=Qt.DashLine, width=1))
        self.p2_err.addItem(self.p2_zero_line)

        # Sync X axes (the angle plot is the master that drives the overview window)
        self.p2_conf.setXLink(self.p2_ang)
        self.p2_err.setXLink(self.p2_ang)
        self.p2_ang.sigRangeChanged.connect(self._on_p2_plot_range_changed)

        # Crosshairs (chrome: grayscale); mouse handling is per panel scene
        self.insp_crosshairs = [
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(PAL['text_2'], width=1, style=Qt.DashLine))
            for _ in range(3)
        ]
        self._p2_leave_watch = set()
        for (key, plot), ch in zip(self.p2_plots.items(), self.insp_crosshairs):
            ch.setZValue(100)
            ch.setVisible(False)
            plot.addItem(ch)
            pw = self.p2_widgets[key]
            pw.scene().sigMouseMoved.connect(lambda pos, p=plot: self._on_p2_mouse_moved(pos, p))
            pw.scene().sigMouseClicked.connect(lambda ev, k=key: self._on_p2_plot_clicked(ev, k))
            pw.viewport().installEventFilter(self)  # hide the crosshair when the mouse leaves
            self._p2_leave_watch.add(pw.viewport())

        sp = self._series_pen
        pens = {
            'target': sp('target', 1.2, dash=True), 'true': sp('true', 1.4),
            'enc': sp('enc', 1.0, 170), 'ahrs': sp('ahrs', 1.0, 170), 'fused': sp('fused', 2.0),
            'conf_enc': sp('enc', 1.6), 'conf_ahrs': sp('ahrs', 1.6),
            'err_enc': sp('enc', 1.0, 170), 'err_ahrs': sp('ahrs', 1.0, 170), 'err_fused': sp('fused', 2.0),
        }
        for key, name, panel, _signal in P2_CURVES:
            self.insp_curves[key] = self.p2_plots[panel].plot(pen=pens[key], name=name)
        self.p2_signal_on = {s: True for s in P2_SIGNALS}
        self._rebuild_p2_legends()

        self.insp_overview_crv = self.p2_overview.plot(pen=sp('fused', 1.0))
        self._p2_prev_panels = None
        self._refresh_p2_axis_labels()
        for key in ('angle', 'error'):
            self.p2_plots[key].vb.sigRangeChangedManually.connect(self._on_p2_manual_y)
        self._apply_p2_auto_y(True)

    # ── Panel visibility ──────────────────────────────────────────────────────
    def _visible_p2_panels(self) -> set:
        return {k for k, b in self.p2_panel_btns.items() if b.isChecked()}

    def _on_p2_panel_btn_toggled(self, _checked=False):
        vis = self._visible_p2_panels()
        if not vis:  # at least one panel must stay visible
            btn = self.sender()
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)
            return
        self._apply_p2_panels(vis)

    def _set_p2_panels(self, keys: set):
        for k, b in self.p2_panel_btns.items():
            b.blockSignals(True)
            b.setChecked(k in keys)
            b.blockSignals(False)
        self._apply_p2_panels(keys)

    def _apply_p2_panels(self, keys: set):
        for k, pw in self.p2_widgets.items():
            pw.setVisible(k in keys)
        # Equal heights for whatever is visible; the splitter handles stay draggable
        n = max(1, len(keys))
        h = max(1, self.p2_plot_split.height())
        self.p2_plot_split.setSizes([h // n if k in keys else 0 for k in self.p2_plots])
        self._refresh_p2_axis_labels()

    def _refresh_p2_axis_labels(self):
        """Time-axis label only on the bottom-most visible panel; tick labels stay on all."""
        vis = [k for k in self.p2_plots if self.p2_panel_btns[k].isChecked()]
        for k, plot in self.p2_plots.items():
            axis = plot.getAxis('bottom')
            if vis and k == vis[-1]:
                plot.setLabel('bottom', 's', color=self.p2_theme['label'], **{'font-size': '9px'})
                axis.showLabel(True)
            else:
                axis.showLabel(False)

    def _on_p2_plot_clicked(self, ev, key):
        """Double-click a plot to focus it; double-click again to restore the previous layout."""
        if not ev.double():
            return
        vis = self._visible_p2_panels()
        if vis == {key} and self._p2_prev_panels:
            self._set_p2_panels(self._p2_prev_panels)
            self._p2_prev_panels = None
        elif vis != {key}:
            self._p2_prev_panels = vis
            self._set_p2_panels({key})
        ev.accept()

    # ── Y-axis behaviour ──────────────────────────────────────────────────────
    def _apply_p2_auto_y(self, on: bool):
        for key in ('angle', 'error'):
            vb = self.p2_plots[key].vb
            if on:
                vb.setAutoVisible(y=True)      # fit to the data inside the time window
                vb.enableAutoRange(axis=pg.ViewBox.YAxis)
            else:
                vb.disableAutoRange(axis=pg.ViewBox.YAxis)
        if on:  # confidence is a weight in [0, 1]; keep its scale fixed
            self.p2_conf.setYRange(-0.05, 1.05, padding=0)

    def _reset_p2_y(self):
        if self.btn_p2_auto_y.isChecked():
            self._apply_p2_auto_y(True)
        else:
            self.btn_p2_auto_y.setChecked(True)  # toggling applies it

    def _on_p2_manual_y(self, *_):
        """Dragging / wheel-zooming Y by hand turns Auto Y off so the view isn't fought."""
        if not self.btn_p2_auto_y.isChecked():
            return
        for key in ('angle', 'error'):
            if not self.p2_plots[key].vb.autoRangeEnabled()[1]:
                self.btn_p2_auto_y.blockSignals(True)
                self.btn_p2_auto_y.setChecked(False)
                self.btn_p2_auto_y.blockSignals(False)
                return

    def _apply_p2_lock_y(self, locked: bool):
        for plot in self.p2_plots.values():
            plot.setMouseEnabled(x=True, y=not locked)

    # ── View options ──────────────────────────────────────────────────────────
    def _apply_p2_grid(self, on: bool):
        for plot in self.p2_plots.values():
            plot.showGrid(x=on, y=on, alpha=0.12)

    def _apply_p2_legend(self, on: bool):
        for plot in self.p2_plots.values():
            plot.legend.setVisible(on)

    def _apply_p2_theme(self, name: str):
        """Restyle the three analysis panels (background, axes, legend, reference curves)."""
        th = self.p2_theme = P2_PLOT_THEMES[name]
        label_style = {'color': th['label'], 'font-size': '9px'}
        for key, title, ylabel in self.p2_specs:
            self.p2_widgets[key].setBackground(th['bg'])
            plot = self.p2_plots[key]
            plot.setTitle(title, color=th['title'], size='12px')
            plot.titleLabel.item.setFont(sans_font(12, QFont.Medium))
            for side in ('left', 'bottom'):
                axis = plot.getAxis(side)
                axis.setPen(pg.mkPen(th['axis']))
                axis.setTextPen(pg.mkPen(th['tick']))
            plot.setLabel('left', ylabel, **label_style)
            plot.legend.setBrush(pg.mkBrush(*th['legend_bg']))
            plot.legend.setPen(pg.mkPen(th['legend_line']))
            plot.legend.setLabelTextColor(th['legend_text'])
        self._apply_p2_grid(self.btn_p2_grid.isChecked())  # grid lines take the new axis colour
        self._refresh_p2_axis_labels()

        # Light references (truth / setpoint) would vanish on white, so they swap with the theme
        self.insp_curves['true'].setPen(pg.mkPen(th['true'], width=1.4))
        self.insp_curves['target'].setPen(pg.mkPen(th['target'], width=1.2, style=Qt.DashLine))
        self.p2_zero_line.setPen(pg.mkPen(th['zero'], style=Qt.DashLine, width=1))
        for ch in self.insp_crosshairs:
            ch.setPen(pg.mkPen(th['crosshair'], width=1, style=Qt.DashLine))
        self._rebuild_p2_legends()
        # No dark seams between panels in the saved image
        self.p2_plot_split.setStyleSheet(f"QSplitter::handle {{ background-color: {th['handle']}; }}")

    def _on_p2_tick_size(self, side: str, px: int, lbl: QLabel):
        """Resize the tick numbers on one axis of every panel; the axis grows to fit them."""
        lbl.setText(value_html(f"{px:2d}", "px"))
        for plot in self.p2_plots.values():
            plot.getAxis(side).setTickFont(mono_font(px))

    def _on_p2_save_png(self):
        base = os.path.splitext(os.path.basename(self.insp_filepath))[0] if self.insp_filepath else "plots"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot image", f"{base}_{self.insp_axis}.png", "PNG Image (*.png)")
        if path:
            if self.p2_plot_split.grab().save(path):
                self.status.showMessage(f"Saved plot image to {path}")
            else:
                self.status.showMessage(f"Failed to save image to {path}")

    # ── Signal visibility ─────────────────────────────────────────────────────
    def _on_p2_signal_toggled(self, signal: str, on: bool):
        self.p2_signal_on[signal] = on
        for key, _name, _panel, sig in P2_CURVES:
            if sig == signal:
                self.insp_curves[key].setVisible(on)
        self._rebuild_p2_legends()
        for key in ('angle', 'error'):  # hidden signals no longer count toward the Y fit
            self.p2_plots[key].vb.updateAutoRange()

    def _rebuild_p2_legends(self):
        """Legends list only the visible signals, in canonical order."""
        for panel, plot in self.p2_plots.items():
            plot.legend.clear()
            for key, name, pnl, sig in P2_CURVES:
                if pnl == panel and self.p2_signal_on[sig]:
                    plot.legend.addItem(self.insp_curves[key], name)
            self._mono_legend(plot.legend)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Leave and obj in getattr(self, '_p2_leave_watch', ()):
            for ch in self.insp_crosshairs:
                ch.setVisible(False)
        return super().eventFilter(obj, event)

    def _readout_row(self, parent_layout, series_key: str = None):
        """One mono readout line; optional series swatch acts as the chart legend key."""
        lbl = QLabel()
        lbl.setStyleSheet(self._mono_qss(size=11))
        parent_layout.addWidget(lbl)
        lbl.swatch = (f"<span style=\"color:{SERIES[series_key]};\">■</span>&nbsp;"
                      if series_key else "")
        return lbl

    def _kv_html(self, key: str, value: str, unit: str = "", key_w: int = 0) -> str:
        key_txt = key.ljust(key_w).replace(' ', '&nbsp;')
        return f"<span style=\"color:{PAL['text_3']};\">{key_txt}</span>&nbsp;{value_html(value, unit)}"

    def _build_p2_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        container = QWidget()
        container.setObjectName("p2SidebarContent")
        container.setStyleSheet(f"QWidget#p2SidebarContent {{ background: {PAL['surface_1']}; }}")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(7, 8, 7, 8)
        layout.setSpacing(8)

        # 1. Cursor inspector
        insp_grp = QGroupBox("Cursor Inspector")
        insp_grp.setStyleSheet(self._grp_style())
        il = QVBoxLayout(insp_grp)
        il.setSpacing(3)

        self.lbl_insp_time = QLabel()
        self.lbl_insp_time.setStyleSheet(self._mono_qss(size=13) + " font-weight: 500;")
        il.addWidget(self.lbl_insp_time)
        il.addWidget(self._hairline())

        self.lbl_insp_true = self._readout_row(il, 'true')
        self.lbl_insp_enc = self._readout_row(il, 'enc')
        self.lbl_insp_ahrs = self._readout_row(il, 'ahrs')
        self.lbl_insp_fused = self._readout_row(il, 'fused')
        il.addWidget(self._hairline())
        self.lbl_insp_conf = self._readout_row(il)
        self._clear_p2_inspector()

        layout.addWidget(insp_grp)

        # 2. Signals: show / hide a signal across every panel (filled box = shown)
        sig_grp = QGroupBox("Signals")
        sig_grp.setStyleSheet(self._grp_style())
        sg = QGridLayout(sig_grp)
        sg.setHorizontalSpacing(9)
        sg.setVerticalSpacing(3)
        self.p2_signal_cbs = {}
        for i, sig in enumerate(P2_SIGNALS):
            cb = QCheckBox(sig)
            cb.setChecked(True)
            cb.setStyleSheet(
                f"QCheckBox {{ color: {PAL['text_1']}; font-family: {MONO_CSS}; font-size: 11px;"
                f" spacing: 6px; background: transparent; }}"
                f"QCheckBox:!checked {{ color: {PAL['text_3']}; }}"
                f"QCheckBox::indicator {{ width: 10px; height: 10px; border: 1px solid {PAL['line_strong']};"
                f" background: {PAL['surface_2']}; }}"
                f"QCheckBox::indicator:checked {{ background: {SERIES[sig]}; border: 1px solid {SERIES[sig]}; }}"
            )
            cb.toggled.connect(lambda on, k=sig: self._on_p2_signal_toggled(k, on))
            sg.addWidget(cb, i // 3, i % 3)
            self.p2_signal_cbs[sig] = cb
        layout.addWidget(sig_grp)

        # 3. Axis number size: tick-label font for the time (x) and value (y) axes of every panel
        tick_grp = QGroupBox("Axis Numbers")
        tick_grp.setStyleSheet(self._grp_style())
        tl = QVBoxLayout(tick_grp)
        tl.setSpacing(5)
        self.p2_tick_sliders = {}
        for side, name in (('bottom', 'x_ticks'), ('left', 'y_ticks')):
            row = QHBoxLayout()
            lbl_name = QLabel(name)
            lbl_name.setStyleSheet(self._mono_qss(PAL['text_2']))
            lbl_name.setFixedWidth(64)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(7, 24)
            sl.setValue(9)
            sl.setToolTip(f"Size of the numbers on the {'time' if side == 'bottom' else 'value'} axis")
            lbl_val = QLabel(value_html(f"{9:2d}", "px"))
            lbl_val.setStyleSheet(self._mono_qss())
            lbl_val.setFixedWidth(44)
            lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            sl.valueChanged.connect(lambda v, s=side, l=lbl_val: self._on_p2_tick_size(s, v, l))
            row.addWidget(lbl_name)
            row.addWidget(sl)
            row.addWidget(lbl_val)
            tl.addLayout(row)
            self.p2_tick_sliders[side] = sl
        layout.addWidget(tick_grp)

        # 4. Time window presets
        pr_grp = QGroupBox("Time Window Presets")
        pr_grp.setStyleSheet(self._grp_style())
        pl = QGridLayout(pr_grp)
        pl.setSpacing(4)

        b_all = QPushButton("Fit All")
        b_all.setStyleSheet(self._action_btn_style())
        b_all.clicked.connect(self._p2_zoom_all)
        pl.addWidget(b_all, 0, 0, 1, 3)

        windows = [("0–5 s", 0, 5), ("5–10 s", 5, 10), ("10–15 s", 10, 15),
                   ("15–20 s", 15, 20), ("20–25 s", 20, 25), ("25–30 s", 25, 30)]
        for i, (name, s, e) in enumerate(windows):
            b = QPushButton(name)
            b.setStyleSheet(self._action_btn_style())
            b.clicked.connect(lambda _, start=s, end=e: self._p2_set_window(start, end))
            pl.addWidget(b, 1 + i // 3, i % 3)

        self.btn_p2_glitch = QPushButton("Jump to Peak Disturbance")
        self.btn_p2_glitch.setStyleSheet(self._action_btn_style())
        self.btn_p2_glitch.clicked.connect(self._p2_jump_to_glitch)
        pl.addWidget(self.btn_p2_glitch, 3, 0, 1, 3)
        layout.addWidget(pr_grp)

        # 5. Fusion quality
        stat_grp = QGroupBox("Fusion Quality")
        stat_grp.setStyleSheet(self._grp_style())
        sl = QVBoxLayout(stat_grp)
        sl.setSpacing(3)

        self.lbl_p2_rmse_enc = self._readout_row(sl, 'enc')
        self.lbl_p2_rmse_ahrs = self._readout_row(sl, 'ahrs')
        self.lbl_p2_rmse_fused = self._readout_row(sl, 'fused')
        sl.addWidget(self._hairline())
        self.lbl_p2_imp_enc = self._readout_row(sl)
        self.lbl_p2_imp_ahrs = self._readout_row(sl)
        for lbl, key in ((self.lbl_p2_rmse_enc, 'rmse_enc'), (self.lbl_p2_rmse_ahrs, 'rmse_ahrs'),
                         (self.lbl_p2_rmse_fused, 'rmse_fused'), (self.lbl_p2_imp_enc, 'gain_vs_enc'),
                         (self.lbl_p2_imp_ahrs, 'gain_vs_ahrs')):
            lbl.setText(lbl.swatch + self._kv_html(key, "--", key_w=12))

        layout.addWidget(stat_grp)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    @staticmethod
    def _hairline():
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {PAL['line']}; border: none;")
        return sep

    def _clear_p2_inspector(self):
        self.lbl_insp_time.setText(self._kv_html("t", "---.---", "s"))
        for lbl, key in ((self.lbl_insp_true, 'true'), (self.lbl_insp_enc, 'enc'),
                         (self.lbl_insp_ahrs, 'ahrs'), (self.lbl_insp_fused, 'fused')):
            lbl.setText(lbl.swatch + self._kv_html(key, "---.--", "deg", key_w=6))
        self.lbl_insp_conf.setText(self._kv_html("c_enc", "--.-", "%") + "&nbsp;&nbsp;" +
                                   self._kv_html("c_ahrs", "--.-", "%"))

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION TICK & ANIMATION
    # ─────────────────────────────────────────────────────────────────────────
    def _on_sim_tick(self):
        if not self.sim.running:
            return

        self.sim.step()
        t = self.sim.sim_time
        self.t_buf.append(t)

        for ax in AXES:
            a = self.sim.axes[ax]
            b = self.buf[ax]
            b['target'].append(a.target)
            b['true'].append(a.true_angle)
            b['enc'].append(a.enc_reading)
            b['ahrs'].append(a.ahrs_reading)
            b['fused'].append(a.fused_angle)
            b['c_enc'].append(a.c_enc)
            b['c_ahrs'].append(a.c_ahrs)
            b['err_enc'].append(a.enc_reading - a.true_angle)
            b['err_ahrs'].append(a.ahrs_reading - a.true_angle)
            b['err_fused'].append(a.fused_angle - a.true_angle)

        # Advance top ribbon trajectory
        self.pipeline_ribbon.advance_step()

        self._update_3d_antenna()
        self._update_live_plots()
        self._update_live_status()

        # Update Lower Console 2x2 Telemetry Quad
        self.diag_quad.update_telemetry(self.sim, self.live_view_ax)

    def _update_3d_antenna(self):
        fk = compute_fk_matrices(
            self.sim.axes['azimuth'].true_angle,
            self.sim.axes['elevation'].true_angle,
            self.sim.axes['polarization'].true_angle
        )
        for name in AXES:
            # Apply kinematic forward transformation
            pts = apply_transform(self.orig_pts[name], fk[name])
            # Apply vertical exploded projection offset if exploded mode active
            if self.is_exploded_view:
                pts = pts + self.layer_offsets[name]
            self.meshes[name].points[:] = pts

            # Joint pivot moves with its parent link; the label reads the true plant angle
            if hasattr(self, 'angle_labels'):
                lbl = self.angle_labels[name]
                anchor = apply_transform(LINKS_CONFIG[name]['pivot'][None, :].astype(np.float32), fk[name])[0]
                if self.is_exploded_view:
                    anchor = anchor + self.layer_offsets[name]
                lbl['anchor'] = anchor.astype(float)
                lbl['value'].SetInput(f"{self.sim.axes[name].true_angle:.1f}")

        self.plotter.update()

    def _update_live_plots(self):
        if not self.t_buf:
            return
        t = list(self.t_buf)
        b = self.buf[self.live_view_ax]

        self.crv_live['target'].setData(t, list(b['target']))
        self.crv_live['true'].setData(t, list(b['true']))
        self.crv_live['enc'].setData(t, list(b['enc']))
        self.crv_live['ahrs'].setData(t, list(b['ahrs']))
        self.crv_live['fused'].setData(t, list(b['fused']))

        self.crv_live['c_enc'].setData(t, list(b['c_enc']))
        self.crv_live['c_ahrs'].setData(t, list(b['c_ahrs']))

        self.crv_live['err_enc'].setData(t, list(b['err_enc']))
        self.crv_live['err_ahrs'].setData(t, list(b['err_ahrs']))
        self.crv_live['err_fused'].setData(t, list(b['err_fused']))

    def _update_live_status(self):
        for ax in AXES:
            a = self.sim.axes[ax]
            self.live_status_lbls[ax].setText(self._bus_row(
                ax, f"{a.true_angle:6.1f}", f"{a.fused_angle:6.1f}", f"{a.fused_angle - a.true_angle:+6.2f}",
                values_color=PAL['text_1']))

        va = self.live_view_ax
        a_v = self.sim.axes[va]
        self.live_conf_lbl.setText(
            self._kv_html("c_enc", f"{a_v.c_enc * 100:5.1f}", "%") + "&nbsp;&nbsp;" +
            self._kv_html("c_ahrs", f"{a_v.c_ahrs * 100:5.1f}", "%") +
            f"<span style=\"color:{PAL['text_3']};\">&nbsp;&nbsp;{va}</span>"
        )

        # Update Callout Overlay Floating Badge
        status_txt = "NOMINAL"
        if a_v.drift_rate > 0:
            status_txt = "DRIFT ACTIVE"
        elif abs(a_v.glitch_offset) > 0.1:
            status_txt = "GLITCH SPIKE"
        self.callout_overlay.update_badge(va, a_v.fused_angle, a_v.c_enc, a_v.c_ahrs, status_txt)

        if self.sim.recording:
            n = len(self.sim.log_data)
            dur = n * self.sim.dt
            self._set_rec_status("RECORDING", n, dur)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_start_sim(self):
        self.sim.running = True
        self.btn_start.setChecked(True)
        self.pipeline_ribbon.set_running(True)
        self.status.showMessage("Simulation running at 50 Hz. Real-time solver active.")

    def _on_stop_sim(self):
        self.sim.running = False
        self.btn_start.setChecked(False)
        self.pipeline_ribbon.set_running(False)
        self.status.showMessage("Simulation paused.")

    def _on_reset_sim(self):
        self.sim.running = False
        self.btn_start.setChecked(False)
        self.pipeline_ribbon.set_running(False)
        self.sim.reset()
        self.t_buf.clear()
        for ax in AXES:
            for q in self.buf[ax].values():
                q.clear()
        for crv in self.crv_live.values():
            crv.setData([], [])

        # Reset motion mode
        self.rb_mode_target.setChecked(True)
        self.sim.set_motion_mode('TARGET')

        # Reset target sliders and simulator targets
        target_defaults = {'azimuth': 90, 'elevation': 45, 'polarization': 45}
        for ax, dflt in target_defaults.items():
            if ax in self.tgt_sliders:
                self.tgt_sliders[ax].blockSignals(True)
                self.tgt_sliders[ax].setValue(dflt)
                self.tgt_sliders[ax].blockSignals(False)
            if ax in self.tgt_labels:
                self.tgt_labels[ax].setText(value_html(f"{dflt:3d}", "deg"))
            self.sim.set_target(ax, float(dflt))

        # Reset sensor noise sliders and clear faults
        self.sl_noise_enc.blockSignals(True)
        self.sl_noise_enc.setValue(5)
        self.sl_noise_enc.blockSignals(False)
        self.lbl_noise_enc.setText(value_html("0.50", "deg"))

        self.sl_noise_ahrs.blockSignals(True)
        self.sl_noise_ahrs.setValue(3)
        self.sl_noise_ahrs.blockSignals(False)
        self.lbl_noise_ahrs.setText(value_html("0.30", "deg"))

        self._on_clear_faults()

        # Stop recording if active
        if self.sim.recording:
            self.sim.stop_recording()
        self.btn_rec_toggle.setChecked(False)
        self.btn_rec_toggle.setText("● Record CSV")
        self._set_rec_status("IDLE", 0)

        # Reset 3D camera to initial isometric position
        self._set_3d_camera('iso')

        # Reset 3D antenna meshes to home kinematic pose
        self._update_3d_antenna()

        # Reset top ribbon
        self.pipeline_ribbon.tick_counter = 0
        self.pipeline_ribbon.set_active_step(0)

        # Reset diagnostic quad, HUD callout, and status
        self.diag_quad.update_telemetry(self.sim, self.live_view_ax)
        self._update_live_status()
        self.status.showMessage("Simulation reset to initial state.")

    def _on_motion_mode_changed(self):
        mode = 'TARGET' if self.rb_mode_target.isChecked() else 'BROWNIAN'
        self.sim.set_motion_mode(mode)
        self.status.showMessage(f"Motion mode set to: {mode}")

    def _on_target_slider(self, axis, value, label):
        label.setText(value_html(f"{value:3d}", "deg"))
        self.sim.set_target(axis, float(value))

    def _on_fault_axis_changed(self, text):
        # Combo items are the axis identifiers themselves (plus "all")
        self.fault_target_ax = text if (text in AXES or text == "all") else "azimuth"

    def _on_enc_noise_slider(self, val):
        sigma = val / 10.0
        self.lbl_noise_enc.setText(value_html(f"{sigma:.2f}", "deg"))
        if self.fault_target_ax == "all":
            self.sim.set_global_noise(sigma_enc=sigma)
        else:
            self.sim.set_noise(self.fault_target_ax, sigma_enc=sigma)

    def _on_ahrs_noise_slider(self, val):
        sigma = val / 10.0
        self.lbl_noise_ahrs.setText(value_html(f"{sigma:.2f}", "deg"))
        if self.fault_target_ax == "all":
            self.sim.set_global_noise(sigma_ahrs=sigma)
        else:
            self.sim.set_noise(self.fault_target_ax, sigma_ahrs=sigma)

    def _on_inject_drift(self):
        targets = AXES if self.fault_target_ax == "all" else [self.fault_target_ax]
        for ax in targets:
            self.sim.inject_fault(ax, 'drift')
        self.status.showMessage(f"Injected continuous drift ramp into {self.fault_target_ax} encoder.")

    def _on_inject_glitch(self):
        targets = AXES if self.fault_target_ax == "all" else [self.fault_target_ax]
        for ax in targets:
            self.sim.inject_fault(ax, 'glitch')
        self.status.showMessage(f"Injected +12° glitch step into {self.fault_target_ax} encoder.")

    def _on_clear_faults(self):
        targets = AXES if self.fault_target_ax == "all" else [self.fault_target_ax]
        for ax in targets:
            self.sim.inject_fault(ax, 'clear')
        self.status.showMessage(f"Cleared all sensor faults on {self.fault_target_ax}.")

    def _set_live_axis(self, axis):
        self.live_view_ax = axis
        for a, b in self.live_ax_btns.items():
            b.setChecked(a == axis)
        self._refresh_mesh_materials()
        self._update_live_plots()
        self.diag_quad.update_telemetry(self.sim, self.live_view_ax)

    def _set_3d_camera(self, view_name):
        if view_name == 'iso':
            self.plotter.camera_position = 'iso'
        elif view_name == 'top':
            self.plotter.view_xy()
        elif view_name == 'front':
            self.plotter.view_xz()
        elif view_name == 'side':
            self.plotter.view_yz()
        self.plotter.reset_camera()
        self.plotter.camera.zoom(1.15)

    def _on_toggle_recording(self, checked):
        if checked:
            self.btn_rec_toggle.setText("■ Stop Recording")
            self.sim.clear_recorded_data()
            self.sim.start_recording()
            self.status.showMessage("Data logging started.")
        else:
            self.btn_rec_toggle.setText("● Record CSV")
            csv_path = self.sim.stop_recording()
            n = len(self.sim.log_data)
            self._set_rec_status("SAVED", n, path=os.path.basename(csv_path))
            self.status.showMessage(f"Recorded data exported to {csv_path}")

    def _on_export_and_inspect(self):
        if self.sim.recording:
            self.btn_rec_toggle.setChecked(False)
            csv_path = self.sim.stop_recording()
        else:
            csv_path = self.sim.export_to_csv()

        if csv_path and os.path.exists(csv_path):
            self.tabs.setCurrentIndex(1)
            self._load_p2_csv(csv_path)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2: CSV INSPECTION HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_inspect_last_recording(self):
        if self.sim.current_csv_path and os.path.exists(self.sim.current_csv_path):
            self._load_p2_csv(self.sim.current_csv_path)
        else:
            latest = find_latest_log(SIM_LOG_PATTERN, TIMESERIES_PATTERN)
            if latest:
                self._load_p2_csv(latest)
            else:
                self.status.showMessage("No recorded CSV found. Run a simulation and record first.")

    def _on_p2_open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Simulation CSV", LOGS_DIR, "CSV Files (*.csv);;All Files (*)")
        if path:
            self._load_p2_csv(path)

    def _on_p2_reload(self):
        if self.insp_filepath and os.path.exists(self.insp_filepath):
            self._load_p2_csv(self.insp_filepath)

    def _on_p2_axis_changed(self, text):
        self.insp_axis = text.lower()
        if self.insp_df is not None:
            self._update_p2_curves()

    def _load_p2_csv(self, filepath):
        try:
            df = pd.read_csv(filepath)
            self.insp_df = df
            self.insp_filepath = os.path.abspath(filepath)
            fname = os.path.basename(filepath)
            n_rows = len(df)
            dur = df['time_s'].iloc[-1] - df['time_s'].iloc[0] if n_rows > 1 else 0
            self.lbl_p2_file_info.setText(f"{fname} | {n_rows:,} samples ({dur:.2f}s)")

            self._update_p2_curves()
            self._p2_zoom_all()
            self.status.showMessage(f"Loaded {n_rows:,} samples from {fname}")
        except Exception as e:
            self.status.showMessage(f"Failed to load CSV: {str(e)}")

    def _update_p2_curves(self):
        if self.insp_df is None:
            return
        df = self.insp_df
        t = df['time_s'].values
        ax = self.insp_axis

        if f'{ax}_true' in df.columns:
            tgt = df[f'{ax}_target'].values if f'{ax}_target' in df.columns else np.zeros_like(t)
            th_true = df[f'{ax}_true'].values
            th_enc = df[f'{ax}_enc'].values
            th_ahrs = df[f'{ax}_ahrs'].values
            th_fused = df[f'{ax}_fused'].values
            c_enc = df[f'{ax}_c_enc'].values
            c_ahrs = df[f'{ax}_c_ahrs'].values
            e_enc = df[f'{ax}_err_enc'].values
            e_ahrs = df[f'{ax}_err_ahrs'].values
            e_fused = df[f'{ax}_err_fused'].values
        else:
            tgt = np.zeros_like(t)
            th_true = df['theta_true_deg'].values
            th_enc = df['theta_enc_deg'].values
            th_ahrs = df['theta_ahrs_deg'].values
            th_fused = df['theta_fused_deg'].values
            c_enc = df['conf_enc'].values if 'conf_enc' in df.columns else np.full_like(t, 0.5)
            c_ahrs = df['conf_ahrs'].values if 'conf_ahrs' in df.columns else np.full_like(t, 0.5)
            e_enc = df['error_enc_deg'].values if 'error_enc_deg' in df.columns else th_enc - th_true
            e_ahrs = df['error_ahrs_deg'].values if 'error_ahrs_deg' in df.columns else th_ahrs - th_true
            e_fused = df['error_fused_deg'].values if 'error_fused_deg' in df.columns else th_fused - th_true

        self.insp_curves['target'].setData(t, tgt)
        self.insp_curves['true'].setData(t, th_true)
        self.insp_curves['enc'].setData(t, th_enc)
        self.insp_curves['ahrs'].setData(t, th_ahrs)
        self.insp_curves['fused'].setData(t, th_fused)

        self.insp_curves['conf_enc'].setData(t, c_enc)
        self.insp_curves['conf_ahrs'].setData(t, c_ahrs)

        self.insp_curves['err_enc'].setData(t, e_enc)
        self.insp_curves['err_ahrs'].setData(t, e_ahrs)
        self.insp_curves['err_fused'].setData(t, e_fused)

        self.insp_overview_crv.setData(t, th_fused)
        self.p2_overview.getPlotItem().setXRange(t[0], t[-1], padding=0)

        # Update Quality Assessment Statistics
        rmse_enc = np.sqrt(np.mean(e_enc ** 2))
        rmse_ahrs = np.sqrt(np.mean(e_ahrs ** 2))
        rmse_fused = np.sqrt(np.mean(e_fused ** 2))

        for lbl, key, val in ((self.lbl_p2_rmse_enc, 'rmse_enc', rmse_enc),
                              (self.lbl_p2_rmse_ahrs, 'rmse_ahrs', rmse_ahrs),
                              (self.lbl_p2_rmse_fused, 'rmse_fused', rmse_fused)):
            lbl.setText(lbl.swatch + self._kv_html(key, f"{val:7.3f}", "deg", key_w=12))

        # Error reduction of the fused estimate relative to each raw sensor
        for lbl, key, ref in ((self.lbl_p2_imp_enc, 'gain_vs_enc', rmse_enc),
                              (self.lbl_p2_imp_ahrs, 'gain_vs_ahrs', rmse_ahrs)):
            if ref > 0:
                lbl.setText(self._kv_html(key, f"{(1 - rmse_fused / ref) * 100:+7.1f}", "%", key_w=12))

    def _on_p2_region_changed(self):
        if self.insp_updating_region:
            return
        self.insp_updating_region = True
        try:
            mn, mx = self.p2_region.getRegion()
            self.p2_ang.setXRange(mn, mx, padding=0)
        finally:
            self.insp_updating_region = False

    def _on_p2_plot_range_changed(self):
        if self.insp_updating_region:
            return
        self.insp_updating_region = True
        try:
            mn, mx = self.p2_ang.viewRange()[0]
            self.p2_region.setRegion([mn, mx])
        finally:
            self.insp_updating_region = False

    def _on_p2_mouse_moved(self, pos, plot):
        if self.insp_df is None:
            return

        mouse_pt = None
        if plot.sceneBoundingRect().contains(pos):
            mouse_pt = plot.vb.mapSceneToView(pos)

        if mouse_pt is None:
            for ch in self.insp_crosshairs:
                ch.setVisible(False)
            return

        t_arr = self.insp_df['time_s'].values
        x_val = mouse_pt.x()
        if x_val < t_arr[0] or x_val > t_arr[-1]:
            for ch in self.insp_crosshairs:
                ch.setVisible(False)
            return

        idx = int(np.searchsorted(t_arr, x_val))
        if idx >= len(t_arr):
            idx = len(t_arr) - 1
        elif idx > 0 and (x_val - t_arr[idx - 1]) < (t_arr[idx] - x_val):
            idx = idx - 1

        snapped_t = t_arr[idx]
        for ch in self.insp_crosshairs:
            ch.setPos(snapped_t)
            ch.setVisible(True)

        row = self.insp_df.iloc[idx]
        ax = self.insp_axis
        if f'{ax}_true' in self.insp_df.columns:
            th_true = row[f'{ax}_true']
            th_enc = row[f'{ax}_enc']
            th_ahrs = row[f'{ax}_ahrs']
            th_fused = row[f'{ax}_fused']
            c_enc = row[f'{ax}_c_enc']
            c_ahrs = row[f'{ax}_c_ahrs']
            e_enc = row[f'{ax}_err_enc']
            e_ahrs = row[f'{ax}_err_ahrs']
            e_fused = row[f'{ax}_err_fused']
        else:
            th_true = row['theta_true_deg']
            th_enc = row['theta_enc_deg']
            th_ahrs = row['theta_ahrs_deg']
            th_fused = row['theta_fused_deg']
            c_enc = row.get('conf_enc', 0.5)
            c_ahrs = row.get('conf_ahrs', 0.5)
            e_enc = row.get('error_enc_deg', th_enc - th_true)
            e_ahrs = row.get('error_ahrs_deg', th_ahrs - th_true)
            e_fused = row.get('error_fused_deg', th_fused - th_true)

        self.lbl_insp_time.setText(self._kv_html("t", f"{snapped_t:7.3f}", "s") +
                                   f"<span style=\"color:{PAL['text_3']}; font-size:10px;\">&nbsp;&nbsp;#{idx}</span>")
        self.lbl_insp_true.setText(self.lbl_insp_true.swatch +
                                   self._kv_html("true", f"{th_true:+7.2f}", "deg", key_w=6))
        for lbl, key, val, err in ((self.lbl_insp_enc, 'enc', th_enc, e_enc),
                                   (self.lbl_insp_ahrs, 'ahrs', th_ahrs, e_ahrs),
                                   (self.lbl_insp_fused, 'fused', th_fused, e_fused)):
            lbl.setText(lbl.swatch + self._kv_html(key, f"{val:+7.2f}", "deg", key_w=6) +
                        f"&nbsp;&nbsp;<span style=\"color:{PAL['text_3']};\">Δ</span>&nbsp;" +
                        value_html(f"{err:+6.2f}"))
        self.lbl_insp_conf.setText(self._kv_html("c_enc", f"{c_enc * 100:5.1f}", "%") + "&nbsp;&nbsp;" +
                                   self._kv_html("c_ahrs", f"{c_ahrs * 100:5.1f}", "%"))

    def _p2_set_window(self, start_s, end_s):
        if self.insp_df is None:
            return
        t = self.insp_df['time_s'].values
        s = max(t[0], start_s)
        e = min(t[-1], end_s)
        self.p2_region.setRegion([s, e])
        self.p2_ang.setXRange(s, e, padding=0)

    def _p2_zoom_all(self):
        if self.insp_df is None:
            return
        t = self.insp_df['time_s'].values
        self._p2_set_window(t[0], t[-1])

    def _p2_jump_to_glitch(self):
        if self.insp_df is None:
            return
        ax = self.insp_axis
        col = f'{ax}_err_enc' if f'{ax}_err_enc' in self.insp_df.columns else 'error_enc_deg'
        if col in self.insp_df.columns:
            peak_idx = int(np.argmax(np.abs(self.insp_df[col].values)))
            peak_t = self.insp_df['time_s'].iloc[peak_idx]
            self._p2_set_window(peak_t - 2.5, peak_t + 2.5)
            self.status.showMessage(f"Centered window around peak disturbance at t = {peak_t:.2f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # STYLESHEET HELPERS (RAZOR-THIN 1PX BENTO DOCKS)
    # ─────────────────────────────────────────────────────────────────────────
    def _grp_style(self):
        """Panel group: surface-1 on a hairline, Title Case sans heading in text-2."""
        return f"""
            QGroupBox {{
                color: {PAL['text_2']};
                font-size: 12px;
                font-weight: 500;
                border: 1px solid {PAL['line']};
                border-radius: 2px;
                margin-top: 9px;
                padding-top: 9px;
                background-color: {PAL['surface_1']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                left: 7px;
                background-color: {PAL['surface_1']};
            }}
            QLabel {{
                background: transparent;
            }}
        """

    def _action_btn_style(self, state_ring: bool = False):
        """
        Grayscale button (chrome never takes hue). With state_ring=True the checked
        state is a live-green hairline ring — used where checked means "running".
        Horizontal padding follows the −7% rule (8px → 7px).
        """
        checked_border = PAL['state_live'] if state_ring else PAL['text_3']
        return f"""
            QPushButton {{
                background-color: {PAL['surface_2']};
                color: {PAL['text_1']};
                border: 1px solid {PAL['line_strong']};
                border-radius: 2px;
                padding: 4px 7px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {PAL['surface_3']};
            }}
            QPushButton:pressed {{
                background-color: {PAL['surface_3']};
                border-color: {PAL['text_3']};
            }}
            QPushButton:checked {{
                background-color: {PAL['surface_3']};
                border: 1px solid {checked_border};
            }}
        """

    def _segment_style(self, mono: bool = False):
        """Segmented chip: flat until active; active chip = surface-3 + primary text."""
        family = f"font-family: {MONO_CSS}; font-size: 11px;" if mono else "font-size: 12px;"
        return f"""
            QPushButton {{
                background-color: {PAL['surface_2']};
                color: {PAL['text_2']};
                border: 1px solid {PAL['line_strong']};
                border-radius: 0px;
                padding: 3px 7px;
                {family}
            }}
            QPushButton:hover {{
                color: {PAL['text_1']};
            }}
            QPushButton:checked {{
                background-color: {PAL['surface_3']};
                color: {PAL['text_1']};
            }}
        """

    @staticmethod
    def _mono_qss(color: str = PAL['text_1'], size: int = 11) -> str:
        return f"color: {color}; font-family: {MONO_CSS}; font-size: {size}px; background: transparent;"

    def _bus_row(self, key: str, true_v: str, fused_v: str, err_v: str, values_color: str = None) -> str:
        """Fixed-width kinematic bus line: identifier + three tabular columns."""
        def col(txt, w):
            return txt.rjust(w).replace(' ', '&nbsp;')
        vals = f"{col(true_v, 7)}{col(fused_v, 7)}{col(err_v, 7)}"
        if values_color:
            vals = f"<span style=\"color:{values_color};\">{vals}</span>"
        return f"<span style=\"color:{PAL['text_2']};\">{key.ljust(12).replace(' ', '&nbsp;')}</span>{vals}"

    def _set_rec_status(self, state: str, n: int, dur: float = None, path: str = None):
        """Recorder state word (mono, uppercase) + sample count; hue only while recording."""
        color = PAL['state_live'] if state == "RECORDING" else PAL['text_2']
        glyph = "●" if state == "RECORDING" else "○"
        txt = (f"<span style=\"color:{color};\">{glyph} {state}</span>&nbsp;&nbsp;"
               f"{value_html(f'{n:,}', 'samples')}")
        if dur is not None:
            txt += "&nbsp;&nbsp;" + value_html(f"{dur:.1f}", "s")
        if path:
            txt += f"<br><span style=\"color:{PAL['text_3']};\">{path}</span>"
        self.lbl_rec_status.setText(txt)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    load_bundled_fonts()
    if sys.platform == 'win32':
        # Own AppUserModelID so the taskbar shows our icon instead of python.exe's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('cotp.antenna_fusion.simulator')
    app.setWindowIcon(load_app_icon())
    win = AntennaFusionApp()
    win.showMaximized()  # Fill the usable screen area (1366x768 minus taskbar)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
