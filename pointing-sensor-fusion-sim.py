"""
COTP Antenna Pointing Simulation
─────────────────────────────────
Architecture:
  Outer loop  (Position PID)  : target_angle → velocity_demand
  Inner loop  (Velocity PID)  : velocity_demand vs actual → PWM
  Sensor fusion               : Kalman filter — AS5600 encoder (θ) + AHRS (ω)
  Stall detection             : monitors velocity under load, applies PWM boost

Dependencies:
  pip install pyvista pyvistaqt pyqtgraph scipy PyQt5
"""

import os, math, random
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from scipy.spatial.transform import Rotation
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel, QPushButton, QGroupBox, QGridLayout,
    QSplitter, QTabWidget, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
STL_FILES = {
    'azimuth':      'CADs-Azimuth-Body.stl',
    'elevation':    'CADs-Elevation-Body.stl',
    'polarization': 'CADs-Polarization-Body.stl',
}
LINKS_CONFIG = {
    'azimuth':     {'pivot': np.array([0., 0.,   0.]), 'axis': np.array([0.,0.,1.]), 'color': 'silver'},
    'elevation':   {'pivot': np.array([0., 0.,  95.]), 'axis': np.array([1.,0.,0.]), 'color': 'skyblue'},
    'polarization':{'pivot': np.array([0.,135.,170.]), 'axis': np.array([0.,0.,1.]), 'color': 'orange'},
}
AXES       = ['azimuth', 'elevation', 'polarization']
AX_COLOR   = {'azimuth': '#378add', 'elevation': '#45a849', 'polarization': '#d97706'}
AX_MAX     = {'azimuth': 360, 'elevation': 90, 'polarization': 180}

DT          = 0.05      # simulation step  [s]
MAX_PTS     = 320       # rolling buffer length
SIM_MS      = int(DT * 1000)

# Motor / mechanical constants
BASE_STALL  = 28.0      # stall PWM at 0° elevation
K_MOTOR     = 0.013     # (°/s) per (PWM – stall) per load_factor
K_FRICTION  = 0.22      # velocity-proportional friction
STATIC_FRIC = 0.04      # static friction (opposes motion below threshold)


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def transform_matrix(pivot: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    R = Rotation.from_rotvec(np.radians(deg) * axis).as_matrix()
    M = np.eye(4); M[:3,:3] = R; M[:3, 3] = pivot - R @ pivot
    return M

def apply_transform(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ h.T).T[:, :3].astype(np.float32)

def load_factor(name: str, el_deg: float) -> float:
    """Gravity / mechanical load scaling. High elevation → lower required PWM."""
    r = math.radians(el_deg)
    if name == 'elevation':   return max(0.30, math.cos(r))
    if name == 'azimuth':     return max(0.75, 1 - 0.15 * math.sin(r))
    return max(0.65, 1 - 0.22 * math.sin(r))


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN FILTER  (2-state: [θ, ω])
# ══════════════════════════════════════════════════════════════════════════════
class KalmanFilter:
    """
    Fuses two sensors:
      AS5600 rotary encoder  →  position measurement  (H = [1, 0])
      AHRS gyroscope         →  rate    measurement   (H = [0, 1])

    State vector x = [angle θ,  angular rate ω]
    Process:       x_k = A·x_{k-1}   (constant-rate model)
    """
    def __init__(self, dt: float, q_ang: float = 3e-4, q_rate: float = 2e-3):
        self.A  = np.array([[1., dt], [0., 1.]])
        self.Q  = np.diag([q_ang, q_rate])
        self.x  = np.zeros(2)
        self.P  = np.eye(2) * 2.
        self.R_enc  = 0.25   # encoder variance  (σ²)
        self.R_ahrs = 0.04   # AHRS rate variance (σ²)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def _update(self, z: float, h_row: list, R: float):
        H   = np.array([h_row])
        inn = z - float(H @ self.x)
        S   = float(H @ self.P @ H.T) + R
        K   = (self.P @ H.T) / S
        self.x += K.ravel() * inn
        self.P  = (np.eye(2) - np.outer(K, H)) @ self.P

    def update_encoder(self, z: float): self._update(z, [1., 0.], self.R_enc)
    def update_ahrs(self,   z: float): self._update(z, [0., 1.], self.R_ahrs)

    def reset(self, angle: float = 0.):
        self.x = np.array([angle, 0.]); self.P = np.eye(2) * 2.

    @property
    def angle(self) -> float: return float(self.x[0])
    @property
    def rate(self)  -> float: return float(self.x[1])


# ══════════════════════════════════════════════════════════════════════════════
#  PID CONTROLLER  (with anti-windup)
# ══════════════════════════════════════════════════════════════════════════════
class PID:
    def __init__(self, kp: float, ki: float, kd: float, lo: float, hi: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.lo, self.hi = lo, hi
        self.integ = self.prev_err = 0.

    def compute(self, err: float, dt: float) -> float:
        self.integ += err * dt
        deriv  = (err - self.prev_err) / max(dt, 1e-9)
        self.prev_err = err
        raw    = self.kp * err + self.ki * self.integ + self.kd * deriv
        # Back-calculation anti-windup
        if   raw > self.hi: self.integ -= err * dt
        elif raw < self.lo: self.integ -= err * dt
        return float(np.clip(raw, self.lo, self.hi))

    def reset(self): self.integ = self.prev_err = 0.


# ══════════════════════════════════════════════════════════════════════════════
#  STALL DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
class StallDetector:
    """
    Watches a rolling window of velocity.  If velocity stays near zero while
    PWM is above the stall threshold, a stall is declared and a one-shot boost
    is injected, then decayed exponentially.
    """
    def __init__(self, window: int = 14, vel_thr: float = 0.25,
                 pwm_thr: float = 20., boost_amt: float = 24.):
        self.window = window; self.vel_thr = vel_thr
        self.pwm_thr = pwm_thr; self.boost_amt = boost_amt
        self.vel_hist = deque(maxlen=window)
        self.boost = 0.; self.detected = False

    def update(self, vel: float, pwm: float) -> tuple[bool, float]:
        self.vel_hist.append(abs(vel))
        self.boost *= 0.87          # exponential decay
        if len(self.vel_hist) == self.window:
            avg_vel = sum(self.vel_hist) / self.window
            self.detected = (avg_vel < self.vel_thr and pwm > self.pwm_thr)
            if self.detected:
                self.boost = self.boost_amt
        return self.detected, self.boost

    def reset(self):
        self.vel_hist.clear(); self.boost = 0.; self.detected = False


# ══════════════════════════════════════════════════════════════════════════════
#  AXIS STATE  (per-axis physics + control)
# ══════════════════════════════════════════════════════════════════════════════
class AxisState:
    """
    Encapsulates one rotational axis.

    Control architecture  (matches the block diagram):
      ┌────────┐   pos_err  ┌──────────────┐  vel_demand  ┌──────────────┐  PWM  ┌───────┐
      │ INPUT  │──────────►│ Outer PID    │─────────────►│ Inner PID    │──────►│ PLANT │
      │(target)│           │ (pos → vel)  │              │ (vel → PWM)  │       │       │
      └────────┘           └──────────────┘              └──────────────┘       └───┬───┘
                                  ▲                             ▲                    │
                                  │ fused angle                 │ fused velocity     │ true motion
                                  └─────────────────────────────┴──── KALMAN ◄───────┘
                                                                        ▲    ▲
                                                                  Encoder   AHRS
    """
    # PID gains — outer: position→velocity [deg/s out]
    OUTER_KP, OUTER_KI, OUTER_KD = 2.8, 0.04, 1.1
    OUTER_LO, OUTER_HI           = -18., 18.

    # PID gains — inner: velocity error→PWM
    INNER_KP, INNER_KI, INNER_KD = 1.05, 1.9, 0.09
    INNER_LO, INNER_HI           = 0., 100.

    def __init__(self, name: str):
        self.name = name
        self.angle = self.vel = 0.
        self.enc_meas = self.ahrs_meas = self.ahrs_bias = 0.
        self.kf    = KalmanFilter(DT)
        self.outer = PID(self.OUTER_KP, self.OUTER_KI, self.OUTER_KD, self.OUTER_LO, self.OUTER_HI)
        self.inner = PID(self.INNER_KP, self.INNER_KI, self.INNER_KD, self.INNER_LO, self.INNER_HI)
        self.stall = StallDetector()
        self.vel_demand = self.pwm = 0.
        self.phase   = 'IDLE'   # IDLE | MOVING | DONE
        self.stalled = False
        self.roc_err = 0.       # rate-of-change error (inner loop input, logged)

    def reset(self):
        self.angle = self.vel = self.vel_demand = self.pwm = self.roc_err = 0.
        self.enc_meas = self.ahrs_meas = self.ahrs_bias = 0.
        self.phase = 'IDLE'; self.stalled = False
        self.kf.reset(); self.outer.reset(); self.inner.reset(); self.stall.reset()

    def step(self, target: float, el_deg: float,
             spd: float, n_enc: float, n_ahrs: float) -> tuple[float, float]:
        """
        One DT step.  Returns (fused_angle, fused_rate).
        spd    — simulation speed multiplier (scales motor force)
        n_enc  — encoder noise σ  [deg]
        n_ahrs — AHRS rate noise σ [deg/s]
        """
        lf        = load_factor(self.name, el_deg)
        stall_pwm = BASE_STALL * lf

        # ── Sensor simulation ────────────────────────────────────────────────
        self.ahrs_bias  += random.gauss(0., 4e-4)          # slow gyro drift
        self.enc_meas    = self.angle + random.gauss(0., n_enc)
        self.ahrs_meas   = self.vel + self.ahrs_bias + random.gauss(0., n_ahrs)

        # ── Kalman filter ────────────────────────────────────────────────────
        self.kf.R_enc  = max(0.01, n_enc  ** 2)
        self.kf.R_ahrs = max(1e-3, n_ahrs ** 2)
        self.kf.predict()
        self.kf.update_encoder(self.enc_meas)
        self.kf.update_ahrs(self.ahrs_meas)
        fa, fv = self.kf.angle, self.kf.rate            # fused estimates

        # ── Phase management ─────────────────────────────────────────────────
        pos_err = target - fa
        if self.phase == 'MOVING' and abs(pos_err) < 0.12:
            self.phase = 'DONE'
            self.angle = target; self.vel = 0.
            self.vel_demand = self.pwm = self.roc_err = 0.
            self.inner.reset(); self.outer.reset()
            return fa, fv
        if self.phase in ('IDLE', 'DONE') and abs(pos_err) > 0.12:
            self.phase = 'MOVING'
        if self.phase != 'MOVING':
            return fa, fv

        # ── OUTER LOOP  (Position → Velocity demand) ─────────────────────────
        # As pos_err shrinks → lower velocity demand → natural deceleration
        self.vel_demand = self.outer.compute(pos_err, DT)

        # ── INNER LOOP  (Velocity error → PWM) ───────────────────────────────
        # "rate-of-change error": demanded speed minus what sensors report
        self.roc_err = abs(self.vel_demand) - abs(fv)
        raw_pwm      = self.inner.compute(self.roc_err, DT)

        # ── Stall detection + boost ───────────────────────────────────────────
        stalled, boost   = self.stall.update(fv, raw_pwm)
        self.stalled     = stalled
        eff_pwm          = min(100., raw_pwm + boost)
        self.pwm         = eff_pwm

        # ── Plant model (motor + friction) ────────────────────────────────────
        direction = math.copysign(1., self.vel_demand) if abs(self.vel_demand) > 0.01 else 0.
        if eff_pwm > stall_pwm:
            motor_force = K_MOTOR * (eff_pwm - stall_pwm) * lf * spd
        else:
            motor_force = 0.
        # Net acceleration: motor − velocity friction − static friction
        friction = K_FRICTION * self.vel + STATIC_FRIC * math.copysign(1., self.vel) if abs(self.vel) > 0.01 else 0.
        accel    = direction * motor_force - friction
        self.vel   += accel  * DT
        self.angle += self.vel * DT

        return fa, fv


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("COTP — Cascade PID + Kalman Filter Simulation")
        self.resize(1420, 900)
        self._init_data()
        self._build_ui()        # splitter, control panel, chart tabs
        self._setup_3d()        # PyVista viewport inside left pane
        self._setup_charts()    # PyQtGraph inside chart tabs
        self._update_status()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(SIM_MS)

    # ─────────────────────────────────────────────────────────────────────────
    def _init_data(self):
        self.running   = False
        self.sim_time  = 0.
        self.targets   = {'azimuth': 88., 'elevation': 44., 'polarization': 90.}
        self.n_enc     = 0.50   # encoder σ [deg]
        self.n_ahrs    = 0.20   # AHRS rate σ [deg/s]
        self.spd       = 2.0    # simulation speed multiplier
        self.view_ax   = 'azimuth'
        self.axes      = {ax: AxisState(ax) for ax in AXES}
        self.t_buf     = deque(maxlen=MAX_PTS)
        KEYS = ('pwm', 'true_ang', 'fused_ang', 'enc_meas',
                'vel_demand', 'true_vel', 'fused_vel', 'roc_err')
        self.buf = {ax: {k: deque(maxlen=MAX_PTS) for k in KEYS} for ax in AXES}
        self.status_lbls: dict[str, QLabel] = {}
        self.ax_btns:     dict[str, QPushButton] = {}

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        # ── Main horizontal splitter ──────────────────────────────────────
        self.main_split = QSplitter(Qt.Horizontal)
        self.main_split.setHandleWidth(6)
        self.main_split.setStyleSheet(
            "QSplitter::handle { background: #2d3748; }"
            "QSplitter::handle:hover { background: #4a5568; }"
        )
        root.addWidget(self.main_split)

        # Left: 3D viewport host
        self.vtk_host = QWidget()
        vtk_layout    = QVBoxLayout(self.vtk_host)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        self.main_split.addWidget(self.vtk_host)

        # Right: controls + charts
        right = QWidget()
        right.setMinimumWidth(360)
        rpl   = QVBoxLayout(right)
        rpl.setContentsMargins(6, 4, 4, 4)
        rpl.setSpacing(6)
        self.main_split.addWidget(right)
        self.main_split.setSizes([800, 560])
        self.main_split.setCollapsible(0, False)
        self.main_split.setCollapsible(1, False)

        rpl.addWidget(self._build_controls())
        self.chart_tabs = QTabWidget()
        rpl.addWidget(self.chart_tabs, stretch=1)

    # ─────────────────────────────────────────────────────────────────────────
    def _build_controls(self) -> QGroupBox:
        box = QGroupBox("Control Panel")
        g   = QGridLayout(box)
        g.setSpacing(3)
        g.setContentsMargins(8, 6, 8, 6)

        mono = QFont("Courier New", 9)

        def hdr(text, col, span=1):
            l = QLabel(text)
            l.setStyleSheet("color:#7c8aaa; font-size:9px; font-weight:bold;")
            g.addWidget(l, 0, col, 1, span)

        hdr("Axis / Parameter", 0); hdr("Slider", 1); hdr("Value", 2); hdr("Live Status", 3)

        row = 1
        sep = lambda: self._hsep(g, row)

        # ── Target angles ────────────────────────────────────────────────────
        for ax, mx, dflt in [('azimuth',360,88), ('elevation',90,44), ('polarization',180,90)]:
            nm = QLabel(f"  {ax[:3].upper()}  target")
            nm.setStyleSheet(f"color:{AX_COLOR[ax]}; font-size:10px;")
            sl = QSlider(Qt.Horizontal); sl.setRange(0, mx); sl.setValue(dflt)
            lv = QLabel(f"{dflt}°"); lv.setFixedWidth(44); lv.setFont(mono)
            st = QLabel("—");       st.setFont(mono)
            st.setStyleSheet(f"color:{AX_COLOR[ax]}; font-size:9px;")
            sl.valueChanged.connect(lambda v, a=ax, l=lv: self._on_target(a, v, l))
            g.addWidget(nm, row, 0); g.addWidget(sl, row, 1)
            g.addWidget(lv, row, 2); g.addWidget(st, row, 3)
            self.status_lbls[ax] = st
            row += 1

        # separator
        row = self._hsep(g, row)

        # ── Parameters ───────────────────────────────────────────────────────
        params = [
            ("Enc noise σ (°)",   0, 30,  5, 10., 'n_enc'),
            ("AHRS noise σ (°/s)",0, 20,  2, 10., 'n_ahrs'),
            ("Sim speed ×",       5, 50, 20, 10., 'spd'),
        ]
        for label, mn, mx, dflt, div, attr in params:
            nm = QLabel(f"  {label}")
            nm.setStyleSheet("font-size:10px;")
            sl = QSlider(Qt.Horizontal); sl.setRange(mn, mx); sl.setValue(dflt)
            v0 = dflt / div
            lv = QLabel(f"{v0:.1f}"); lv.setFixedWidth(44); lv.setFont(mono)
            sl.valueChanged.connect(lambda v, a=attr, l=lv, d=div: self._on_param(a, v/d, l))
            g.addWidget(nm, row, 0); g.addWidget(sl, row, 1); g.addWidget(lv, row, 2)
            row += 1

        # separator
        row = self._hsep(g, row)

        # ── Buttons ──────────────────────────────────────────────────────────
        bw = QWidget(); bl = QHBoxLayout(bw); bl.setContentsMargins(0, 0, 0, 0)
        self.btn_start = QPushButton("▶  START")
        self.btn_stop  = QPushButton("■  STOP")
        self.btn_reset = QPushButton("↺  RESET")
        self.btn_start.setStyleSheet("color:#22c55e; font-weight:bold;")
        self.btn_stop.setStyleSheet("color:#ef4444; font-weight:bold;")
        for b in [self.btn_start, self.btn_stop, self.btn_reset]: bl.addWidget(b)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reset.clicked.connect(self._reset)
        g.addWidget(bw, row, 0, 1, 4)

        return box

    def _hsep(self, grid: QGridLayout, row: int) -> int:
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2d3748; margin: 2px 0;")
        grid.addWidget(sep, row, 0, 1, 4)
        return row + 1

    # ─────────────────────────────────────────────────────────────────────────
    def _setup_3d(self):
        self.plotter = QtInteractor(self.vtk_host)
        self.vtk_host.layout().addWidget(self.plotter)

        self.orig_pts: dict[str, np.ndarray] = {}
        self.meshes:   dict[str, pv.DataSet] = {}

        self.plotter.set_background('#0f172a')

        for name, cfg in LINKS_CONFIG.items():
            path = STL_FILES[name]
            if os.path.exists(path):
                mesh = pv.read(path)
                if mesh.n_cells > 10_000:
                    mesh = mesh.decimate(0.5)
            else:
                # Fallback placeholder geometry
                c = cfg['pivot']
                if name == 'azimuth':
                    mesh = pv.Cylinder(center=c, direction=[0,0,1], radius=40, height=20, resolution=32)
                elif name == 'elevation':
                    mesh = pv.Box(bounds=(c[0]-30, c[0]+30, c[1]-10, c[1]+10, c[2]-40, c[2]+40))
                else:
                    mesh = pv.Cylinder(center=c, direction=[0,1,0], radius=20, height=60, resolution=20)

            mesh.points = mesh.points.astype(np.float32)
            self.orig_pts[name] = mesh.points.copy()
            self.meshes[name]   = mesh
            self.plotter.add_mesh(mesh, color=cfg['color'], smooth_shading=True,
                                  opacity=0.92, name=name)

        self.plotter.add_axes()
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()

    # ─────────────────────────────────────────────────────────────────────────
    def _setup_charts(self):
        pg.setConfigOptions(antialias=True, background='#0f172a', foreground='#94a3b8')

        # ── Tab A: PWM ──────────────────────────────────────────────────────
        pw = QWidget(); pl = QVBoxLayout(pw); pl.setContentsMargins(4,4,4,4)
        self.plt_pwm = pg.PlotWidget(title="A — PWM Duty Cycle (%)")
        self.plt_pwm.setLabel('left', 'PWM (%)')
        self.plt_pwm.setLabel('bottom', 'Time (s)')
        self.plt_pwm.showGrid(x=True, y=True, alpha=0.12)
        self.plt_pwm.setYRange(0, 110)
        self.plt_pwm.addLegend(offset=(-10, 10))
        # Stall threshold reference line
        self.plt_pwm.addLine(
            y=BASE_STALL,
            pen=pg.mkPen('#ef4444', style=Qt.DashLine, width=1),
            label=f"Stall threshold ({BASE_STALL:.0f}%)"
        )
        self.crv_pwm = {
            ax: self.plt_pwm.plot(pen=pg.mkPen(AX_COLOR[ax], width=2), name=ax.capitalize())
            for ax in AXES
        }
        pl.addWidget(self.plt_pwm)
        self.chart_tabs.addTab(pw, "A: PWM")

        # ── Tab B: Position ─────────────────────────────────────────────────
        aw = QWidget(); al = QVBoxLayout(aw); al.setContentsMargins(4,4,4,4)
        # Axis selector row
        sel_row = QHBoxLayout()
        for ax in AXES:
            b = QPushButton(ax.capitalize())
            b.setCheckable(True); b.setChecked(ax == 'azimuth')
            b.setMaximumWidth(100)
            b.setStyleSheet(
                f"QPushButton:checked {{ background:{AX_COLOR[ax]}; color:#fff; border-radius:4px; }}"
                f"QPushButton {{ color:{AX_COLOR[ax]}; }}"
            )
            b.clicked.connect(lambda _, a=ax: self._set_view_ax(a))
            sel_row.addWidget(b)
            self.ax_btns[ax] = b
        sel_row.addStretch()
        sw = QWidget(); sw.setLayout(sel_row)
        al.addWidget(sw)

        self.plt_ang = pg.PlotWidget(title="B — Angle Response — Azimuth (°)")
        self.plt_ang.setLabel('left', 'Angle (°)')
        self.plt_ang.setLabel('bottom', 'Time (s)')
        self.plt_ang.showGrid(x=True, y=True, alpha=0.12)
        self.plt_ang.setYRange(0, AX_MAX['azimuth'] * 1.05)
        self.plt_ang.addLegend(offset=(-10, 10))
        self.crv_ang = {
            'target':    self.plt_ang.plot(pen=pg.mkPen('#475569', style=Qt.DashLine, width=1.5), name='Target'),
            'enc_meas':  self.plt_ang.plot(pen=pg.mkPen(color=(100,160,255,55), width=1),         name='Encoder raw'),
            'fused_ang': self.plt_ang.plot(pen=pg.mkPen('#a78bfa', width=2),                       name='Kalman fused'),
            'true_ang':  self.plt_ang.plot(pen=pg.mkPen(AX_COLOR['azimuth'], width=3),            name='True angle'),
        }
        al.addWidget(self.plt_ang)
        self.chart_tabs.addTab(aw, "B: Position")

        # ── Tab C: Velocity (inner loop) ────────────────────────────────────
        vw = QWidget(); vl = QVBoxLayout(vw); vl.setContentsMargins(4,4,4,4)
        self.plt_vel = pg.PlotWidget(title="C — Velocity Response — Azimuth (°/s)")
        self.plt_vel.setLabel('left', 'Angular Velocity (°/s)')
        self.plt_vel.setLabel('bottom', 'Time (s)')
        self.plt_vel.showGrid(x=True, y=True, alpha=0.12)
        self.plt_vel.addLegend(offset=(-10, 10))
        self.crv_vel = {
            'vel_demand': self.plt_vel.plot(pen=pg.mkPen('#f97316', style=Qt.DashLine, width=2), name='Demand (outer loop)'),
            'fused_vel':  self.plt_vel.plot(pen=pg.mkPen('#a78bfa', width=2),                    name='Fused velocity (Kalman)'),
            'true_vel':   self.plt_vel.plot(pen=pg.mkPen('#22d3ee', width=2.5),                  name='True velocity'),
        }
        vl.addWidget(self.plt_vel)
        self.chart_tabs.addTab(vw, "C: Velocity")

        # ── Tab D: Rate-of-change error (inner loop input) ───────────────────
        rw = QWidget(); rl = QVBoxLayout(rw); rl.setContentsMargins(4,4,4,4)
        self.plt_roc = pg.PlotWidget(title="D — Rate-of-Change Error (Inner Loop Input)")
        self.plt_roc.setLabel('left', 'Velocity Error (°/s)')
        self.plt_roc.setLabel('bottom', 'Time (s)')
        self.plt_roc.showGrid(x=True, y=True, alpha=0.12)
        self.plt_roc.addLegend(offset=(-10, 10))
        self.crv_roc = {
            ax: self.plt_roc.plot(pen=pg.mkPen(AX_COLOR[ax], width=2), name=ax.capitalize())
            for ax in AXES
        }
        rl.addWidget(self.plt_roc)
        self.chart_tabs.addTab(rw, "D: ROC Error")

    # ─────────────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_target(self, ax: str, v: int, lbl: QLabel):
        self.targets[ax] = float(v); lbl.setText(f"{v}°")

    def _on_param(self, attr: str, v: float, lbl: QLabel):
        setattr(self, attr, v); lbl.setText(f"{v:.1f}")

    def _set_view_ax(self, ax: str):
        self.view_ax = ax
        for a, b in self.ax_btns.items():
            b.setChecked(a == ax)
        c = AX_COLOR[ax]
        self.crv_ang['true_ang'].setPen(pg.mkPen(c, width=3))
        self.plt_ang.setTitle(f"B — Angle Response — {ax.capitalize()} (°)")
        self.plt_ang.setYRange(0, AX_MAX[ax] * 1.05)
        self.plt_vel.setTitle(f"C — Velocity Response — {ax.capitalize()} (°/s)")

    # ─────────────────────────────────────────────────────────────────────────
    #  SIMULATION CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    def _start(self):
        if self.running:
            return
        # Reset physics and logs — keep target sliders as-is
        for ax in AXES:
            self.axes[ax].reset()
        self.t_buf.clear()
        for ax in AXES:
            for q in self.buf[ax].values(): q.clear()
        self.sim_time = 0.
        self.running  = True

    def _stop(self):
        self.running = False

    def _reset(self):
        self.running  = False
        self.sim_time = 0.
        for ax in AXES:
            self.axes[ax].reset()
        self.t_buf.clear()
        for ax in AXES:
            for q in self.buf[ax].values(): q.clear()
        # Clear curves
        for crv_dict in [self.crv_pwm, self.crv_ang, self.crv_vel, self.crv_roc]:
            for c in crv_dict.values():
                c.setData([], [])
        self._update_status()
        self._update_3d()

    # ─────────────────────────────────────────────────────────────────────────
    #  SIMULATION TICK
    # ─────────────────────────────────────────────────────────────────────────
    def _tick(self):
        if not self.running:
            return

        self.sim_time += DT
        self.t_buf.append(self.sim_time)

        el_deg  = self.axes['elevation'].angle
        all_done = True

        for ax in AXES:
            a  = self.axes[ax]
            fa, fv = a.step(self.targets[ax], el_deg, self.spd, self.n_enc, self.n_ahrs)
            b  = self.buf[ax]
            b['pwm'].append(a.pwm)
            b['true_ang'].append(a.angle)
            b['fused_ang'].append(fa)
            b['enc_meas'].append(a.enc_meas)
            b['vel_demand'].append(a.vel_demand)
            b['true_vel'].append(a.vel)
            b['fused_vel'].append(fv)
            b['roc_err'].append(a.roc_err)
            if a.phase != 'DONE':
                all_done = False

        if all_done:
            self.running = False

        self._update_plots()
        self._update_3d()
        self._update_status()

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDERING
    # ─────────────────────────────────────────────────────────────────────────
    def _update_plots(self):
        if not self.t_buf:
            return
        t = list(self.t_buf)

        # Tab A — PWM
        for ax in AXES:
            self.crv_pwm[ax].setData(t, list(self.buf[ax]['pwm']))

        # Tab B — Position (selected axis)
        va = self.view_ax; b = self.buf[va]
        tgt_line = [self.targets[va]] * len(t)
        self.crv_ang['target'].setData(t, tgt_line)
        self.crv_ang['enc_meas'].setData(t,  list(b['enc_meas']))
        self.crv_ang['fused_ang'].setData(t, list(b['fused_ang']))
        self.crv_ang['true_ang'].setData(t,  list(b['true_ang']))

        # Tab C — Velocity
        self.crv_vel['vel_demand'].setData(t, list(b['vel_demand']))
        self.crv_vel['fused_vel'].setData(t,  list(b['fused_vel']))
        self.crv_vel['true_vel'].setData(t,   list(b['true_vel']))

        # Tab D — Rate-of-change error
        for ax in AXES:
            self.crv_roc[ax].setData(t, list(self.buf[ax]['roc_err']))

    def _update_3d(self):
        M_az  = transform_matrix(LINKS_CONFIG['azimuth']['pivot'],
                                 LINKS_CONFIG['azimuth']['axis'],
                                 self.axes['azimuth'].angle)
        M_el  = transform_matrix(LINKS_CONFIG['elevation']['pivot'],
                                 LINKS_CONFIG['elevation']['axis'],
                                 self.axes['elevation'].angle)
        M_pol = transform_matrix(LINKS_CONFIG['polarization']['pivot'],
                                 LINKS_CONFIG['polarization']['axis'],
                                 self.axes['polarization'].angle)

        T = {
            'azimuth':     M_az,
            'elevation':   M_az @ M_el,
            'polarization':M_az @ M_el @ M_pol,
        }
        for name in AXES:
            self.meshes[name].points[:] = apply_transform(self.orig_pts[name], T[name])
        self.plotter.update()

    def _update_status(self):
        el = self.axes['elevation'].angle
        for ax in AXES:
            a  = self.axes[ax]
            lf = load_factor(ax, el)
            if a.stalled:
                badge = "⚠ STALL"
            elif a.phase == 'MOVING':
                badge = f"MOVING  roc_err:{a.roc_err:+.2f}°/s"
            else:
                badge = a.phase
            self.status_lbls[ax].setText(
                f"θ:{a.angle:6.1f}°  V:{a.vel:+5.2f}°/s  "
                f"PWM:{a.pwm:4.0f}%  lf:{lf:.2f}  {badge}"
            )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())