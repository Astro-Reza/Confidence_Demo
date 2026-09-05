"""
COTP Antenna Pointing System: 3D Kinematics & Confidence-Score Sensor Fusion
=============================================================================
A comprehensive interactive simulation environment featuring:
- Page 1: Live 3D Simulation & Sensor Noise Control "Game"
    * 3D PyVista viewport with real CAD STL mesh kinematics (Azimuth, Elevation, Polarization)
    * Real-time noise sliders (σ_enc, σ_ahrs) & interactive fault injection (Drift ramp, Glitch step)
    * Real-time causal confidence-score quaternion sensor fusion (from `development.ipynb`)
    * Target chasing (Cascade PID plant) vs Brownian random motion
    * Live telemetry plots & real-time CSV recording
- Page 2: Post-Run Analysis & Data Inspector
    * 1-click inspection of recorded simulation runs or any exported CSV
    * Multi-axis switching (Azimuth, Elevation, Polarization)
    * Synchronized multi-panel graphs with draggable overview time window
    * Live mouse hover crosshair with exact numeric readout
    * Comprehensive quality score card (RMSE, MAE, Max Error, SNR, % Improvement)

Usage:
    python antenna_fusion_app.py
"""

import os
import sys
import glob
from collections import deque
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QCheckBox, QRadioButton,
    QButtonGroup, QComboBox, QFileDialog, QGroupBox, QFrame, QSplitter,
    QTabWidget, QStatusBar, QSizePolicy, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg
import pyvista as pv
from pyvistaqt import QtInteractor

from antenna_sim_core import (
    AntennaSimulationManager, compute_fk_matrices, apply_transform,
    AXES, AX_LIMITS, AX_COLORS, LINKS_CONFIG, STL_FILES, BASE_STALL
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
pg.setConfigOption('background', '#0f172a')  # Slate 900
pg.setConfigOption('foreground', '#cbd5e1')  # Slate 300
pg.setConfigOptions(antialias=True)

COLORS = {
    'true':       '#94a3b8',   # Slate 400 (Reference)
    'target':     '#e2e8f0',   # Slate 200 (Dashed target)
    'enc':        '#f97316',   # Bright Orange (Encoder)
    'ahrs':       '#38bdf8',   # Sky Blue (AHRS)
    'fused':      '#10b981',   # Emerald Green (Fused output)
    'conf_enc':   '#fb923c',   # Light Orange
    'conf_ahrs':  '#7dd3fc',   # Light Blue
    'err_enc':    '#ea580c',   # Deep Orange
    'err_ahrs':   '#0284c7',   # Deep Cyan
    'err_fused':  '#059669',   # Deep Green
    'crosshair':  '#facc15',   # Yellow crosshair
    'region':     '#38bdf833', # Translucent cyan for region selection
}


class AntennaFusionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antenna Pointing System — 3D Kinematics & Confidence-Score Sensor Fusion")
        self.resize(1560, 960)

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

        self._build_main_ui()

        # Simulation Timer (50 Hz / 20ms)
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self._on_sim_tick)
        self.sim_timer.start(20)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN APPLICATION UI (TABBED CONTAINER)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_main_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #1e293b; background: #0b1120; }
            QTabBar::tab {
                background: #0f172a; color: #94a3b8; font-weight: bold; font-size: 12px;
                padding: 8px 24px; border: 1px solid #1e293b; border-bottom: none;
                border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #1e293b; color: #38bdf8; border-top: 2px solid #38bdf8;
            }
            QTabBar::tab:hover { background: #1a2333; color: #f8fafc; }
        """)

        # Page 1: Live 3D Simulation & Noise Game
        self.page1 = QWidget()
        self._build_page1_ui(self.page1)
        self.tabs.addTab(self.page1, "🛰  PAGE 1: Live Simulation & Noise Control (Game)")

        # Page 2: History & Post-Run Analysis
        self.page2 = QWidget()
        self._build_page2_ui(self.page2)
        self.tabs.addTab(self.page2, "📊  PAGE 2: Post-Run Analysis & CSV Inspector")

        layout.addWidget(self.tabs)

        # Global Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setStyleSheet("color: #cbd5e1; background: #090d16; font-size: 11px;")
        self.status.showMessage("Simulation ready. Set targets and press ▶ START.")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1: LIVE SIMULATION & NOISE CONTROL
    # ─────────────────────────────────────────────────────────────────────────
    def _build_page1_ui(self, parent):
        p1_layout = QHBoxLayout(parent)
        p1_layout.setContentsMargins(4, 4, 4, 4)
        p1_layout.setSpacing(4)

        p1_split = QSplitter(Qt.Horizontal)
        p1_split.setHandleWidth(6)
        p1_split.setStyleSheet("QSplitter::handle { background: #1e293b; }")

        # 1. Left Sidebar: Controls & Game Settings
        sidebar = self._build_p1_sidebar()
        p1_split.addWidget(sidebar)

        # 2. Center Panel: 3D PyVista Viewport
        viewport_container = self._build_p1_3d_viewport()
        p1_split.addWidget(viewport_container)

        # 3. Right Panel: Real-Time Telemetry Plots
        plots_container = self._build_p1_live_plots()
        p1_split.addWidget(plots_container)

        p1_split.setStretchFactor(0, 2)
        p1_split.setStretchFactor(1, 4)
        p1_split.setStretchFactor(2, 4)
        p1_layout.addWidget(p1_split)

    def _build_p1_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border: 1px solid #1e293b; background: #0b1120; border-radius: 6px; }
            QWidget#p1SidebarContent { background: #0b1120; }
        """)

        container = QWidget()
        container.setObjectName("p1SidebarContent")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        mono = QFont("Consolas", 9)

        # ── Group 1: Motion & Target Controls ──────────────────────────────
        grp_tgt = QGroupBox("Motion & Target Controls")
        grp_tgt.setStyleSheet(self._grp_style("#38bdf8"))
        gt_l = QVBoxLayout(grp_tgt)
        gt_l.setSpacing(6)

        # Motion Mode Toggle
        mode_row = QHBoxLayout()
        self.rb_mode_target = QRadioButton("Target Chasing (PID)")
        self.rb_mode_brownian = QRadioButton("Brownian Motion")
        self.rb_mode_target.setChecked(True)
        self.rb_mode_target.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        self.rb_mode_brownian.setStyleSheet("color: #a78bfa; font-size: 11px; font-weight: bold;")
        self.rb_mode_target.toggled.connect(self._on_motion_mode_changed)
        mode_row.addWidget(self.rb_mode_target)
        mode_row.addWidget(self.rb_mode_brownian)
        gt_l.addLayout(mode_row)

        # Target Sliders
        self.tgt_sliders = {}
        self.tgt_labels = {}
        target_defs = [
            ('azimuth',      'Azimuth (AZ)', 0, 360, 90,  AX_COLORS['azimuth']),
            ('elevation',    'Elevation (EL)', 0, 90, 45, AX_COLORS['elevation']),
            ('polarization', 'Polarization (POL)', 0, 180, 45, AX_COLORS['polarization']),
        ]
        for ax, label, mn, mx, dflt, col in target_defs:
            row_l = QHBoxLayout()
            lbl_name = QLabel(f"{label}:")
            lbl_name.setStyleSheet(f"color: {col}; font-weight: bold; font-size: 11px; width: 110px;")
            lbl_name.setFixedWidth(115)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(mn, mx)
            sl.setValue(dflt)
            lbl_val = QLabel(f"{dflt:3d}°")
            lbl_val.setFont(mono)
            lbl_val.setFixedWidth(40)
            lbl_val.setStyleSheet(f"color: {col};")

            sl.valueChanged.connect(lambda v, a=ax, l=lbl_val: self._on_target_slider(a, v, l))
            row_l.addWidget(lbl_name)
            row_l.addWidget(sl)
            row_l.addWidget(lbl_val)
            gt_l.addLayout(row_l)
            self.tgt_sliders[ax] = sl
            self.tgt_labels[ax] = lbl_val
            self.sim.set_target(ax, dflt)

        # Run / Stop / Reset Buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ START")
        self.btn_stop = QPushButton("■ STOP")
        self.btn_reset = QPushButton("↺ RESET")
        self.btn_start.setStyleSheet(self._btn_style("#16a34a"))
        self.btn_stop.setStyleSheet(self._btn_style("#dc2626"))
        self.btn_reset.setStyleSheet(self._btn_style("#334155"))
        self.btn_start.clicked.connect(self._on_start_sim)
        self.btn_stop.clicked.connect(self._on_stop_sim)
        self.btn_reset.clicked.connect(self._on_reset_sim)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_reset)
        gt_l.addLayout(btn_row)

        layout.addWidget(grp_tgt)

        # ── Group 2: CSV Data Recording ────────────────────────────────────
        grp_rec = QGroupBox("CSV Data Recording")
        grp_rec.setStyleSheet(self._grp_style("#ef4444"))
        gr_l = QVBoxLayout(grp_rec)
        gr_l.setSpacing(6)

        rec_btn_row = QHBoxLayout()
        self.btn_rec_toggle = QPushButton("🔴 START RECORDING")
        self.btn_rec_toggle.setCheckable(True)
        self.btn_rec_toggle.setStyleSheet(self._btn_style("#991b1b"))
        self.btn_rec_toggle.clicked.connect(self._on_toggle_recording)
        rec_btn_row.addWidget(self.btn_rec_toggle)

        self.btn_export_inspect = QPushButton("Export & Inspect ➔")
        self.btn_export_inspect.setStyleSheet(self._btn_style("#0284c7"))
        self.btn_export_inspect.clicked.connect(self._on_export_and_inspect)
        rec_btn_row.addWidget(self.btn_export_inspect)
        gr_l.addLayout(rec_btn_row)

        self.lbl_rec_status = QLabel("Recording: Inactive (0 samples)")
        self.lbl_rec_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        gr_l.addWidget(self.lbl_rec_status)

        layout.addWidget(grp_rec)

        # ── Group 3: Live Sensor Noise & Fault Injection ("Game") ──────────
        grp_game = QGroupBox("Live Noise & Fault Control (Game)")
        grp_game.setStyleSheet(self._grp_style("#facc15"))
        gg_l = QVBoxLayout(grp_game)
        gg_l.setSpacing(6)

        # Fault Target Axis Selector
        f_axis_row = QHBoxLayout()
        f_axis_lbl = QLabel("Inject Target:")
        f_axis_lbl.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        f_axis_row.addWidget(f_axis_lbl)
        self.cb_fault_axis = QComboBox()
        self.cb_fault_axis.addItems(["Azimuth", "Elevation", "Polarization", "All Axes"])
        self.cb_fault_axis.setStyleSheet("background: #1e293b; color: #f8fafc; padding: 4px; border-radius: 4px;")
        self.cb_fault_axis.currentTextChanged.connect(self._on_fault_axis_changed)
        f_axis_row.addWidget(self.cb_fault_axis)
        gg_l.addLayout(f_axis_row)

        # Noise Sliders
        # Encoder Noise
        enc_noise_row = QHBoxLayout()
        lbl_en_name = QLabel("Encoder Noise σ:")
        lbl_en_name.setStyleSheet(f"color: {COLORS['enc']}; font-size: 11px;")
        lbl_en_name.setFixedWidth(115)
        self.sl_noise_enc = QSlider(Qt.Horizontal)
        self.sl_noise_enc.setRange(0, 50)  # 0.0 to 5.0 deg
        self.sl_noise_enc.setValue(5)       # 0.5 deg
        self.lbl_noise_enc = QLabel("0.50°")
        self.lbl_noise_enc.setFont(mono)
        self.lbl_noise_enc.setFixedWidth(40)
        self.lbl_noise_enc.setStyleSheet(f"color: {COLORS['enc']};")
        self.sl_noise_enc.valueChanged.connect(self._on_enc_noise_slider)
        enc_noise_row.addWidget(lbl_en_name)
        enc_noise_row.addWidget(self.sl_noise_enc)
        enc_noise_row.addWidget(self.lbl_noise_enc)
        gg_l.addLayout(enc_noise_row)

        # AHRS Noise
        ahrs_noise_row = QHBoxLayout()
        lbl_an_name = QLabel("AHRS Noise σ:")
        lbl_an_name.setStyleSheet(f"color: {COLORS['ahrs']}; font-size: 11px;")
        lbl_an_name.setFixedWidth(115)
        self.sl_noise_ahrs = QSlider(Qt.Horizontal)
        self.sl_noise_ahrs.setRange(0, 50)  # 0.0 to 5.0 deg
        self.sl_noise_ahrs.setValue(3)       # 0.3 deg
        self.lbl_noise_ahrs = QLabel("0.30°")
        self.lbl_noise_ahrs.setFont(mono)
        self.lbl_noise_ahrs.setFixedWidth(40)
        self.lbl_noise_ahrs.setStyleSheet(f"color: {COLORS['ahrs']};")
        self.sl_noise_ahrs.valueChanged.connect(self._on_ahrs_noise_slider)
        ahrs_noise_row.addWidget(lbl_an_name)
        ahrs_noise_row.addWidget(self.sl_noise_ahrs)
        ahrs_noise_row.addWidget(self.lbl_noise_ahrs)
        gg_l.addLayout(ahrs_noise_row)

        # Fault Injection Buttons
        fault_btn_row = QHBoxLayout()
        self.btn_fault_drift = QPushButton("+ Drift Ramp (1.5°/s)")
        self.btn_fault_drift.setStyleSheet(self._btn_style("#c2410c"))
        self.btn_fault_drift.clicked.connect(self._on_inject_drift)
        fault_btn_row.addWidget(self.btn_fault_drift)

        self.btn_fault_glitch = QPushButton("+ Glitch Spike (+12°)")
        self.btn_fault_glitch.setStyleSheet(self._btn_style("#b91c1c"))
        self.btn_fault_glitch.clicked.connect(self._on_inject_glitch)
        fault_btn_row.addWidget(self.btn_fault_glitch)
        gg_l.addLayout(fault_btn_row)

        self.btn_fault_clear = QPushButton("↺ Clear Faults / Recover Sensor")
        self.btn_fault_clear.setStyleSheet(self._btn_style("#15803d"))
        self.btn_fault_clear.clicked.connect(self._on_clear_faults)
        gg_l.addWidget(self.btn_fault_clear)

        layout.addWidget(grp_game)

        # ── Group 4: Live Status Readout ───────────────────────────────────
        grp_stat = QGroupBox("Live Kinematic Status")
        grp_stat.setStyleSheet(self._grp_style("#10b981"))
        gs_l = QVBoxLayout(grp_stat)
        gs_l.setSpacing(4)

        self.live_status_lbls = {}
        for ax in AXES:
            l = QLabel(f"{ax.upper()[:3]}: True: --° | Fused: --°")
            l.setFont(mono)
            l.setStyleSheet(f"color: {AX_COLORS[ax]}; font-size: 11px;")
            gs_l.addWidget(l)
            self.live_status_lbls[ax] = l

        self.live_conf_lbl = QLabel("Confidence (Azimuth): Enc 50% | AHRS 50%")
        self.live_conf_lbl.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: bold;")
        gs_l.addWidget(self.live_conf_lbl)

        layout.addWidget(grp_stat)
        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_p1_3d_viewport(self):
        vp_widget = QWidget()
        layout = QVBoxLayout(vp_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # 3D Toolbar Header
        tb = QHBoxLayout()
        title = QLabel("<b>3D ANTENNA MODEL (REAL STL KINEMATICS)</b>")
        title.setStyleSheet("color: #38bdf8; font-size: 11px; padding: 4px;")
        tb.addWidget(title)
        tb.addStretch()

        for view_name in ['Iso', 'Top', 'Front', 'Side']:
            b = QPushButton(view_name)
            b.setStyleSheet(self._btn_style("#1e293b"))
            b.clicked.connect(lambda _, vn=view_name.lower(): self._set_3d_camera(vn))
            tb.addWidget(b)

        layout.addLayout(tb)

        # PyVista QtInteractor
        self.plotter = QtInteractor(vp_widget)
        self.plotter.set_background('#0b1120')
        layout.addWidget(self.plotter, stretch=1)

        # Load STL meshes with kinematics configuration
        self.orig_pts = {}
        self.meshes = {}

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
            self.plotter.add_mesh(mesh, color=cfg['color'], smooth_shading=True, opacity=0.92, name=name)

        self.plotter.add_axes()
        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()

        return vp_widget

    def _build_p1_live_plots(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Axis Selector Row
        sel_row = QHBoxLayout()
        sel_lbl = QLabel("<b>TELEMETRY AXIS:</b>")
        sel_lbl.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        sel_row.addWidget(sel_lbl)

        self.live_ax_btns = {}
        for ax in AXES:
            b = QPushButton(ax.capitalize())
            b.setCheckable(True)
            b.setChecked(ax == 'azimuth')
            b.setStyleSheet(
                f"QPushButton:checked {{ background:{AX_COLORS[ax]}; color:#fff; font-weight:bold; border-radius:4px; padding: 4px 12px; }}"
                f"QPushButton {{ background:#1e293b; color:{AX_COLORS[ax]}; border-radius:4px; padding: 4px 12px; }}"
            )
            b.clicked.connect(lambda _, a=ax: self._set_live_axis(a))
            sel_row.addWidget(b)
            self.live_ax_btns[ax] = b
        sel_row.addStretch()
        layout.addLayout(sel_row)

        # PyQtGraph GraphicsLayoutWidget
        self.live_gw = pg.GraphicsLayoutWidget()
        self.live_gw.ci.layout.setContentsMargins(2, 2, 2, 2)
        self.live_gw.ci.layout.setSpacing(2)
        layout.addWidget(self.live_gw, stretch=1)

        # Plot 1: Angle Tracking
        self.plt_live_ang = self.live_gw.addPlot(row=0, col=0)
        self.plt_live_ang.setTitle("<span style='font-size: 10pt; color: #f8fafc; font-weight: bold;'>1. Angle Response (°): True vs Raw Sensors vs Fused</span>")
        self.plt_live_ang.setLabel('left', 'Angle (°)')
        self.plt_live_ang.showGrid(x=True, y=True, alpha=0.15)
        self.plt_live_ang.addLegend(offset=(-8, 8))

        # Plot 2: Confidence Weights
        self.plt_live_conf = self.live_gw.addPlot(row=1, col=0)
        self.plt_live_conf.setTitle("<span style='font-size: 10pt; color: #f8fafc; font-weight: bold;'>2. Confidence Scoring (c_enc vs c_ahrs)</span>")
        self.plt_live_conf.setLabel('left', 'Weight')
        self.plt_live_conf.setYRange(-0.05, 1.05, padding=0)
        self.plt_live_conf.showGrid(x=True, y=True, alpha=0.15)
        self.plt_live_conf.addLegend(offset=(-8, 8))

        # Plot 3: Error
        self.plt_live_err = self.live_gw.addPlot(row=2, col=0)
        self.plt_live_err.setTitle("<span style='font-size: 10pt; color: #f8fafc; font-weight: bold;'>3. Real-Time Tracking Error (Sensor - Ground Truth)</span>")
        self.plt_live_err.setLabel('left', 'Error (°)')
        self.plt_live_err.setLabel('bottom', 'Time (s)')
        self.plt_live_err.showGrid(x=True, y=True, alpha=0.15)
        self.plt_live_err.addLegend(offset=(-8, 8))
        self.plt_live_err.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#475569', style=Qt.DashLine, width=1)))

        # Synchronize X axes
        self.plt_live_conf.setXLink(self.plt_live_ang)
        self.plt_live_err.setXLink(self.plt_live_ang)

        # Curves
        self.crv_live = {
            'target': self.plt_live_ang.plot(pen=pg.mkPen(COLORS['target'], style=Qt.DashLine, width=1.5), name='Target'),
            'true':   self.plt_live_ang.plot(pen=pg.mkPen(COLORS['true'], width=2), name='True Plant'),
            'enc':    self.plt_live_ang.plot(pen=pg.mkPen(color=(249, 115, 22, 120), width=1.2), name='Encoder (Noisy/Drift)'),
            'ahrs':   self.plt_live_ang.plot(pen=pg.mkPen(color=(56, 189, 248, 120), width=1.2), name='AHRS'),
            'fused':  self.plt_live_ang.plot(pen=pg.mkPen(COLORS['fused'], width=2.5), name='Quaternion Fused'),

            'c_enc':  self.plt_live_conf.plot(pen=pg.mkPen(COLORS['conf_enc'], width=1.8), name='c_enc (Weight)'),
            'c_ahrs': self.plt_live_conf.plot(pen=pg.mkPen(COLORS['conf_ahrs'], width=1.8), name='c_ahrs (Weight)'),

            'err_enc':   self.plt_live_err.plot(pen=pg.mkPen(color=(234, 88, 12, 140), width=1.2), name='Encoder Error'),
            'err_ahrs':  self.plt_live_err.plot(pen=pg.mkPen(color=(2, 132, 199, 140), width=1.2), name='AHRS Error'),
            'err_fused': self.plt_live_err.plot(pen=pg.mkPen(COLORS['err_fused'], width=2.2), name='Fused Error'),
        }

        return container

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2: POST-RUN ANALYSIS & CSV INSPECTOR
    # ─────────────────────────────────────────────────────────────────────────
    def _build_page2_ui(self, parent):
        p2_layout = QHBoxLayout(parent)
        p2_layout.setContentsMargins(4, 4, 4, 4)
        p2_layout.setSpacing(4)

        p2_split = QSplitter(Qt.Horizontal)
        p2_split.setHandleWidth(6)
        p2_split.setStyleSheet("QSplitter::handle { background: #1e293b; }")

        # ── Left / Main Plots Area ──────────────────────────────────────────
        plot_col = QWidget()
        pl_layout = QVBoxLayout(plot_col)
        pl_layout.setContentsMargins(0, 0, 0, 0)
        pl_layout.setSpacing(4)

        # Top Bar: File & Axis Selection
        top_bar = QHBoxLayout()
        self.btn_p2_inspect_last = QPushButton("📥 Inspect Last Recording")
        self.btn_p2_inspect_last.setStyleSheet(self._btn_style("#16a34a"))
        self.btn_p2_inspect_last.clicked.connect(self._on_inspect_last_recording)
        top_bar.addWidget(self.btn_p2_inspect_last)

        self.btn_p2_open = QPushButton("📂 Open CSV...")
        self.btn_p2_open.setStyleSheet(self._btn_style("#0284c7"))
        self.btn_p2_open.clicked.connect(self._on_p2_open_file)
        top_bar.addWidget(self.btn_p2_open)

        self.btn_p2_reload = QPushButton("↺ Reload")
        self.btn_p2_reload.setStyleSheet(self._btn_style("#334155"))
        self.btn_p2_reload.clicked.connect(self._on_p2_reload)
        top_bar.addWidget(self.btn_p2_reload)

        top_bar.addSpacing(12)
        ax_lbl = QLabel("Axis:")
        ax_lbl.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px;")
        top_bar.addWidget(ax_lbl)

        self.cb_p2_axis = QComboBox()
        self.cb_p2_axis.addItems(["Azimuth", "Elevation", "Polarization"])
        self.cb_p2_axis.setStyleSheet("background: #1e293b; color: #f8fafc; padding: 4px; border-radius: 4px;")
        self.cb_p2_axis.currentTextChanged.connect(self._on_p2_axis_changed)
        top_bar.addWidget(self.cb_p2_axis)

        self.lbl_p2_file_info = QLabel("No CSV loaded")
        self.lbl_p2_file_info.setStyleSheet("color: #64748b; font-size: 11px; margin-left: 10px;")
        top_bar.addWidget(self.lbl_p2_file_info, stretch=1)
        pl_layout.addLayout(top_bar)

        # Graphics Layout Widget for Synchronized Plots
        self.p2_gw = pg.GraphicsLayoutWidget()
        self.p2_gw.ci.layout.setContentsMargins(4, 4, 4, 4)
        self.p2_gw.ci.layout.setSpacing(4)
        pl_layout.addWidget(self.p2_gw, stretch=1)

        # Overview Timeline Strip with Draggable Region
        ov_title = QLabel("<b>TIMELINE OVERVIEW & DRAGGABLE TIME WINDOW:</b> (Drag highlighted window or edges to pan/zoom)")
        ov_title.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 4px;")
        pl_layout.addWidget(ov_title)

        self.p2_overview = pg.PlotWidget(name="P2Overview")
        self.p2_overview.setFixedHeight(75)
        self.p2_overview.showGrid(x=True, y=True, alpha=0.15)
        self.p2_overview.setMouseEnabled(x=False, y=False)
        self.p2_overview.getPlotItem().hideAxis('left')
        self.p2_overview.getPlotItem().hideAxis('bottom')

        self.p2_region = pg.LinearRegionItem([0, 5], brush=pg.mkBrush(COLORS['region']))
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
        p2_layout.addWidget(p2_split)

    def _setup_p2_plots(self):
        # Panel 1: Angle Tracking
        self.p2_ang = self.p2_gw.addPlot(row=0, col=0)
        self.p2_ang.setTitle("<span style='font-size: 11pt; color: #f8fafc; font-weight: bold;'>1. Angle Tracking: Ground Truth vs Sensors vs Fused</span>")
        self.p2_ang.setLabel('left', 'Angle (°)')
        self.p2_ang.showGrid(x=True, y=True, alpha=0.18)
        self.p2_ang.addLegend(offset=(-10, 10))

        # Panel 2: Confidences
        self.p2_conf = self.p2_gw.addPlot(row=1, col=0)
        self.p2_conf.setTitle("<span style='font-size: 10pt; color: #f8fafc; font-weight: bold;'>2. Confidence Scoring Weights (c_enc vs c_ahrs)</span>")
        self.p2_conf.setLabel('left', 'Confidence')
        self.p2_conf.setYRange(-0.05, 1.05, padding=0)
        self.p2_conf.showGrid(x=True, y=True, alpha=0.18)
        self.p2_conf.addLegend(offset=(-10, 10))

        # Panel 3: Errors
        self.p2_err = self.p2_gw.addPlot(row=2, col=0)
        self.p2_err.setTitle("<span style='font-size: 10pt; color: #f8fafc; font-weight: bold;'>3. Tracking Errors (Sensor - Ground Truth)</span>")
        self.p2_err.setLabel('left', 'Error (°)')
        self.p2_err.setLabel('bottom', 'Time (s)')
        self.p2_err.showGrid(x=True, y=True, alpha=0.18)
        self.p2_err.addLegend(offset=(-10, 10))
        self.p2_err.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#475569', style=Qt.DashLine, width=1)))

        # Sync X axes
        self.p2_conf.setXLink(self.p2_ang)
        self.p2_err.setXLink(self.p2_ang)
        self.p2_ang.sigRangeChanged.connect(self._on_p2_plot_range_changed)

        # Crosshairs
        self.insp_crosshairs = [
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS['crosshair'], width=1.2, style=Qt.DashLine)),
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS['crosshair'], width=1.2, style=Qt.DashLine)),
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS['crosshair'], width=1.2, style=Qt.DashLine)),
        ]
        for plot, ch in zip([self.p2_ang, self.p2_conf, self.p2_err], self.insp_crosshairs):
            ch.setZValue(100)
            ch.setVisible(False)
            plot.addItem(ch)

        self.p2_gw.scene().sigMouseMoved.connect(self._on_p2_mouse_moved)

        # Plot Curves
        self.insp_curves['target'] = self.p2_ang.plot(pen=pg.mkPen(COLORS['target'], style=Qt.DashLine, width=1.5), name="Target")
        self.insp_curves['true'] = self.p2_ang.plot(pen=pg.mkPen(COLORS['true'], width=2), name="Ground Truth")
        self.insp_curves['enc'] = self.p2_ang.plot(pen=pg.mkPen(color=(249, 115, 22, 120), width=1.2), name="Encoder")
        self.insp_curves['ahrs'] = self.p2_ang.plot(pen=pg.mkPen(color=(56, 189, 248, 120), width=1.2), name="AHRS")
        self.insp_curves['fused'] = self.p2_ang.plot(pen=pg.mkPen(COLORS['fused'], width=2.5), name="Quaternion Fused")

        self.insp_curves['conf_enc'] = self.p2_conf.plot(pen=pg.mkPen(COLORS['conf_enc'], width=1.8), name="Encoder Confidence")
        self.insp_curves['conf_ahrs'] = self.p2_conf.plot(pen=pg.mkPen(COLORS['conf_ahrs'], width=1.8), name="AHRS Confidence")

        self.insp_curves['err_enc'] = self.p2_err.plot(pen=pg.mkPen(color=(234, 88, 12, 140), width=1.2), name="Encoder Error")
        self.insp_curves['err_ahrs'] = self.p2_err.plot(pen=pg.mkPen(color=(2, 132, 199, 140), width=1.2), name="AHRS Error")
        self.insp_curves['err_fused'] = self.p2_err.plot(pen=pg.mkPen(COLORS['err_fused'], width=2.2), name="Fused Error")

        self.insp_overview_crv = self.p2_overview.plot(pen=pg.mkPen('#64748b', width=1))

    def _build_p2_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border: 1px solid #1e293b; background: #0b1120; border-radius: 6px; }
            QWidget#p2SidebarContent { background: #0b1120; }
        """)

        container = QWidget()
        container.setObjectName("p2SidebarContent")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # 1. Live Cursor Inspector
        insp_grp = QGroupBox("Live Cursor Inspector")
        insp_grp.setStyleSheet(self._grp_style("#facc15"))
        il = QVBoxLayout(insp_grp)
        il.setSpacing(4)

        self.lbl_insp_time = QLabel("Time:  --- s")
        self.lbl_insp_time.setStyleSheet("color: #facc15; font-weight: bold; font-size: 13px;")
        il.addWidget(self.lbl_insp_time)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine); sep1.setStyleSheet("color: #334155;"); il.addWidget(sep1)

        self.lbl_insp_true = QLabel("True:   --- °")
        self.lbl_insp_true.setStyleSheet(f"color: {COLORS['true']};")
        il.addWidget(self.lbl_insp_true)

        self.lbl_insp_enc = QLabel("Enc:    --- °  (Δ: ---°)")
        self.lbl_insp_enc.setStyleSheet(f"color: {COLORS['enc']};")
        il.addWidget(self.lbl_insp_enc)

        self.lbl_insp_ahrs = QLabel("AHRS:   --- °  (Δ: ---°)")
        self.lbl_insp_ahrs.setStyleSheet(f"color: {COLORS['ahrs']};")
        il.addWidget(self.lbl_insp_ahrs)

        self.lbl_insp_fused = QLabel("Fused:  --- °  (Δ: ---°)")
        self.lbl_insp_fused.setStyleSheet(f"color: {COLORS['fused']}; font-weight: bold;")
        il.addWidget(self.lbl_insp_fused)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setStyleSheet("color: #334155;"); il.addWidget(sep2)

        self.lbl_insp_conf = QLabel("Confidence: Enc: -% | AHRS: -%")
        self.lbl_insp_conf.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        il.addWidget(self.lbl_insp_conf)

        layout.addWidget(insp_grp)

        # 2. Presets
        pr_grp = QGroupBox("Time Window Presets")
        pr_grp.setStyleSheet(self._grp_style("#a855f7"))
        pl = QGridLayout(pr_grp)
        pl.setSpacing(4)

        b_all = QPushButton("Fit All (Full)")
        b_all.setStyleSheet(self._btn_style("#475569"))
        b_all.clicked.connect(self._p2_zoom_all)
        pl.addWidget(b_all, 0, 0, 1, 2)

        windows = [("0 - 5s", 0, 5), ("5 - 10s", 5, 10), ("10 - 15s", 10, 15),
                   ("15 - 20s", 15, 20), ("20 - 25s", 20, 25), ("25 - 30s", 25, 30)]
        for i, (name, s, e) in enumerate(windows):
            b = QPushButton(name)
            b.setStyleSheet(self._btn_style("#334155"))
            b.clicked.connect(lambda _, start=s, end=e: self._p2_set_window(start, end))
            pl.addWidget(b, 1 + i // 2, i % 2)

        self.btn_p2_glitch = QPushButton("Jump to Peak Disturbance")
        self.btn_p2_glitch.setStyleSheet(self._btn_style("#b91c1c"))
        self.btn_p2_glitch.clicked.connect(self._p2_jump_to_glitch)
        pl.addWidget(self.btn_p2_glitch, 4, 0, 1, 2)
        layout.addWidget(pr_grp)

        # 3. Quality Metrics Card
        stat_grp = QGroupBox("Fusion Quality Assessment")
        stat_grp.setStyleSheet(self._grp_style("#10b981"))
        sl = QVBoxLayout(stat_grp)
        sl.setSpacing(4)

        self.lbl_p2_rmse_enc = QLabel("RMSE Encoder: --")
        self.lbl_p2_rmse_enc.setStyleSheet(f"color: {COLORS['enc']}; font-size: 11px;")
        sl.addWidget(self.lbl_p2_rmse_enc)

        self.lbl_p2_rmse_ahrs = QLabel("RMSE AHRS:    --")
        self.lbl_p2_rmse_ahrs.setStyleSheet(f"color: {COLORS['ahrs']}; font-size: 11px;")
        sl.addWidget(self.lbl_p2_rmse_ahrs)

        self.lbl_p2_rmse_fused = QLabel("RMSE Fused:   --")
        self.lbl_p2_rmse_fused.setStyleSheet(f"color: {COLORS['fused']}; font-weight: bold; font-size: 12px;")
        sl.addWidget(self.lbl_p2_rmse_fused)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine); sep3.setStyleSheet("color: #334155;"); sl.addWidget(sep3)

        self.lbl_p2_imp_enc = QLabel("Fused vs Enc:  --")
        self.lbl_p2_imp_enc.setStyleSheet("color: #a7f3d0; font-size: 11px;")
        sl.addWidget(self.lbl_p2_imp_enc)

        self.lbl_p2_imp_ahrs = QLabel("Fused vs AHRS: --")
        self.lbl_p2_imp_ahrs.setStyleSheet("color: #a7f3d0; font-size: 11px;")
        sl.addWidget(self.lbl_p2_imp_ahrs)

        layout.addWidget(stat_grp)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

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

        self._update_3d_antenna()
        self._update_live_plots()
        self._update_live_status()

    def _update_3d_antenna(self):
        fk = compute_fk_matrices(
            self.sim.axes['azimuth'].true_angle,
            self.sim.axes['elevation'].true_angle,
            self.sim.axes['polarization'].true_angle
        )
        for name in AXES:
            self.meshes[name].points[:] = apply_transform(self.orig_pts[name], fk[name])
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
            self.live_status_lbls[ax].setText(
                f"{ax.upper()[:3]}: True {a.true_angle:5.1f}° | Fused {a.fused_angle:5.1f}° | Δ {a.fused_angle - a.true_angle:+5.2f}°"
            )

        va = self.live_view_ax
        a_v = self.sim.axes[va]
        self.live_conf_lbl.setText(
            f"Confidence ({va.capitalize()}): Enc {a_v.c_enc*100:4.1f}% | AHRS {a_v.c_ahrs*100:4.1f}%"
        )

        if self.sim.recording:
            n = len(self.sim.log_data)
            dur = n * self.sim.dt
            self.lbl_rec_status.setText(f"🔴 RECORDING: {n:,} samples ({dur:.1f}s)")

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_start_sim(self):
        self.sim.running = True
        self.status.showMessage("Simulation running.")

    def _on_stop_sim(self):
        self.sim.running = False
        self.status.showMessage("Simulation paused.")

    def _on_reset_sim(self):
        self.sim.reset()
        self.t_buf.clear()
        for ax in AXES:
            for q in self.buf[ax].values():
                q.clear()
        for crv in self.crv_live.values():
            crv.setData([], [])
        self._update_3d_antenna()
        self.status.showMessage("Simulation reset.")

    def _on_motion_mode_changed(self):
        mode = 'TARGET' if self.rb_mode_target.isChecked() else 'BROWNIAN'
        self.sim.set_motion_mode(mode)
        self.status.showMessage(f"Motion mode set to: {mode}")

    def _on_target_slider(self, axis, value, label):
        label.setText(f"{value:3d}°")
        self.sim.set_target(axis, float(value))

    def _on_fault_axis_changed(self, text):
        mapping = {"Azimuth": "azimuth", "Elevation": "elevation", "Polarization": "polarization", "All Axes": "all"}
        self.fault_target_ax = mapping.get(text, "azimuth")

    def _on_enc_noise_slider(self, val):
        sigma = val / 10.0
        self.lbl_noise_enc.setText(f"{sigma:.2f}°")
        if self.fault_target_ax == "all":
            self.sim.set_global_noise(sigma_enc=sigma)
        else:
            self.sim.set_noise(self.fault_target_ax, sigma_enc=sigma)

    def _on_ahrs_noise_slider(self, val):
        sigma = val / 10.0
        self.lbl_noise_ahrs.setText(f"{sigma:.2f}°")
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
        self._update_live_plots()

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

    def _on_toggle_recording(self, checked):
        if checked:
            self.btn_rec_toggle.setText("■ STOP RECORDING")
            self.btn_rec_toggle.setStyleSheet(self._btn_style("#dc2626"))
            self.sim.clear_recorded_data()
            self.sim.start_recording()
            self.status.showMessage("Data logging started.")
        else:
            self.btn_rec_toggle.setText("🔴 START RECORDING")
            self.btn_rec_toggle.setStyleSheet(self._btn_style("#991b1b"))
            csv_path = self.sim.stop_recording()
            n = len(self.sim.log_data)
            self.lbl_rec_status.setText(f"Saved {n:,} samples to {os.path.basename(csv_path)}")
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
            # Look for newest sim_antenna_*.csv
            candidates = glob.glob("sim_antenna_*.csv")
            if not candidates:
                candidates = glob.glob("sim_timeseries_*.csv")
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                self._load_p2_csv(candidates[0])
            else:
                self.status.showMessage("No recorded CSV found. Run a simulation and record first.")

    def _on_p2_open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Simulation CSV", "", "CSV Files (*.csv);;All Files (*)")
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

        # Map columns (handle multi-axis format or legacy 1-axis format)
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
            # Fallback to single-axis columns (development.ipynb / csv_logger.py)
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

        self.lbl_p2_rmse_enc.setText(f"RMSE Encoder: {rmse_enc:.3f}°")
        self.lbl_p2_rmse_ahrs.setText(f"RMSE AHRS:    {rmse_ahrs:.3f}°")
        self.lbl_p2_rmse_fused.setText(f"RMSE Fused:   {rmse_fused:.3f}°")

        if rmse_enc > 0:
            self.lbl_p2_imp_enc.setText(f"Fused vs Enc:  {(1 - rmse_fused / rmse_enc) * 100:+.1f}% error")
        if rmse_ahrs > 0:
            self.lbl_p2_imp_ahrs.setText(f"Fused vs AHRS: {(1 - rmse_fused / rmse_ahrs) * 100:+.1f}% error")

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

    def _on_p2_mouse_moved(self, pos):
        if self.insp_df is None:
            return

        mouse_pt = None
        for plot in [self.p2_ang, self.p2_conf, self.p2_err]:
            if plot.sceneBoundingRect().contains(pos):
                mouse_pt = plot.vb.mapSceneToView(pos)
                break

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

        self.lbl_insp_time.setText(f"Time:  {snapped_t:06.3f}s (pt #{idx})")
        self.lbl_insp_true.setText(f"True:   {th_true:+7.2f}°")
        self.lbl_insp_enc.setText(f"Enc:    {th_enc:+7.2f}°  (Δ: {e_enc:+5.2f}°)")
        self.lbl_insp_ahrs.setText(f"AHRS:   {th_ahrs:+7.2f}°  (Δ: {e_ahrs:+5.2f}°)")
        self.lbl_insp_fused.setText(f"Fused:  {th_fused:+7.2f}°  (Δ: {e_fused:+5.2f}°)")
        self.lbl_insp_conf.setText(f"Confidence: Enc {c_enc*100:4.1f}% | AHRS {c_ahrs*100:4.1f}%")

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
    # STYLESHEET HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _grp_style(self, accent_color):
        return f"""
            QGroupBox {{
                color: {accent_color}; font-weight: bold; font-size: 11px;
                border: 1px solid #1e293b; border-radius: 6px;
                margin-top: 10px; padding-top: 10px; background-color: #0f172a;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 6px; left: 10px;
            }}
        """

    def _btn_style(self, bg_color):
        return f"""
            QPushButton {{
                background-color: {bg_color}; color: #ffffff;
                border: none; border-radius: 4px; padding: 6px 12px;
                font-weight: bold; font-size: 11px;
            }}
            QPushButton:hover {{ opacity: 0.85; filter: brightness(115%); }}
            QPushButton:pressed {{ background-color: #0284c7; }}
        """


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = AntennaFusionApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
