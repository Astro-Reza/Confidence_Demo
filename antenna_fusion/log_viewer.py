"""
Interactive Sensor Fusion CSV Viewer
====================================
GUI application to inspect, pan, zoom, and drag time windows across time-series
data exported from `notebooks/fusion_development.ipynb` (via `csv_export.py`).

Features:
- Multi-plot layout:
    1. Angle Tracking (True, Encoder, AHRS, Fused)
    2. Confidence Weights (Encoder confidence, AHRS confidence)
    3. Tracking Errors (Encoder error, AHRS error, Fused error)
    4. Overview timeline strip with draggable / resizable Time Window (LinearRegionItem)
- Synchronized X-axis (pan / zoom in one plot updates all)
- Vertical crosshair tracking mouse position with real-time value HUD
- Preset time window buttons (0-5s, 5-10s, Fit All, and "Jump to Glitch/Drift")
- Curve visibility toggles (show/hide individual sensors)
- Summary statistics card (RMSE, Max Error, % improvement)
- File picker dialog & auto-load latest CSV in logs/

Usage:
    python -m antenna_fusion.log_viewer [optional_path_to_file.csv]   (or run_log_viewer.bat)
"""

import os
import sys
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QFileDialog,
    QGroupBox, QFrame, QSplitter, QStatusBar, QSizePolicy,
    QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg

from .paths import LOGS_DIR, TIMESERIES_PATTERN, find_latest_log

# Configure PyQtGraph dark theme & antialiasing
pg.setConfigOption('background', '#0f172a')  # Slate 900
pg.setConfigOption('foreground', '#e2e8f0')  # Slate 200
pg.setConfigOptions(antialias=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'true':       '#94a3b8',   # Slate 400 (Dashed reference)
    'enc':        '#f97316',   # Bright Orange (Encoder)
    'ahrs':       '#38bdf8',   # Sky Blue (AHRS)
    'fused':      '#10b981',   # Emerald Green (Fused output)
    'conf_enc':   '#fb923c',   # Light Orange
    'conf_ahrs':  '#7dd3fc',   # Light Blue
    'err_enc':    '#ea580c',   # Deep Orange
    'err_ahrs':   '#0284c7',   # Deep Cyan
    'err_fused':  '#059669',   # Deep Green
    'grid':       '#334155',   # Slate 700
    'crosshair':  '#facc15',   # Yellow crosshair
    'region':     '#38bdf844', # Translucent cyan for region selection
}


class CSVViewerWindow(QMainWindow):
    def __init__(self, initial_csv=None):
        super().__init__()
        self.setWindowTitle("Sensor Fusion Timeseries Viewer")
        self.resize(1360, 880)

        self.df = None
        self.filepath = None
        self.curves = {}
        self.crosshair_vlines = []
        self.glitch_idx = None
        self._updating_region = False

        self._init_ui()

        # Load initial file or look for latest CSV
        if initial_csv and os.path.exists(initial_csv):
            self.load_csv(initial_csv)
        else:
            latest = self._find_latest_csv()
            if latest:
                self.load_csv(latest)
            else:
                self.statusBar().showMessage("No CSV file loaded. Click 'Open CSV...' to select data.")

    # ─────────────────────────────────────────────────────────────────────────
    # UI SETUP
    # ─────────────────────────────────────────────────────────────────────────
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── Left / Main Content: Plot Area ──────────────────────────────────
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)

        # Graphics layout widget for multi-panel synchronized plots
        self.gw = pg.GraphicsLayoutWidget()
        self.gw.ci.layout.setContentsMargins(4, 4, 4, 4)
        self.gw.ci.layout.setSpacing(4)
        plot_layout.addWidget(self.gw, stretch=1)

        # Overview Strip (for dragging time window)
        overview_header = QLabel("<b>TIMELINE OVERVIEW & DRAGGABLE TIME WINDOW:</b> (Drag the highlighted window or its edges)")
        overview_header.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 4px;")
        plot_layout.addWidget(overview_header)

        self.plt_overview = pg.PlotWidget(name="Overview")
        self.plt_overview.setFixedHeight(75)
        self.plt_overview.showGrid(x=True, y=True, alpha=0.15)
        self.plt_overview.setMouseEnabled(x=False, y=False)
        self.plt_overview.getPlotItem().hideAxis('left')
        self.plt_overview.getPlotItem().hideAxis('bottom')
        self.region = pg.LinearRegionItem([0, 5], brush=pg.mkBrush(COLORS['region']))
        self.region.setZValue(10)
        self.plt_overview.addItem(self.region)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        plot_layout.addWidget(self.plt_overview)

        # ── Right Sidebar: Controls & Inspector ─────────────────────────────
        sidebar = self._create_sidebar()

        # Splitter to allow resizing sidebar vs plots
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(plot_container)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # ── Build Plots in GraphicsLayout ────────────────────────────────────
        self._setup_plots()

        # Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setStyleSheet("color: #cbd5e1; background: #0b1120;")

    def _create_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea { border: 1px solid #1e293b; background: #0b1120; border-radius: 6px; }
            QWidget#sidebarContent { background: #0b1120; }
        """)

        container = QWidget()
        container.setObjectName("sidebarContent")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # 1. File Group
        file_grp = QGroupBox("Data Source")
        file_grp.setStyleSheet(self._grp_style("#38bdf8"))
        fl = QVBoxLayout(file_grp)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("Open CSV...")
        self.btn_open.setStyleSheet(self._btn_style("#0284c7"))
        self.btn_open.clicked.connect(self._on_open_file)
        btn_row.addWidget(self.btn_open)

        self.btn_reload = QPushButton("Reload")
        self.btn_reload.setStyleSheet(self._btn_style("#334155"))
        self.btn_reload.clicked.connect(self._on_reload)
        btn_row.addWidget(self.btn_reload)
        fl.addLayout(btn_row)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color: #94a3b8; font-size: 11px; font-family: monospace;")
        fl.addWidget(self.lbl_file)

        self.lbl_rows = QLabel("Rows: - | Duration: -")
        self.lbl_rows.setStyleSheet("color: #64748b; font-size: 10px;")
        fl.addWidget(self.lbl_rows)
        layout.addWidget(file_grp)

        # 2. Live Cursor Inspector Group
        insp_grp = QGroupBox("Live Cursor Inspector")
        insp_grp.setStyleSheet(self._grp_style("#facc15"))
        il = QVBoxLayout(insp_grp)
        il.setSpacing(4)

        self.insp_time = QLabel("Time:  --- s")
        self.insp_time.setStyleSheet("color: #facc15; font-weight: bold; font-size: 13px;")
        il.addWidget(self.insp_time)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine); sep1.setStyleSheet("color: #334155;"); il.addWidget(sep1)

        self.insp_true = QLabel("True:   --- °")
        self.insp_true.setStyleSheet(f"color: {COLORS['true']};")
        il.addWidget(self.insp_true)

        self.insp_enc = QLabel("Enc:    --- °  (err: ---°)")
        self.insp_enc.setStyleSheet(f"color: {COLORS['enc']};")
        il.addWidget(self.insp_enc)

        self.insp_ahrs = QLabel("AHRS:   --- °  (err: ---°)")
        self.insp_ahrs.setStyleSheet(f"color: {COLORS['ahrs']};")
        il.addWidget(self.insp_ahrs)

        self.insp_fused = QLabel("Fused:  --- °  (err: ---°)")
        self.insp_fused.setStyleSheet(f"color: {COLORS['fused']}; font-weight: bold;")
        il.addWidget(self.insp_fused)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine); sep2.setStyleSheet("color: #334155;"); il.addWidget(sep2)

        self.insp_conf = QLabel("Confidence: Enc: -% | AHRS: -%")
        self.insp_conf.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        il.addWidget(self.insp_conf)

        layout.addWidget(insp_grp)

        # 3. Quick Window Presets
        preset_grp = QGroupBox("Time Window Presets")
        preset_grp.setStyleSheet(self._grp_style("#a855f7"))
        pl = QGridLayout(preset_grp)
        pl.setSpacing(6)

        b_all = QPushButton("Fit All (Full)")
        b_all.setStyleSheet(self._btn_style("#475569"))
        b_all.clicked.connect(self._zoom_all)
        pl.addWidget(b_all, 0, 0, 1, 2)

        windows = [("0 - 5s", 0, 5), ("5 - 10s", 5, 10), ("10 - 15s", 10, 15),
                   ("15 - 20s", 15, 20), ("20 - 25s", 20, 25), ("25 - 30s", 25, 30)]
        for i, (name, s, e) in enumerate(windows):
            b = QPushButton(name)
            b.setStyleSheet(self._btn_style("#334155"))
            b.clicked.connect(lambda _, start=s, end=e: self._set_window(start, end))
            pl.addWidget(b, 1 + i // 2, i % 2)

        self.btn_glitch = QPushButton("Jump to Drift/Glitch")
        self.btn_glitch.setStyleSheet(self._btn_style("#b91c1c"))
        self.btn_glitch.clicked.connect(self._jump_to_glitch)
        pl.addWidget(self.btn_glitch, 4, 0, 1, 2)

        layout.addWidget(preset_grp)

        # 4. Curve Visibility Toggles
        vis_grp = QGroupBox("Visible Signals")
        vis_grp.setStyleSheet(self._grp_style("#10b981"))
        vl = QVBoxLayout(vis_grp)
        vl.setSpacing(3)

        self.checks = {}
        items = [
            ('true',      "Ground Truth (True)", True,  COLORS['true']),
            ('enc',       "Encoder (Raw / Drift)", True, COLORS['enc']),
            ('ahrs',      "AHRS (Noisy)",         True, COLORS['ahrs']),
            ('fused',     "Fused Angle",          True, COLORS['fused']),
            ('conf_enc',  "Confidence: Encoder",  True, COLORS['conf_enc']),
            ('conf_ahrs', "Confidence: AHRS",     True, COLORS['conf_ahrs']),
            ('err_enc',   "Error: Encoder",       True, COLORS['err_enc']),
            ('err_ahrs',  "Error: AHRS",          True, COLORS['err_ahrs']),
            ('err_fused', "Error: Fused",         True, COLORS['err_fused']),
        ]
        for key, label, default, color in items:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; font-size: 11px; }} QCheckBox::indicator {{ width: 14px; height: 14px; }}")
            cb.stateChanged.connect(self._update_visibility)
            vl.addWidget(cb)
            self.checks[key] = cb

        layout.addWidget(vis_grp)

        # 5. Accuracy & Performance Statistics
        stat_grp = QGroupBox("Performance Metrics")
        stat_grp.setStyleSheet(self._grp_style("#6366f1"))
        sl = QVBoxLayout(stat_grp)
        sl.setSpacing(4)

        self.lbl_rmse_enc = QLabel("RMSE Encoder: --")
        self.lbl_rmse_enc.setStyleSheet(f"color: {COLORS['enc']}; font-size: 11px;")
        sl.addWidget(self.lbl_rmse_enc)

        self.lbl_rmse_ahrs = QLabel("RMSE AHRS:    --")
        self.lbl_rmse_ahrs.setStyleSheet(f"color: {COLORS['ahrs']}; font-size: 11px;")
        sl.addWidget(self.lbl_rmse_ahrs)

        self.lbl_rmse_fused = QLabel("RMSE Fused:   --")
        self.lbl_rmse_fused.setStyleSheet(f"color: {COLORS['fused']}; font-weight: bold; font-size: 12px;")
        sl.addWidget(self.lbl_rmse_fused)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine); sep3.setStyleSheet("color: #334155;"); sl.addWidget(sep3)

        self.lbl_improv_enc = QLabel("Fused vs Enc:  --")
        self.lbl_improv_enc.setStyleSheet("color: #a7f3d0; font-size: 11px;")
        sl.addWidget(self.lbl_improv_enc)

        self.lbl_improv_ahrs = QLabel("Fused vs AHRS: --")
        self.lbl_improv_ahrs.setStyleSheet("color: #a7f3d0; font-size: 11px;")
        sl.addWidget(self.lbl_improv_ahrs)

        layout.addWidget(stat_grp)

        # Controls hint
        hint = QLabel("<b>Mouse Controls:</b><br>"
                      "• <b>Left-drag:</b> Pan view<br>"
                      "• <b>Right-drag / Wheel:</b> Zoom<br>"
                      "• <b>Hover:</b> Crosshair readout<br>"
                      "• <b>Bottom strip:</b> Drag window")
        hint.setStyleSheet("color: #64748b; font-size: 10px; line-height: 140%;")
        layout.addWidget(hint)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _setup_plots(self):
        # Panel 1: Angle Tracking
        self.p_ang = self.gw.addPlot(row=0, col=0)
        self.p_ang.setTitle("<span style='font-size: 12pt; color: #f8fafc; font-weight: bold;'>1. Angle Tracking: Ground Truth vs Sensors vs Fused</span>")
        self.p_ang.setLabel('left', 'Angle', units='deg')
        self.p_ang.showGrid(x=True, y=True, alpha=0.18)
        self.p_ang.addLegend(offset=(-10, 10))

        # Panel 2: Confidences
        self.p_conf = self.gw.addPlot(row=1, col=0)
        self.p_conf.setTitle("<span style='font-size: 11pt; color: #f8fafc; font-weight: bold;'>2. Confidence Scoring Weights (c_enc vs c_ahrs)</span>")
        self.p_conf.setLabel('left', 'Confidence')
        self.p_conf.setYRange(-0.05, 1.05, padding=0)
        self.p_conf.showGrid(x=True, y=True, alpha=0.18)
        self.p_conf.addLegend(offset=(-10, 10))

        # Panel 3: Errors
        self.p_err = self.gw.addPlot(row=2, col=0)
        self.p_err.setTitle("<span style='font-size: 11pt; color: #f8fafc; font-weight: bold;'>3. Tracking Errors (Sensor - Ground Truth)</span>")
        self.p_err.setLabel('left', 'Error', units='deg')
        self.p_err.setLabel('bottom', 'Time', units='s')
        self.p_err.showGrid(x=True, y=True, alpha=0.18)
        self.p_err.addLegend(offset=(-10, 10))

        # Zero reference line for error plot
        self.err_zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#475569', style=Qt.DashLine, width=1))
        self.p_err.addItem(self.err_zero)

        # Synchronize X-axes across all three plots
        self.p_conf.setXLink(self.p_ang)
        self.p_err.setXLink(self.p_ang)

        # When the user zooms/pans the main plots, update the overview region
        self.p_ang.sigRangeChanged.connect(self._on_plot_range_changed)

        # Create synchronized vertical crosshair lines
        self.crosshair_vlines = [
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS['crosshair'], width=1.2, style=Qt.DashLine)),
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS['crosshair'], width=1.2, style=Qt.DashLine)),
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS['crosshair'], width=1.2, style=Qt.DashLine)),
        ]
        for plot, vline in zip([self.p_ang, self.p_conf, self.p_err], self.crosshair_vlines):
            vline.setZValue(100)
            vline.setVisible(False)
            plot.addItem(vline)

        # Connect mouse motion on the graphics layout scene
        self.gw.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # Create Plot Curves (empty until data is loaded)
        self.curves['true'] = self.p_ang.plot(
            pen=pg.mkPen(COLORS['true'], width=2, style=Qt.DashLine), name="Ground Truth"
        )
        self.curves['enc'] = self.p_ang.plot(
            pen=pg.mkPen(color=(249, 115, 22, 120), width=1.2), name="Encoder"
        )
        self.curves['ahrs'] = self.p_ang.plot(
            pen=pg.mkPen(color=(56, 189, 248, 120), width=1.2), name="AHRS"
        )
        self.curves['fused'] = self.p_ang.plot(
            pen=pg.mkPen(COLORS['fused'], width=2.5), name="Quaternion Fused"
        )

        self.curves['conf_enc'] = self.p_conf.plot(
            pen=pg.mkPen(COLORS['conf_enc'], width=1.8), name="Encoder Confidence"
        )
        self.curves['conf_ahrs'] = self.p_conf.plot(
            pen=pg.mkPen(COLORS['conf_ahrs'], width=1.8), name="AHRS Confidence"
        )

        self.curves['err_enc'] = self.p_err.plot(
            pen=pg.mkPen(color=(234, 88, 12, 140), width=1.2), name="Encoder Error"
        )
        self.curves['err_ahrs'] = self.p_err.plot(
            pen=pg.mkPen(color=(2, 132, 199, 140), width=1.2), name="AHRS Error"
        )
        self.curves['err_fused'] = self.p_err.plot(
            pen=pg.mkPen(COLORS['err_fused'], width=2.2), name="Fused Error"
        )

        # Overview curves (simplified)
        self.curve_overview = self.plt_overview.plot(pen=pg.mkPen('#64748b', width=1))

    # ─────────────────────────────────────────────────────────────────────────
    # DATA LOADING & PARSING
    # ─────────────────────────────────────────────────────────────────────────
    def load_csv(self, filepath):
        try:
            self.statusBar().showMessage(f"Loading {filepath}...")
            df = pd.read_csv(filepath)

            # Validate minimum required columns
            req = ['time_s', 'theta_true_deg', 'theta_enc_deg', 'theta_ahrs_deg', 'theta_fused_deg']
            missing = [c for c in req if c not in df.columns]
            if missing:
                self.statusBar().showMessage(f"Error: Missing columns {missing}")
                return

            self.df = df
            self.filepath = os.path.abspath(filepath)
            filename = os.path.basename(filepath)
            self.lbl_file.setText(filename)
            self.lbl_file.setToolTip(filepath)

            # Compute error columns if missing
            if 'error_enc_deg' not in df.columns:
                df['error_enc_deg'] = df['theta_enc_deg'] - df['theta_true_deg']
            if 'error_ahrs_deg' not in df.columns:
                df['error_ahrs_deg'] = df['theta_ahrs_deg'] - df['theta_true_deg']
            if 'error_fused_deg' not in df.columns:
                df['error_fused_deg'] = df['theta_fused_deg'] - df['theta_true_deg']

            # Compute confidences if missing
            if 'conf_enc' not in df.columns:
                df['conf_enc'] = 0.5
            if 'conf_ahrs' not in df.columns:
                df['conf_ahrs'] = 0.5

            t = df['time_s'].values
            n_rows = len(t)
            duration = t[-1] - t[0] if n_rows > 1 else 0
            dt = t[1] - t[0] if n_rows > 1 else 0.01

            self.lbl_rows.setText(f"Rows: {n_rows:,} | Duration: {duration:.2f}s (dt: {dt*1000:.1f}ms)")

            # Update curves data
            self.curves['true'].setData(t, df['theta_true_deg'].values)
            self.curves['enc'].setData(t, df['theta_enc_deg'].values)
            self.curves['ahrs'].setData(t, df['theta_ahrs_deg'].values)
            self.curves['fused'].setData(t, df['theta_fused_deg'].values)

            self.curves['conf_enc'].setData(t, df['conf_enc'].values)
            self.curves['conf_ahrs'].setData(t, df['conf_ahrs'].values)

            self.curves['err_enc'].setData(t, df['error_enc_deg'].values)
            self.curves['err_ahrs'].setData(t, df['error_ahrs_deg'].values)
            self.curves['err_fused'].setData(t, df['error_fused_deg'].values)

            # Overview data (fused angle)
            self.curve_overview.setData(t, df['theta_fused_deg'].values)
            self.plt_overview.getPlotItem().setXRange(t[0], t[-1], padding=0)

            # Locate glitch/drift (sample with maximum encoder absolute error)
            err_enc_abs = np.abs(df['error_enc_deg'].values)
            self.glitch_idx = int(np.argmax(err_enc_abs))

            # Update Statistics
            self._update_stats()

            # Set initial view to first 6 seconds or full range if shorter
            initial_end = min(t[0] + 6.0, t[-1])
            self.region.setRegion([t[0], initial_end])
            self.p_ang.setXRange(t[0], initial_end, padding=0)

            self.statusBar().showMessage(f"Successfully loaded {n_rows:,} samples from {filename}")

        except Exception as e:
            self.statusBar().showMessage(f"Failed to load CSV: {str(e)}")

    def _update_stats(self):
        if self.df is None:
            return

        err_enc = self.df['error_enc_deg'].values
        err_ahrs = self.df['error_ahrs_deg'].values
        err_fused = self.df['error_fused_deg'].values

        rmse_enc = np.sqrt(np.mean(err_enc ** 2))
        rmse_ahrs = np.sqrt(np.mean(err_ahrs ** 2))
        rmse_fused = np.sqrt(np.mean(err_fused ** 2))

        self.lbl_rmse_enc.setText(f"RMSE Encoder: {rmse_enc:.3f}°")
        self.lbl_rmse_ahrs.setText(f"RMSE AHRS:    {rmse_ahrs:.3f}°")
        self.lbl_rmse_fused.setText(f"RMSE Fused:   {rmse_fused:.3f}°")

        if rmse_enc > 0:
            imp_enc = (1 - rmse_fused / rmse_enc) * 100
            self.lbl_improv_enc.setText(f"Fused vs Enc:  {imp_enc:+.1f}% error")
        if rmse_ahrs > 0:
            imp_ahrs = (1 - rmse_fused / rmse_ahrs) * 100
            self.lbl_improv_ahrs.setText(f"Fused vs AHRS: {imp_ahrs:+.1f}% error")

    # ─────────────────────────────────────────────────────────────────────────
    # INTERACTION HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_region_changed(self):
        """Called when user drags or resizes the overview region item."""
        if self._updating_region:
            return
        self._updating_region = True
        try:
            min_x, max_x = self.region.getRegion()
            self.p_ang.setXRange(min_x, max_x, padding=0)
        finally:
            self._updating_region = False

    def _on_plot_range_changed(self):
        """Called when user pans or zooms in any of the main plots."""
        if self._updating_region:
            return
        self._updating_region = True
        try:
            view_range = self.p_ang.viewRange()
            min_x, max_x = view_range[0]
            self.region.setRegion([min_x, max_x])
        finally:
            self._updating_region = False

    def _on_mouse_moved(self, pos):
        """Crosshair tracking and real-time live value inspection."""
        if self.df is None:
            return

        # Find which plot the cursor is over
        mouse_point = None
        for plot in [self.p_ang, self.p_conf, self.p_err]:
            if plot.sceneBoundingRect().contains(pos):
                mouse_point = plot.vb.mapSceneToView(pos)
                break

        if mouse_point is None:
            for vline in self.crosshair_vlines:
                vline.setVisible(False)
            return

        x_val = mouse_point.x()
        t_arr = self.df['time_s'].values

        # Check bounds
        if x_val < t_arr[0] or x_val > t_arr[-1]:
            for vline in self.crosshair_vlines:
                vline.setVisible(False)
            return

        # Snap to nearest timestamp
        idx = int(np.searchsorted(t_arr, x_val))
        if idx >= len(t_arr):
            idx = len(t_arr) - 1
        elif idx > 0 and (x_val - t_arr[idx - 1]) < (t_arr[idx] - x_val):
            idx = idx - 1

        snapped_t = t_arr[idx]

        # Update vertical lines
        for vline in self.crosshair_vlines:
            vline.setPos(snapped_t)
            vline.setVisible(True)

        # Extract values
        row = self.df.iloc[idx]
        t = row['time_s']
        th_true = row['theta_true_deg']
        th_enc = row['theta_enc_deg']
        th_ahrs = row['theta_ahrs_deg']
        th_fused = row['theta_fused_deg']
        c_enc = row['conf_enc']
        c_ahrs = row['conf_ahrs']
        e_enc = row['error_enc_deg']
        e_ahrs = row['error_ahrs_deg']
        e_fused = row['error_fused_deg']

        # Update inspector labels
        self.insp_time.setText(f"Time:  {t:06.3f}s  (pt #{idx})")
        self.insp_true.setText(f"True:   {th_true:+7.2f}°")
        self.insp_enc.setText(f"Enc:    {th_enc:+7.2f}°  (Δ {e_enc:+5.2f}°)")
        self.insp_ahrs.setText(f"AHRS:   {th_ahrs:+7.2f}°  (Δ {e_ahrs:+5.2f}°)")
        self.insp_fused.setText(f"Fused:  {th_fused:+7.2f}°  (Δ {e_fused:+5.2f}°)")
        self.insp_conf.setText(f"Confidence: Enc {c_enc*100:4.1f}% | AHRS {c_ahrs*100:4.1f}%")

    def _update_visibility(self):
        """Toggle curve visibility based on checkboxes."""
        for key, cb in self.checks.items():
            if key in self.curves:
                self.curves[key].setVisible(cb.isChecked())

    def _set_window(self, start_s, end_s):
        """Set the active view window in seconds."""
        if self.df is None:
            return
        t = self.df['time_s'].values
        start_s = max(t[0], start_s)
        end_s = min(t[-1], end_s)
        self.region.setRegion([start_s, end_s])
        self.p_ang.setXRange(start_s, end_s, padding=0)

    def _zoom_all(self):
        """Fit all data into view."""
        if self.df is None:
            return
        t = self.df['time_s'].values
        self._set_window(t[0], t[-1])

    def _jump_to_glitch(self):
        """Center a 4-second time window around the largest drift/glitch event."""
        if self.df is None or self.glitch_idx is None:
            return
        t_glitch = self.df['time_s'].iloc[self.glitch_idx]
        self._set_window(t_glitch - 2.0, t_glitch + 2.0)
        self.statusBar().showMessage(f"Jumped to peak disturbance event at t = {t_glitch:.2f}s")

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Simulation CSV", LOGS_DIR, "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.load_csv(path)

    def _on_reload(self):
        if self.filepath and os.path.exists(self.filepath):
            self.load_csv(self.filepath)
        else:
            latest = self._find_latest_csv()
            if latest:
                self.load_csv(latest)

    def _find_latest_csv(self):
        """Find the newest sim_timeseries_*.csv (else any CSV) in logs/."""
        return find_latest_log(TIMESERIES_PATTERN, '*.csv')

    # ─────────────────────────────────────────────────────────────────────────
    # STYLESHEET HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _grp_style(self, accent_color):
        return f"""
            QGroupBox {{
                color: {accent_color};
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #1e293b;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #0f172a;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                left: 10px;
            }}
        """

    def _btn_style(self, bg_color):
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{
                opacity: 0.85;
                filter: brightness(115%);
            }}
            QPushButton:pressed {{
                background-color: #0284c7;
            }}
        """


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Optional CLI file argument
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    window = CSVViewerWindow(initial_csv=csv_file)
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
