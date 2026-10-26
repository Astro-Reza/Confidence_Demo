"""
Antenna Simulation & Confidence-Score Sensor Fusion Core
========================================================
Implements:
1. 3-Axis Kinematics (Azimuth, Elevation, Polarization) with forward
   kinematics transformation matrices for 3D STL mesh assembly.
2. Dual Motion Models:
   - Target Chasing (Cascade PID with motor plant, friction, and gravity load factor)
   - Brownian Random Motion with damping and reflective mechanical boundaries (from `notebooks/fusion_development.ipynb`)
3. Real-Time Sensor Modeling with dynamic noise and fault injection:
   - Encoder: base Gaussian noise σ_enc, drift ramp, and electrical glitch offset
   - AHRS: base Gaussian noise σ_ahrs (zero drift)
4. Real-Time Causal Confidence-Score Quaternion Fusion (from `notebooks/fusion_development.ipynb`):
   - Sliding window detrended variance calculation (isolates noise from physical motion)
   - Cross-sensor divergence trend filter and drift penalty
   - Inverse variance weighting and normalized confidence weights
   - Confidence-weighted quaternion fusion
5. Extended Kalman Filter baseline (angle, rate, encoder bias) run on the same readings for comparison
6. Real-Time Data Recording & Multi-Axis CSV Export
7. Virtual limit switches at both ends of each axis' operational range (AX_LIMITS)
8. Startup limit-switch calibration: the antenna is installed with its azimuth limit seam on
   true north (lo stop = 0 deg, hi stop = 360 deg), up to a small unknown installation error. It
   drives every axis into both limit switches and reads the AHRS (world frame) at each stop to
   verify its operational range and measure that error. Readings and targets are world angles;
   azimuth stays continuous over the full lo..hi range, so it never wraps inside the range.
"""

import math
import os
import random
from collections import deque
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation

from .paths import CAD_DIR, LOGS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# KINEMATICS & AXES CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
STL_FILES = {
    'azimuth':      os.path.join(CAD_DIR, 'azimuth_body.stl'),
    'elevation':    os.path.join(CAD_DIR, 'elevation_body.stl'),
    'polarization': os.path.join(CAD_DIR, 'polarization_body.stl'),
}

LINKS_CONFIG = {
    'azimuth':     {'pivot': np.array([0., 0.,   0.]), 'axis': np.array([0., 0., 1.]), 'color': '#64748b'},
    'elevation':   {'pivot': np.array([0., 0.,  95.]), 'axis': np.array([1., 0., 0.]), 'color': '#38bdf8'},
    'polarization':{'pivot': np.array([0., 135., 170.]), 'axis': np.array([0., 0., 1.]), 'color': '#fb923c'},
}

AXES = ['azimuth', 'elevation', 'polarization']

AX_LIMITS = {
    'azimuth':      (0.0, 360.0),
    'elevation':    (0.0, 90.0),
    'polarization': (0.0, 180.0),
}

AX_COLORS = {
    'azimuth':      '#38bdf8',  # Sky Blue
    'elevation':    '#4ade80',  # Green
    'polarization': '#fb923c',  # Amber/Orange
}

# Virtual limit switches at both ends of each axis' operational range
LIMIT_SW_BAND = 0.25    # actuation zone inside the limit (°): switch closes once the axis is this close
LIMIT_SW_HYST = 0.25    # release differential (°): switch reopens at BAND + HYST, so it can't chatter

# Startup calibration
INITIAL_POSE = {'azimuth': 180.0, 'elevation': 45.0, 'polarization': 90.0}  # mechanical angles at power-on (°)
CAL_RATE  = {'azimuth': 45.0, 'elevation': 18.0, 'polarization': 30.0}      # slew toward a limit (°/s)
CAL_CREEP = 3.0         # approach speed once the switch has closed, until the hard stop (°/s)
CAL_DWELL = 50          # AHRS samples averaged at each stop (1 s at 50 Hz)
MOUNT_TOL = 5.0         # installation error of the azimuth seam about true north (± deg)


def wrap180(deg: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0

BASE_STALL  = 28.0      # stall PWM at 0° elevation
K_MOTOR     = 0.060     # (°/s) per (PWM – stall) per load_factor (responsive slew response)
K_FRICTION  = 0.22      # velocity-proportional friction
STATIC_FRIC = 0.04      # static friction


def transform_matrix(pivot: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix around a pivot point."""
    r = Rotation.from_rotvec(np.radians(deg) * axis).as_matrix()
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = pivot - r @ pivot
    return m


def apply_transform(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Transform 3D point cloud using a 4x4 homogeneous transformation matrix."""
    h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    return (m @ h.T).T[:, :3].astype(np.float32)


def compute_fk_matrices(az_deg: float, el_deg: float, pol_deg: float, mount_deg: float = 0.0) -> dict:
    """
    Compute forward kinematics cumulative matrices for the 3-link antenna chain.
    Azimuth is a compass bearing (clockwise from north = +Y viewed from above), so the
    rotation about +Z is negated. `mount_deg` is the world heading of the mechanical zero.
    """
    m_az = transform_matrix(LINKS_CONFIG['azimuth']['pivot'],
                            LINKS_CONFIG['azimuth']['axis'], -(az_deg + mount_deg))
    m_el = transform_matrix(LINKS_CONFIG['elevation']['pivot'],
                            LINKS_CONFIG['elevation']['axis'], el_deg)
    m_pol = transform_matrix(LINKS_CONFIG['polarization']['pivot'],
                             LINKS_CONFIG['polarization']['axis'], pol_deg)

    t_az = m_az
    t_el = m_az @ m_el
    t_pol = m_az @ m_el @ m_pol

    return {
        'azimuth': t_az,
        'elevation': t_el,
        'polarization': t_pol,
    }


def load_factor(name: str, el_deg: float) -> float:
    """Gravity and mechanical load factor as a function of elevation."""
    r = math.radians(el_deg)
    if name == 'elevation':
        return max(0.30, math.cos(r))
    if name == 'azimuth':
        return max(0.75, 1.0 - 0.15 * math.sin(r))
    return max(0.65, 1.0 - 0.22 * math.sin(r))


# ─────────────────────────────────────────────────────────────────────────────
# CASCADE CONTROL SYSTEM & STALL DETECTOR (Plant Dynamics)
# ─────────────────────────────────────────────────────────────────────────────
class CascadeOuterLoop:
    """
    Cascade Outer Loop: Position Error -> Demanded Rate of Change (Velocity Demand).
    Generates high demanded rate of change when far from target (fast, responsive initial slew),
    and progressively reduces rate of change as the antenna approaches the target angle
    ('slower and slower') via kinematic deceleration profiling to prevent overshoot.
    """
    def __init__(self, v_max: float = 16.0, a_profile: float = 1.1, k_pos: float = 0.75):
        self.v_max = v_max
        self.a_profile = a_profile
        self.k_pos = k_pos
        self.kp = k_pos
        self.ki = 0.0
        self.kd = 0.0
        self.d_crit = (2.0 * self.a_profile) / (self.k_pos ** 2)
        self.prev_err = 0.0

    def compute(self, pos_err: float, dt: float) -> float:
        dist = abs(pos_err)
        if dist > self.d_crit:
            # Kinematic smooth deceleration curve: v = sqrt(2 * a * dist)
            v_profile = math.sqrt(2.0 * self.a_profile * dist)
        else:
            # Exponential smooth arrival: perfectly critically damped, zero overshoot
            v_profile = self.k_pos * dist

        dem_speed = min(self.v_max, v_profile)
        self.prev_err = pos_err
        return float(math.copysign(dem_speed, pos_err))

    def reset(self):
        self.prev_err = 0.0


class CascadeInnerLoop:
    """
    Cascade Inner Loop: Rate of Change -> Actuator PWM.
    Dynamically adjusts motor PWM based on rate of change:
    - Provides velocity feedforward to break stiction and sustain commanded rate.
    - Applies closed-loop rate error feedback (proportional-derivative).
    - As rate-of-change demand decreases, PWM reduces toward stall/zero levels,
      with active counter-torque / braking when decelerating into target lock.
    """
    def __init__(self, kp: float = 14.0, kd: float = 0.30):
        self.kp = kp
        self.ki = 0.0
        self.kd = kd
        self.prev_rate_err = 0.0

    def compute(self, rate_err: float, dt: float) -> float:
        rate_deriv = (rate_err - self.prev_rate_err) / max(dt, 1e-9)
        self.prev_rate_err = rate_err
        fb = self.kp * rate_err + self.kd * rate_deriv
        return float(fb)

    def reset(self):
        self.prev_rate_err = 0.0


class PID:
    def __init__(self, kp: float, ki: float, kd: float, lo: float, hi: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.lo, self.hi = lo, hi
        self.integ = self.prev_err = 0.0

    def compute(self, err: float, dt: float) -> float:
        self.integ += err * dt
        deriv = (err - self.prev_err) / max(dt, 1e-9)
        self.prev_err = err
        raw = self.kp * err + self.ki * self.integ + self.kd * deriv
        # Anti-windup clamping
        if raw > self.hi or raw < self.lo:
            self.integ -= err * dt
        return float(np.clip(raw, self.lo, self.hi))

    def reset(self):
        self.integ = self.prev_err = 0.0


class StallDetector:
    def __init__(self, window: int = 14, vel_thr: float = 0.25,
                 pwm_thr: float = 20.0, boost_amt: float = 24.0):
        self.window = window
        self.vel_thr = vel_thr
        self.pwm_thr = pwm_thr
        self.boost_amt = boost_amt
        self.vel_hist = deque(maxlen=window)
        self.boost = 0.0
        self.detected = False

    def update(self, vel: float, pwm: float) -> tuple[bool, float]:
        self.vel_hist.append(abs(vel))
        self.boost *= 0.87
        if len(self.vel_hist) == self.window:
            avg_vel = sum(self.vel_hist) / self.window
            self.detected = (avg_vel < self.vel_thr and pwm > self.pwm_thr)
            if self.detected:
                self.boost = self.boost_amt
        return self.detected, self.boost

    def reset(self):
        self.vel_hist.clear()
        self.boost = 0.0
        self.detected = False


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL CONFIDENCE-SCORE QUATERNION FUSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceFusionEngine:
    """
    Confidence-score based sensor fusion adapted from `notebooks/fusion_development.ipynb`.
    Maintains a rolling causal buffer to calculate:
      1. Detrended rolling variance for each sensor (isolating noise from physical motion)
      2. Cross-sensor divergence trend (low-pass filter)
      3. Encoder drift penalty
      4. Normalized confidence weights (c_enc, c_ahrs)
      5. Confidence-weighted quaternion fusion
    """
    def __init__(self, window: int = 50, dt: float = 0.02,
                 sigma_prior_enc: float = 1.5, sigma_prior_ahrs: float = 1.2,
                 drift_penalty_gain: float = 5.0):
        self.window = window
        self.dt = dt
        self.sigma_prior_enc = sigma_prior_enc
        self.sigma_prior_ahrs = sigma_prior_ahrs
        self.drift_penalty_gain = drift_penalty_gain

        self.buf_enc = deque(maxlen=window)
        self.buf_ahrs = deque(maxlen=window)
        self.buf_div = deque(maxlen=window)

        self.c_enc = 0.5
        self.c_ahrs = 0.5
        self.var_enc = 0.0
        self.var_ahrs = 0.0
        self.drift_penalty = 0.0
        self.fused_angle = 0.0

    def reset(self, initial_angle: float = 0.0):
        self.buf_enc.clear()
        self.buf_ahrs.clear()
        self.buf_div.clear()
        self.c_enc = 0.5
        self.c_ahrs = 0.5
        self.var_enc = 0.0
        self.var_ahrs = 0.0
        self.drift_penalty = 0.0
        self.fused_angle = initial_angle

    def _detrended_variance(self, buf: deque) -> float:
        n = len(buf)
        if n < 4:
            return 0.0
        arr = np.array(buf, dtype=np.float64)
        x = np.arange(n, dtype=np.float64)
        # Fast 1st order linear regression detrending
        # slope = cov(x, y) / var(x)
        x_mean = (n - 1.0) / 2.0
        y_mean = np.mean(arr)
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-9:
            return float(np.var(arr))
        slope = np.sum((x - x_mean) * (arr - y_mean)) / denom
        intercept = y_mean - slope * x_mean
        detrended = arr - (slope * x + intercept)
        return float(np.var(detrended))

    def update(self, theta_enc: float, theta_ahrs: float, axis_vector: np.ndarray = None) -> tuple[float, float, float]:
        """
        Processes one sample of raw sensor readings.
        Returns: (fused_angle, c_enc, c_ahrs)
        """
        self.buf_enc.append(theta_enc)
        self.buf_ahrs.append(theta_ahrs)

        # 1. Detrended rolling variance
        self.var_enc = self._detrended_variance(self.buf_enc)
        self.var_ahrs = self._detrended_variance(self.buf_ahrs)

        # 2. Cross-sensor divergence trend (LPF)
        divergence = theta_enc - theta_ahrs
        self.buf_div.append(divergence)
        div_trend = sum(self.buf_div) / len(self.buf_div)

        # 3. Encoder drift penalty
        self.drift_penalty = abs(div_trend) * self.drift_penalty_gain

        # 4. Inverse variance confidence scoring
        c_enc_raw = 1.0 / (self.sigma_prior_enc**2 + self.var_enc + self.drift_penalty + 1e-5)
        c_ahrs_raw = 1.0 / (self.sigma_prior_ahrs**2 + self.var_ahrs + 1e-5)

        total_c = c_enc_raw + c_ahrs_raw
        if total_c > 1e-9:
            self.c_enc = c_enc_raw / total_c
            self.c_ahrs = c_ahrs_raw / total_c
        else:
            self.c_enc = 0.5
            self.c_ahrs = 0.5

        # 5. Quaternion Fusion (or 1D weighted sum)
        if axis_vector is None:
            # Direct 1D confidence-weighted angle fusion
            self.fused_angle = self.c_enc * theta_enc + self.c_ahrs * theta_ahrs
        else:
            # Full Quaternion spherical/weighted fusion
            r_enc = Rotation.from_rotvec(np.radians(theta_enc) * axis_vector)
            r_ahrs = Rotation.from_rotvec(np.radians(theta_ahrs) * axis_vector)
            q_enc = r_enc.as_quat()
            q_ahrs = r_ahrs.as_quat()
            # Ensure consistent hemisphere (inner product >= 0)
            if np.dot(q_enc, q_ahrs) < 0:
                q_ahrs = -q_ahrs
            q_fused = self.c_enc * q_enc + self.c_ahrs * q_ahrs
            norm = np.linalg.norm(q_fused)
            if norm > 1e-9:
                q_fused /= norm
                rot_fused = Rotation.from_quat(q_fused)
                rotvec = rot_fused.as_rotvec()
                # Projection along axis vector; as_rotvec() folds into (-180, 180], so unwrap
                # onto the branch of the sensors (the axis can travel the full 0-360)
                fused = float(np.degrees(np.dot(rotvec, axis_vector)))
                ref = self.c_enc * theta_enc + self.c_ahrs * theta_ahrs
                self.fused_angle = fused + 360.0 * round((ref - fused) / 360.0)
            else:
                self.fused_angle = self.c_enc * theta_enc + self.c_ahrs * theta_ahrs

        return self.fused_angle, self.c_enc, self.c_ahrs


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED KALMAN FILTER BASELINE (comparison against the confidence fusion)
# ─────────────────────────────────────────────────────────────────────────────
class EKFFusionEngine:
    """
    Causal EKF baseline from `notebooks/fusion_development.ipynb` (§5), one sample per update.
      State       x = [θ, ω, b_enc]   (angle, angular rate, encoder bias)
      Process     constant velocity, random-walk encoder bias
      Measurement z = [θ + b_enc, θ]  (encoder, drift-free AHRS)
    R uses the same sensor priors as the confidence fusion, so both methods see the same trust levels.
    """
    def __init__(self, dt: float = 0.02,
                 sigma_prior_enc: float = 1.5, sigma_prior_ahrs: float = 1.2,
                 q_acc: float = 1e4, q_bias: float = 10.0):
        self.dt = dt
        # q_acc is large on purpose: wall bounces and target steps reverse ω within one step
        self.Q = np.array([[q_acc * dt**3 / 3, q_acc * dt**2 / 2, 0.0],
                           [q_acc * dt**2 / 2, q_acc * dt,        0.0],
                           [0.0,               0.0,               q_bias * dt]])
        self.R = np.diag([sigma_prior_enc**2, sigma_prior_ahrs**2])
        self.F = np.array([[1.0, dt, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        self.H = np.array([[1.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0]])
        self.P0 = np.diag([sigma_prior_ahrs**2, 50.0**2, 2.0**2])
        self.reset()

    def reset(self, initial_angle: float = 0.0):
        self.x = np.array([initial_angle, 0.0, 0.0])
        self.P = self.P0.copy()
        self.initialized = False
        self.angle = initial_angle
        self.bias = 0.0
        self.sigma = math.sqrt(self.P0[0, 0])

    def update(self, theta_enc: float, theta_ahrs: float) -> tuple[float, float, float]:
        """
        Processes one sample of raw sensor readings.
        Returns: (angle, encoder bias estimate, 1σ angle uncertainty)
        """
        if not self.initialized:
            self.x = np.array([theta_ahrs, 0.0, 0.0])  # start at the drift-free sensor
            self.initialized = True
        else:
            # Predict (f and h are linear for one axis, so the Jacobians are the constant F, H)
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        H = self.H
        y = np.array([theta_enc, theta_ahrs]) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        IKH = np.eye(3) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T  # Joseph form: keeps P symmetric PSD

        self.angle = float(self.x[0])
        self.bias = float(self.x[2])
        self.sigma = float(math.sqrt(max(self.P[0, 0], 0.0)))
        return self.angle, self.bias, self.sigma


# ─────────────────────────────────────────────────────────────────────────────
# VIRTUAL LIMIT SWITCH
# ─────────────────────────────────────────────────────────────────────────────
class LimitSwitch:
    """
    End-of-travel switch at one end of an axis' operational range.
    Reads the true (mechanical) angle like a physical switch would, so it is immune
    to sensor noise and encoder faults. Closes inside LIMIT_SW_BAND of the limit and
    reopens only beyond LIMIT_SW_BAND + LIMIT_SW_HYST.
    `hit` is True only on the step the switch closes (rising edge).
    """
    def __init__(self, limit_deg: float, side: str):
        self.limit = limit_deg
        self.side = side  # 'lo' | 'hi'
        self.pressed = False
        self.hit = False

    def _depth(self, angle: float) -> float:
        """Distance (°) still to travel before reaching the limit; <= 0 means at/past it."""
        return angle - self.limit if self.side == 'lo' else self.limit - angle

    def reset(self, angle: float):
        # Starting on the switch counts as pressed, but not as a fresh hit
        self.pressed = self._depth(angle) <= LIMIT_SW_BAND
        self.hit = False

    def update(self, angle: float) -> bool:
        depth = self._depth(angle)
        was = self.pressed
        if depth <= LIMIT_SW_BAND:
            self.pressed = True
        elif depth > LIMIT_SW_BAND + LIMIT_SW_HYST:
            self.pressed = False
        self.hit = self.pressed and not was
        return self.hit


# ─────────────────────────────────────────────────────────────────────────────
# AXIS SIMULATION MODEL (Physics, Motor, Sensors, and Fault Injection)
# ─────────────────────────────────────────────────────────────────────────────
class AxisSimulation:
    """
    Simulates one rotational axis of the antenna.
    Supports:
      - Mode A: Target Chasing (Cascade PID)
      - Mode B: Brownian Random Motion (Wiener process with damping & bounds)
      - Dynamic sensor noise (σ_enc, σ_ahrs)
      - Interactive fault injection: continuous drift ramp & glitch step offset
    """
    OUTER_KP, OUTER_KI, OUTER_KD = 0.75, 0.0, 0.0
    OUTER_LO, OUTER_HI           = -16., 16.

    INNER_KP, INNER_KI, INNER_KD = 14.0, 0.0, 0.30
    INNER_LO, INNER_HI           = 0., 100.

    def __init__(self, name: str, dt: float = 0.02):
        self.name = name
        self.dt = dt
        self.limits = AX_LIMITS[name]
        self.axis_vector = LINKS_CONFIG[name]['axis']

        # Plant states
        self.true_angle = 0.0
        self.true_vel = 0.0
        self.target = 45.0
        self.motion_mode = 'TARGET'  # 'TARGET' | 'BROWNIAN'

        # Brownian process state
        self.brownian_alpha = 0.998
        self.brownian_sigma = 3.5

        # Motor / Cascade Control
        self.outer = CascadeOuterLoop(v_max=16.0, a_profile=1.1, k_pos=0.75)
        self.inner = CascadeInnerLoop(kp=14.0, kd=0.30)
        self.stall = StallDetector()
        self.vel_demand = 0.0
        self.pwm = 0.0
        self.roc_err = 0.0
        self.phase = 'IDLE'

        # Limit switches at both ends of the operational range
        lo, hi = self.limits
        self.lim_lo = LimitSwitch(lo, 'lo')
        self.lim_hi = LimitSwitch(hi, 'hi')
        self.lim_lo.reset(self.true_angle)
        self.lim_hi.reset(self.true_angle)

        # Startup calibration (see begin_calibration)
        self.world_offset = 0.0     # TRUE world heading of this axis' mechanical zero (hidden from the controller)
        self.cal_phase = 'IDLE'     # IDLE | SEEK_LO | DWELL_LO | SEEK_HI | DWELL_HI | DONE
        self.cal_samples = []
        self.cal_world_lo = None    # AHRS world reading at the lo / hi stop
        self.cal_world_hi = None
        self.cal_offset = None      # estimated world offset of mechanical zero, set when DONE

        # Sensors & Faults
        self.sigma_enc = 0.50      # Encoder Gaussian noise σ (°)
        self.sigma_ahrs = 0.30     # AHRS Gaussian noise σ (°)
        self.drift_rate = 0.0      # Active continuous drift ramp (°/s)
        self.drift_offset = 0.0    # Accumulated drift (° )
        self.glitch_offset = 0.0   # Sudden glitch step offset (°)

        # Sensor readings
        self.enc_reading = 0.0
        self.ahrs_reading = 0.0

        # Confidence Fusion Engine
        self.fusion = ConfidenceFusionEngine(window=50, dt=dt)
        self.fused_angle = 0.0
        self.c_enc = 0.5
        self.c_ahrs = 0.5

        # EKF baseline, run on the same sensor readings for comparison
        self.ekf = EKFFusionEngine(dt=dt)
        self.ekf_angle = 0.0
        self.ekf_bias = 0.0
        self.ekf_sigma = self.ekf.sigma

        self.reset(INITIAL_POSE[name])

    @property
    def calibrating(self) -> bool:
        return self.cal_phase not in ('IDLE', 'DONE')

    @property
    def calibrated(self) -> bool:
        return self.cal_phase == 'DONE'

    def begin_calibration(self):
        """Forget any previous calibration and start seeking the lo limit switch."""
        self.cal_phase = 'SEEK_LO'
        self.cal_samples = []
        self.cal_world_lo = self.cal_world_hi = self.cal_offset = None

    def reset(self, initial_angle: float = 0.0):
        self.true_angle = initial_angle
        self.true_vel = 0.0
        self.vel_demand = self.pwm = self.roc_err = 0.0
        self.phase = 'IDLE'
        self.outer.reset()
        self.inner.reset()
        self.stall.reset()
        self.cal_phase = 'IDLE'
        self.cal_samples = []
        self.cal_world_lo = self.cal_world_hi = self.cal_offset = None
        self.lim_lo.reset(initial_angle)
        self.lim_hi.reset(initial_angle)

        self.drift_rate = 0.0
        self.drift_offset = 0.0
        self.glitch_offset = 0.0
        self.enc_reading = initial_angle
        self.ahrs_reading = initial_angle

        self.fusion.reset(initial_angle)
        self.fused_angle = initial_angle
        self.c_enc = 0.5
        self.c_ahrs = 0.5

        self.ekf.reset(initial_angle)
        self.ekf_angle = initial_angle
        self.ekf_bias = 0.0
        self.ekf_sigma = self.ekf.sigma

    def inject_drift_ramp(self, rate_deg_per_sec: float = 1.5):
        """Starts injecting continuous mechanical slip drift."""
        self.drift_rate = rate_deg_per_sec

    def inject_glitch_step(self, offset_deg: float = 12.0):
        """Injects a sudden electrical glitch / step offset."""
        self.glitch_offset += offset_deg

    def clear_faults(self):
        """Clears all injected drift and glitch offsets."""
        self.drift_rate = 0.0
        self.drift_offset = 0.0
        self.glitch_offset = 0.0

    # ── Startup calibration ────────────────────────────────────────────────
    def _step_calibration_motion(self):
        """Slew into the current limit switch, creep onto the hard stop, then hold for the dwell."""
        lo, hi = self.limits
        if self.cal_phase in ('SEEK_LO', 'SEEK_HI'):
            seek_lo = self.cal_phase == 'SEEK_LO'
            sw = self.lim_lo if seek_lo else self.lim_hi
            rate = CAL_CREEP if sw.pressed else CAL_RATE[self.name]
            self.true_vel = -rate if seek_lo else rate
            self.true_angle = float(np.clip(self.true_angle + self.true_vel * self.dt, lo, hi))
            if self.true_angle in (lo, hi) and sw.pressed:
                self.true_vel = 0.0
                self.cal_samples = []
                self.cal_phase = 'DWELL_LO' if seek_lo else 'DWELL_HI'
        else:
            self.true_vel = 0.0
        self.phase = 'CAL'
        self.pwm = self.vel_demand = self.roc_err = 0.0

    def _mean_world(self, samples: list) -> float:
        """Azimuth wraps at 360°, so it is averaged on the circle; the others linearly."""
        if self.name == 'azimuth':
            r = np.radians(samples)
            return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360.0)
        return float(np.mean(samples))

    def _finish_calibration_dwell(self):
        world = self._mean_world(self.cal_samples)
        self.cal_samples = []
        if self.cal_phase == 'DWELL_LO':
            self.cal_world_lo = world
            self.cal_phase = 'SEEK_HI'
            return

        self.cal_world_hi = world
        lo, hi = self.limits
        offsets = [self.cal_world_lo - lo, self.cal_world_hi - hi]
        if self.name == 'azimuth':
            # Signed offset of the seam from north; the stops are then reported unwrapped (lo ~ 0, hi ~ 360)
            offsets = [wrap180(o) for o in offsets]
            self.cal_offset = wrap180(self._mean_world(offsets))
            self.cal_world_lo, self.cal_world_hi = lo + offsets[0], hi + offsets[1]
        else:
            self.cal_offset = float(np.mean(offsets))
        self.cal_phase = 'DONE'
        # Hand over to the position loops from a clean state
        self.outer.reset()
        self.inner.reset()
        self.stall.reset()

    def world_to_mech(self, world_deg: float) -> float:
        """Mechanical angle that points the axis at a world angle, using the calibrated offset."""
        mech = world_deg - self.cal_offset
        lo, hi = self.limits
        if self.name == 'azimuth':
            mech = (mech - lo) % 360.0 + lo
        return float(np.clip(mech, lo, hi))

    def target_world(self) -> float:
        """Commanded world angle; NaN while calibrating, when the setpoint has no meaning yet."""
        return float('nan') if self.calibrating else self.mech_to_world(self.target)

    def mech_to_world(self, mech_deg: float) -> float:
        """
        World angle of a mechanical angle (azimuth: bearing from true north). Uses the
        calibrated offset; until then the AHRS' own world-referenced heading stands in for it,
        so readings stay continuous when calibration completes (the two agree to ~0.05 deg).
        Not wrapped: the seam sits on north, so azimuth runs continuously from ~0 to ~360.
        """
        offset = self.world_offset if self.cal_offset is None else self.cal_offset
        return mech_deg + offset

    def step(self, el_deg: float, sim_speed: float = 1.5):
        """Executes one simulation step for this axis."""
        dt = self.dt

        # ── 1. True Motion Update ──────────────────────────────────────────
        if self.calibrating:
            self._step_calibration_motion()

        elif self.motion_mode == 'BROWNIAN':
            # Brownian motion with damping & reflective boundaries (fusion_development.ipynb)
            d_omega = random.gauss(0.0, self.brownian_sigma) * math.sqrt(dt) * sim_speed
            self.true_vel = self.true_vel * self.brownian_alpha + d_omega
            new_angle = self.true_angle + self.true_vel * dt

            # Reflective boundaries
            lo, hi = self.limits
            if new_angle > hi:
                new_angle = 2.0 * hi - new_angle
                self.true_vel = -self.true_vel * 0.7
            elif new_angle < lo:
                new_angle = 2.0 * lo - new_angle
                self.true_vel = -self.true_vel * 0.7

            self.true_angle = float(np.clip(new_angle, lo, hi))
            self.phase = 'BROWNIAN'

        else:
            # Target Chasing with Motor & Friction Plant (Cascade Control)
            lf = load_factor(self.name, el_deg)
            stall_pwm = BASE_STALL * lf

            pos_err = self.target - self.true_angle
            dist = abs(pos_err)

            if dist < 0.15 and abs(self.true_vel) < 0.15:
                self.phase = 'HOLDING'
                self.true_vel = 0.0
                self.pwm = 0.0
                self.vel_demand = 0.0
                self.roc_err = 0.0
                self.stall.update(0.0, 0.0)
            else:
                self.phase = 'TRACKING'
                # 1. Outer Loop: Position error -> Demanded Rate of Change
                # Moves fast initially, then tapers smoothly as it nears target ('slower and slower')
                self.vel_demand = self.outer.compute(pos_err, dt)

                # Rate tracking error (signed) & rate-of-change error (magnitude)
                rate_err = self.vel_demand - self.true_vel
                self.roc_err = abs(self.vel_demand) - abs(self.true_vel)

                # 2. Inner Loop: Rate of change -> PWM adjustment
                # Feedforward effort to overcome friction at demanded rate-of-change
                ff_req_force = (K_FRICTION * self.vel_demand + STATIC_FRIC * math.copysign(1.0, self.vel_demand)) if abs(self.vel_demand) > 0.02 else 0.0
                ff_effort = ff_req_force / (K_MOTOR * lf * sim_speed)

                # Closed-loop rate feedback correction
                fb_effort = self.inner.compute(rate_err, dt)
                total_effort = ff_effort + fb_effort
                total_effort = float(np.clip(total_effort, -100.0, 100.0))

                effort_mag = abs(total_effort)
                if effort_mag > 0.2:
                    raw_pwm = min(100.0, stall_pwm + effort_mag * (1.0 - stall_pwm / 100.0))
                else:
                    raw_pwm = 0.0

                # Stall detector monitoring & boost
                stalled, boost = self.stall.update(self.true_vel, raw_pwm)
                eff_pwm = min(100.0, raw_pwm + boost)
                self.pwm = eff_pwm

                motor_dir = math.copysign(1.0, total_effort) if effort_mag > 0.2 else 0.0
                if eff_pwm > stall_pwm:
                    motor_force = K_MOTOR * (eff_pwm - stall_pwm) * lf * sim_speed
                else:
                    motor_force = 0.0

                friction = K_FRICTION * self.true_vel + STATIC_FRIC * math.copysign(1.0, self.true_vel) if abs(self.true_vel) > 0.01 else 0.0
                accel = motor_dir * motor_force - friction
                self.true_vel += accel * dt
                self.true_angle += self.true_vel * dt

                # Mechanical limits clamping
                lo, hi = self.limits
                if self.true_angle > hi:
                    self.true_angle = hi
                    self.true_vel = 0.0
                elif self.true_angle < lo:
                    self.true_angle = lo
                    self.true_vel = 0.0

        # ── 1b. Limit Switches (mechanical position, not sensor readings) ───
        self.lim_lo.update(self.true_angle)
        self.lim_hi.update(self.true_angle)

        # ── 2. Sensor Readings (Gaussian Noise + Faults) ────────────────────
        self.drift_offset += self.drift_rate * dt
        total_encoder_error = self.drift_offset + self.glitch_offset + random.gauss(0.0, max(1e-4, self.sigma_enc))
        self.enc_reading = self.true_angle + total_encoder_error

        ahrs_noise = random.gauss(0.0, max(1e-4, self.sigma_ahrs))
        self.ahrs_reading = self.true_angle + ahrs_noise

        # Calibration dwell: the AHRS reports the world-frame angle of the stop
        if self.cal_phase in ('DWELL_LO', 'DWELL_HI'):
            self.cal_samples.append(self.true_angle + self.world_offset + ahrs_noise)
            if len(self.cal_samples) >= CAL_DWELL:
                self._finish_calibration_dwell()

        # ── 3. Confidence-Weighted Sensor Fusion ────────────────────────────
        self.fused_angle, self.c_enc, self.c_ahrs = self.fusion.update(
            self.enc_reading, self.ahrs_reading, self.axis_vector
        )

        # ── 4. EKF Baseline (same readings) ─────────────────────────────────
        self.ekf_angle, self.ekf_bias, self.ekf_sigma = self.ekf.update(
            self.enc_reading, self.ahrs_reading
        )

        return self.fused_angle


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-AXIS SIMULATION MANAGER & DATA RECORDER
# ─────────────────────────────────────────────────────────────────────────────
class AntennaSimulationManager:
    """Coordinates the 3-axis antenna system, kinematics, and CSV logging."""
    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.sim_time = 0.0
        self.running = False
        self.recording = False
        self.sim_speed = 1.5

        self.axes = {ax: AxisSimulation(ax, dt=dt) for ax in AXES}

        # Installation: world heading (from true north) of the azimuth mechanical zero.
        # Unknown to the controller until calibration has found it.
        self.mount_heading = 0.0
        self.world_targets = {}          # last commanded world angles, re-applied after calibration
        self.cal_just_done = False       # set for one step when the last axis finishes
        self._cal_pending = set()        # axes still waiting to take their world target
        self.start_calibration(new_mount=True)

        # Real-time recorded data buffers for CSV export
        self.log_data = []
        self.current_csv_path = None

    def reset(self):
        """Power-cycle into a fresh installation: initial pose, new random mount heading, recalibrate."""
        self.sim_time = 0.0
        self.running = False
        for ax, a in self.axes.items():
            a.reset(initial_angle=INITIAL_POSE[ax])
        self.clear_recorded_data()
        self.start_calibration(new_mount=True)

    def start_calibration(self, new_mount: bool = False, mount_heading: float = None):
        """
        Begin the limit-switch calibration from the current pose. A new mount models installing
        the antenna with its seam on true north, off by a small random error; keep it for a
        re-calibration in place.
        """
        if mount_heading is not None:
            self.mount_heading = wrap180(float(mount_heading))
        elif new_mount:
            self.mount_heading = random.uniform(-MOUNT_TOL, MOUNT_TOL)
        self.cal_just_done = False
        self._cal_pending = set(self.axes)
        for ax, a in self.axes.items():
            a.world_offset = self.mount_heading if ax == 'azimuth' else 0.0
            a.begin_calibration()

    @property
    def calibrating(self) -> bool:
        return any(a.calibrating for a in self.axes.values())

    @property
    def calibrated(self) -> bool:
        return all(a.calibrated for a in self.axes.values())

    def set_world_target(self, axis_name: str, world_deg: float):
        """Command an axis in world angles (azimuth = bearing from true north); applied once calibrated."""
        self.world_targets[axis_name] = float(world_deg)
        if self.axes[axis_name].calibrated:
            self.set_target(axis_name, self.axes[axis_name].world_to_mech(world_deg))

    def set_motion_mode(self, mode: str):
        for a in self.axes.values():
            a.motion_mode = mode

    def set_target(self, axis_name: str, target_deg: float):
        if axis_name in self.axes:
            lo, hi = AX_LIMITS[axis_name]
            self.axes[axis_name].target = float(np.clip(target_deg, lo, hi))

    def set_noise(self, axis_name: str, sigma_enc: float = None, sigma_ahrs: float = None):
        if axis_name in self.axes:
            if sigma_enc is not None:
                self.axes[axis_name].sigma_enc = max(0.0, sigma_enc)
            if sigma_ahrs is not None:
                self.axes[axis_name].sigma_ahrs = max(0.0, sigma_ahrs)

    def set_global_noise(self, sigma_enc: float = None, sigma_ahrs: float = None):
        for a in self.axes.values():
            if sigma_enc is not None:
                a.sigma_enc = max(0.0, sigma_enc)
            if sigma_ahrs is not None:
                a.sigma_ahrs = max(0.0, sigma_ahrs)

    def inject_fault(self, axis_name: str, fault_type: str):
        if axis_name in self.axes:
            a = self.axes[axis_name]
            if fault_type == 'drift':
                a.inject_drift_ramp(1.5)
            elif fault_type == 'glitch':
                a.inject_glitch_step(12.0)
            elif fault_type == 'clear':
                a.clear_faults()

    def step(self):
        """Advances simulation by one dt."""
        self.sim_time += self.dt
        el_deg = self.axes['elevation'].true_angle

        was_calibrating = self.calibrating

        # Step each axis
        for ax in AXES:
            self.axes[ax].step(el_deg, self.sim_speed)

        # Each axis starts chasing its commanded world angle as soon as its own calibration finishes
        for ax in list(self._cal_pending):
            if self.axes[ax].calibrated:
                self._cal_pending.discard(ax)
                if ax in self.world_targets:
                    self.set_world_target(ax, self.world_targets[ax])
        self.cal_just_done = was_calibrating and not self.calibrating

        # Record sample if recording is active
        if self.recording:
            row = {'time_s': self.sim_time}
            for ax in AXES:
                a = self.axes[ax]
                # Absolute angles in the world frame once calibrated; errors are frame-independent
                w = a.mech_to_world
                row[f'{ax}_target'] = a.target_world()
                row[f'{ax}_true'] = w(a.true_angle)
                row[f'{ax}_enc'] = w(a.enc_reading)
                row[f'{ax}_ahrs'] = w(a.ahrs_reading)
                row[f'{ax}_fused'] = w(a.fused_angle)
                row[f'{ax}_c_enc'] = a.c_enc
                row[f'{ax}_c_ahrs'] = a.c_ahrs
                row[f'{ax}_err_enc'] = a.enc_reading - a.true_angle
                row[f'{ax}_err_ahrs'] = a.ahrs_reading - a.true_angle
                row[f'{ax}_err_fused'] = a.fused_angle - a.true_angle
                row[f'{ax}_ekf'] = w(a.ekf_angle)
                row[f'{ax}_ekf_bias'] = a.ekf_bias
                row[f'{ax}_ekf_sigma'] = a.ekf_sigma
                row[f'{ax}_err_ekf'] = a.ekf_angle - a.true_angle
                # Noise / fault settings in effect, so post-run analysis can explain the confidence
                row[f'{ax}_sigma_enc'] = a.sigma_enc
                row[f'{ax}_sigma_ahrs'] = a.sigma_ahrs
                row[f'{ax}_drift_rate'] = a.drift_rate
                row[f'{ax}_drift_offset'] = a.drift_offset
                row[f'{ax}_glitch_offset'] = a.glitch_offset
                row[f'{ax}_lim_lo'] = int(a.lim_lo.pressed)
                row[f'{ax}_lim_hi'] = int(a.lim_hi.pressed)
                row[f'{ax}_calibrating'] = int(a.calibrating)

            # Also provide single-axis legacy column names for the primary (azimuth) axis
            az = self.axes['azimuth']
            row['theta_true_deg'] = az.mech_to_world(az.true_angle)
            row['theta_enc_deg'] = az.mech_to_world(az.enc_reading)
            row['theta_ahrs_deg'] = az.mech_to_world(az.ahrs_reading)
            row['theta_fused_deg'] = az.mech_to_world(az.fused_angle)
            row['conf_enc'] = self.axes['azimuth'].c_enc
            row['conf_ahrs'] = self.axes['azimuth'].c_ahrs
            row['error_enc_deg'] = self.axes['azimuth'].enc_reading - self.axes['azimuth'].true_angle
            row['error_ahrs_deg'] = self.axes['azimuth'].ahrs_reading - self.axes['azimuth'].true_angle
            row['error_fused_deg'] = self.axes['azimuth'].fused_angle - self.axes['azimuth'].true_angle
            row['theta_ekf_deg'] = az.mech_to_world(az.ekf_angle)
            row['error_ekf_deg'] = self.axes['azimuth'].ekf_angle - self.axes['azimuth'].true_angle

            self.log_data.append(row)

    def start_recording(self):
        self.recording = True

    def stop_recording(self) -> str:
        self.recording = False
        return self.export_to_csv()

    def clear_recorded_data(self):
        self.log_data.clear()

    def export_to_csv(self, filename: str = None) -> str:
        """Exports all recorded data to a CSV file."""
        if not self.log_data:
            return None

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sim_antenna_{ts}.csv"

        os.makedirs(LOGS_DIR, exist_ok=True)
        filepath = os.path.join(LOGS_DIR, filename)  # bare names land in logs/, absolute paths are kept
        import pandas as pd
        df = pd.DataFrame(self.log_data)
        df.to_csv(filepath, index=False, float_format="%.6f")
        self.current_csv_path = filepath
        return filepath
