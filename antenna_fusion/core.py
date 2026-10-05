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
5. Real-Time Data Recording & Multi-Axis CSV Export
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


def compute_fk_matrices(az_deg: float, el_deg: float, pol_deg: float) -> dict:
    """Compute forward kinematics cumulative matrices for the 3-link antenna chain."""
    m_az = transform_matrix(LINKS_CONFIG['azimuth']['pivot'],
                            LINKS_CONFIG['azimuth']['axis'], az_deg)
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
                # Projection along axis vector
                self.fused_angle = float(np.degrees(np.dot(rotvec, axis_vector)))
            else:
                self.fused_angle = self.c_enc * theta_enc + self.c_ahrs * theta_ahrs

        return self.fused_angle, self.c_enc, self.c_ahrs


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

    def reset(self, initial_angle: float = 0.0):
        self.true_angle = initial_angle
        self.true_vel = 0.0
        self.vel_demand = self.pwm = self.roc_err = 0.0
        self.phase = 'IDLE'
        self.outer.reset()
        self.inner.reset()
        self.stall.reset()

        self.drift_rate = 0.0
        self.drift_offset = 0.0
        self.glitch_offset = 0.0
        self.enc_reading = initial_angle
        self.ahrs_reading = initial_angle

        self.fusion.reset(initial_angle)
        self.fused_angle = initial_angle
        self.c_enc = 0.5
        self.c_ahrs = 0.5

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

    def step(self, el_deg: float, sim_speed: float = 1.5):
        """Executes one simulation step for this axis."""
        dt = self.dt

        # ── 1. True Motion Update ──────────────────────────────────────────
        if self.motion_mode == 'BROWNIAN':
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

        # ── 2. Sensor Readings (Gaussian Noise + Faults) ────────────────────
        self.drift_offset += self.drift_rate * dt
        total_encoder_error = self.drift_offset + self.glitch_offset + random.gauss(0.0, max(1e-4, self.sigma_enc))
        self.enc_reading = self.true_angle + total_encoder_error

        ahrs_noise = random.gauss(0.0, max(1e-4, self.sigma_ahrs))
        self.ahrs_reading = self.true_angle + ahrs_noise

        # ── 3. Confidence-Weighted Sensor Fusion ────────────────────────────
        self.fused_angle, self.c_enc, self.c_ahrs = self.fusion.update(
            self.enc_reading, self.ahrs_reading, self.axis_vector
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

        # Real-time recorded data buffers for CSV export
        self.log_data = []
        self.current_csv_path = None

    def reset(self):
        self.sim_time = 0.0
        self.running = False
        for ax, a in self.axes.items():
            a.reset(initial_angle=0.0)
        self.clear_recorded_data()

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

        # Step each axis
        for ax in AXES:
            self.axes[ax].step(el_deg, self.sim_speed)

        # Record sample if recording is active
        if self.recording:
            row = {'time_s': self.sim_time}
            for ax in AXES:
                a = self.axes[ax]
                row[f'{ax}_target'] = a.target
                row[f'{ax}_true'] = a.true_angle
                row[f'{ax}_enc'] = a.enc_reading
                row[f'{ax}_ahrs'] = a.ahrs_reading
                row[f'{ax}_fused'] = a.fused_angle
                row[f'{ax}_c_enc'] = a.c_enc
                row[f'{ax}_c_ahrs'] = a.c_ahrs
                row[f'{ax}_err_enc'] = a.enc_reading - a.true_angle
                row[f'{ax}_err_ahrs'] = a.ahrs_reading - a.true_angle
                row[f'{ax}_err_fused'] = a.fused_angle - a.true_angle

            # Also provide single-axis legacy column names for the primary (azimuth) axis
            row['theta_true_deg'] = self.axes['azimuth'].true_angle
            row['theta_enc_deg'] = self.axes['azimuth'].enc_reading
            row['theta_ahrs_deg'] = self.axes['azimuth'].ahrs_reading
            row['theta_fused_deg'] = self.axes['azimuth'].fused_angle
            row['conf_enc'] = self.axes['azimuth'].c_enc
            row['conf_ahrs'] = self.axes['azimuth'].c_ahrs
            row['error_enc_deg'] = self.axes['azimuth'].enc_reading - self.axes['azimuth'].true_angle
            row['error_ahrs_deg'] = self.axes['azimuth'].ahrs_reading - self.axes['azimuth'].true_angle
            row['error_fused_deg'] = self.axes['azimuth'].fused_angle - self.axes['azimuth'].true_angle

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
