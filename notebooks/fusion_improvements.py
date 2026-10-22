"""Drop-in additions for fusion_development.ipynb.

Paste as a new cell after section 8 (Quality Score Card). Uses the notebook's globals:
dt, N, t, theta_true, theta_enc, theta_ahrs, sigma_prior_enc, sigma_prior_ahrs, WINDOW, run_ekf, compute_metrics.

1. robust_noise_var  - noise variance that ignores motion and wall bounces (fixes the main flaw)
2. confidence_v2     - quadratic (unit-consistent) drift penalty + optional bias compensation
3. run_ekf_adaptive  - EKF whose R comes from the confidence statistics (adaptive-R hybrid)
4. jolt test         - abrupt-motion test, where the confidence method beats the EKF
"""
import numpy as np


# ── 1. Motion-insensitive noise variance ──────────────────────────────────
def robust_noise_var(x, w=WINDOW):
    """MAD of the 2nd difference in a causal window.
    For white noise var(x[k] - 2x[k-1] + x[k-2]) = 6 sigma^2. Smooth motion cancels in the
    2nd difference, and the median ignores the 1-2 samples hit by a wall bounce, so (unlike the
    detrended variance) it does not jump to ~50-150 deg^2 at every bounce."""
    d2 = np.zeros(len(x)); d2[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    v = np.full(len(x), np.nan)
    for i in range(len(x)):
        s = d2[max(2, i - w + 1):i + 1]
        if len(s) >= 5:
            v[i] = (1.4826 * np.median(np.abs(s - np.median(s)))) ** 2 / 6
    return v


def causal_mean(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(len(x)); a = np.maximum(0, idx - w + 1)
    return (cs[idx + 1] - cs[a]) / (idx + 1 - a)


# ── 2. Confidence fusion v2 ───────────────────────────────────────────────
def confidence_v2(enc, ahrs, w=WINDOW, w_drift=20, k=3.0, compensate=True):
    ve = np.nan_to_num(robust_noise_var(enc, w), nan=sigma_prior_enc ** 2)
    va = np.nan_to_num(robust_noise_var(ahrs, w), nan=sigma_prior_ahrs ** 2)
    se2 = 0.5 * (sigma_prior_enc ** 2 + ve)        # prior and measured variance, averaged
    sa2 = 0.5 * (sigma_prior_ahrs ** 2 + va)
    b = causal_mean(enc - ahrs, w_drift)            # encoder bias estimate (AHRS = reference)
    if compensate:
        enc_use = enc - b                           # remove the bias instead of only down-weighting
        db = np.zeros_like(b); db[w_drift:] = b[w_drift:] - b[:-w_drift]
        penalty = (se2 + sa2) / w_drift + k * (db / 2) ** 2   # bias-estimate noise + lag while it moves
    else:
        enc_use = enc
        penalty = k * np.maximum(b ** 2 - (se2 + sa2) / w_drift, 0)  # bias adds b^2 to the MSE (deg^2)
    c_e = 1 / (se2 + penalty); c_a = 1 / sa2
    c = c_e / (c_e + c_a)
    return c * enc_use + (1 - c) * ahrs, c


# ── 3. Adaptive-R EKF: confidence statistics drive the measurement noise ──
def run_ekf_adaptive(z_enc, z_ahrs, R_enc, R_ahrs, q_acc=1e4, q_bias=10.0):
    F = np.array([[1, dt, 0], [0, 1, 0], [0, 0, 1.]]); H = np.array([[1, 0, 1], [1, 0, 0.]])
    Q = np.array([[q_acc * dt**3 / 3, q_acc * dt**2 / 2, 0],
                  [q_acc * dt**2 / 2, q_acc * dt, 0], [0, 0, q_bias * dt]])
    I = np.eye(3); x = np.array([z_ahrs[0], 0, 0.]); P = np.diag([sigma_prior_ahrs**2, 50.**2, 2.**2])
    out = np.zeros(len(z_enc))
    for k in range(len(z_enc)):
        if k > 0:
            x = F @ x; P = F @ P @ F.T + Q
        R = np.diag([R_enc[k], R_ahrs[k]])
        y = np.array([z_enc[k], z_ahrs[k]]) - H @ x
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ y; IKH = I - K @ H; P = IKH @ P @ IKH.T + K @ R @ K.T
        out[k] = x[0]
    return out


theta_fused_v2, c_enc_v2 = confidence_v2(theta_enc, theta_ahrs)
R_e = np.nan_to_num(robust_noise_var(theta_enc), nan=sigma_prior_enc ** 2)
R_a = np.nan_to_num(robust_noise_var(theta_ahrs), nan=sigma_prior_ahrs ** 2)
theta_aekf = run_ekf_adaptive(theta_enc, theta_ahrs, R_e, R_a)
for lbl, est in (('Confidence v2', theta_fused_v2), ('Adaptive-R EKF', theta_aekf)):
    m = compute_metrics(theta_true, est, lbl)
    print(f"{lbl:<15s} RMSE {m['rmse']:.3f}  MAE {m['mae']:.3f}  Max {m['max_error']:.3f}")


# ── 4. Abrupt-motion (jolt) test ──────────────────────────────────────────
# The true angle jumps by `amp` degrees (mechanical shock, backlash, slip). Both sensors see it,
# so the convex combination follows it at once; the EKF's constant-velocity model needs several
# samples to catch up.
def jolt_test(amp=8.0, jolts=(500, 1600, 2700), seed=7):
    rng = np.random.RandomState(seed)
    th = theta_true.copy()
    for j in jolts:
        th[j:] += amp * rng.choice([-1, 1])
    enc = th + rng.normal(0, sigma_prior_enc, N)
    ahrs = th + rng.normal(0, sigma_prior_ahrs, N)
    est = {'Confidence v2': confidence_v2(enc, ahrs)[0], 'EKF': run_ekf(enc, ahrs, q_acc, q_bias)[0][:, 0]}
    for name, e in est.items():
        err = np.abs(e - th)
        peak = max(err[j:j + 30].max() for j in jolts)
        print(f'{name:<15s} peak error after {amp:g} deg jolt: {peak:.2f} deg')

jolt_test()
