# Kinematics and Dynamics — Equations

This document collects the core kinematics/dynamics equations used in `pointing-sensor-fusion-sim.py`.

**Coordinate transform (link rotation about pivot)**

Let pivot p \in R^3, unit axis u, and rotation angle \(\theta\) in degrees. Convert to radians: \(\phi=\theta\cdot\pi/180\).

Rotation matrix: $$R = R(u,\phi)\quad\text{(3x3 rotation matrix)}$$

Homogeneous transform (4x4):
$$
M=\begin{bmatrix}R & t\\[4pt]0_{1\times3} & 1\end{bmatrix},\qquad t = p - R\,p
$$

Point transform: for point \(x\in\mathbb{R}^3\),
$$x' = (M\,[x;1])_{1:3} = R x + t$$

**Load factor (gravity / mechanical load scaling)**

Let \(r=\mathrm{rad}(\text{el\,deg})\).

Azimuth:
$$lf_{az} = \max\big(0.75,\;1 - 0.15\sin r\big)$$
Elevation:
$$lf_{el} = \max\big(0.30,\;\cos r\big)$$
Polarization:
$$lf_{pol} = \max\big(0.65,\;1 - 0.22\sin r\big)$$

**State & process model (Kalman filter)**

State vector: \(x=\begin{bmatrix}\theta\\[2pt]\omega\end{bmatrix}\)

Discrete-time constant-rate model (sampling interval \(\Delta t\)):
$$A=\begin{bmatrix}1 & \Delta t\\[4pt]0 & 1\end{bmatrix},\qquad x_k = A\,x_{k-1} + w,\quad w\sim\mathcal{N}(0,Q)$$

Process noise covariance:
$$Q=\operatorname{diag}(q_{ang},\;q_{rate})$$

Prediction step:
$$\hat x^{-}=A\hat x^{+},\qquad P^{-}=A P^{+} A^{\top}+Q$$

Measurements:
Encoder (angle):
$$z_{enc}=\theta + v_{enc},\quad H_{enc}=[1\;\;0],\;R_{enc}=\sigma_{enc}^2$$
AHRS (rate):
$$z_{ahrs}=\omega + b + v_{ahrs},\quad H_{ahrs}=[0\;\;1],\;R_{ahrs}=\sigma_{ahrs}^2$$

Update (general H,R):
$$S = H P^{-} H^{\top} + R$$
$$K = P^{-} H^{\top} S^{-1}$$
$$\hat x^{+} = \hat x^{-} + K\,(z - H\hat x^{-})$$
$$P^{+} = (I - K H) P^{-}$$

**PID control (outer & inner loops)**

Continuous form (implemented discretely):
$$u(t)=K_p e(t) + K_i \int e(t)\,dt + K_d \frac{de}{dt}$$

Discrete approximations used:
$$\mathrm{integ} \leftarrow \mathrm{integ} + e\,\Delta t$$
$$\mathrm{deriv} \approx \frac{e - e_{prev}}{\Delta t}$$
$$u_{raw} = K_p e + K_i\,\mathrm{integ} + K_d\,\mathrm{deriv}$$
Anti-windup:
if \(u_{raw}>u_{hi}\) or \(u_{raw}<u_{lo}\) then \(\mathrm{integ}\leftarrow\mathrm{integ} - e\,\Delta t\)

Saturation:
$$u = \operatorname{clip}(u_{raw},\;u_{lo},\;u_{hi})$$

Outer loop (position→velocity demand):
$$e_{pos}=\theta_{target} - \hat\theta_{fused}\quad\Rightarrow\quad v_{demand}=\mathrm{PID}_{outer}(e_{pos})$$

Inner loop (velocity→PWM): rate-of-change error is used as inner input:
$$e_{roc}=|v_{demand}| - |\hat\omega_{fused}|$$
$$\mathrm{raw\_pwm}=\mathrm{PID}_{inner}(e_{roc})$$

**Stall detection & boost**

Maintain windowed history of absolute velocities; compute average:
$$\bar v = \frac{1}{N}\sum_{i=1}^{N}|v_i|$$
If \(\bar v < v_{thr}\) and \(\mathrm{raw\_pwm} > pwm_{thr}\) then declare stall and set a one-shot boost
$$\text{boost} \leftarrow B_{0}$$
Boost decays each step by a factor \(\alpha\) (code uses \(\alpha=0.87\)):
$$\text{boost}\leftarrow \alpha\cdot\text{boost}$$
Effective PWM:
$$\mathrm{eff\_pwm}=\min(100,\;\mathrm{raw\_pwm}+\text{boost})$$

**Plant model (motor + friction → kinematics)**

Stall threshold (depends on load factor):
$$pwm_{stall} = PWM_{base}\cdot lf$$

Motor force (only when above stall):
$$\text{motor\_force}=\begin{cases}K_{motor}\,(\mathrm{eff\_pwm}-pwm_{stall})\cdot lf\cdot S &\text{if }\mathrm{eff\_pwm}>pwm_{stall}\\[4pt]0 &\text{otherwise}\end{cases}$$
where $S$ is the simulation speed multiplier.

Friction model (velocity-proportional + static):
$$\text{friction} = \begin{cases}K_{friction}\,v + F_{static}\,\operatorname{sgn}(v) &\text{if }|v|>\epsilon\\[4pt]0 &\text{otherwise}\end{cases}$$

Direction (sign of demanded speed):
$$d=\operatorname{sgn}(v_{demand})\quad\text{(zero if }|v_{demand}|\le 0.01\text{)}$$

Net acceleration:
$$a = d\cdot\text{motor\_force} - \text{friction}$$

Euler integration (explicit):
$$v_{t+\Delta t} = v_t + a\,\Delta t$$
$$\theta_{t+\Delta t} = \theta_t + v_{t+\Delta t}\,\Delta t$$
(Code uses the update order: update velocity first, then angle using the new velocity.)

**Sensor models (simulated measurements)**

Encoder measurement:
$$z_{enc}=\theta + \mathcal{N}(0,\sigma_{enc}^2)$$
AHRS measurement (gyro rate with bias):
$$z_{ahrs}=\omega + b + \mathcal{N}(0,\sigma_{ahrs}^2)$$

Kalman measurement covariances are set from the noise stddevs:
$$R_{enc}=\max(0.01,\;\sigma_{enc}^2),\qquad R_{ahrs}=\max(10^{-3},\;\sigma_{ahrs}^2)$$

---

Constants (as used in the script):
- $\Delta t = 0.05\,\mathrm{s}$
- $\text{BASE\_STALL}=28.0$ (percent PWM at 0° el)
- $K_{motor}=0.013\; (\mathrm{°/s})/(\%\;\mathrm{PWM})$
- $K_{friction}=0.22$
- $F_{static}=0.04$ 

(For implementation details, see the code in `pointing-sensor-fusion-sim.py`.)
