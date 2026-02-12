"""
vehicle_model.py

Deterministic kinematic bicycle vehicle model with:
- friction-limited tire slip (combined-slip / friction circle)
- lateral acceleration limits (curvature saturation)
- aerodynamic drag + rolling resistance
- optimized vectorized stepping for evolution mode
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from rc_racer.core.state import State, StateArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VehicleParams:
    """
    Physical parameters of the vehicle model.

    Parameters
    ----------
    wheelbase : float
        Distance between front and rear axle (meters).
    rear_axle_ratio : float
        Ratio lr/L, where lr is distance from CoM to rear axle.
        Used for kinematic slip angle approximation (beta).
        Typical value: 0.5 for symmetric bicycle.
    max_steering_angle : float
        Maximum absolute steering angle (radians).
    max_steering_rate : float
        Maximum absolute steering rate (radians/s).
    max_acceleration : float
        Maximum commanded longitudinal acceleration magnitude (m/s^2),
        before friction and drag are applied.
    max_velocity : float
        Maximum forward velocity (m/s).

    mu : float
        Tire-road friction coefficient (dimensionless). Used for combined-slip.
    g : float
        Gravitational acceleration (m/s^2).

    a_lat_max : float
        Maximum allowed lateral acceleration magnitude (m/s^2).
        This acts as a curvature limit: |a_lat| = v^2 * |kappa|.

    mass : float
        Vehicle mass (kg).
    c_rr : float
        Rolling resistance coefficient (dimensionless). Rolling decel ~ c_rr * g.
    c_d_a_over_m : float
        Aerodynamic drag constant divided by mass (1/m):
        a_drag = c_d_a_over_m * v^2.
        (This corresponds to 0.5*rho*Cd*A / m if desired.)
    """

    wheelbase: float
    rear_axle_ratio: float
    max_steering_angle: float
    max_steering_rate: float
    max_acceleration: float
    max_velocity: float

    mu: float
    g: float
    a_lat_max: float

    mass: float
    c_rr: float
    c_d_a_over_m: float


class VehicleModel:
    """
    Deterministic kinematic bicycle model with friction-limited slip and drag.

    Model Highlights
    ----------------
    1) Kinematic bicycle geometry:
       - heading rate ~ v/L * tan(delta_eff)
       - slip angle beta ~ arctan(lr/L * tan(delta_eff))

    2) Lateral acceleration / curvature saturation:
       - kappa_geom = tan(delta) / L
       - a_lat_geom = v^2 * kappa_geom
       - if |a_lat_geom| > a_lat_max, reduce effective curvature:
         kappa_eff = sign(kappa_geom) * a_lat_max / max(v^2, eps)

    3) Combined-slip (friction circle) on longitudinal acceleration:
       - available total accel magnitude ~ mu*g
       - given lateral accel demand a_lat_eff, cap longitudinal accel:
         |a_long| <= sqrt(max((mu*g)^2 - a_lat_eff^2, 0))

    4) Energy/drag model:
       - rolling resistance decel: a_roll = c_rr * g
       - aero drag decel: a_drag = c_d_a_over_m * v^2
       - net longitudinal accel: a_net = a_cmd_clipped - sign(v)*a_roll - a_drag
       - velocity is clamped to [0, max_velocity]

    Notes
    -----
    - Deterministic and stateless (no hidden memory).
    - Uses Euler integration with fixed dt.
    - progress_s is passed through; environment/track logic updates it.
    """

    def __init__(self, params: VehicleParams) -> None:
        """
        Initialize vehicle model.

        Parameters
        ----------
        params : VehicleParams
            Model parameters.
        """
        self._p: VehicleParams = params
        if not (0.0 < self._p.rear_axle_ratio < 1.0):
            raise ValueError("rear_axle_ratio must be in (0, 1).")
        if self._p.wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive.")
        if self._p.mass <= 0.0:
            raise ValueError("mass must be positive.")
        if self._p.g <= 0.0:
            raise ValueError("g must be positive.")
        if self._p.mu < 0.0:
            raise ValueError("mu must be non-negative.")
        if self._p.a_lat_max <= 0.0:
            raise ValueError("a_lat_max must be positive.")
        if self._p.max_velocity <= 0.0:
            raise ValueError("max_velocity must be positive.")

    # ------------------------------------------------------------------
    # Internal helpers (scalar)
    # ------------------------------------------------------------------

    def _apply_steering_limits(self, steering_angle: float, steering_rate: float, dt: float) -> float:
        """
        Integrate steering angle with rate and clamp to bounds.

        Parameters
        ----------
        steering_angle : float
            Current steering angle (rad).
        steering_rate : float
            Commanded steering rate (rad/s).
        dt : float
            Timestep (s).

        Returns
        -------
        float
            New clamped steering angle (rad).
        """
        sr = float(np.clip(steering_rate, -self._p.max_steering_rate, self._p.max_steering_rate))
        delta = steering_angle + sr * dt
        return float(np.clip(delta, -self._p.max_steering_angle, self._p.max_steering_angle))

    def _effective_curvature_and_lateral_accel(self, v: float, steering_angle: float) -> tuple[float, float]:
        """
        Compute effective curvature and lateral acceleration with saturation.

        Parameters
        ----------
        v : float
            Speed (m/s).
        steering_angle : float
            Steering angle (rad).

        Returns
        -------
        kappa_eff : float
            Effective curvature (1/m) after lateral acceleration limit.
        a_lat_eff : float
            Effective lateral acceleration (m/s^2), capped by a_lat_max.
        """
        # geometric curvature
        kappa_geom = float(np.tan(steering_angle) / self._p.wheelbase)

        # a_lat = v^2 * kappa
        v2 = v * v
        a_lat_geom = v2 * kappa_geom

        # saturate lateral acceleration by limiting curvature
        a_lat_max = self._p.a_lat_max
        if abs(a_lat_geom) <= a_lat_max:
            return kappa_geom, a_lat_geom

        # kappa_eff = sign(kappa_geom) * a_lat_max / max(v^2, eps)
        eps = 1e-9
        denom = max(v2, eps)
        kappa_eff = np.sign(kappa_geom) * (a_lat_max / denom)
        a_lat_eff = v2 * kappa_eff
        return float(kappa_eff), float(a_lat_eff)

    def _friction_limited_longitudinal_accel(self, a_cmd: float, a_lat: float) -> float:
        """
        Apply combined-slip friction circle constraint to longitudinal acceleration.

        Parameters
        ----------
        a_cmd : float
            Commanded longitudinal acceleration (m/s^2) (already clipped by max_acceleration).
        a_lat : float
            Lateral acceleration demand (m/s^2).

        Returns
        -------
        float
            Longitudinal acceleration after friction circle limit (m/s^2).
        """
        # total available acceleration magnitude
        a_total_max = self._p.mu * self._p.g
        if a_total_max <= 0.0:
            return 0.0

        # remaining budget for longitudinal after lateral usage
        rem2 = a_total_max * a_total_max - a_lat * a_lat
        if rem2 <= 0.0:
            return 0.0

        a_long_max = float(np.sqrt(rem2))
        return float(np.clip(a_cmd, -a_long_max, a_long_max))

    def _drag_and_rolling_decel(self, v: float) -> float:
        """
        Compute deterministic opposing acceleration from drag and rolling resistance.

        Parameters
        ----------
        v : float
            Speed (m/s).

        Returns
        -------
        float
            Deceleration term (m/s^2), non-negative.
        """
        a_roll = self._p.c_rr * self._p.g
        a_drag = self._p.c_d_a_over_m * (v * v)
        return float(max(0.0, a_roll + a_drag))

    # ------------------------------------------------------------------
    # Public API (scalar)
    # ------------------------------------------------------------------

    def step(self, state: State, action: Tuple[float, float], dt: float) -> State:
        """
        Advance vehicle state by one fixed timestep.

        Parameters
        ----------
        state : State
            Current vehicle state.
        action : tuple[float, float]
            (acceleration_command, steering_rate_command)
            acceleration_command is in m/s^2
            steering_rate_command is in rad/s
        dt : float
            Fixed timestep (seconds).

        Returns
        -------
        State
            New immutable state.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        a_cmd_raw, steering_rate = action

        # integrate steering with rate limits
        new_steering = self._apply_steering_limits(state.steering_angle, steering_rate, dt)

        # compute effective curvature and lateral accel with saturation
        v = float(state.velocity)
        kappa_eff, a_lat_eff = self._effective_curvature_and_lateral_accel(v, new_steering)

        # clip commanded longitudinal accel
        a_cmd = float(np.clip(a_cmd_raw, -self._p.max_acceleration, self._p.max_acceleration))

        # friction circle combined-slip limit
        a_fric = self._friction_limited_longitudinal_accel(a_cmd, a_lat_eff)

        # apply drag + rolling resistance
        a_resist = self._drag_and_rolling_decel(v)
        a_net = a_fric - a_resist

        # integrate speed (clamp to [0, max_velocity])
        new_velocity = v + a_net * dt
        new_velocity = float(np.clip(new_velocity, 0.0, self._p.max_velocity))

        # slip angle approximation beta (kinematic)
        tan_delta = float(np.tan(new_steering))
        beta = float(np.arctan(self._p.rear_axle_ratio * tan_delta))

        # heading rate from effective curvature (more stable under lateral saturation)
        d_heading = new_velocity * kappa_eff

        # integrate pose
        new_heading = float(state.heading + d_heading * dt)
        dx = new_velocity * float(np.cos(new_heading + beta))
        dy = new_velocity * float(np.sin(new_heading + beta))
        new_x = float(state.x + dx * dt)
        new_y = float(state.y + dy * dt)

        return state.copy_with(
            x=new_x,
            y=new_y,
            heading=new_heading,
            velocity=new_velocity,
            steering_angle=new_steering,
        )

    # ------------------------------------------------------------------
    # Public API (vectorized)
    # ------------------------------------------------------------------

    def step_array(self, states: StateArray, actions: FloatArray, dt: float) -> StateArray:
        """
        Vectorized vehicle step for evolution / batch evaluation.

        Parameters
        ----------
        states : StateArray
            Batched vehicle states.
        actions : ndarray of shape (N, 2)
            Columns: [acceleration_command, steering_rate_command]
        dt : float
            Fixed timestep.

        Returns
        -------
        StateArray
            New batched state.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("actions must have shape (N, 2).")
        if actions.shape[0] != states.batch_size:
            raise ValueError("actions batch size must match states.batch_size.")

        p = self._p
        n = states.batch_size

        # --- Unpack (views) ---
        x = states.x
        y = states.y
        heading = states.heading
        v = states.velocity
        delta = states.steering_angle

        # --- Commands ---
        a_cmd_raw = actions[:, 0]
        steer_rate_raw = actions[:, 1]

        # --- Steering integrate + clamp (optimized: single clip per op) ---
        steer_rate = np.clip(steer_rate_raw, -p.max_steering_rate, p.max_steering_rate)
        delta_new = delta + steer_rate * dt
        delta_new = np.clip(delta_new, -p.max_steering_angle, p.max_steering_angle)

        # --- Geometry curvature ---
        tan_delta = np.tan(delta_new)
        kappa_geom = tan_delta / p.wheelbase  # (N,)

        # --- Lateral accel saturation ---
        v2 = v * v
        a_lat_geom = v2 * kappa_geom

        # kappa_eff initialized to geom; then replace where saturated
        kappa_eff = kappa_geom.copy()
        a_lat_eff = a_lat_geom.copy()

        # mask where |a_lat| > a_lat_max
        a_lat_max = p.a_lat_max
        sat_mask = np.abs(a_lat_geom) > a_lat_max
        if np.any(sat_mask):
            eps = 1e-9
            denom = np.maximum(v2, eps)
            kappa_sat = np.sign(kappa_geom) * (a_lat_max / denom)
            # apply saturation only where needed
            kappa_eff[sat_mask] = kappa_sat[sat_mask]
            a_lat_eff[sat_mask] = (v2 * kappa_eff)[sat_mask]

        # --- Longitudinal accel clip ---
        a_cmd = np.clip(a_cmd_raw, -p.max_acceleration, p.max_acceleration)

        # --- Friction circle combined-slip ---
        a_total_max = p.mu * p.g
        if a_total_max <= 0.0:
            a_fric = np.zeros((n,), dtype=np.float64)
        else:
            rem2 = (a_total_max * a_total_max) - (a_lat_eff * a_lat_eff)
            rem2 = np.maximum(rem2, 0.0)
            a_long_max = np.sqrt(rem2)
            a_fric = np.clip(a_cmd, -a_long_max, a_long_max)

        # --- Drag + rolling resistance (always opposing forward motion) ---
        a_roll = p.c_rr * p.g
        a_drag = p.c_d_a_over_m * v2
        a_resist = a_roll + a_drag

        a_net = a_fric - a_resist

        # --- Integrate speed ---
        v_new = v + a_net * dt
        v_new = np.clip(v_new, 0.0, p.max_velocity)

        # --- Slip angle beta ---
        beta = np.arctan(p.rear_axle_ratio * tan_delta)

        # --- Integrate heading using effective curvature ---
        heading_new = heading + (v_new * kappa_eff) * dt

        # --- Integrate position (use heading_new for slightly better stability) ---
        cos_term = np.cos(heading_new + beta)
        sin_term = np.sin(heading_new + beta)

        x_new = x + (v_new * cos_term) * dt
        y_new = y + (v_new * sin_term) * dt

        return StateArray(
            x=x_new,
            y=y_new,
            heading=heading_new,
            velocity=v_new,
            steering_angle=delta_new,
            progress_s=states.progress_s,
        )
