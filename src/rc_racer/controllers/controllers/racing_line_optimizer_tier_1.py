"""
Deterministic offline racing-line optimizer + global speed profile + rollout.

This module computes an approximate fastest lap plan by:
1) Optimizing a racing line as a lateral offset from the Track centerline.
2) Computing curvature along that optimized path.
3) Computing a global time-optimal speed profile v(s) with a forward-backward pass.
4) Generating per-step actions and rolling out the provided VehicleModel.

Design Goals
------------
- Offline (batch) and deterministic.
- No GUI, no threading, no randomness.
- Compatible with fast online tracking (e.g., MPCC tracking this precomputed line).

Notes
-----
This is a heuristic racing line optimizer (outside–inside–outside style). It is not
a full nonlinear OCP solver, but it’s fast and provides strong references.

See Also
--------
rc_racer.core.track.Track
rc_racer.core.vehicle_model.VehicleModel
rc_racer.core.vehicle_model.VehicleParams
rc_racer.core.state.State
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from rc_racer.core.state import State
from rc_racer.core.track import Track
from rc_racer.core.vehicle_model import VehicleModel, VehicleParams

FloatArray = NDArray[np.float64]
Action = Tuple[float, float]


@dataclass(frozen=True)
class Tier1RacingLineOptimizerConfig:
    """
    Configuration for the offline racing-line optimizer.

    Parameters
    ----------
    ds : float
        Uniform arc-length step for internal line optimization grid.
    margin : float
        Safety margin inside track half-width (meters).
    iterations : int
        Number of smoothing iterations.
    beta_outside : float
        Strength of "go outside in corners" target offset in [0, 1].
    kappa_scale_quantile : float
        Quantile used to scale curvature magnitude into [0,1] for target offset.
        Values in [0.7, 0.95] work well.
    smooth_alpha : float
        Smoothing factor per iteration in [0, 1]. Higher -> smoother.
    attract_gamma : float
        Attraction factor to target offset in [0, 1]. Higher -> follows target more.
    """

    ds: float = 0.25
    margin: float = 0.20
    iterations: int = 400

    beta_outside: float = 0.95
    kappa_scale_quantile: float = 0.90

    smooth_alpha: float = 0.25
    attract_gamma: float = 0.08


@dataclass(frozen=True)
class SpeedProfileConfig:
    """
    Configuration for the global speed profile + rollout.

    Parameters
    ----------
    dt : float
        Output timestep for action/state sequence.
    v_start : float
        Start speed [m/s].
    v_end : float | None
        Optional end speed constraint [m/s]. If None, no explicit constraint.
    speed_kp : float
        P gain converting speed error to net accel demand during rollout.
    steer_kp : float
        P gain converting steering angle error to steering rate during rollout.
    safety_eps : float
        Numerical epsilon.
    """

    dt: float = 0.05
    v_start: float = 0.0
    v_end: float | None = 0.0

    speed_kp: float = 1.0
    steer_kp: float = 6.0

    safety_eps: float = 1e-9


@dataclass(frozen=True)
class RacingLinePlan:
    """
    Result container for the offline racing line plan.

    Parameters
    ----------
    path_points : ndarray of shape (M, 2)
        Optimized racing line points sampled on a uniform internal s-grid.
    s_grid : ndarray of shape (M,)
        Arc-length parameter along the optimized path (cumulative).
    kappa_s : ndarray of shape (M,)
        Curvature along the optimized path.
    v_s : ndarray of shape (M,)
        Global time-optimal speed profile along s_grid.
    states : list[State]
        Rolled-out states at dt spacing.
    actions : ndarray of shape (T, 2)
        Actions (accel_cmd, steer_rate_cmd) applied between states.
    times_s : ndarray of shape (T+1,)
        Simulation time stamps.
    """

    path_points: FloatArray
    s_grid: FloatArray
    kappa_s: FloatArray
    v_s: FloatArray
    states: list[State]
    actions: FloatArray
    times_s: FloatArray


class Tier1RacingLineOptimizer:
    """
    Offline deterministic racing-line optimizer + speed profile planner.
    """

    def __init__(
        self,
        *,
        line_cfg: Tier1RacingLineOptimizerConfig | None = None,
        speed_cfg: SpeedProfileConfig | None = None,
        show_progress: bool = True,
    ) -> None:
        self._line_cfg = line_cfg if line_cfg is not None else Tier1RacingLineOptimizerConfig()
        self._speed_cfg = speed_cfg if speed_cfg is not None else SpeedProfileConfig()
        self._show_progress = bool(show_progress)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        *,
        track: Track,
        vehicle_model: VehicleModel,
        vehicle_params: VehicleParams,
    ) -> RacingLinePlan:
        """
        Optimize racing line and compute global fastest-lap plan.

        Parameters
        ----------
        track : Track
            Immutable track.
        vehicle_model : VehicleModel
            Vehicle model used for rollout.
        vehicle_params : VehicleParams
            Parameters used for speed profile + action limits.

        Returns
        -------
        RacingLinePlan
            Offline plan result.
        """
        # 1) Optimize path geometry
        path_points = self._optimize_racing_line(track=track)

        # 2) Compute s_grid and curvature for optimized path
        s_grid = self._cumulative_arclength(path_points)
        kappa_s = self._polyline_curvature(path_points, eps=self._speed_cfg.safety_eps)

        # 3) Global time-optimal speed profile along s_grid
        v_s = self._time_optimal_speed_profile(
            s_grid=s_grid,
            kappa_s=kappa_s,
            params=vehicle_params,
            v_start=float(self._speed_cfg.v_start),
            v_end=self._speed_cfg.v_end,
            eps=float(self._speed_cfg.safety_eps),
        )

        # 4) Convert to dt actions and roll out vehicle model
        states, actions, times = self._rollout(
            track=track,
            vehicle_model=vehicle_model,
            params=vehicle_params,
            path_points=path_points,
            s_grid=s_grid,
            kappa_s=kappa_s,
            v_s=v_s,
        )

        return RacingLinePlan(
            path_points=path_points,
            s_grid=s_grid,
            kappa_s=kappa_s,
            v_s=v_s,
            states=states,
            actions=actions,
            times_s=times,
        )

    # ------------------------------------------------------------------
    # Racing line geometry
    # ------------------------------------------------------------------

    def _optimize_racing_line(self, *, track: Track) -> FloatArray:
        """
        Optimize a racing line as an offset from the centerline.

        Parameters
        ----------
        track : Track

        Returns
        -------
        ndarray of shape (M, 2)
            Optimized path points.
        """
        cfg = self._line_cfg
        eps = float(self._speed_cfg.safety_eps)

        # Resample centerline to uniform s
        s_u, cl_u = self._resample_centerline(track=track, ds=float(cfg.ds))
        tang_u, norm_u = self._tangent_and_normal(cl_u, eps=eps)

        # Curvature of centerline (used to define "go outside" target)
        kappa_c = self._polyline_curvature(cl_u, eps=eps)
        k_abs = np.abs(kappa_c)

        # Scale curvature magnitude into [0,1]
        scale = float(np.quantile(k_abs, float(cfg.kappa_scale_quantile)))
        scale = max(scale, eps)
        k01 = np.clip(k_abs / scale, 0.0, 1.0)

        half_width = 0.5 * float(track.width)
        bound = max(0.0, half_width - float(cfg.margin))

        # Target offset: go to outside of corner
        # If kappa > 0 (left normal points "left"), we typically want offset negative (to the right side)
        # so we use -sign(kappa).
        target = (-np.sign(kappa_c) * float(cfg.beta_outside) * bound * k01).astype(np.float64)

        # Initialize offsets at zero
        y = np.zeros_like(target, dtype=np.float64)

        it_range = range(int(cfg.iterations))
        if self._show_progress:
            it_range = tqdm(it_range, desc="Optimizing racing line", unit="iter")

        # Iterative smoothing with attraction to target, periodic boundary
        a = float(cfg.smooth_alpha)
        g = float(cfg.attract_gamma)

        n = int(y.shape[0])
        for _ in it_range:
            y_prev = np.roll(y, 1)
            y_next = np.roll(y, -1)

            # Smooth toward neighbor average (elastic band)
            y_smooth = (1.0 - a) * y + a * (0.5 * (y_prev + y_next))

            # Attract toward target in corners
            y = (1.0 - g) * y_smooth + g * target

            # Project back into bounds
            y = np.clip(y, -bound, bound)

            # Optional: tiny damping to avoid jitter
            # (keeps deterministic, helps convergence)
            y *= 0.9995

        # Build final path: cl + y*n
        path = cl_u + y[:, None] * norm_u
        return path.astype(np.float64)

    @staticmethod
    def _resample_centerline(*, track: Track, ds: float) -> tuple[FloatArray, FloatArray]:
        """
        Uniformly resample track centerline by arc-length.

        Parameters
        ----------
        track : Track
        ds : float

        Returns
        -------
        s_u : ndarray of shape (M,)
        cl_u : ndarray of shape (M, 2)
        """
        if ds <= 0.0:
            raise ValueError("ds must be positive.")

        s0 = 0.0
        s1 = float(track.total_length)
        m = int(np.floor((s1 - s0) / ds)) + 1
        m = max(m, 3)

        s_u = np.linspace(s0, s1, m, dtype=np.float64)

        s_src = np.asarray(track.arc_lengths, dtype=np.float64)
        x_src = np.asarray(track.centerline[:, 0], dtype=np.float64)
        y_src = np.asarray(track.centerline[:, 1], dtype=np.float64)

        x_u = np.interp(s_u, s_src, x_src).astype(np.float64)
        y_u = np.interp(s_u, s_src, y_src).astype(np.float64)

        cl_u = np.column_stack([x_u, y_u]).astype(np.float64)
        return s_u, cl_u

    @staticmethod
    def _tangent_and_normal(points: FloatArray, *, eps: float) -> tuple[FloatArray, FloatArray]:
        """
        Compute unit tangents and normals along a polyline.

        Parameters
        ----------
        points : ndarray of shape (M, 2)
        eps : float

        Returns
        -------
        tangents : ndarray of shape (M, 2)
        normals : ndarray of shape (M, 2)
        """
        m = int(points.shape[0])
        if m < 2:
            raise ValueError("Need at least 2 points.")

        # Central differences for tangent
        d = np.zeros_like(points)
        d[1:-1] = points[2:] - points[:-2]
        d[0] = points[1] - points[0]
        d[-1] = points[-1] - points[-2]

        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        nrm = np.maximum(nrm, eps)
        t = d / nrm

        # Left normal
        n = np.column_stack([-t[:, 1], t[:, 0]]).astype(np.float64)
        return t.astype(np.float64), n

    @staticmethod
    def _cumulative_arclength(points: FloatArray) -> FloatArray:
        """
        Compute cumulative arc-length of a polyline.

        Parameters
        ----------
        points : ndarray of shape (M, 2)

        Returns
        -------
        ndarray of shape (M,)
        """
        diffs = np.diff(points, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        s = np.zeros((points.shape[0],), dtype=np.float64)
        s[1:] = np.cumsum(seg)
        return s

    @staticmethod
    def _polyline_curvature(points: FloatArray, *, eps: float) -> FloatArray:
        """
        Approximate signed curvature along a 2D polyline using a 3-point formula.

        Parameters
        ----------
        points : ndarray of shape (M, 2)
        eps : float

        Returns
        -------
        ndarray of shape (M,)
        """
        m = int(points.shape[0])
        if m < 3:
            return np.zeros((m,), dtype=np.float64)

        x = points[:, 0]
        y = points[:, 1]
        kappa = np.zeros((m,), dtype=np.float64)

        for i in range(1, m - 1):
            x1, y1 = float(x[i - 1]), float(y[i - 1])
            x2, y2 = float(x[i]), float(y[i])
            x3, y3 = float(x[i + 1]), float(y[i + 1])

            ax, ay = x2 - x1, y2 - y1
            bx, by = x3 - x2, y3 - y2
            cx, cy = x3 - x1, y3 - y1

            la = float(np.hypot(ax, ay))
            lb = float(np.hypot(bx, by))
            lc = float(np.hypot(cx, cy))
            denom = max(la * lb * lc, eps)

            cross = ax * (y3 - y1) - ay * (x3 - x1)  # 2*Area signed
            kappa[i] = float(cross) / denom

        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]
        return kappa

    # ------------------------------------------------------------------
    # Speed profile (global v(s))
    # ------------------------------------------------------------------

    @staticmethod
    def _resistance_decel(params: VehicleParams, v: float) -> float:
        """
        Opposing acceleration from rolling + aero.

        Parameters
        ----------
        params : VehicleParams
        v : float

        Returns
        -------
        float
            Nonnegative deceleration [m/s^2].
        """
        a_roll = float(params.c_rr) * float(params.g)
        a_drag = float(params.c_d_a_over_m) * (v * v)
        return float(max(0.0, a_roll + a_drag))

    @staticmethod
    def _friction_longitudinal_limit(params: VehicleParams, a_lat: float) -> float:
        """
        Max longitudinal accel magnitude allowed by friction circle.

        Parameters
        ----------
        params : VehicleParams
        a_lat : float

        Returns
        -------
        float
            Max |a_long| [m/s^2].
        """
        a_total = float(params.mu) * float(params.g)
        if a_total <= 0.0:
            return 0.0
        rem2 = a_total * a_total - float(a_lat) * float(a_lat)
        if rem2 <= 0.0:
            return 0.0
        return float(np.sqrt(rem2))

    def _time_optimal_speed_profile(
        self,
        *,
        s_grid: FloatArray,
        kappa_s: FloatArray,
        params: VehicleParams,
        v_start: float,
        v_end: float | None,
        eps: float,
    ) -> FloatArray:
        """
        Compute global speed profile v(s) via forward-backward pass.

        Parameters
        ----------
        s_grid : ndarray of shape (M,)
        kappa_s : ndarray of shape (M,)
        params : VehicleParams
        v_start : float
        v_end : float | None
        eps : float

        Returns
        -------
        ndarray of shape (M,)
        """
        m = int(s_grid.shape[0])
        if m < 2:
            raise ValueError("s_grid must have at least 2 points.")

        ds = float(np.mean(np.diff(s_grid)))
        ds = max(ds, eps)

        # Lateral limit -> curvature speed cap
        k_abs = np.abs(kappa_s)
        v_curve = np.sqrt(float(params.a_lat_max) / np.maximum(k_abs, eps))
        v_lim = np.minimum(v_curve, float(params.max_velocity)).astype(np.float64)

        v = v_lim.copy()
        v[0] = float(np.clip(v_start, 0.0, v_lim[0]))
        if v_end is not None:
            v[-1] = float(np.clip(float(v_end), 0.0, v_lim[-1]))

        # Forward pass: accel-limited
        for i in range(0, m - 1):
            vi = float(v[i])
            a_lat = (vi * vi) * float(kappa_s[i])
            a_long_fric = self._friction_longitudinal_limit(params, a_lat)
            a_cmd_max = min(float(params.max_acceleration), a_long_fric)

            a_net_max = max(0.0, a_cmd_max - self._resistance_decel(params, vi))

            v_next_max = np.sqrt(max(0.0, vi * vi + 2.0 * a_net_max * ds))
            v[i + 1] = float(min(v[i + 1], v_next_max, v_lim[i + 1]))

        # Backward pass: braking-limited (resistance helps)
        for i in range(m - 2, -1, -1):
            vi1 = float(v[i + 1])
            a_lat = (vi1 * vi1) * float(kappa_s[i + 1])
            a_long_fric = self._friction_longitudinal_limit(params, a_lat)
            a_cmd_max = min(float(params.max_acceleration), a_long_fric)

            a_net_brake = max(0.0, a_cmd_max + self._resistance_decel(params, vi1))

            v_i_max = np.sqrt(max(0.0, vi1 * vi1 + 2.0 * a_net_brake * ds))
            v[i] = float(min(v[i], v_i_max, v_lim[i]))

        return v.astype(np.float64)

    # ------------------------------------------------------------------
    # Convert to dt actions and roll out vehicle model
    # ------------------------------------------------------------------

    def _rollout(
        self,
        *,
        track: Track,
        vehicle_model: VehicleModel,
        params: VehicleParams,
        path_points: FloatArray,
        s_grid: FloatArray,
        kappa_s: FloatArray,
        v_s: FloatArray,
    ) -> tuple[list[State], FloatArray, FloatArray]:
        """
        Roll out vehicle model to produce states and actions at fixed dt.

        Parameters
        ----------
        track : Track
        vehicle_model : VehicleModel
        params : VehicleParams
        path_points : ndarray of shape (M,2)
        s_grid : ndarray of shape (M,)
        kappa_s : ndarray of shape (M,)
        v_s : ndarray of shape (M,)

        Returns
        -------
        states : list[State]
        actions : ndarray of shape (T,2)
        times : ndarray of shape (T+1,)
        """
        cfg = self._speed_cfg
        dt = float(cfg.dt)
        eps = float(cfg.safety_eps)

        # Initial state placed at first optimized point, heading from first segment
        p0 = path_points[0]
        p1 = path_points[1]
        heading0 = float(np.arctan2(float(p1[1] - p0[1]), float(p1[0] - p0[0])))

        state = State(
            x=float(p0[0]),
            y=float(p0[1]),
            heading=heading0,
            velocity=float(max(0.0, cfg.v_start)),
            steering_angle=0.0,
            progress_s=0.0,
        )

        actions: list[Action] = []
        states: list[State] = [state]
        times: list[float] = [0.0]

        t = 0.0

        # ------------------------------------------------------------------
        # Better max step estimate using integral of 1/v(s)
        # ------------------------------------------------------------------

        lap_time_est = float(np.trapz(1.0 / np.maximum(v_s, 1e-3), s_grid))
        max_steps = int(1.2 * lap_time_est / dt)  # 20% buffer
        max_steps = max(max_steps, 2000)

        step_iter = range(max_steps)
        if self._show_progress:
            step_iter = tqdm(step_iter, desc="Rollout", unit="step")

        for _ in step_iter:
            s = float(state.progress_s)

            if s >= float(track.total_length):
                break

            # Desired speed and curvature
            v_tgt = float(np.interp(s, s_grid, v_s))
            k_tgt = float(np.interp(s, s_grid, kappa_s))

            # Steering target
            delta_des = float(np.arctan(float(params.wheelbase) * k_tgt))
            delta_des = float(np.clip(
                delta_des,
                -float(params.max_steering_angle),
                float(params.max_steering_angle),
            ))

            delta_err = delta_des - float(state.steering_angle)
            steer_rate_cmd = float(cfg.steer_kp) * (delta_err / dt)
            steer_rate_cmd = float(np.clip(
                steer_rate_cmd,
                -float(params.max_steering_rate),
                float(params.max_steering_rate),
            ))

            # Speed tracking
            v_err = v_tgt - float(state.velocity)
            a_net_des = float(cfg.speed_kp) * (v_err / dt)

            a_cmd = a_net_des + self._resistance_decel(params, float(state.velocity))
            a_cmd = float(np.clip(
                a_cmd,
                -float(params.max_acceleration),
                float(params.max_acceleration),
            ))

            a_lat = (float(state.velocity) ** 2) * k_tgt
            a_long_max = self._friction_longitudinal_limit(params, a_lat)
            a_cmd = float(np.clip(a_cmd, -a_long_max, a_long_max))

            action: Action = (float(a_cmd), float(steer_rate_cmd))

            next_state = vehicle_model.step(state, action, dt)

            # Track projection (authoritative progress)
            s_next, _ = track.project(
                np.asarray([next_state.x, next_state.y], dtype=np.float64)
            )
            next_state = next_state.copy_with(progress_s=float(s_next))

            actions.append(action)
            states.append(next_state)

            t += dt
            times.append(t)
            state = next_state

        return (
            states,
            np.asarray(actions, dtype=np.float64),
            np.asarray(times, dtype=np.float64),
        )