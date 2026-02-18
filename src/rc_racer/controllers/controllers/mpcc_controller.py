"""
mpcc_controller.py

High-accuracy lap-time MPCC-style controller using CasADi with IPOPT
(nonlinear interior-point).

AGENTS Layer (Competition Layer)
--------------------------------
This module implements a deterministic controller that follows the
:class:`rc_racer.agents.base_controller.BaseController` interface.

Design Goals
------------
- Prioritize robustness and solution quality over speed.
- Solve the full nonlinear program (NLP) with IPOPT.
- Make staying on track the top priority (hard constraints).
- Optimize for fast lap progress (maximize forward progress) rather than
  staying near the centerline.
- Use a curvature-aware speed target: high top speed on straights, lower speed
  through corners.

Solver Strategy
---------------
- CasADi builds a symbolic NLP once (decision variables are controls only).
- IPOPT solves the nonlinear problem with barrier methods and Newton steps.
- Warm-start is implemented by shifting the previous optimal control sequence.

Notes
-----
- This controller predicts using a symbolic clone of the VehicleModel's scalar
  step logic (steering limits, curvature saturation, friction circle, drag).
- Control blocking is supported: each control decision is held constant for
  ``control_block_steps`` internal integration steps, increasing prediction span
  without increasing decision variables.
- Braking is allowed to be stronger than acceleration by a configurable ratio.

Dependencies
------------
- casadi (pip install casadi)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from rc_racer.agents.base_controller import BaseController
from rc_racer.core.state import State
from rc_racer.core.track import Track
from rc_racer.core.vehicle_factory import VehicleFactory
from rc_racer.core.vehicle_model import VehicleParams

try:
    import casadi as ca
except Exception as exc:  # pragma: no cover
    raise ImportError("MpccController requires CasADi. Install with: pip install casadi") from exc


FloatArray = NDArray[np.float64]
Action = Tuple[float, float]


@dataclass(frozen=True)
class MpccConfig:
    """
    Configuration for the IPOPT lap-time MPCC controller.

    Parameters
    ----------
    dt : float
        Prediction integration timestep (seconds). Should match environment dt.
    horizon_steps : int
        Number of MPC decision stages (control knots).
    control_block_steps : int
        Number of internal integration steps per decision stage (move blocking factor).
        Total predicted time horizon is ``horizon_steps * control_block_steps * dt``.

    w_progress : float
        Weight for progress maximization (higher -> more aggressive forward motion).
    w_speed : float
        Weight for speed tracking to curvature-aware reference speed.
    w_u_acc : float
        Weight for acceleration command squared.
    w_u_steer_rate : float
        Weight for steering rate command squared.
    w_v_min : float
        Weight for standstill prevention penalty.
    v_min : float
        Soft minimum speed for standstill prevention (m/s).

    v_ref_max : float
        Maximum (top) target speed on straights (m/s).
    a_lat_target_ratio : float
        Fraction of vehicle lateral acceleration limit used to compute curvature-aware speed target.
        Typical range: 0.75 .. 0.95.

    ds_eps : float
        Finite-difference step in meters for computing track tangent via interpolants.
        Must be meters-scale (e.g., 0.25..1.0) for numerical stability.

    brake_ratio : float
        Maximum braking is ``brake_ratio * max_acceleration`` (as a positive factor).
        Example: 2.0 means you can brake twice as hard as you accelerate.

    ipopt_max_iter : int
        Maximum IPOPT iterations per solve.
    ipopt_tol : float
        IPOPT convergence tolerance.
    ipopt_print_level : int
        IPOPT verbosity level (0 silent, higher is more verbose).
    linear_solver : str
        IPOPT linear solver (e.g., "mumps").

    solver_verbosity : bool
        If True, enables more printing from CasADi / IPOPT.
    """

    dt: float = 0.02
    horizon_steps: int = 20
    control_block_steps: int = 5

    # Lap time / progress objective
    w_progress: float = 80.0

    # Curvature-aware speed target
    v_ref_max: float = 50.0
    a_lat_target_ratio: float = 0.85
    w_speed: float = 6.0

    # Control regularization
    w_u_acc: float = 0.03
    w_u_steer_rate: float = 0.05

    # Standstill prevention
    v_min: float = 0.5
    w_v_min: float = 5000.0

    # Track tangent finite difference (meters)
    ds_eps: float = 0.5

    # Braking vs acceleration
    brake_ratio: float = 2.0

    # IPOPT settings
    ipopt_max_iter: int = 200
    ipopt_tol: float = 1e-6
    ipopt_print_level: int = 0
    linear_solver: str = "mumps"

    solver_verbosity: bool = False


class MpccController(BaseController):
    """
    High-accuracy lap-time MPCC-style controller using IPOPT.

    Parameters
    ----------
    track : Track
        Immutable track geometry.
    config : MpccConfig | None
        Controller configuration.
    vehicle_params : VehicleParams | None
        Vehicle parameters used inside the prediction model. If None, uses VehicleFactory default.

    Notes
    -----
    - Deterministic and synchronous.
    - Hard track boundary constraints are enforced at every internal substep.
    - The objective maximizes progress along the track tangent and follows a
      curvature-aware speed target (high on straights, lower in corners).
    """

    def __init__(
        self,
        *,
        track: Track,
        config: MpccConfig | None = None,
        vehicle_params: VehicleParams | None = None,
    ) -> None:
        super().__init__()
        self._track: Track = track
        self._cfg: MpccConfig = config if config is not None else MpccConfig()
        self._p: VehicleParams = (
            vehicle_params if vehicle_params is not None else VehicleFactory.create_params("default")
        )

        # Warm-start storage: flattened U = [a0, sr0, a1, sr1, ...]
        self._u_prev: FloatArray | None = None

        # Debug values from the last solve
        self._debug_last: Dict[str, float] = {}

        # Last predicted path (from last optimal controls)
        self._last_predicted_path: FloatArray | None = None

        # Build solver once
        self._solver, self._nlp_struct, self._cost_fun = self._build_solver()

    # ==========================================================
    # Lifecycle
    # ==========================================================

    def reset(self) -> None:
        """
        Reset internal controller state.
        """
        self._u_prev = None
        self._debug_last = {}
        self._last_predicted_path = None

    # ==========================================================
    # Core API
    # ==========================================================

    def compute_action(self, state: State) -> Action:
        """
        Compute control action from current state.

        Parameters
        ----------
        state : State
            Immutable vehicle state.

        Returns
        -------
        tuple[float, float]
            (acceleration_command, steering_rate_command)
        """
        n = int(self._cfg.horizon_steps)

        s0, proj = self._track.project(np.array([state.x, state.y], dtype=np.float64))

        # Parameters: [x0, y0, psi0, v0, delta0, s0]
        pvec = np.array(
            [state.x, state.y, state.heading, state.velocity, state.steering_angle, float(s0)],
            dtype=np.float64,
        )

        z0 = self._initial_guess(n=n)

        try:
            sol = self._solver(
                x0=z0,
                lbx=self._nlp_struct["lbx"],
                ubx=self._nlp_struct["ubx"],
                lbg=self._nlp_struct["lbg"],
                ubg=self._nlp_struct["ubg"],
                p=pvec,
            )

            z_opt = np.array(sol["x"]).reshape((-1,), order="F").astype(np.float64)

            u_opt = z_opt[: 2 * n]
            self._u_prev = u_opt.copy()
            self._last_predicted_path = self._rollout_trajectory(state=state, u_opt=u_opt)

            a0 = float(u_opt[0])
            sr0 = float(u_opt[1])

            stats = self._solver.stats()
            costs = self._cost_fun(z_opt, pvec)

            self._debug_last = {
                "success": float(bool(stats.get("success", False))),
                "iters": float(stats.get("iter_count", 0)),
                "cost": float(np.array(sol["f"]).item()),
                "a_cmd": a0,
                "steer_rate_cmd": sr0,
                "s0": float(s0),
                "x_proj": float(proj[0]),
                "y_proj": float(proj[1]),
                "cost_total": float(costs[0]),
                "cost_speed": float(costs[1]),
                "cost_control": float(costs[2]),
                "cost_vmin": float(costs[3]),
                "cost_progress": float(costs[4]),
            }

            return a0, sr0

        except Exception:
            # If solve fails, prefer neutral safe action.
            a0 = 0.0
            sr0 = 0.0
            self._debug_last = {
                "success": 0.0,
                "iters": 0.0,
                "cost": float("nan"),
                "a_cmd": float(a0),
                "steer_rate_cmd": float(sr0),
                "s0": float(s0),
                "x_proj": float(proj[0]),
                "y_proj": float(proj[1]),
            }
            self._last_predicted_path = None
            return float(a0), float(sr0)

    @property
    def debug_values(self) -> Dict[str, float]:
        """
        Last-step debug values.

        Returns
        -------
        dict[str, float]
            Debug scalars such as last cost and action.
        """
        return dict(self._debug_last)

    def get_last_predicted_path(self) -> FloatArray | None:
        """
        Return last predicted (x, y) trajectory.

        Returns
        -------
        ndarray or None
            Shape (N, 2) if available, where N = horizon_steps * control_block_steps.
        """
        return self._last_predicted_path

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _initial_guess(self, *, n: int) -> FloatArray:
        """
        Create an initial guess for the decision vector Z = U(2n).

        Parameters
        ----------
        n : int
            Horizon length (decision stages).

        Returns
        -------
        ndarray
            Flattened decision vector of length 2*n.
        """
        u_dim = 2 * n

        if self._u_prev is None or self._u_prev.shape != (u_dim,):
            u_guess = np.zeros((u_dim,), dtype=np.float64)
            u_guess[0::2] = 0.3 * float(self._p.max_acceleration)
        else:
            u = self._u_prev
            u_guess = np.empty_like(u)
            u_guess[:-2] = u[2:]
            u_guess[-2:] = u[-2:]

        return u_guess

    def _rollout_trajectory(self, *, state: State, u_opt: FloatArray) -> FloatArray:
        """
        Roll out predicted trajectory using optimal control sequence.

        Parameters
        ----------
        state : State
            Current vehicle state.
        u_opt : ndarray
            Optimal flattened control vector of shape (2*n,).

        Returns
        -------
        ndarray
            Array of shape (N, 2) containing predicted (x, y) positions in world meters.
        """
        cfg = self._cfg
        p = self._p

        dt = float(cfg.dt)
        n = int(cfg.horizon_steps)
        block_steps = int(cfg.control_block_steps)

        max_acc = float(p.max_acceleration)
        max_brake = float(cfg.brake_ratio) * max_acc

        x = float(state.x)
        y = float(state.y)
        psi = float(state.heading)
        v = float(state.velocity)
        delta = float(state.steering_angle)

        path: list[list[float]] = []

        eps = 1e-9

        for k in range(n):
            a_cmd = float(u_opt[2 * k + 0])
            sr_cmd = float(u_opt[2 * k + 1])

            a_cmd = float(np.clip(a_cmd, -max_brake, max_acc))
            sr_cmd = float(np.clip(sr_cmd, -p.max_steering_rate, p.max_steering_rate))

            for _ in range(block_steps):
                delta = delta + sr_cmd * dt
                delta = float(np.clip(delta, -p.max_steering_angle, p.max_steering_angle))

                kappa_geom = float(np.tan(delta) / p.wheelbase)
                v2 = v * v
                kappa_max = float(p.a_lat_max / (v2 + eps))
                kappa_eff = float(np.clip(kappa_geom, -kappa_max, kappa_max))
                a_lat_eff = v2 * kappa_eff

                a_total_max = float(p.mu * p.g)
                rem2 = max(0.0, a_total_max * a_total_max - a_lat_eff * a_lat_eff)
                a_long_max = float(np.sqrt(rem2))
                a_fric = float(np.clip(a_cmd, -a_long_max, a_long_max))

                a_roll = float(p.c_rr * p.g)
                a_drag = float(p.c_d_a_over_m * v2)
                a_resist = max(0.0, a_roll + a_drag)
                a_net = a_fric - a_resist

                v = float(np.clip(v + a_net * dt, 0.0, p.max_velocity))

                beta = float(np.arctan(p.rear_axle_ratio * np.tan(delta)))

                psi = psi + (v * kappa_eff) * dt
                x = x + v * np.cos(psi + beta) * dt
                y = y + v * np.sin(psi + beta) * dt

                path.append([x, y])

        return np.asarray(path, dtype=np.float64)

    def _build_solver(self) -> tuple[ca.Function, Dict[str, FloatArray], ca.Function]:
        """
        Build the CasADi NLP solver and static bound structures using IPOPT.

        Returns
        -------
        solver : casadi.Function
            Configured CasADi solver.
        nlp_struct : dict[str, ndarray]
            Bounds and constraint arrays.
        cost_fun : casadi.Function
            Cost breakdown function ``cost_fun(Z, P)``.
        """
        cfg = self._cfg
        p = self._p

        n = int(cfg.horizon_steps)
        dt = float(cfg.dt)
        block_steps = max(1, int(cfg.control_block_steps))

        max_acc = float(p.max_acceleration)
        max_brake = float(cfg.brake_ratio) * max_acc

        # Track interpolants x(s), y(s)
        s_grid = np.asarray(self._track.arc_lengths, dtype=np.float64)
        x_grid = np.asarray(self._track.centerline[:, 0], dtype=np.float64)
        y_grid = np.asarray(self._track.centerline[:, 1], dtype=np.float64)

        # Curvature profile kappa(s) from centerline (numpy, once)
        dx_ds = np.gradient(x_grid, s_grid)
        dy_ds = np.gradient(y_grid, s_grid)
        ddx_ds2 = np.gradient(dx_ds, s_grid)
        ddy_ds2 = np.gradient(dy_ds, s_grid)

        denom = (dx_ds * dx_ds + dy_ds * dy_ds) ** 1.5
        denom = np.maximum(denom, 1e-9)

        kappa_grid = (dx_ds * ddy_ds2 - dy_ds * ddx_ds2) / denom
        kappa_grid = kappa_grid.astype(np.float64)

        x_ref_fun = ca.interpolant("x_ref", "bspline", [s_grid], x_grid)
        y_ref_fun = ca.interpolant("y_ref", "bspline", [s_grid], y_grid)
        kappa_fun = ca.interpolant("kappa_ref", "bspline", [s_grid], kappa_grid)

        # Decision variables: U = [a0, sr0, a1, sr1, ...]
        U = ca.MX.sym("U", 2 * n, 1)
        Z = U

        # Parameters: [x0, y0, psi0, v0, delta0, s0]
        P = ca.MX.sym("P", 6, 1)
        x0 = P[0]
        y0 = P[1]
        psi0 = P[2]
        v0 = P[3]
        delta0 = P[4]
        s0 = P[5]

        eps = 1e-9
        total_len = float(self._track.total_length)
        half_width = 0.5 * float(self._track.width)
        ds_eps = float(max(1e-6, cfg.ds_eps))

        def _clip(z: ca.MX, lo: float, hi: float) -> ca.MX:
            return ca.fmin(ca.fmax(z, lo), hi)

        def _norm2(vec2: ca.MX) -> ca.MX:
            return ca.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + eps)

        # Rollout state
        xk = x0
        yk = y0
        psik = psi0
        vk = v0
        deltak = delta0

        # Track progress state (integrated using projected forward speed)
        s_k = s0

        # Cost terms (time-scaled)
        cost_speed = 0.0
        cost_control = 0.0
        cost_vmin = 0.0
        cost_progress = 0.0

        # Hard constraints: g <= 0
        g_list: list[ca.MX] = []

        for k in range(n):
            a_cmd = U[2 * k + 0]
            sr_cmd = U[2 * k + 1]

            a_cmd = _clip(a_cmd, -max_brake, max_acc)
            sr_cmd = _clip(sr_cmd, -float(p.max_steering_rate), float(p.max_steering_rate))

            for _ in range(block_steps):
                # Steering integrate + clamp
                delta_next = deltak + sr_cmd * dt
                delta_next = _clip(delta_next, -float(p.max_steering_angle), float(p.max_steering_angle))

                # Curvature + lateral accel saturation
                kappa_geom = ca.tan(delta_next) / float(p.wheelbase)
                v2 = vk * vk
                kappa_max = float(p.a_lat_max) / (v2 + eps)
                kappa_eff = _clip(kappa_geom, -kappa_max, kappa_max)
                a_lat_eff = v2 * kappa_eff

                # Friction circle on longitudinal accel
                a_total_max = float(p.mu) * float(p.g)
                rem2 = a_total_max * a_total_max - a_lat_eff * a_lat_eff
                rem2 = ca.fmax(0.0, rem2)
                a_long_max = ca.sqrt(rem2)
                a_fric = _clip(a_cmd, -a_long_max, a_long_max)

                # Drag + rolling
                a_roll = float(p.c_rr) * float(p.g)
                a_drag = float(p.c_d_a_over_m) * (vk * vk)
                a_resist = ca.fmax(0.0, a_roll + a_drag)
                a_net = a_fric - a_resist

                # Speed integrate + clamp
                v_next = vk + a_net * dt
                v_next = _clip(v_next, 0.0, float(p.max_velocity))

                # Slip angle beta
                beta = ca.atan(float(p.rear_axle_ratio) * ca.tan(delta_next))

                # Heading integrate
                psi_next = psik + (v_next * kappa_eff) * dt

                # Pose integrate
                x_next = xk + v_next * ca.cos(psi_next + beta) * dt
                y_next = yk + v_next * ca.sin(psi_next + beta) * dt

                # Reference based on integrated progress
                s_ref = _clip(s_k, 0.0, total_len)
                px = x_ref_fun(s_ref)
                py = y_ref_fun(s_ref)

                # Tangent via finite differences (meters-scale + clipped)
                s_ref_fwd = _clip(s_ref + ds_eps, 0.0, total_len)
                px_fwd = x_ref_fun(s_ref_fwd)
                py_fwd = y_ref_fun(s_ref_fwd)

                tvec = ca.vertcat(px_fwd - px, py_fwd - py)
                t_hat = tvec / (_norm2(tvec) + 1e-9)
                n_hat = ca.vertcat(-t_hat[1], t_hat[0])

                # Errors in track frame
                e = ca.vertcat(x_next - px, y_next - py)
                e_cont = ca.dot(e, n_hat)

                # Hard track boundary: |e_cont| <= half_width
                g_list.append(e_cont - half_width)
                g_list.append(-e_cont - half_width)

                # Curvature-aware speed reference
                kappa_ref = kappa_fun(s_ref)
                a_lat_target = float(cfg.a_lat_target_ratio) * float(p.a_lat_max)
                v_ref_s = ca.sqrt(a_lat_target / (ca.fabs(kappa_ref) + 1e-6))
                v_ref_s = ca.fmin(v_ref_s, float(cfg.v_ref_max))

                cost_speed += float(cfg.w_speed) * ((v_next - v_ref_s) ** 2) * dt

                # Control cost (u held constant in time, accumulate per substep with dt)
                cost_control += float(cfg.w_u_acc) * (a_cmd * a_cmd) * dt
                cost_control += float(cfg.w_u_steer_rate) * (sr_cmd * sr_cmd) * dt

                # Standstill prevention
                v_def = ca.fmax(0.0, float(cfg.v_min) - v_next)
                cost_vmin += float(cfg.w_v_min) * (v_def * v_def) * dt

                # Progress maximization: project velocity onto tangent
                v_vec = ca.vertcat(
                    v_next * ca.cos(psi_next + beta),
                    v_next * ca.sin(psi_next + beta),
                )
                progress_rate = ca.dot(v_vec, t_hat)
                cost_progress += -float(cfg.w_progress) * progress_rate * dt

                # Integrate track progress
                s_k = s_k + progress_rate * dt

                # Advance internal state
                xk, yk, psik, vk, deltak = x_next, y_next, psi_next, v_next, delta_next

        cost_total = cost_speed + cost_control + cost_vmin + cost_progress

        g = ca.vertcat(*g_list) if len(g_list) > 0 else ca.MX.zeros(0, 1)
        nlp = {"x": Z, "f": cost_total, "g": g, "p": P}

        # Bounds for decision variables
        lbx = np.zeros((2 * n,), dtype=np.float64)
        ubx = np.zeros((2 * n,), dtype=np.float64)
        for k in range(n):
            lbx[2 * k + 0] = -max_brake
            ubx[2 * k + 0] = max_acc
            lbx[2 * k + 1] = -float(p.max_steering_rate)
            ubx[2 * k + 1] = float(p.max_steering_rate)

        # Bounds for constraints g <= 0
        m = 2 * n * block_steps
        lbg = -np.inf * np.ones((m,), dtype=np.float64)
        ubg = np.zeros((m,), dtype=np.float64)

        ipopt_opts: Dict[str, object] = {
            "ipopt.max_iter": int(cfg.ipopt_max_iter),
            "ipopt.tol": float(cfg.ipopt_tol),
            "ipopt.print_level": int(cfg.ipopt_print_level if cfg.solver_verbosity else 0),
            "ipopt.linear_solver": str(cfg.linear_solver),
            "ipopt.hessian_approximation": "exact",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.bound_relax_factor": 0.0,
            "ipopt.constr_viol_tol": float(cfg.ipopt_tol),
            "ipopt.compl_inf_tol": float(cfg.ipopt_tol),
            "ipopt.dual_inf_tol": float(cfg.ipopt_tol),
            "print_time": bool(cfg.solver_verbosity),
            "error_on_fail": False,
        }

        solver = ca.nlpsol("mpcc_ipopt", "ipopt", nlp, ipopt_opts)

        nlp_struct: Dict[str, FloatArray] = {
            "lbx": lbx,
            "ubx": ubx,
            "lbg": lbg,
            "ubg": ubg,
        }

        cost_fun = ca.Function(
            "cost_breakdown",
            [Z, P],
            [
                cost_total,
                cost_speed,
                cost_control,
                cost_vmin,
                cost_progress,
            ],
        )

        return solver, nlp_struct, cost_fun
