"""
mpcc_controller.py

High-accuracy MPCC-style controller using CasADi with IPOPT (nonlinear interior-point).

AGENTS Layer (Competition Layer)
--------------------------------
This module implements a deterministic MPCC-style controller following the
:class:`rc_racer.agents.base_controller.BaseController` interface.

Design Goals
------------
- Prioritize solution accuracy and robustness over speed.
- Solve the full nonlinear program (NLP) with IPOPT (interior-point method).
- Deterministic behavior (given deterministic inputs / track projection).

Solver Strategy
---------------
- CasADi builds a symbolic NLP once (decision variables are controls + slacks).
- IPOPT solves the nonlinear problem with barrier methods and Newton steps.
- Warm-start is implemented by shifting the previous optimal control sequence.

Notes
-----
- This controller predicts using a symbolic clone of the VehicleModel's scalar
  step logic (steering limits, curvature saturation, friction circle, drag).
- The Environment remains authoritative for stepping; best results occur when
  cfg.dt matches environment dt and vehicle_params match the environment vehicle model.
- "Control blocking" is supported: each control decision is held constant for
  ``control_block_steps`` internal integration steps, increasing prediction span
  without increasing decision variables.

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
    Configuration for the IPOPT MPCC controller.

    Parameters
    ----------
    dt : float
        Prediction integration timestep (seconds). Should match environment dt for best results.
    horizon_steps : int
        Number of MPC decision stages (control knots).
    control_block_steps : int
        Number of internal integration steps per decision stage (move blocking factor).
        Total predicted time horizon is ``horizon_steps * control_block_steps * dt``.
    v_ref : float
        Moving reference speed along the centerline (m/s).

    w_contour : float
        Weight for contouring (normal) error squared.
    w_lag : float
        Weight for lag (tangent) error squared.
    w_speed : float
        Weight for speed tracking (v - v_ref)^2.
    w_u_acc : float
        Weight for acceleration command squared.
    w_u_steer_rate : float
        Weight for steering rate command squared.

    v_min : float
        Soft minimum speed for standstill prevention (m/s).
    w_v_min : float
        Weight for standstill-prevention penalty.

    w_slack : float
        Penalty weight for track-boundary slack (near-hard constraint when large).
    slack_max : float
        Upper bound on slack (meters). Prevents "buying" infinite constraint violation.
        A good default is a few track widths when debugging.

    ds_eps : float
        Finite-difference step in meters for computing track tangent via interpolants.
        Must be meters-scale (e.g., 0.25..1.0) for numerical stability.

    ipopt_max_iter : int
        Maximum IPOPT iterations per solve.
    ipopt_tol : float
        IPOPT convergence tolerance.
    ipopt_print_level : int
        IPOPT verbosity level (0 silent, higher is more verbose).
    linear_solver : str
        IPOPT linear solver (commonly "mumps"; if unavailable, IPOPT will error).

    solver_verbosity : bool
        If True, enables more printing from CasADi / IPOPT.
    """

    dt: float = 0.02
    horizon_steps: int = 20
    control_block_steps: int = 5
    v_ref: float = 14.0

    w_contour: float = 8.0
    w_lag: float = 2.0
    w_speed: float = 1.5
    w_u_acc: float = 0.03
    w_u_steer_rate: float = 0.05

    v_min: float = 0.5
    w_v_min: float = 5000.0

    w_slack: float = 1.0e8
    slack_max: float = 0.1

    ds_eps: float = 0.5

    ipopt_max_iter: int = 200
    ipopt_tol: float = 1e-6
    ipopt_print_level: int = 0
    linear_solver: str = "mumps"

    solver_verbosity: bool = False


class MpccController(BaseController):
    """
    High-accuracy MPCC-style controller using IPOPT.

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
    - Uses Track centerline interpolation for reference and tangent/normal frame.
    - Uses soft track boundary constraints with slack variables (near-hard when w_slack is large).
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

        # Build solver once
        self._solver, self._nlp_struct, self._cost_fun = self._build_solver()

        # Last predicted path
        self._last_predicted_path: FloatArray | None = None

    # ==========================================================
    # Lifecycle
    # ==========================================================

    def reset(self) -> None:
        """
        Reset internal controller state.
        """
        self._u_prev = None
        self._debug_last = {}

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

        # params = [x0, y0, psi0, v0, delta0, s0]
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
            self._last_predicted_path = self._rollout_trajectory(state, u_opt)

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
                "cost_contour": float(costs[1]),
                "cost_lag": float(costs[2]),
                "cost_speed": float(costs[3]),
                "cost_control": float(costs[4]),
                "cost_vmin": float(costs[5]),
                "cost_slack": float(costs[6]),
            }

            return a0, sr0

        except Exception:
            # Accuracy-first controller: if solve fails, prefer neutral + safe action
            # rather than reusing possibly-saturated previous action.
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
            Shape (N, 2) if available.
        """
        return self._last_predicted_path


    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _initial_guess(self, *, n: int) -> FloatArray:
        """
        Create an initial guess for the decision vector Z = [U(2n), S(n)].

        Parameters
        ----------
        n : int
            Horizon length (decision stages).

        Returns
        -------
        ndarray
            Flattened decision vector of length 3*n.
        """
        u_dim = 2 * n
        z_dim = 3 * n

        if self._u_prev is None or self._u_prev.shape != (u_dim,):
            u_guess = np.zeros((u_dim,), dtype=np.float64)
            u_guess[0::2] = 0.3 * float(self._p.max_acceleration)
        else:
            u = self._u_prev
            u_guess = np.empty_like(u)
            u_guess[:-2] = u[2:]
            u_guess[-2:] = u[-2:]

        s_guess = np.zeros((n,), dtype=np.float64)

        z0 = np.zeros((z_dim,), dtype=np.float64)
        z0[:u_dim] = u_guess
        z0[u_dim:] = s_guess
        return z0

    def _rollout_trajectory(
        self,
        state: State,
        u_opt: FloatArray,
    ) -> FloatArray:
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
            Array of shape (N, 2) containing predicted (x, y) positions.
        """
        cfg = self._cfg
        p = self._p

        dt = float(cfg.dt)
        n = int(cfg.horizon_steps)
        block_steps = int(cfg.control_block_steps)

        x = float(state.x)
        y = float(state.y)
        psi = float(state.heading)
        v = float(state.velocity)
        delta = float(state.steering_angle)

        path = []

        eps = 1e-9

        for k in range(n):
            a_cmd = float(u_opt[2 * k + 0])
            sr_cmd = float(u_opt[2 * k + 1])

            for _ in range(block_steps):

                # Steering integrate
                delta = delta + sr_cmd * dt
                delta = np.clip(delta, -p.max_steering_angle, p.max_steering_angle)

                # Curvature saturation
                kappa_geom = np.tan(delta) / p.wheelbase
                v2 = v * v
                kappa_max = p.a_lat_max / (v2 + eps)
                kappa_eff = np.clip(kappa_geom, -kappa_max, kappa_max)
                a_lat_eff = v2 * kappa_eff

                # Friction circle
                a_total_max = p.mu * p.g
                rem2 = max(0.0, a_total_max**2 - a_lat_eff**2)
                a_long_max = np.sqrt(rem2)
                a_fric = np.clip(a_cmd, -a_long_max, a_long_max)

                # Drag + rolling
                a_roll = p.c_rr * p.g
                a_drag = p.c_d_a_over_m * v2
                a_resist = max(0.0, a_roll + a_drag)
                a_net = a_fric - a_resist

                # Speed integrate
                v = v + a_net * dt
                v = np.clip(v, 0.0, p.max_velocity)

                # Slip angle beta
                beta = np.arctan(p.rear_axle_ratio * np.tan(delta))

                # Heading integrate
                psi = psi + (v * kappa_eff) * dt

                # Pose integrate
                x = x + v * np.cos(psi + beta) * dt
                y = y + v * np.sin(psi + beta) * dt

                path.append([x, y])

        return np.array(path, dtype=np.float64)


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

        # Track interpolants x(s), y(s)
        s_grid = np.asarray(self._track.arc_lengths, dtype=np.float64)
        x_grid = np.asarray(self._track.centerline[:, 0], dtype=np.float64)
        y_grid = np.asarray(self._track.centerline[:, 1], dtype=np.float64)

        x_ref_fun = ca.interpolant("x_ref", "bspline", [s_grid], x_grid)
        y_ref_fun = ca.interpolant("y_ref", "bspline", [s_grid], y_grid)

        # Decision variables
        U = ca.MX.sym("U", 2 * n, 1)
        S = ca.MX.sym("S", n, 1)  # slack per stage (shared across its block)
        Z = ca.vertcat(U, S)

        # Parameters: initial state + s0
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
        w_slack = float(cfg.w_slack)

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

        # Cost terms (time-scaled)
        cost_contour = 0.0
        cost_lag = 0.0
        cost_speed = 0.0
        cost_control = 0.0
        cost_vmin = 0.0
        cost_slack_term = 0.0

        # Constraints: enforce at every internal substep for "accuracy"
        g_list: list[ca.MX] = []

        for k in range(n):
            a_cmd = U[2 * k + 0]
            sr_cmd = U[2 * k + 1]
            slack_k = S[k]

            a_cmd = _clip(a_cmd, -float(2*p.max_acceleration), float(p.max_acceleration))
            sr_cmd = _clip(sr_cmd, -float(p.max_steering_rate), float(p.max_steering_rate))

            for i in range(block_steps):
                # Steering integrate
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

                # Speed integrate
                v_next = vk + a_net * dt
                v_next = _clip(v_next, 0.0, float(p.max_velocity))

                # Slip angle beta
                beta = ca.atan(float(p.rear_axle_ratio) * ca.tan(delta_next))

                # Heading integrate
                psi_next = psik + (v_next * kappa_eff) * dt

                # Pose integrate
                x_next = xk + v_next * ca.cos(psi_next + beta) * dt
                y_next = yk + v_next * ca.sin(psi_next + beta) * dt

                # Moving reference along centerline
                step_index = (k * block_steps) + (i + 1)
                s_ref = s0 + float(cfg.v_ref) * dt * float(step_index)
                s_ref = _clip(s_ref, 0.0, total_len)

                px = x_ref_fun(s_ref)
                py = y_ref_fun(s_ref)

                # Tangent via finite differences (meters-scale + clipped)
                s_ref_fwd = _clip(s_ref + ds_eps, 0.0, total_len)
                px_fwd = x_ref_fun(s_ref_fwd)
                py_fwd = y_ref_fun(s_ref_fwd)

                tvec = ca.vertcat(px_fwd - px, py_fwd - py)
                t_hat = tvec / (_norm2(tvec) + 1e-9)
                n_hat = ca.vertcat(-t_hat[1], t_hat[0])

                e = ca.vertcat(x_next - px, y_next - py)
                e_lag = ca.dot(e, t_hat)
                e_cont = ca.dot(e, n_hat)

                # Track boundaries with slack: |e_cont| <= half_width + slack_k
                g_list.append(e_cont - (half_width + slack_k))
                g_list.append(-e_cont - (half_width + slack_k))

                # Time-scaled stage cost (consistent with blocking)
                cost_contour += float(cfg.w_contour) * (e_cont * e_cont) * dt
                cost_lag += float(cfg.w_lag) * (e_lag * e_lag) * dt
                cost_speed += float(cfg.w_speed) * ((v_next - float(cfg.v_ref)) ** 2) * dt

                # Control cost (also time-scaled; applied per substep because u is held constant in time)
                cost_control += float(cfg.w_u_acc) * (a_cmd * a_cmd) * dt
                cost_control += float(cfg.w_u_steer_rate) * (sr_cmd * sr_cmd) * dt

                # Standstill prevention (time-scaled)
                v_def = ca.fmax(0.0, float(cfg.v_min) - v_next)
                cost_vmin += float(cfg.w_v_min) * (v_def * v_def) * dt

                # Slack penalty (time-scaled)
                cost_slack_term += w_slack * (slack_k * slack_k) * dt

                # advance
                xk, yk, psik, vk, deltak = x_next, y_next, psi_next, v_next, delta_next

        cost_total = cost_contour + cost_lag + cost_speed + cost_control + cost_vmin + cost_slack_term

        g = ca.vertcat(*g_list) if len(g_list) > 0 else ca.MX.zeros(0, 1)
        nlp = {"x": Z, "f": cost_total, "g": g, "p": P}

        # ---- Bounds for decision variables ----
        lbx = np.zeros((3 * n,), dtype=np.float64)
        ubx = np.zeros((3 * n,), dtype=np.float64)

        for k in range(n):
            lbx[2 * k + 0] = -float(p.max_acceleration)
            ubx[2 * k + 0] = float(p.max_acceleration)
            lbx[2 * k + 1] = -float(p.max_steering_rate)
            ubx[2 * k + 1] = float(p.max_steering_rate)

        slack_max = float(cfg.slack_max) if float(cfg.slack_max) > 0.0 else float(self._track.width)
        lbx[2 * n :] = 0.0
        ubx[2 * n :] = slack_max

        # ---- Bounds for constraints g <= 0 ----
        m = 2 * n * block_steps
        lbg = -np.inf * np.ones((m,), dtype=np.float64)
        ubg = np.zeros((m,), dtype=np.float64)

        # ---- IPOPT options ----
        # Notes:
        # - For maximum robustness/accuracy, we avoid aggressive "acceptable" shortcuts.
        # - Increase ipopt_max_iter if you want "more accurate but slower".
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
                cost_contour,
                cost_lag,
                cost_speed,
                cost_control,
                cost_vmin,
                cost_slack_term,
            ],
        )

        return solver, nlp_struct, cost_fun
