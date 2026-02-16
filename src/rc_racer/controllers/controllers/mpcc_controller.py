"""
mpcc_casadi_controller.py

Fast MPCC-style controller using CasADi with an SQP backend.

AGENTS Layer (Competition Layer)
--------------------------------
This module implements a high-performance, deterministic MPCC-style controller
that follows the BaseController interface.

Solver Strategy
---------------
- CasADi builds a symbolic optimal control problem once.
- Uses `sqpmethod` (SQP) with `qpoases` (QP solver) for speed.
- Warm-starts by shifting the previous optimal control sequence.

Standstill Prevention
---------------------
A soft minimum-speed penalty removes the "zero-velocity stationary optimum"
that can occur in simplified MPCC formulations.

Dependencies
------------
- casadi (pip install casadi)

Notes
-----
- This controller predicts using a symbolic clone of the VehicleModel's scalar
  step logic (steering limits, curvature saturation, friction circle, drag).
- The environment still applies its own dynamics; best results occur when dt
  matches env dt and vehicle_params match env vehicle model params.
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
    raise ImportError(
        "MpccCasadiController requires CasADi. Install with: pip install casadi"
    ) from exc


FloatArray = NDArray[np.float64]
Action = Tuple[float, float]


@dataclass(frozen=True)
class MpccConfig:
    """
    Configuration for the CasADi MPCC controller.

    Parameters
    ----------
    dt : float
        Prediction timestep. MUST match environment dt for best results.
    horizon_steps : int
        MPC horizon length in steps.
    v_ref : float
        Moving-reference speed along the centerline (m/s).

    w_contour : float
        Weight for contouring (lateral / normal) error squared.
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
        Weight for the standstill-prevention penalty.

    max_sqp_iter : int
        Maximum SQP iterations per solve.
    tol : float
        Convergence tolerance for SQP.

    solver_verbosity : bool
        If True, prints solver diagnostics (noisy).
    """

    dt: float = 0.02
    horizon_steps: int = 25
    v_ref: float = 14.0

    w_contour: float = 8.0
    w_lag: float = 2.0
    w_speed: float = 1.5
    w_u_acc: float = 0.03
    w_u_steer_rate: float = 0.05

    v_min: float = 0.5
    w_v_min: float = 5000.0

    max_sqp_iter: int = 20
    tol: float = 1e-3

    solver_verbosity: bool = False


class MpccController(BaseController):
    """
    Fast MPCC-style controller using CasADi SQP + qpOASES.

    Parameters
    ----------
    track : Track
        Immutable track geometry.
    config : MpccConfig | None
        Controller configuration.
    vehicle_params : VehicleParams | None
        Vehicle parameters for prediction. If None, uses VehicleFactory default.

    Notes
    -----
    - Deterministic and synchronous.
    - Uses Track centerline interpolation for reference and tangent/normal frame.
    - Standstill prevention is enabled via a soft minimum-speed penalty.
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

        # Warm start: flattened U = [a0, sr0, a1, sr1, ...]
        self._u_prev: FloatArray | None = None

        # Debug values from the last solve
        self._debug_last: Dict[str, float] = {}

        # Build solver once
        self._solver, self._nlp_struct = self._build_solver()

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

        x0 = self._initial_guess(n=n)

        try:
            sol = self._solver(
                x0=x0,
                lbx=self._nlp_struct["lbx"],
                ubx=self._nlp_struct["ubx"],
                lbg=self._nlp_struct["lbg"],
                ubg=self._nlp_struct["ubg"],
                p=pvec,
            )

            z_opt = np.array(sol["x"]).reshape((-1,), order="F").astype(np.float64)

            # Decision vector layout: [U(2n), slack(n)]
            u_opt = z_opt[: 2 * n]
            self._u_prev = u_opt.copy()

            a0 = float(u_opt[0])
            sr0 = float(u_opt[1])

            stats = self._solver.stats()
            self._debug_last = {
                "success": float(bool(stats.get("success", False))),
                "iters": float(stats.get("iter_count", 0)),
                "cost": float(np.array(sol["f"]).item()),
                "a_cmd": a0,
                "steer_rate_cmd": sr0,
                "s0": float(s0),
                "x_proj": float(proj[0]),
                "y_proj": float(proj[1]),
            }

            return a0, sr0

        except Exception:
            # If the solver fails, prefer a safe "get moving" action.
            a0 = 0.3 * float(self._p.max_acceleration)
            sr0 = 0.0

            # If we have a previous solution, use it instead.
            if self._u_prev is not None and self._u_prev.shape[0] >= 2:
                a0 = float(self._u_prev[0])
                sr0 = float(self._u_prev[1])

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

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _initial_guess(self, *, n: int) -> FloatArray:
        """
        Create an initial guess for the decision vector Z = [U(2n), S(n)].

        Parameters
        ----------
        n : int
            Horizon length.

        Returns
        -------
        ndarray
            Flattened decision vector of length 3*n.
        """
        u_dim = 2 * n
        z_dim = 3 * n

        # Controls warm start
        if self._u_prev is None or self._u_prev.shape != (u_dim,):
            u_guess = np.zeros((u_dim,), dtype=np.float64)
            u_guess[0::2] = 0.5 * float(self._p.max_acceleration)
        else:
            # Shift previous controls: [u1, u2, ..., u_{n-1}, u_{n-1}]
            u = self._u_prev
            u_guess = np.empty_like(u)
            u_guess[:-2] = u[2:]
            u_guess[-2:] = u[-2:]

        # Slack warm start: start feasible-ish (small slack) but not huge
        s_guess = np.zeros((n,), dtype=np.float64)

        z0 = np.zeros((z_dim,), dtype=np.float64)
        z0[:u_dim] = u_guess
        z0[u_dim:] = s_guess
        return z0

    def _build_solver(self) -> tuple[ca.Function, Dict[str, FloatArray]]:
        """
        Build the CasADi NLP solver and static bound structures.

        This version includes:
        - Finite-difference tangent (avoids CasADi jacobian() constraints)
        - SQP + qpOASES fast backend with correct option names
        - Track boundary constraints with nonnegative slacks (prevents QP infeasibility)
        while behaving nearly hard via a large slack penalty.

        Returns
        -------
        solver : casadi.Function
            Configured CasADi solver.
        nlp_struct : dict[str, ndarray]
            Bounds and constraint arrays.
        """
        cfg = self._cfg
        p = self._p

        n = int(cfg.horizon_steps)
        dt = float(cfg.dt)

        # ---- Track interpolants (x(s), y(s)) ----
        s_grid = np.asarray(self._track.arc_lengths, dtype=np.float64)
        x_grid = np.asarray(self._track.centerline[:, 0], dtype=np.float64)
        y_grid = np.asarray(self._track.centerline[:, 1], dtype=np.float64)

        x_ref_fun = ca.interpolant("x_ref", "bspline", [s_grid], x_grid)
        y_ref_fun = ca.interpolant("y_ref", "bspline", [s_grid], y_grid)

        # ---- Decision variables ----
        # U = [a0,sr0, a1,sr1, ...], slack S = [s0..s_{n-1}] >= 0
        U = ca.MX.sym("U", 2 * n, 1)
        S = ca.MX.sym("S", n, 1)

        Z = ca.vertcat(U, S)

        # ---- Parameters: initial state + s0 ----
        # [x0, y0, psi0, v0, delta0, s0]
        P = ca.MX.sym("P", 6, 1)
        x0 = P[0]
        y0 = P[1]
        psi0 = P[2]
        v0 = P[3]
        delta0 = P[4]
        s0 = P[5]

        eps = 1e-9

        def _clip(z: ca.MX, lo: float, hi: float) -> ca.MX:
            return ca.fmin(ca.fmax(z, lo), hi)

        def _norm2(vec2: ca.MX) -> ca.MX:
            return ca.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + eps)

        # ---- Rollout + objective + constraints ----
        xk = x0
        yk = y0
        psik = psi0
        vk = v0
        deltak = delta0

        cost = 0.0

        half_width = 0.5 * float(self._track.width)

        # “Near-hard” slack penalty. Make it big.
        # You can move this into config if you want.
        w_slack = 5.0e5

        ds_eps = 1e-3

        g_list: list[ca.MX] = []

        for k in range(n):
            a_cmd = U[2 * k + 0]
            sr_cmd = U[2 * k + 1]
            slack_k = S[k]

            a_cmd = _clip(a_cmd, -float(p.max_acceleration), float(p.max_acceleration))
            sr_cmd = _clip(sr_cmd, -float(p.max_steering_rate), float(p.max_steering_rate))

            # Steering integrate + clamp
            delta_next = deltak + sr_cmd * dt
            delta_next = _clip(delta_next, -float(p.max_steering_angle), float(p.max_steering_angle))

            # Curvature + lateral accel saturation
            kappa_geom = ca.tan(delta_next) / float(p.wheelbase)
            v2 = vk * vk
            kappa_max = float(p.a_lat_max) / (v2 + eps)
            kappa_eff = _clip(kappa_geom, -kappa_max, kappa_max)
            a_lat_eff = v2 * kappa_eff

            # Friction circle
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

            # Moving reference along centerline
            s_ref = s0 + float(cfg.v_ref) * dt * float(k + 1)
            s_ref = _clip(s_ref, 0.0, float(self._track.total_length))

            px = x_ref_fun(s_ref)
            py = y_ref_fun(s_ref)

            # Tangent via finite differences in s
            px_fwd = x_ref_fun(s_ref + ds_eps)
            py_fwd = y_ref_fun(s_ref + ds_eps)
            tvec = ca.vertcat(px_fwd - px, py_fwd - py)
            t_hat = tvec / (_norm2(tvec) + 1e-9)
            n_hat = ca.vertcat(-t_hat[1], t_hat[0])

            e = ca.vertcat(x_next - px, y_next - py)
            e_lag = ca.dot(e, t_hat)
            e_cont = ca.dot(e, n_hat)

            # ------------------------------------------------------
            # Track boundaries with slack (prevents infeasible QPs)
            # We enforce: |e_cont| <= half_width + slack_k, slack_k >= 0
            # Encode as two inequalities <= 0:
            #   e_cont - (half_width + slack_k) <= 0
            #  -e_cont - (half_width + slack_k) <= 0
            # ------------------------------------------------------
            g_list.append(e_cont - (half_width + slack_k))
            g_list.append(-e_cont - (half_width + slack_k))

            # MPCC stage cost
            cost = cost + float(cfg.w_contour) * (e_cont * e_cont)
            cost = cost + float(cfg.w_lag) * (e_lag * e_lag)
            cost = cost + float(cfg.w_speed) * ((v_next - float(cfg.v_ref)) ** 2)
            cost = cost + float(cfg.w_u_acc) * (a_cmd * a_cmd)
            cost = cost + float(cfg.w_u_steer_rate) * (sr_cmd * sr_cmd)

            # Standstill prevention
            v_def = ca.fmax(0.0, float(cfg.v_min) - v_next)
            cost = cost + float(cfg.w_v_min) * (v_def * v_def)

            # Slack penalty (near-hard)
            cost = cost + float(w_slack) * (slack_k * slack_k)

            # advance
            xk, yk, psik, vk, deltak = x_next, y_next, psi_next, v_next, delta_next

        g = ca.vertcat(*g_list) if len(g_list) > 0 else ca.MX.zeros(0, 1)
        nlp = {"x": Z, "f": cost, "g": g, "p": P}

        # ---- Bounds for decision vars ----
        # Z = [U(2n), S(n)]
        lbx = np.zeros((2 * n + n,), dtype=np.float64)
        ubx = np.zeros((2 * n + n,), dtype=np.float64)

        for k in range(n):
            lbx[2 * k + 0] = -float(p.max_acceleration)
            ubx[2 * k + 0] = float(p.max_acceleration)
            lbx[2 * k + 1] = -float(p.max_steering_rate)
            ubx[2 * k + 1] = float(p.max_steering_rate)

        # Slack bounds: slack_k >= 0, and cap it so the solver can't “buy” infinite violation.
        slack_max = float(self._track.width)  # generous cap
        lbx[2 * n :] = 0.0
        ubx[2 * n :] = slack_max

        # ---- Bounds for constraints ----
        # We constructed inequalities of form g_i <= 0
        m = 2 * n
        lbg = -np.inf * np.ones((m,), dtype=np.float64)
        ubg = np.zeros((m,), dtype=np.float64)

        # ---- Solver options (sqpmethod) ----
        opts = {
            "print_time": bool(cfg.solver_verbosity),
            "verbose": bool(cfg.solver_verbosity),
            "error_on_fail": False,  # <- important; we also catch exceptions in compute_action
            "qpsol": "qpoases",
            "jit": True,
            "max_iter": int(cfg.max_sqp_iter),
            "tol_pr": float(cfg.tol),
            "tol_du": float(cfg.tol),
        }

        solver = ca.nlpsol("mpcc_sqp", "sqpmethod", nlp, opts)

        nlp_struct: Dict[str, FloatArray] = {
            "lbx": lbx,
            "ubx": ubx,
            "lbg": lbg,
            "ubg": ubg,
        }
        return solver, nlp_struct
