"""
track.py

Core track representation for the racing simulation.

This module defines the immutable Track class used by the simulation core.
The track provides:

- Arc-length parametrized centerline
- Left/right boundaries
- Projection of a point onto the centerline
- Inside/outside boundary test

Notes
-----
This module belongs to the CORE layer and must not depend on:
- Controllers
- GUI
- Threading
- Real-time clocks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Track:
    """
    Immutable spline-based race track.

    Parameters
    ----------
    centerline : NDArray[np.float64]
        Array of shape (N, 2) containing ordered (x, y) points.
    width : float
        Total track width (meters).

    Attributes
    ----------
    centerline : NDArray[np.float64]
        Discrete centerline points.
    width : float
        Track width.
    left_boundary : NDArray[np.float64]
        Left boundary polyline.
    right_boundary : NDArray[np.float64]
        Right boundary polyline.
    arc_lengths : NDArray[np.float64]
        Cumulative arc-length parameterization.
    total_length : float
        Total track length.
    """

    centerline: FloatArray
    width: float

    def __post_init__(self) -> None:
        """
        Compute arc-length parametrization and boundaries.

        Raises
        ------
        ValueError
            If centerline is invalid.
        """
        if self.centerline.ndim != 2 or self.centerline.shape[1] != 2:
            raise ValueError("centerline must be of shape (N, 2)")

        if self.centerline.shape[0] < 2:
            raise ValueError("centerline must contain at least two points")

        if self.width <= 0.0:
            raise ValueError("width must be positive")

        object.__setattr__(self, "centerline", np.asarray(self.centerline, dtype=np.float64))

        arc_lengths = self._compute_arc_lengths(self.centerline)
        object.__setattr__(self, "arc_lengths", arc_lengths)
        object.__setattr__(self, "total_length", float(arc_lengths[-1]))

        left, right = self._compute_boundaries(self.centerline, self.width)
        object.__setattr__(self, "left_boundary", left)
        object.__setattr__(self, "right_boundary", right)

    @staticmethod
    def _compute_arc_lengths(points: FloatArray) -> FloatArray:
        """
        Compute cumulative arc-length parameterization.

        Parameters
        ----------
        points : NDArray[np.float64]
            Centerline points.

        Returns
        -------
        NDArray[np.float64]
            Cumulative arc-length array of shape (N,).
        """
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        arc_lengths = np.zeros(points.shape[0], dtype=np.float64)
        arc_lengths[1:] = np.cumsum(segment_lengths)
        return arc_lengths

    @staticmethod
    def _compute_boundaries(
        points: FloatArray,
        width: float,
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Compute left and right track boundaries.

        Parameters
        ----------
        points : NDArray[np.float64]
            Centerline points.
        width : float
            Track width.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            Left and right boundary arrays.
        """
        tangents = np.zeros_like(points)
        tangents[1:-1] = points[2:] - points[:-2]
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]

        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / norms

        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        half_width = width / 2.0
        left_boundary = points + half_width * normals
        right_boundary = points - half_width * normals

        return left_boundary, right_boundary

    def project(self, position: FloatArray) -> Tuple[float, FloatArray]:
        """
        Project a point onto the centerline.

        Parameters
        ----------
        position : NDArray[np.float64]
            Array of shape (2,) representing (x, y).

        Returns
        -------
        progress_s : float
            Arc-length coordinate of projection.
        projected_point : NDArray[np.float64]
            Closest point on centerline.
        """
        if position.shape != (2,):
            raise ValueError("position must be shape (2,)")

        min_dist_sq = np.inf
        best_s = 0.0
        best_point = self.centerline[0]

        for i in range(len(self.centerline) - 1):
            p0 = self.centerline[i]
            p1 = self.centerline[i + 1]
            segment = p1 - p0
            length_sq = float(np.dot(segment, segment))

            if length_sq == 0.0:
                continue

            t = float(np.dot(position - p0, segment) / length_sq)
            t_clamped = np.clip(t, 0.0, 1.0)
            proj = p0 + t_clamped * segment

            dist_sq = float(np.sum((position - proj) ** 2))

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_point = proj
                seg_length = np.sqrt(length_sq)
                best_s = self.arc_lengths[i] + t_clamped * seg_length

        return best_s, best_point

    def is_inside(self, position: FloatArray) -> bool:
        """
        Check if a point lies inside track boundaries.

        Parameters
        ----------
        position : NDArray[np.float64]
            Array of shape (2,) representing (x, y).

        Returns
        -------
        bool
            True if inside track width, False otherwise.
        """
        s, proj = self.project(position)
        lateral_dist = float(np.linalg.norm(position - proj))
        return lateral_dist <= self.width / 2.0
