"""
track_factory.py

Factory system for generating standard deterministic track configurations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rc_racer.core.track import Track
from rc_racer.utils.registry import Registry


FloatArray = NDArray[np.float64]

_track_registry: Registry[Track] = Registry()


# ================================================================
# Track Generators
# ================================================================


def _straight_line(
    length: float = 100.0,
    num_points: int = 200,
    width: float = 10.0,
) -> Track:
    xs = np.linspace(0.0, length, num_points)
    ys = np.zeros_like(xs)
    centerline = np.column_stack((xs, ys))
    return Track(centerline=centerline, width=width)


def _simple_curve_open(
    radius: float = 50.0,
    angle: float = np.pi / 2.0,
    num_points: int = 200,
    width: float = 10.0,
) -> Track:
    angles = np.linspace(0.0, angle, num_points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    centerline = np.column_stack((x, y))
    return Track(centerline=centerline, width=width)

def _curved_s_track( ) -> Track:
    xs = np.linspace(0.0, 120.0, 600)

    ys = (
        8.0 * np.sin(0.08 * xs) +
        4.0 * np.sin(0.18 * xs)
    )

    centerline = np.column_stack((xs, ys)).astype(np.float64)

    return Track(centerline=centerline, width=10.0)

def _sinusoidal_open(
    length: float = 150.0,
    amplitude: float = 20.0,
    waves: int = 3,
    num_points: int = 400,
    width: float = 10.0,
) -> Track:
    xs = np.linspace(0.0, length, num_points)
    ys = amplitude * np.sin(2.0 * np.pi * waves * xs / length)
    centerline = np.column_stack((xs, ys))
    return Track(centerline=centerline, width=width)


def _closed_circle(
    radius: float = 60.0,
    num_points: int = 400,
    width: float = 10.0,
) -> Track:
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    centerline = np.column_stack((x, y))
    return Track(centerline=centerline, width=width)


def _closed_challenging(
    base_radius: float = 80.0,
    num_points: int = 800,
    width: float = 12.0,
) -> Track:
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    radius = (
        base_radius
        + 20.0 * np.sin(2.0 * angles)
        + 15.0 * np.sin(5.0 * angles)
        + 10.0 * np.sin(9.0 * angles)
    )

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    centerline = np.column_stack((x, y))
    return Track(centerline=centerline, width=width)


# ================================================================
# Registry Setup
# ================================================================

_track_registry.register("straight_line", _straight_line)
_track_registry.register("simple_curve_open", _simple_curve_open)
_track_registry.register("curved_s_track", _curved_s_track)
_track_registry.register("sinusoidal_open", _sinusoidal_open)
_track_registry.register("closed_circle", _closed_circle)
_track_registry.register("closed_challenging", _closed_challenging)


# ================================================================
# Public API
# ================================================================


class TrackFactory:
    """
    Public track factory interface.
    """

    @staticmethod
    def create(name: str, **kwargs) -> Track:
        return _track_registry.create(name, **kwargs)

    @staticmethod
    def available() -> list[str]:
        return _track_registry.available
