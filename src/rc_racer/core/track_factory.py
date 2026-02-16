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
# Section Generators
# ================================================================

def _hermite_s_section(
    *,
    start: tuple[float, float],
    start_angle: float,
    end: tuple[float, float],
    end_angle: float,
    num_points: int,
    tangent_scale: float = 0.3,
) -> np.ndarray:
    """
    Generate a C¹ continuous cubic Hermite curve between two
    points with specified heading angles.

    Parameters
    ----------
    start : tuple[float, float]
    start_angle : float
        Heading angle in radians.
    end : tuple[float, float]
    end_angle : float
        Heading angle in radians.
    num_points : int
    tangent_scale : float
        Controls curve tightness.

    Returns
    -------
    ndarray (N,2)
    """

    p0 = np.array(start, dtype=np.float64)
    p1 = np.array(end, dtype=np.float64)

    chord = p1 - p0
    chord_length = np.linalg.norm(chord)

    # Tangent magnitudes proportional to chord length
    m0 = chord_length * tangent_scale * np.array(
        [np.cos(start_angle), np.sin(start_angle)]
    )

    m1 = chord_length * tangent_scale * np.array(
        [np.cos(end_angle), np.sin(end_angle)]
    )

    t = np.linspace(0.0, 1.0, num_points)

    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2

    curve = (
        h00[:, None] * p0 +
        h10[:, None] * m0 +
        h01[:, None] * p1 +
        h11[:, None] * m1
    )

    return curve.astype(np.float64)


def _curved_s_track_section(
    *,
    start: tuple[float, float],
    length: float,
    num_points: int,
    direction: str,
    amp1: float,
    freq1: float,
    amp2: float,
    freq2: float,
) -> np.ndarray:
    """
    Generate a curved S track section with zero initial slope
    and support for signed axis directions.

    Parameters
    ----------
    start : tuple[float, float]
        Starting coordinate (x0, y0).
    length : float
        Positive segment length.
    num_points : int
        Number of discretization points.
    direction : str
        One of {"x", "-x", "y", "-y"}.
    amp1, freq1 : float
        First cosine component.
    amp2, freq2 : float
        Second cosine component.

    Returns
    -------
    ndarray of shape (N, 2)
        Centerline section.
    """
    if direction not in ("x", "-x", "y", "-y"):
        raise ValueError("direction must be one of {'x', '-x', 'y', '-y'}")

    if length <= 0.0:
        raise ValueError("length must be positive")

    t = np.linspace(0.0, length, num_points, dtype=np.float64)

    # C¹ continuous oscillation (zero slope at t=0)
    oscillation = (
        amp1 * (1.0 - np.cos(freq1 * t))
        + amp2 * (1.0 - np.cos(freq2 * t))
    )

    x0, y0 = start

    if direction == "x":
        xs = x0 + t
        ys = y0 + oscillation

    elif direction == "-x":
        xs = x0 - t
        ys = y0 + oscillation

    elif direction == "y":
        xs = x0 + oscillation
        ys = y0 + t

    else:  # "-y"
        xs = x0 + oscillation
        ys = y0 - t

    return np.column_stack((xs, ys)).astype(np.float64)


def _arc_section(
    center: np.ndarray,
    radius: float,
    start_angle: float,
    end_angle: float,
    num_points_per_segment: int
) -> np.ndarray:
    angles = np.linspace(start_angle, end_angle, num_points_per_segment)
    x = center[0]  + (radius ) * np.cos(angles)
    y = center[1]  + (radius ) * np.sin(angles)
    return np.column_stack((x, y))

def _straight_section(p0: np.ndarray, p1: np.ndarray, num_points_per_segment: int) -> np.ndarray:
    xs = np.linspace(p0[0] , p1[0] , num_points_per_segment)
    ys = np.linspace(p0[1] , p1[1] , num_points_per_segment)
    return np.column_stack((xs, ys))


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

def _f1_like_closed(
    num_points_per_segment: int = 200,
    width: float = 20.0,
) -> Track:
    """
    Create a Barcelona-inspired closed circuit scaled down by factor 5.

    All geometric dimensions from the original Barcelona-like layout
    are divided by 5 while preserving proportions.

    Parameters
    ----------
    num_points_per_segment : int
        Resolution per geometric segment.
    width : float
        Track width in meters.

    Returns
    -------
    Track
        Immutable closed Track instance.
    """
    segments: list[np.ndarray] = []

    # ------------------------------------------------------------
    # Start/Finish Straight
    # ------------------------------------------------------------
    segments.append(_straight_section(np.array([0.0, 0.0]), np.array([-100.0, 0.0]), num_points_per_segment))

    segments.append(_arc_section(center=np.array([-100.0, 20.0]), radius=20.0, start_angle=-np.pi / 2.0, end_angle=-np.pi, num_points_per_segment=num_points_per_segment) )

    segments.append(_arc_section(center=np.array([-140.0, 20.0]), radius=20.0,  start_angle=0.0, end_angle=np.pi/2, num_points_per_segment=num_points_per_segment) )

    segments.append(_arc_section(center=np.array([-140.0, 100.0]), radius=60.0,  start_angle=-np.pi / 2.0, end_angle=-3*np.pi/2, num_points_per_segment=num_points_per_segment) )

    segments.append(_straight_section(np.array([-140.0, 160.0]), np.array([-60.0, 160.0]), num_points_per_segment))
    
    segments.append(_arc_section(center=np.array([-60.0, 150.0]), radius=10.0, start_angle=np.pi / 2.0, end_angle=0.0, num_points_per_segment=num_points_per_segment) )

    segments.append(_arc_section(center=np.array([-80.0, 150.0]), radius=30.0, start_angle=0.0, end_angle=-np.pi/2, num_points_per_segment=num_points_per_segment) )
    
    segments.append(_straight_section(np.array([-80.0, 120.0]), np.array([-100.0, 120.0]), num_points_per_segment))

    segments.append(_arc_section(center=np.array([-100.0, 110.0]), radius=10.0, start_angle=np.pi / 2.0, end_angle=5*np.pi/4, num_points_per_segment=num_points_per_segment) )

    segments.append(_straight_section(np.array([-100 - 10 * np.sqrt(2) / 2, 110 - 10*np.sqrt(2) / 2]), np.array([-80 - 10*np.sqrt(2) / 2, 90 - 10*np.sqrt(2) / 2]), num_points_per_segment))

    segments.append(_arc_section(center=np.array([-80 + 10*np.sqrt(2), 90 + 20*np.sqrt(2)/2]), radius=30.0, start_angle=5*np.pi/4, end_angle=3*np.pi/2, num_points_per_segment=num_points_per_segment) )

    segments.append(_straight_section(np.array([-80 + 10*np.sqrt(2) , 60 + 20*np.sqrt(2)/2]), np.array([-60 + 10*np.sqrt(2) , 60 + 20*np.sqrt(2)/2]), num_points_per_segment))

    segments.append(_arc_section(center=np.array([-60 + 10*np.sqrt(2) , 80 + 20*np.sqrt(2)/2]), radius=20.0, start_angle=-np.pi/2, end_angle=0.0, num_points_per_segment=num_points_per_segment) )

    segments.append(_arc_section(center=np.array([-10 + 10*np.sqrt(2) , 80 + 20*np.sqrt(2)/2]), radius=30.0, start_angle=np.pi, end_angle=5*np.pi/6, num_points_per_segment=num_points_per_segment) )

    segments.append(_straight_section(np.array([-10 + 10*np.sqrt(2) - 30*np.sqrt(3)/2 , 95 + 20*np.sqrt(2)/2]), np.array([5 + 10*np.sqrt(2) - 30*np.sqrt(3)/2 , 95 + 15*np.sqrt(3) + 20*np.sqrt(2)/2]), num_points_per_segment))

    segments.append(_arc_section(center=np.array([5 + 10*np.sqrt(2), 80 + 15*np.sqrt(3) + 20*np.sqrt(2)/2]), radius=30.0, start_angle=5*np.pi/6, end_angle=np.pi/3, num_points_per_segment=num_points_per_segment) )

    segments.append(_straight_section(np.array([20 + 10*np.sqrt(2), 80 + 30*np.sqrt(3) + 20*np.sqrt(2)/2]), np.array([20 + 10*np.sqrt(2)+60*np.sqrt(3), 20 + 30*np.sqrt(3) + 20*np.sqrt(2)/2]), num_points_per_segment))

    segments.append(_arc_section(center=np.array([35 + 10*np.sqrt(2)+60*np.sqrt(3), 20 + 45*np.sqrt(3) + 20*np.sqrt(2)/2]), radius=30.0, start_angle=4*np.pi/3, end_angle=5*np.pi/2, num_points_per_segment=num_points_per_segment) )

    segments.append(_arc_section(center=np.array([35 + 10*np.sqrt(2)+60*np.sqrt(3), 70 + 45*np.sqrt(3) + 20*np.sqrt(2)/2]), radius=20.0, start_angle=3*np.pi/2, end_angle=np.pi/2, num_points_per_segment=num_points_per_segment) )

    segments.append(_straight_section(np.array([35 + 10*np.sqrt(2)+60*np.sqrt(3), 90 + 45*np.sqrt(3) + 20*np.sqrt(2)/2]), np.array([60 + 10*np.sqrt(2)+60*np.sqrt(3), 90 + 45*np.sqrt(3) + 20*np.sqrt(2)/2]), num_points_per_segment))

    segments.append(_arc_section(center=np.array([60 + 10*np.sqrt(2)+60*np.sqrt(3), 50 + 45*np.sqrt(3) + 20*np.sqrt(2)/2]), radius=40.0, start_angle=np.pi/2, end_angle=0.0, num_points_per_segment=num_points_per_segment) )

    segments.append(_straight_section(np.array([100 + 10*np.sqrt(2)+60*np.sqrt(3), 50 + 45*np.sqrt(3) + 20*np.sqrt(2)/2]), np.array([100 + 10*np.sqrt(2)+60*np.sqrt(3), 60]), num_points_per_segment))

    segments.append(_arc_section(center=np.array([40 + 10*np.sqrt(2)+60*np.sqrt(3), 60]), radius=60.0, start_angle=0, end_angle=-np.pi/2, num_points_per_segment=num_points_per_segment) )

    segments.append(_straight_section(np.array([40 + 10*np.sqrt(2)+60*np.sqrt(3), 0]), np.array([0, 0]), num_points_per_segment))
    # segments.append(_hermite_s_section(
    #     start=(60.0, -10.0),
    #     start_angle=-np.pi/2,   # pointing down
    #     end=(70.0, -30.0),
    #     end_angle=-np.pi/2,     # also pointing down
    #     num_points=num_points_per_segment,
    #     tangent_scale=0.8,
    # ))

    
    centerline = np.vstack(segments).astype(np.float64)

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
_track_registry.register("f1_like_closed", _f1_like_closed)


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
