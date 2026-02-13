"""
vehicle_factory.py

Factory system for generating deterministic vehicle physics configurations.

FACTORY LAYER
-------------
Produces:
    - VehicleParams
    - VehicleModel
"""

from __future__ import annotations

from rc_racer.core.vehicle_model import VehicleModel, VehicleParams
from rc_racer.utils.registry import Registry


# ================================================================
# Registry
# ================================================================

_vehicle_registry: Registry[VehicleParams] = Registry()


# ================================================================
# Vehicle Presets
# ================================================================


def _default() -> VehicleParams:
    """
    Balanced default vehicle.
    """
    return VehicleParams(
        wheelbase=2.6,
        rear_axle_ratio=0.5,
        max_steering_angle=0.6,
        max_steering_rate=1.5,
        max_acceleration=6.0,
        max_velocity=25.0,
        mu=1.1,
        g=9.81,
        a_lat_max=7.0,
        mass=1200.0,
        c_rr=0.015,
        c_d_a_over_m=0.0005,
    )


# ================================================================
# Register Presets
# ================================================================

_vehicle_registry.register("default", _default)


# ================================================================
# Public API
# ================================================================


class VehicleFactory:
    """
    Factory for creating vehicle models from named presets.
    """

    @staticmethod
    def create_params(name: str) -> VehicleParams:
        """
        Create VehicleParams from preset.

        Parameters
        ----------
        name : str
            Preset name.

        Returns
        -------
        VehicleParams
        """
        return _vehicle_registry.create(name)

    @staticmethod
    def create_model(name: str) -> VehicleModel:
        """
        Create full VehicleModel from preset.

        Parameters
        ----------
        name : str
            Preset name.

        Returns
        -------
        VehicleModel
        """
        params = VehicleFactory.create_params(name)
        return VehicleModel(params)

    @staticmethod
    def available() -> list[str]:
        """
        Available preset names.

        Returns
        -------
        list[str]
        """
        return _vehicle_registry.available
