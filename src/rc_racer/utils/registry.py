"""
registry.py

Generic registry implementation for track and agent factories.

FACTORY LAYER
-------------
This module provides a reusable registry pattern implementation
used by track_factory.py and agent_factory.py.

Design Principles
-----------------
- Deterministic
- No randomness
- No environment dependency
- No GUI dependency
- Immutable outputs expected from registered builders
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """
    Simple string-key registry.

    Allows registering named builders and retrieving them later.

    Notes
    -----
    - Deterministic
    - Thread-safe for read operations (not concurrent writes)
    """

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., T]] = {}

    # ------------------------------------------------------------

    def register(self, name: str, builder: Callable[..., T]) -> None:
        """
        Register a new builder.

        Parameters
        ----------
        name : str
            Unique identifier.
        builder : Callable[..., T]
            Factory function.
        """
        if name in self._builders:
            raise ValueError(f"Builder '{name}' already registered.")

        self._builders[name] = builder

    # ------------------------------------------------------------

    def create(self, name: str, **kwargs) -> T:
        """
        Create instance from registry.

        Parameters
        ----------
        name : str
            Registered builder name.

        Returns
        -------
        T
        """
        if name not in self._builders:
            raise ValueError(f"Unknown builder '{name}'.")

        return self._builders[name](**kwargs)

    # ------------------------------------------------------------

    @property
    def available(self) -> list[str]:
        """
        List available builder names.

        Returns
        -------
        list[str]
        """
        return sorted(self._builders.keys())
