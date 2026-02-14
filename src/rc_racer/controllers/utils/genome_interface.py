"""
genome_interface.py

Generic genome interface for evolutionary controllers.

AGENTS Layer
------------
- Controller-agnostic genome contract
- Used by evolution layer
- Does not depend on CMA-ES
- No environment access
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Dict, Any, Type, TypeVar

import json
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

T = TypeVar("T", bound="GenomeInterface")


class GenomeInterface(ABC):
    """
    Abstract base class for controller genome/config objects.

    Any controller config that wants to be optimized
    must implement this interface.
    """

    # ==========================================================
    # Genome Dimension
    # ==========================================================

    @classmethod
    @abstractmethod
    def genome_size(cls) -> int:
        """
        Return genome dimensionality.

        Returns
        -------
        int
        """

    # ==========================================================
    # Decode From Vector
    # ==========================================================

    @classmethod
    @abstractmethod
    def decode(cls: Type[T], z: FloatArray) -> T:
        """
        Decode raw genome vector into configuration object.

        Parameters
        ----------
        z : ndarray
            Raw genome vector.

        Returns
        -------
        GenomeInterface
        """

    # ==========================================================
    # Serialization
    # ==========================================================

    def serialize(self) -> Dict[str, float]:
        """
        Serialize genome to dictionary.

        Returns
        -------
        dict[str, float]
        """
        return asdict(self)  # requires dataclass implementation

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize genome to JSON string.

        Parameters
        ----------
        indent : int

        Returns
        -------
        str
        """
        return json.dumps(self.serialize(), indent=indent)

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Construct genome from dictionary.

        Parameters
        ----------
        data : dict

        Returns
        -------
        GenomeInterface
        """
