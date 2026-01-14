"""Interface for the composable mapping module."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence

from torch import device as torch_device
from torch import dtype as torch_dtype

if TYPE_CHECKING:
    from torchmorph.coordinate_system import CoordinateSystem


class ICoordinateSystemContainer(ABC):
    """Interfaces for classes holding a unique coordinate system."""

    @property
    @abstractmethod
    def coordinate_system(
        self,
    ) -> "CoordinateSystem":
        """Coordinate system of the container."""


class ICoordinateSystemFactory(ABC):
    """Interface for classes that can be used to build coordinate systems.

    Coordinate system factories can be used for creating samplable volumes
    without having to specify the spatial shape, dtype, and device (duplicate
    information).
    """

    @abstractmethod
    def build(
        self,
        spatial_shape: Sequence[int],
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Build the coordinate system.

        Args:
            spatial_shape: Spatial shape of the grid
            dtype: Dtype of the associated data.
            device: Device of the associated data.

        Returns:
            Concrete coordinate system.
        """
