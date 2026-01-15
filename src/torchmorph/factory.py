"""Factory functions."""

from typing import Optional

from nibabel import load as nib_load  # type: ignore
from numpy import memmap as np_memmap
from numpy import ndarray as np_ndarray
from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import equal, from_numpy

from torchmorph.util import numpy_dtype_to_torch_dtype

from .composable_mapping import SamplableVolume
from .coordinate_system import CoordinateSystem
from .sampler import DataFormat, ISampler


def from_file(
    data_path: str,
    mask_path: Optional[str] = None,
    data_format: DataFormat = DataFormat.world_coordinates(),
    sampler: Optional[ISampler] = None,
    dtype: Optional[torch_dtype] = None,
    mask_dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
    memory_map: bool = False,
) -> SamplableVolume:
    """Create a samplable volume from file.

    Args:
        data_path: Path to the file from which to load the data (the file is
            read using nibabel).
        mask_path: Path to the file from which to load the mask (the file is
            read using nibabel). If None, no mask is used. Note that his should
            be in same coordinate system as the data, and that no check is done
            to ensure that.
        data_format: Data format of the grid values.
        sampler: Sampler turning the grid values into a continuously defined mapping
            over spatial coordinates.
        dtype: Cast loaded data to this data type.
        mask_dtype: Cast loaded mask to this data type.
        device: Move loaded data to this device.
        memory_map: Whether to use memory mapping when loading the data. Note that this
            can only be used when loading to CPU and if the dtype is not changed.
    """
    image = nib_load(data_path)
    affine = from_numpy(image.affine)  # type: ignore
    data_np = image.dataobj[...]  # type: ignore
    if memory_map:
        if not _is_memory_mapped(data_np):
            raise ValueError("Memory mapping was not succesfull.")
        if device is not None and device.type != "cpu":
            raise ValueError("Memory mapping can only be used when loading to CPU.")
        if dtype is not None and dtype != numpy_dtype_to_torch_dtype(data_np.dtype):
            raise ValueError("Memory mapping can only be used when not changing the dtype.")
    data = from_numpy(data_np).to(dtype=dtype, device=device, copy=not memory_map)  # type: ignore

    n_dims = affine.size(1) - 1
    if data.ndim > n_dims:
        n_channel_dims = data.ndim - n_dims
        data = data.movedim(
            tuple(range(data.ndim - n_channel_dims, data.ndim)), tuple(range(n_channel_dims))
        )
    else:
        n_channel_dims = 1
        data = data[None]
    data = data[None]
    if mask_path is None:
        mask: Optional[Tensor] = None
    else:
        mask_np = nib_load(mask_path).dataobj[...]  # type: ignore
        if memory_map:
            if not _is_memory_mapped(mask_np):
                raise ValueError("Memory mapping was not succesfull.")
            if mask_dtype is not None and mask_dtype != numpy_dtype_to_torch_dtype(mask_np.dtype):
                raise ValueError("Memory mapping can only be used when not changing the dtype.")
        mask = from_numpy(mask_np).to(dtype=mask_dtype, device=device, copy=not memory_map)
        if mask.ndim > n_dims:
            n_channel_dims_mask = mask.ndim - n_dims
            mask = mask.movedim(
                tuple(range(mask.ndim - n_channel_dims_mask, mask.ndim)),
                tuple(range(n_channel_dims_mask)),
            )
        mask = mask[(None,) * (data.ndim - mask.ndim)]
    if equal(affine[:-1, :-1], affine[:-1, :-1].diag().diag()):
        coordinate_system = CoordinateSystem.from_diagonal_affine_matrix(
            spatial_shape=data.shape[-n_dims:],
            diagonal=affine[:-1, :-1].diag().to(dtype=data.dtype),
            translation=affine[:-1, -1].to(dtype=data.dtype),
            device=data.device,
        )
    else:
        coordinate_system = CoordinateSystem.from_affine_matrix(
            spatial_shape=data.shape[-n_dims:],
            affine_matrix=affine.to(dtype=data.dtype),
            device=data.device,
        )
    return SamplableVolume.from_tensor(
        data=data,
        coordinate_system=coordinate_system,
        mask=mask,
        data_format=data_format,
        sampler=sampler,
        n_channel_dims=n_channel_dims,
    )


def _is_memory_mapped(array: np_ndarray) -> bool:
    base = array
    while hasattr(base, "base") and base.base is not None:
        base = base.base
    return isinstance(base, np_memmap)
