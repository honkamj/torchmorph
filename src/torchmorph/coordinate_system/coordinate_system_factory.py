"""Coordinate system factory for creating coordinate systems."""

from typing import Callable, Optional, Sequence, Union, overload

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

from torchmorph.composable_mapping.interface import ICoordinateSystemFactory
from torchmorph.interface import Number

from .coordinate_system import CoordinateSystem, ensure_tensor_on_host
from .reformatting_reference import Center, ReformattingReference
from .reformatting_spatial_shape import OriginalFOV, ReformattingSpatialShape


class CoordinateSystemFactory(ICoordinateSystemFactory):
    """Factory for creating coordinate systems.

    Coordinate system factories can be used for creating samplable volumes
    without having to specify the spatial shape, dtype, and device (duplicate
    information).

    Recommended way to create a coordinate system is to use the factory
    methods provided as class method of this class:
    `CoordinateSystemFactory.centered_normalized`,
    `CoordinateSystemFactory.centered`, `CoordinateSystem.voxel`,
    `CoordinateSystemFactory.torch_grid_sample`,
    `CoordinateSystemFactory.from_affine_matrix`,
    `CoordinateSystemFactory.from_diagonal_affine_matrix`.

    Arguments:
        factory_function: Method which actually initializes the coordinate
            system given spatial shape, dtype, and device.
    """

    def __init__(
        self,
        factory_function: Callable[
            [Sequence[int], Optional[torch_dtype], Optional[torch_device]], CoordinateSystem
        ],
    ) -> None:
        self._factory_function = factory_function

    def build(
        self,
        spatial_shape: Sequence[int],
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> CoordinateSystem:
        """Build the coordinate system.

        Args:
            spatial_shape: Spatial shape of the grid
            dtype: Dtype of the associated data.
            device: Device of the associated data.
        """
        return self._factory_function(spatial_shape, dtype, device)

    @classmethod
    def centered_normalized(
        cls,
        voxel_size: Union[Sequence[Number], Number, Tensor] = 1.0,
        fov_convention: str = "full_voxels",
    ) -> "CoordinateSystemFactory":
        """Create a centered normalized coordinate system factory.

        The coordinate system is normalized such that the grid just fits into the
        cube from -1 to 1 in each dimension. The grid is centered in the cube.

        Args:
            voxel_size: Relative voxel size of the grid.
            fov_convention: Convention for defining the field of view, either "full_voxels"
                or "voxel_centers". If voxels are seens as cubes with the value at the
                center, the convention "full voxels" includes the full cubes in the field
                of view, while the convention "voxel_centers" includes only the centers.
                The latter results in a field of view that is one voxel smaller in each
                dimension. Similar to the align_corners option in
                torch.nn.functional.grid_sample.

        Returns:
            Centered normalized coordinate system factory.
        """

        def factory_function(
            spatial_shape: Sequence[int],
            dtype: Optional[torch_dtype],
            device: Optional[torch_device],
        ) -> CoordinateSystem:
            return CoordinateSystem.centered_normalized(
                spatial_shape=spatial_shape,
                voxel_size=voxel_size,
                fov_convention=fov_convention,
                dtype=dtype,
                device=device,
            )

        return CoordinateSystemFactory(factory_function)

    @classmethod
    def centered(
        cls,
        voxel_size: Union[Sequence[Number], Number, Tensor] = 1.0,
    ) -> "CoordinateSystemFactory":
        """Create centered coordinate factory system with a given voxel size.

        Args:
            voxel_size: Voxel size of the grid.

        Returns:
            Centered coordinate system factory.
        """

        def factory_function(
            spatial_shape: Sequence[int],
            dtype: Optional[torch_dtype],
            device: Optional[torch_device],
        ) -> CoordinateSystem:
            return CoordinateSystem.centered(
                spatial_shape=spatial_shape,
                voxel_size=voxel_size,
                dtype=dtype,
                device=device,
            )

        return CoordinateSystemFactory(factory_function)

    @classmethod
    def voxel(
        cls,
        voxel_size: Optional[Union[Tensor, Sequence[Number], Number]] = None,
    ) -> "CoordinateSystemFactory":
        """Create coordinate system factory corresponding to the voxel
        coordinates potentially scaled by the voxel size.

        Args:
            voxel_size: Voxel size of the grid.

        Returns:
            Voxel coordinate system factory.
        """

        def factory_function(
            spatial_shape: Sequence[int],
            dtype: Optional[torch_dtype],
            device: Optional[torch_device],
        ) -> CoordinateSystem:
            return CoordinateSystem.voxel(
                spatial_shape=spatial_shape,
                voxel_size=voxel_size,
                dtype=dtype,
                device=device,
            )

        return CoordinateSystemFactory(factory_function)

    @classmethod
    def torch_grid_sample(cls, fov_convention: str = "full_voxels") -> "CoordinateSystemFactory":
        """Create coordinate system factory corresponding to the one used by the
        torch.nn.functional.grid_sample.

        The coordinate system is normalized along each dimensions to the range
        from -1 to 1.

        Args:
            fov_convention: Convention for defining the field of view, either "full_voxels"
                or "voxel_centers". If voxels are seens as cubes with the value at the
                center, the convention "full voxels" includes the full cubes in the field
                of view, while the convention "voxel_centers" includes only the centers.
                The latter results in a field of view that is one voxel smaller in each
                dimension. Similar to the align_corners option in
                torch.nn.functional.grid_sample.

        Returns:
            Coordinate system factory corresponding to the one used by the
            torch.nn.functional.grid_sample.
        """

        def factory_function(
            spatial_shape: Sequence[int],
            dtype: Optional[torch_dtype],
            device: Optional[torch_device],
        ) -> CoordinateSystem:
            return CoordinateSystem.torch_grid_sample(
                spatial_shape=spatial_shape,
                fov_convention=fov_convention,
                dtype=dtype,
                device=device,
            )

        return CoordinateSystemFactory(factory_function)

    @classmethod
    def from_affine_matrix(cls, affine_matrix: Tensor) -> "CoordinateSystemFactory":
        """Create coordinate system factory from an affine matrix

        Args:
            affine_matrix: Affine matrix describing the transformation
                from voxel to world coordinates. Tensor with shape
                ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1).
                The affine matrix tensor should be provided on the host (cpu).
                That is, since the transformation may be used in control flow
                decisions, and we want to avoid synchronizing the target device
                with the host. The matrix is moved to the target device
                asynchronously, if needed.

        Returns:
            Coordinate system factory with the given affine matrix.
        """
        ensure_tensor_on_host(affine_matrix)

        def factory_function(
            spatial_shape: Sequence[int],
            dtype: Optional[torch_dtype],  # pylint: disable=unused-argument
            device: Optional[torch_device],
        ) -> CoordinateSystem:
            return CoordinateSystem.from_affine_matrix(
                spatial_shape=spatial_shape,
                affine_matrix=affine_matrix,
                device=device,
            )

        return CoordinateSystemFactory(factory_function)

    @classmethod
    def from_diagonal_affine_matrix(
        cls,
        diagonal: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        translation: Optional[Union[Tensor, Sequence[Number], Number]] = None,
    ) -> "CoordinateSystemFactory":
        """Create coordinate system factory from diagonal affine matrix

        The diagonal and the translation should be provided on the host (cpu).
        That is, since the transformation may be used in control flow decisions,
        and we want to avoid synchronizing the target device with the host. The
        matrix is moved to the target device asynchronously, if needed.

        Args:
            diagonal: Diagonal of the affine transformation matrix describing
                the transformation from voxel to world coordinates. Tensor
                with shape ([*batch_shape, ]diagonal_length[, *spatial_shape]).
            translation: Translation of the affine transformation describing
                the transformation from voxel to world coordinates. Tensor with
                shape ([*batch_shape, ]n_output_dims[, *spatial_shape]).

        Returns:
            Coordinate system factory with the given diagonal affine matrix.
        """
        ensure_tensor_on_host(diagonal)
        ensure_tensor_on_host(translation)

        def factory_function(
            spatial_shape: Sequence[int],
            dtype: Optional[torch_dtype],  # pylint: disable=unused-argument
            device: Optional[torch_device],
        ) -> CoordinateSystem:
            return CoordinateSystem.from_diagonal_affine_matrix(
                spatial_shape=spatial_shape,
                diagonal=diagonal,
                translation=translation,
                device=device,
            )

        return CoordinateSystemFactory(factory_function)

    def __repr__(self) -> str:
        return "CoordinateSystemFactory()"

    def _apply_operator(
        self, operation: Callable[[CoordinateSystem], CoordinateSystem]
    ) -> "CoordinateSystemFactory":
        def factory_function(
            spatial_shape: Sequence[int],
            dtype: Optional[torch_dtype],  # pylint: disable=unused-variable
            device: Optional[torch_device],
        ) -> CoordinateSystem:
            return operation(self._factory_function(spatial_shape, dtype, device))

        return CoordinateSystemFactory(factory_function)

    def transform_voxel(self, affine_matrix: Tensor) -> "CoordinateSystemFactory":
        """Transform the coordinates with an affine matrix in the
        voxel coordinates before applying the current affine transformation

        Args:
            affine_matrix: Affine matrix describing the transformation
                in the voxel coordinates. Tensor with shape
                ([*batch_shape, ]n_output_dims + 1, n_spatial_dims + 1).
                The affine matrix tensor should be provided on the host (cpu).
                That is, since the transformation may be used in control flow
                decisions, and we want to avoid synchronizing the target device
                with the host. The matrix is moved to the target device
                asynchronously, if needed.

        Returns:
            Coordinate system factory with a modified affine transformation from
            voxel to world coordinates.
        """
        return self._apply_operator(
            lambda coordinate_system: coordinate_system.transform_voxel(affine_matrix=affine_matrix)
        )

    def transform_voxel_with_diagonal_matrix(
        self,
        diagonal: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        translation: Optional[Union[Tensor, Sequence[Number], Number]] = None,
    ) -> "CoordinateSystemFactory":
        """Transform the coordinates with a diagonal affine matrix in the
        voxel coordinates before applying the current affine transformation.

        The diagonal and the translation should be provided on the host (cpu).
        That is, since the transformation may be used in control flow decisions,
        and we want to avoid synchronizing the target device with the host. The
        matrix is moved to the target device asynchronously, if needed.

        Args:
            diagonal: Diagonal of the affine transformation matrix with shape
                ([*batch_shape, ]n_spatial_dims). Can be also
                given as a number or sequence of numbers. If None, corresponds
                to the diagonal of ones.
            translation: Translation of the affine transformation matrix
                with shape ([*batch_shape, ]n_spatial_dims).
                Can be also given as a number or sequence of numbers. If None,
                corresponds to the zero translation.

        Returns:
            Coordinate system factory with a modified affine transformation from
            voxel to world coordinates.
        """

        return self._apply_operator(
            lambda coordinate_system: coordinate_system.transform_voxel_with_diagonal_matrix(
                diagonal=diagonal, translation=translation
            )
        )

    def transform_world(self, affine_matrix: Tensor) -> "CoordinateSystemFactory":
        """Transform the coordinates with an affine matrix in the
        world coordinates after applying the current affine transformation

        Args:
            affine_matrix: Affine matrix describing the transformation
                in the world coordinates. Tensor with shape
                ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1).
                The affine matrix tensor should be provided on the host (cpu).
                That is, since the transformation may be used in control flow
                decisions, and we want to avoid synchronizing the target device
                with the host. The matrix is moved to the target device
                asynchronously, if needed.

        Returns:
            Coordinate system factory with a modified affine transformation from
            voxel to world coordinates.
        """
        return self._apply_operator(
            lambda coordinate_system: coordinate_system.transform_world(affine_matrix=affine_matrix)
        )

    def __rmatmul__(self, affine_matrix: Tensor) -> "CoordinateSystemFactory":
        return self._apply_operator(
            lambda coordinate_system: coordinate_system.__rmatmul__(affine_matrix)
        )

    def transform_world_with_diagonal_matrix(
        self,
        diagonal: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        translation: Optional[Union[Tensor, Sequence[Number], Number]] = None,
        n_output_dims: Optional[int] = None,
    ) -> "CoordinateSystemFactory":
        """Transform the coordinates with a diagonal affine matrix in the
        world coordinates after applying the current affine transformation

        The diagonal and the translation should be provided on the host (cpu).
        That is, since the transformation may be used in control flow decisions,
        and we want to avoid synchronizing the target device with the host. The
        matrix is moved to the target device asynchronously, if needed.

        Args:
            diagonal: Diagonal of the affine transformation matrix with shape
                ([*batch_shape, ]diagonal_length). Can be also
                given as a number or sequence of numbers. If None, corresponds
                to the diagonal of ones.
            translation: Translation of the affine transformation matrix
                with shape ([*batch_shape, ]n_output_dims).
                Can be also given as a number or sequence of numbers. If None,
                corresponds to the zero translation.
            n_output_dims: Number of output dimensions of the transformation.

        Returns:
            Coordinate system factory with a modified affine transformation from
            voxel to world coordinates.
        """
        return self._apply_operator(
            lambda coordinate_system: coordinate_system.transform_world_with_diagonal_matrix(
                diagonal=diagonal, translation=translation, n_output_dims=n_output_dims
            )
        )

    def translate_voxel(
        self, translation: Union[Sequence[Number], Number, Tensor]
    ) -> "CoordinateSystemFactory":
        """Translate the coordinates in the voxel coordinates before applying
        the current affine transformation

        Args:
            translation: Translation with shape ([*batch_shape, ]n_spatial_dims).
                Can be also given as a number or sequence of numbers.

        Returns:
            Coordinate system factory with translated coordinates.
        """
        return self._apply_operator(
            lambda coordinate_system: coordinate_system.translate_voxel(translation=translation)
        )

    def translate_world(
        self, translation: Union[Sequence[Number], Number, Tensor]
    ) -> "CoordinateSystemFactory":
        """Translate the coordinates in the world coordinates after applying
        the current affine transformation

        Args:
            translation: Translation with shape ([*batch_shape, ]n_output_dims).
                Can be also given as a number or sequence of numbers.

        Returns:
            Coordinate system factory with translated coordinates.
        """
        return self._apply_operator(
            lambda coordinate_system: coordinate_system.translate_world(translation=translation)
        )

    def __add__(self, other: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystemFactory":
        return self._apply_operator(lambda coordinate_system: coordinate_system + other)

    def __sub__(self, other: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystemFactory":
        return self._apply_operator(lambda coordinate_system: coordinate_system - other)

    def __neg__(self) -> "CoordinateSystemFactory":
        return self._apply_operator(lambda coordinate_system: -coordinate_system)

    def multiply_voxel(
        self, factor: Union[Sequence[Number], Number, Tensor]
    ) -> "CoordinateSystemFactory":
        """Multiply the coordinates in the voxel coordinates (before applying
        the current affine transformation)

        Args:
            factor: Factor to multiply the grid spacing. A tensor with shape
                ([*batch_shape, ]n_spatial_dims). Can be also given as a number
                or sequence of numbers.

        Returns:
            Coordinate system with a modified affine transformation from voxel
            to world coordinates.
        """
        return self._apply_operator(
            lambda coordinate_system: coordinate_system.multiply_voxel(factor)
        )

    def multiply_world(
        self, factor: Union[Sequence[Number], Number, Tensor]
    ) -> "CoordinateSystemFactory":
        """Multiply the coordinates in the world coordinates after applying
        the current affine transformation.

        Args:
            factor: Factor to multiply each dimension of the grid. A tensor with
                shape ([*batch_shape, ]n_output_dims). Can be also given as a number
                or sequence of numbers.

        Returns:
            Coordinate system factory with a modified affine transformation from
            voxel to world coordinates.
        """
        return self._apply_operator(
            lambda coordinate_system: coordinate_system.multiply_world(factor)
        )

    def __mul__(self, other: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystemFactory":
        return self._apply_operator(lambda coordinate_system: coordinate_system * other)

    @overload
    def reformat(
        self,
        *,
        downsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        upsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        spatial_shape: Union[
            Sequence[Union[ReformattingSpatialShape, int]],
            ReformattingSpatialShape,
            int,
            Tensor,
        ] = OriginalFOV(fitting_method="round", fov_convention="full_voxels"),
        reference: Union[
            Sequence[Union[ReformattingReference, Number]],
            ReformattingReference,
            Number,
        ] = Center(),
        target_reference: Optional[
            Union[
                Sequence[Union[ReformattingReference, Number]],
                ReformattingReference,
                Number,
            ]
        ] = None,
    ) -> "CoordinateSystemFactory": ...

    @overload
    def reformat(
        self,
        *,
        voxel_size: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
        spatial_shape: Union[
            Sequence[Union[ReformattingSpatialShape, int]],
            ReformattingSpatialShape,
            int,
            Tensor,
        ] = OriginalFOV(fitting_method="round", fov_convention="full_voxels"),
        reference: Union[
            Sequence[Union[ReformattingReference, Number]],
            ReformattingReference,
            Number,
        ] = Center(),
        target_reference: Optional[
            Union[
                Sequence[Union[ReformattingReference, Number]],
                ReformattingReference,
                Number,
            ]
        ] = None,
    ) -> "CoordinateSystemFactory": ...

    def reformat(
        self,
        *,
        downsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        upsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        voxel_size: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        spatial_shape: Union[
            Sequence[Union[ReformattingSpatialShape, int]],
            ReformattingSpatialShape,
            int,
            Tensor,
        ] = OriginalFOV(fitting_method="round", fov_convention="full_voxels"),
        reference: Union[
            Sequence[Union[ReformattingReference, Number]],
            ReformattingReference,
            Number,
        ] = Center(),
        target_reference: Optional[
            Union[
                Sequence[Union[ReformattingReference, Number]],
                ReformattingReference,
                Number,
            ]
        ] = None,
    ) -> "CoordinateSystemFactory":
        """Reformat the coordinate system factory.

        All tensors should be provided on the host (cpu). That is, since the
        transformation may be used in control flow decisions, and we want to
        avoid synchronizing the target device with the host. The matrix is moved
        to the target device asynchronously, if needed.

        Args:
            downsampling_factor: Factor to downsample the grid voxel size. A tensor
                with shape ([*batch_shape, ]n_spatial_dims). Can be also given as a
                number or sequence of numbers.
            upsampling_factor: Factor to upsample the grid voxel size. A tensor with
                shape ([*batch_shape, ]n_spatial_dims). Can be also given as a number
                or sequence of numbers.
            voxel_size: Voxel size of the reformatted grid. A tensor with shape
                ([*batch_shape, ]n_spatial_dims). Can be also given as a number or
                sequence of numbers.
            spatial_shape: Defines spatial_shape of the target grid, either
                given separately for each dimension or as a single value in
                which case the same value is used for all the dimensions.
                Defaults to OriginalFOV("round", "full_voxels").
            reference: Defines the point in the original voxel
                coordinates which will be aligned with the target reference in
                the reformatted coordinates. Either given separately for each
                dimension or as a single value in which case the same value is
                used for all the dimensions. Defaults to Center().
            target_reference: Defaults to reference. Defines the point in the
                reformatted voxel coordinates which will be aligned with the
                source reference in the original coordinates. Either given
                separately for each dimension or as a single value in which case
                the same value is used for all the dimensions.

        Returns:
            Reformatted coordinate system factory.
        """
        return self._apply_operator(
            lambda coordinate_system: coordinate_system.reformat(  # type: ignore
                downsampling_factor=downsampling_factor,
                upsampling_factor=upsampling_factor,
                voxel_size=voxel_size,
                spatial_shape=spatial_shape,
                reference=reference,
                target_reference=target_reference,
            )
        )
