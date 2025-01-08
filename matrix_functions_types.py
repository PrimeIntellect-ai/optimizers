"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from dataclasses import dataclass

from commons import AbstractDataclass


@dataclass(init=False)
class MatrixFunctionConfig(AbstractDataclass):
    """Base dataclass for matrix function configurations."""


@dataclass(kw_only=True)
class EigenvalueDecompositionConfig(MatrixFunctionConfig):
    """Configuration for eigenvalue decomposition.

    Args:
        retry_double_precision (bool): Whether to re-trying eigendecomposition with higher (double) precision if lower precision fails due
            to CuSOLVER failure. (Default: True)

    """

    retry_double_precision: bool = True


@dataclass(init=False)
class RootInvConfig(MatrixFunctionConfig):
    """Base dataclass for matrix root inverse method configurations."""


@dataclass(kw_only=True)
class EigenConfig(RootInvConfig, EigenvalueDecompositionConfig):
    """Configuration for matrix root inverse via an eigendecomposition.

    Args:
        retry_double_precision (bool): Whether to re-trying eigendecomposition with higher (double) precision if lower precision fails due
            to CuSOLVER failure. (Default: True)
        make_positive_semidefinite (bool): Perturbs matrix eigenvalues to ensure it is numerically positive semi-definite. (Default: True)
        exponent_multiplier (float): Number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: 1.0)

    """

    make_positive_semidefinite: bool = True
    exponent_multiplier: float = 1.0


DefaultEigenConfig = EigenConfig()


@dataclass(kw_only=True)
class CoupledNewtonConfig(RootInvConfig):
    """Configuration for matrix root inverse via coupled Newton method.

    Args:
        max_iterations (int): Maximum number of iterations for coupled Newton iteration. (Default: 100)
        tolerance (float): Tolerance for computing root inverse using coupled Newton iteration. (Default: 1e-6)

    """

    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass(kw_only=True)
class CoupledHigherOrderConfig(RootInvConfig):
    """Configuration for matrix root inverse via coupled higher-order method.

    Args:
        rel_epsilon (float): Relative epsilon for coupled higher order method. Adds epsilon * lambda_max * I to matrix
            before taking matrix root, where lambda_max is an upper bound on maximum eigenvalue. (Default: 0.0)
        max_iterations (int): Maximum number of iterations for coupled higher order method. (Default: 100)
        tolerance (float): Tolerance for computing root inverse using coupled higher order method. (Default: 1e-8)
        order (int): Order of the method. Order must be >= 2.  Higher order methods accelerate convergence (fewer iterations),
            but can take more matmuls per iteration. order=2 represents Newton's method. (Default: 3)
        disable_tf32 (bool): Whether to disable tf32 matmuls or not internally. Highly recommend keeping True,
            since tf32 is challenging numerically here. (Default: True)

    """

    rel_epsilon: float = 0.0
    max_iterations: int = 100
    tolerance: float = 1e-8
    order: int = 3
    disable_tf32: bool = True


@dataclass(init=False)
class EigenvectorConfig(MatrixFunctionConfig):
    """Base dataclass for matrix eigenvector method configurations."""


@dataclass(kw_only=True)
class EighEigenvectorConfig(EigenvectorConfig, EigenvalueDecompositionConfig):
    """Configuration for eigenvectors via an eigendecomposition.

    Args:
        retry_double_precision (bool): Whether to re-trying eigendecomposition with higher (double) precision if lower precision fails due
            to CuSOLVER failure. (Default: True)

    """


DefaultEighEigenvectorConfig = EighEigenvectorConfig()


@dataclass(kw_only=True)
class QRConfig(EigenvectorConfig):
    """Configuration for eigenvectors via orthogonal/simultaneous iterations/QR algorithm.

    Args:
        max_iterations (int): The maximum number of iterations to perform. (Default: 1)
        tolerance (float): The tolerance for determining convergence in terms of the relative change of the eigenvectors estimate.
            (Default: 1e-5)

    """

    max_iterations: int = 1
    tolerance: float = 1e-5


@dataclass(kw_only=True)
class TopKCompressionEigenvectorConfig(EighEigenvectorConfig):
    """Configuration for compression of eigenvectors by only keeping the top k eigenvectors.

    NOTE: For now the compression is fake and we just zero out all but the top k eigenvectors. Only use it to test convergence rather
    than actual performance.

    Args:
        topk_compression (int | float ): The number of eigenvectors to keep, if float then it is a fraction of the number of eigenvectors
        min_compression_ratio (float): The minimum compression ratio available to actually do compression.
        min_dim (int): The minimum dimension to actually do compression.

        auto (bool): If set, the compression ratio is automatically determined based on the eigenvalues up to compression_t value. (Default: False)
        compression_t (float): The threshold value to use for auto compression.

    """

    topk_compression: int | float = 0.999
    min_compression_ratio: float = 0.0
    min_dim: int = 1024

    auto: bool = False

    compression_t: float = 0.95

    warmup_steps: int = 0

    def __post_init__(self):
        if isinstance(self.topk_compression, float):
            if not 0 < self.topk_compression <= 1:
                raise ValueError("If topk_compression is float, it must be between 0 and 1")

            if 0 < self.compression_t and self.compression_t > 1:
                raise ValueError("compression_value must be between 0 and 1")
