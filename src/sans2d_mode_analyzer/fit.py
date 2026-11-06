"""Linear mean-squared-error fits for explicitly selected annulus modes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import design_matrix
from .indices import ModeIndex, validate_indices


@dataclass(frozen=True)
class MSEFitResult:
    """Result of an unweighted linear least-squares/MSE fit."""

    indices: tuple[ModeIndex, ...]
    coefficients: NDArray[np.float64]
    fitted_values: NDArray[np.float64]
    residuals: NDArray[np.float64]
    mse: float
    residual_variance: float
    degrees_of_freedom: int
    coefficient_covariance: NDArray[np.float64]
    rank: int
    singular_values: NDArray[np.float64]

    def predict(
        self,
        q: ArrayLike,
        theta: ArrayLike,
        *,
        q_min: float,
        q_max: float,
    ) -> NDArray[np.float64]:
        """Evaluate the fitted expansion at new annulus coordinates."""
        matrix = design_matrix(q, theta, self.indices, q_min=q_min, q_max=q_max)
        return np.asarray(matrix @ self.coefficients, dtype=float)


def fit_mse(
    q: ArrayLike,
    theta: ArrayLike,
    values: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    q_min: float,
    q_max: float,
    rcond: float | None = None,
) -> MSEFitResult:
    """Fit exactly the supplied modes by minimizing the unweighted MSE.

    ``q``, ``theta``, and ``values`` may be one- or multi-dimensional but must
    broadcast to a common shape.  No modes are added implicitly.
    """
    modes = validate_indices(indices)
    q_array, theta_array, value_array = np.broadcast_arrays(
        np.asarray(q, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(values, dtype=float),
    )
    if not np.all(np.isfinite(value_array)):
        raise ValueError("values must contain only finite values")
    matrix = design_matrix(
        q_array, theta_array, modes, q_min=q_min, q_max=q_max
    ).reshape(-1, len(modes))
    target = value_array.reshape(-1)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        matrix, target, rcond=rcond
    )
    if rank != len(modes):
        raise ValueError(
            f"selected design matrix is rank deficient: rank {rank} for {len(modes)} modes"
        )
    fitted_flat = matrix @ coefficients
    residual_flat = target - fitted_flat
    degrees_of_freedom = target.size - rank
    if degrees_of_freedom > 0:
        residual_variance = float(
            np.sum(residual_flat**2) / degrees_of_freedom
        )
        coefficient_covariance = residual_variance * np.linalg.inv(
            matrix.T @ matrix
        )
    else:
        residual_variance = float("nan")
        coefficient_covariance = np.full((len(modes), len(modes)), np.nan)
    shape = value_array.shape
    return MSEFitResult(
        indices=modes,
        coefficients=np.asarray(coefficients, dtype=float),
        fitted_values=fitted_flat.reshape(shape),
        residuals=residual_flat.reshape(shape),
        mse=float(np.mean(residual_flat**2)),
        residual_variance=residual_variance,
        degrees_of_freedom=int(degrees_of_freedom),
        coefficient_covariance=coefficient_covariance,
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
    )
