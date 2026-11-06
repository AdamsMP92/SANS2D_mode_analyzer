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
    """Result of a linear least-squares fit.

    ``mse`` always refers to the unweighted detector-space residuals.
    ``weighted_sum_squares`` and ``reduced_chi_square`` describe the metric
    actually minimized.  They therefore coincide with the ordinary residual
    sum of squares and its degree-of-freedom normalization for ``fit_mse``.
    """

    indices: tuple[ModeIndex, ...]
    coefficients: NDArray[np.float64]
    fitted_values: NDArray[np.float64]
    residuals: NDArray[np.float64]
    mse: float
    residual_variance: float
    weighted_sum_squares: float
    reduced_chi_square: float
    degrees_of_freedom: int
    coefficient_covariance: NDArray[np.float64]
    covariance_scaled_by_residual: bool
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
    return _fit_linear(
        q,
        theta,
        values,
        indices,
        q_min=q_min,
        q_max=q_max,
        whitening=None,
        absolute_uncertainty=False,
        rcond=rcond,
    )


def fit_wls(
    q: ArrayLike,
    theta: ArrayLike,
    values: ArrayLike,
    indices: Iterable[Sequence[int]],
    uncertainty: ArrayLike,
    *,
    q_min: float,
    q_max: float,
    sample_weight: ArrayLike | None = None,
    absolute_uncertainty: bool = True,
    rcond: float | None = None,
) -> MSEFitResult:
    r"""Fit selected modes by uncertainty-weighted least squares.

    The minimized statistic is

    .. math::

       \sum_p w_p \left[(I_p-I_{\mathrm{fit},p})/\sigma_p\right]^2,

    where ``sample_weight`` supplies :math:`w_p` and defaults to one.  A
    fractional annulus coverage can therefore be passed directly as the
    sample weight.  Nonpositive uncertainties and weights are rejected rather
    than silently clipped.

    With ``absolute_uncertainty=True`` the coefficient covariance is the
    unscaled inverse weighted normal matrix.  Set it to false only when the
    supplied uncertainties define relative weights and the residual scatter
    should estimate their common scale.
    """
    q_array, theta_array, value_array, uncertainty_array = np.broadcast_arrays(
        np.asarray(q, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(values, dtype=float),
        np.asarray(uncertainty, dtype=float),
    )
    if not np.all(np.isfinite(uncertainty_array)) or np.any(
        uncertainty_array <= 0.0
    ):
        raise ValueError("uncertainty must contain finite positive values")
    if sample_weight is None:
        weight_array = np.ones_like(value_array)
    else:
        weight_array = np.broadcast_to(
            np.asarray(sample_weight, dtype=float), value_array.shape
        )
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0.0):
            raise ValueError("sample_weight must contain finite positive values")
    whitening = np.sqrt(weight_array) / uncertainty_array
    return _fit_linear(
        q_array,
        theta_array,
        value_array,
        indices,
        q_min=q_min,
        q_max=q_max,
        whitening=whitening,
        absolute_uncertainty=absolute_uncertainty,
        rcond=rcond,
    )


def fit_design_matrix(
    values: ArrayLike,
    indices: Iterable[Sequence[int]],
    matrix: ArrayLike,
    *,
    uncertainty: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    absolute_uncertainty: bool = True,
    rcond: float | None = None,
) -> MSEFitResult:
    """Fit a precomputed, possibly pixel-integrated design matrix.

    The matrix must have shape ``values.shape + (number_of_modes,)``.  This
    entry point allows detector-pixel integration to be performed separately
    without changing the linear least-squares solver.
    """
    modes = validate_indices(indices)
    value_array = np.asarray(values, dtype=float)
    matrix_array = np.asarray(matrix, dtype=float)
    expected_shape = value_array.shape + (len(modes),)
    if matrix_array.shape != expected_shape:
        raise ValueError(
            f"matrix must have shape {expected_shape}, got {matrix_array.shape}"
        )
    if not np.all(np.isfinite(value_array)) or not np.all(np.isfinite(matrix_array)):
        raise ValueError("values and matrix must contain only finite values")
    whitening = _whitening_factors(
        value_array.shape,
        uncertainty=uncertainty,
        sample_weight=sample_weight,
    )
    fitted_with_absolute_errors = uncertainty is not None and absolute_uncertainty
    return _solve_design_matrix(
        matrix_array.reshape(-1, len(modes)),
        value_array.reshape(-1),
        value_array.shape,
        modes,
        whitening.reshape(-1),
        absolute_uncertainty=fitted_with_absolute_errors,
        rcond=rcond,
    )


def _fit_linear(
    q: ArrayLike,
    theta: ArrayLike,
    values: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    q_min: float,
    q_max: float,
    whitening: ArrayLike | None,
    absolute_uncertainty: bool,
    rcond: float | None,
) -> MSEFitResult:
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
    if whitening is None:
        whitening_flat = np.ones_like(target)
    else:
        whitening_flat = np.broadcast_to(
            np.asarray(whitening, dtype=float), value_array.shape
        ).reshape(-1)
        if not np.all(np.isfinite(whitening_flat)) or np.any(
            whitening_flat <= 0.0
        ):
            raise ValueError("whitening factors must be finite and positive")
    return _solve_design_matrix(
        matrix,
        target,
        value_array.shape,
        modes,
        whitening_flat,
        absolute_uncertainty=absolute_uncertainty,
        rcond=rcond,
    )


def _whitening_factors(shape, *, uncertainty, sample_weight):
    whitening = np.ones(shape, dtype=float)
    if uncertainty is not None:
        uncertainty_array = np.broadcast_to(
            np.asarray(uncertainty, dtype=float), shape
        )
        if not np.all(np.isfinite(uncertainty_array)) or np.any(
            uncertainty_array <= 0.0
        ):
            raise ValueError("uncertainty must contain finite positive values")
        whitening /= uncertainty_array
    if sample_weight is not None:
        weight_array = np.broadcast_to(np.asarray(sample_weight, dtype=float), shape)
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0.0):
            raise ValueError("sample_weight must contain finite positive values")
        whitening *= np.sqrt(weight_array)
    return whitening


def _solve_design_matrix(
    matrix,
    target,
    shape,
    modes,
    whitening_flat,
    *,
    absolute_uncertainty,
    rcond,
):
    weighted_matrix = whitening_flat[:, None] * matrix
    weighted_target = whitening_flat * target
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted_matrix, weighted_target, rcond=rcond
    )
    if rank != len(modes):
        raise ValueError(
            f"selected design matrix is rank deficient: rank {rank} for {len(modes)} modes"
        )
    fitted_flat = matrix @ coefficients
    residual_flat = target - fitted_flat
    degrees_of_freedom = target.size - rank
    weighted_residual = whitening_flat * residual_flat
    weighted_sum_squares = float(np.sum(weighted_residual**2))
    if degrees_of_freedom > 0:
        residual_variance = weighted_sum_squares / degrees_of_freedom
        covariance_scale = 1.0 if absolute_uncertainty else residual_variance
        coefficient_covariance = covariance_scale * np.linalg.inv(
            weighted_matrix.T @ weighted_matrix
        )
    else:
        residual_variance = float("nan")
        coefficient_covariance = np.full((len(modes), len(modes)), np.nan)
    return MSEFitResult(
        indices=modes,
        coefficients=np.asarray(coefficients, dtype=float),
        fitted_values=fitted_flat.reshape(shape),
        residuals=residual_flat.reshape(shape),
        mse=float(np.mean(residual_flat**2)),
        residual_variance=residual_variance,
        weighted_sum_squares=weighted_sum_squares,
        reduced_chi_square=residual_variance,
        degrees_of_freedom=int(degrees_of_freedom),
        coefficient_covariance=coefficient_covariance,
        covariance_scaled_by_residual=not absolute_uncertainty,
        rank=int(rank),
        singular_values=np.asarray(singular_values, dtype=float),
    )
