"""Linear mean-squared-error fits for explicitly selected annulus modes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import design_matrix
from .indices import ModeIndex, validate_indices


@dataclass(frozen=True)
class LinearFitResult:
    """Result of a linear least-squares fit.

    ``mse`` always refers to the unweighted detector-space residuals.
    ``weighted_sum_squares`` and ``reduced_chi_square`` describe the metric
    actually minimized.  They therefore coincide with the ordinary residual
    sum of squares and its degree-of-freedom normalization for ``fit_mse``.
    With absolute uncertainties, ``coefficient_covariance`` propagates the
    supplied independent measurement errors through the fitted linear map.
    With residual scaling it is conditional on noise proportional to the
    inverse combined fit weights.
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
    radial_measure_exponent: float

    def predict(
        self,
        q: ArrayLike,
        theta: ArrayLike,
        *,
        q_min: float,
        q_max: float,
    ) -> NDArray[np.float64]:
        """Evaluate the fitted expansion at new annulus coordinates."""
        matrix = design_matrix(
            q,
            theta,
            self.indices,
            q_min=q_min,
            q_max=q_max,
            radial_measure_exponent=self.radial_measure_exponent,
        )
        return np.asarray(matrix @ self.coefficients, dtype=float)


def fit_mse(
    q: ArrayLike,
    theta: ArrayLike,
    values: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    q_min: float,
    q_max: float,
    radial_measure_exponent: float = 1.0,
    rcond: float | None = None,
) -> LinearFitResult:
    """Fit exactly the supplied modes by minimizing the unweighted MSE.

    ``q``, ``theta``, and ``values`` may be one- or multi-dimensional but must
    broadcast to a common shape.  No modes are added implicitly.
    ``radial_measure_exponent`` selects the radial basis normalization but does
    not infer quadrature weights from an arbitrary sample list.
    """
    return _fit_linear(
        q,
        theta,
        values,
        indices,
        q_min=q_min,
        q_max=q_max,
        radial_measure_exponent=radial_measure_exponent,
        whitening=None,
        measurement_uncertainty=None,
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
    radial_measure_exponent: float = 1.0,
    sample_weight: ArrayLike | None = None,
    absolute_uncertainty: bool = True,
    rcond: float | None = None,
) -> LinearFitResult:
    r"""Fit selected modes by uncertainty-weighted least squares.

    The minimized statistic is

    .. math::

       \sum_p w_p \left[(I_p-I_{\mathrm{fit},p})/\sigma_p\right]^2,

    where ``sample_weight`` supplies :math:`w_p` and defaults to one.  A
    fractional annulus coverage can therefore be passed directly as the
    sample weight.  Nonpositive uncertainties and weights are rejected rather
    than silently clipped.
    ``radial_measure_exponent`` selects the matching radial basis; callers
    working below :func:`standard_analyzer` supply any desired quadrature
    factor explicitly through ``sample_weight``.

    With ``absolute_uncertainty=True``, ``uncertainty`` specifies the actual
    independent measurement standard deviations. The covariance propagates
    these errors through the weighted fit; additional ``sample_weight``
    values change the fit metric, not the measurement variance. To express
    statistical precision weights, include them in ``uncertainty`` itself.

    With ``absolute_uncertainty=False`` the combined weights instead define
    a relative precision model: Var(I_p) = scale * sigma_p**2 / w_p. The
    residuals estimate that scale. This working model does not provide a
    geometric-weight sampling covariance for unknown measurement noise.
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
        radial_measure_exponent=radial_measure_exponent,
        whitening=whitening,
        measurement_uncertainty=uncertainty_array,
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
    radial_measure_exponent: float = 1.0,
    rcond: float | None = None,
) -> LinearFitResult:
    """Fit a precomputed, possibly pixel-integrated design matrix.

    The matrix must have shape ``values.shape + (number_of_modes,)``.  This
    entry point allows detector-pixel integration to be performed separately
    without changing the linear least-squares solver. Absolute uncertainties
    are propagated as independent pixel errors, regardless of sample weights.
    Without absolute uncertainties, the combined weights specify the same
    residual-scaled relative precision model as :func:`fit_wls`.
    """
    modes = validate_indices(indices)
    radial_measure_exponent = float(radial_measure_exponent)
    if (
        not np.isfinite(radial_measure_exponent)
        or radial_measure_exponent <= -1.0
    ):
        raise ValueError(
            "radial_measure_exponent must be finite and greater than -1"
        )
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
        measurement_uncertainty=(
            None if uncertainty is None else np.broadcast_to(
                np.asarray(uncertainty, dtype=float), value_array.shape
            ).reshape(-1)
        ),
        absolute_uncertainty=fitted_with_absolute_errors,
        radial_measure_exponent=radial_measure_exponent,
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
    radial_measure_exponent: float,
    whitening: ArrayLike | None,
    measurement_uncertainty: ArrayLike | None,
    absolute_uncertainty: bool,
    rcond: float | None,
) -> LinearFitResult:
    modes = validate_indices(indices)
    q_array, theta_array, value_array = np.broadcast_arrays(
        np.asarray(q, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(values, dtype=float),
    )
    if not np.all(np.isfinite(value_array)):
        raise ValueError("values must contain only finite values")
    matrix = design_matrix(
        q_array,
        theta_array,
        modes,
        q_min=q_min,
        q_max=q_max,
        radial_measure_exponent=radial_measure_exponent,
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
        measurement_uncertainty=(
            None if measurement_uncertainty is None else np.broadcast_to(
                np.asarray(measurement_uncertainty, dtype=float), value_array.shape
            ).reshape(-1)
        ),
        absolute_uncertainty=absolute_uncertainty,
        radial_measure_exponent=radial_measure_exponent,
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
    measurement_uncertainty,
    absolute_uncertainty,
    radial_measure_exponent,
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
        # QR avoids squaring the design's condition number in a normal-matrix
        # inverse. No N-by-N observation covariance or hat matrix is formed.
        if absolute_uncertainty:
            orthogonal, triangular = np.linalg.qr(weighted_matrix, mode="reduced")
            # For A = R^-1 Q.T sqrt(W), propagate diag(sigma**2):
            # C = (A diag(sigma)) (A diag(sigma)).T. In particular, a
            # fractional area weight does not change the supplied pixel noise.
            noise_response = np.linalg.solve(
                triangular,
                orthogonal.T * (whitening_flat * measurement_uncertainty),
            )
            coefficient_covariance = noise_response @ noise_response.T
        else:
            # Legacy relative precision model, Var(y) = scale * W^-1.
            triangular = np.linalg.qr(weighted_matrix, mode="r")
            inverse_factor = np.linalg.solve(triangular, np.eye(len(modes)))
            coefficient_covariance = residual_variance * (
                inverse_factor @ inverse_factor.T
            )
    else:
        residual_variance = float("nan")
        coefficient_covariance = np.full((len(modes), len(modes)), np.nan)
    return LinearFitResult(
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
        radial_measure_exponent=float(radial_measure_exponent),
    )
