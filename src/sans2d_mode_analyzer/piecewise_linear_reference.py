"""Piecewise-linear radial reference analysis for SANS mode fits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fit import LinearFitResult, fit_design_matrix
from .indices import ModeIndex
from .piecewise_linear_basis import (
    piecewise_linear_design_matrix,
    piecewise_linear_gram_matrix,
    piecewise_linear_indices,
    piecewise_linear_radial_basis,
)
from .power import AngularModePowers
from .spectra import AngularRadialSpectra


@dataclass(frozen=True)
class PiecewiseLinearFitResult:
    """Linear-fit output whose radial coefficients are values at the knots."""

    indices: tuple[ModeIndex, ...]
    knots: NDArray[np.float64]
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

    def predict(self, q: ArrayLike, theta: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the fitted polygonal expansion at new annulus coordinates."""
        matrix = piecewise_linear_design_matrix(
            q,
            theta,
            self.indices,
            knots=self.knots,
        )
        return np.asarray(matrix @ self.coefficients, dtype=float)


@dataclass(frozen=True)
class PiecewiseLinearReferenceResult:
    """Fit, radial spectra, and Gram-corrected powers of the reference branch."""

    fit: PiecewiseLinearFitResult
    knots: NDArray[np.float64]
    gram_matrix: NDArray[np.float64]
    spectra: AngularRadialSpectra
    powers: AngularModePowers


def fit_piecewise_linear_reference(
    q: ArrayLike,
    theta: ArrayLike,
    values: ArrayLike,
    angular_modes: Iterable[int],
    *,
    knots: ArrayLike,
    radial_measure_exponent: float = 1.0,
    uncertainty: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    absolute_uncertainty: bool = True,
    design_matrix: ArrayLike | None = None,
    spectrum_q: ArrayLike | None = None,
) -> PiecewiseLinearReferenceResult:
    """Fit a polygonal radial profile for every selected angular mode."""
    knot_values = np.asarray(knots, dtype=float)
    radial_segments = knot_values.size - 1
    indices = piecewise_linear_indices(
        angular_modes,
        radial_segments=radial_segments,
    )
    q_values, theta_values, intensity_values = np.broadcast_arrays(
        np.asarray(q, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(values, dtype=float),
    )
    if design_matrix is None:
        matrix = piecewise_linear_design_matrix(
            q_values,
            theta_values,
            indices,
            knots=knot_values,
        )
    else:
        matrix = np.asarray(design_matrix, dtype=float)
    linear_fit = fit_design_matrix(
        intensity_values,
        indices,
        matrix,
        uncertainty=uncertainty,
        sample_weight=sample_weight,
        absolute_uncertainty=absolute_uncertainty,
        radial_measure_exponent=radial_measure_exponent,
    )
    fit = _piecewise_fit_result(linear_fit, knot_values)
    gram = piecewise_linear_gram_matrix(
        knot_values,
        radial_measure_exponent=radial_measure_exponent,
    )
    powers = piecewise_linear_angular_powers(
        fit.coefficients,
        fit.indices,
        gram,
        fit.coefficient_covariance,
    )
    if spectrum_q is None:
        spectrum_values = np.linspace(knot_values[0], knot_values[-1], 300)
    else:
        spectrum_values = np.asarray(spectrum_q, dtype=float)
    spectra = piecewise_linear_angular_spectra(
        spectrum_values,
        fit.coefficients,
        fit.indices,
        knot_values,
        radial_measure_exponent=radial_measure_exponent,
        coefficient_covariance=fit.coefficient_covariance,
    )
    return PiecewiseLinearReferenceResult(
        fit=fit,
        knots=knot_values.copy(),
        gram_matrix=gram,
        spectra=spectra,
        powers=powers,
    )


def piecewise_linear_angular_spectra(
    q: ArrayLike,
    coefficients: ArrayLike,
    indices: tuple[ModeIndex, ...],
    knots: ArrayLike,
    *,
    radial_measure_exponent: float = 1.0,
    coefficient_covariance: ArrayLike | None = None,
) -> AngularRadialSpectra:
    """Evaluate cosine and sine polygonal spectra and their covariance bands."""
    values = np.asarray(coefficients, dtype=float)
    knot_values = np.asarray(knots, dtype=float)
    q_values = np.asarray(q, dtype=float)
    radial = piecewise_linear_radial_basis(q_values, knot_values)
    covariance = None
    if coefficient_covariance is not None:
        covariance = np.asarray(coefficient_covariance, dtype=float)
    n_max = max(abs(order) for order, _ in indices)
    cosine = np.zeros((n_max + 1, q_values.size))
    sine = np.zeros_like(cosine)
    cosine_uncertainty = None if covariance is None else np.zeros_like(cosine)
    sine_uncertainty = None if covariance is None else np.zeros_like(sine)
    for order in range(n_max + 1):
        cosine_positions = [
            position for position, (n, _) in enumerate(indices) if n == order
        ]
        sine_positions = [
            position
            for position, (n, _) in enumerate(indices)
            if order > 0 and n == -order
        ]
        if cosine_positions:
            cosine[order] = _branch_spectrum(
                radial, values, indices, cosine_positions
            )
            if covariance is not None:
                cosine_uncertainty[order] = _branch_uncertainty(
                    radial, covariance, indices, cosine_positions
                )
        if sine_positions:
            sine[order] = _branch_spectrum(radial, values, indices, sine_positions)
            if covariance is not None:
                sine_uncertainty[order] = _branch_uncertainty(
                    radial, covariance, indices, sine_positions
                )
    return AngularRadialSpectra(
        q=q_values,
        orders=np.arange(n_max + 1, dtype=np.int64),
        cosine=cosine,
        sine=sine,
        radial_measure_exponent=float(radial_measure_exponent),
        cosine_uncertainty=cosine_uncertainty,
        sine_uncertainty=sine_uncertainty,
    )


def piecewise_linear_angular_powers(
    coefficients: ArrayLike,
    indices: tuple[ModeIndex, ...],
    gram_matrix: ArrayLike,
    coefficient_covariance: ArrayLike | None = None,
) -> AngularModePowers:
    r"""Compute :math:`a_n^T G a_n` for every angular branch."""
    values = np.asarray(coefficients, dtype=float)
    gram = np.asarray(gram_matrix, dtype=float)
    covariance = None
    if coefficient_covariance is not None:
        covariance = np.asarray(coefficient_covariance, dtype=float)
    n_max = max(abs(order) for order, _ in indices)
    cosine = np.zeros(n_max + 1)
    sine = np.zeros(n_max + 1)
    cosine_uncertainty = None if covariance is None else np.zeros(n_max + 1)
    sine_uncertainty = None if covariance is None else np.zeros(n_max + 1)
    for order in range(n_max + 1):
        for branch_order, power, uncertainty in (
            (order, cosine, cosine_uncertainty),
            (-order, sine, sine_uncertainty),
        ):
            if order == 0 and branch_order < 0:
                continue
            positions = [
                position for position, (n, _) in enumerate(indices) if n == branch_order
            ]
            if not positions:
                continue
            knot_indices = [indices[position][1] for position in positions]
            branch_gram = gram[np.ix_(knot_indices, knot_indices)]
            branch_coefficients = values[positions]
            power[order] = branch_coefficients @ branch_gram @ branch_coefficients
            if covariance is not None:
                branch_covariance = covariance[np.ix_(positions, positions)]
                uncertainty[order] = _quadratic_form_uncertainty(
                    branch_coefficients,
                    branch_covariance,
                    branch_gram,
                )
    return AngularModePowers(
        orders=np.arange(n_max + 1, dtype=np.int64),
        cosine=cosine,
        sine=sine,
        cosine_uncertainty=cosine_uncertainty,
        sine_uncertainty=sine_uncertainty,
    )


def _piecewise_fit_result(
    result: LinearFitResult,
    knots: NDArray[np.float64],
) -> PiecewiseLinearFitResult:
    return PiecewiseLinearFitResult(
        indices=result.indices,
        knots=knots.copy(),
        coefficients=result.coefficients,
        fitted_values=result.fitted_values,
        residuals=result.residuals,
        mse=result.mse,
        residual_variance=result.residual_variance,
        weighted_sum_squares=result.weighted_sum_squares,
        reduced_chi_square=result.reduced_chi_square,
        degrees_of_freedom=result.degrees_of_freedom,
        coefficient_covariance=result.coefficient_covariance,
        covariance_scaled_by_residual=result.covariance_scaled_by_residual,
        rank=result.rank,
        singular_values=result.singular_values,
        radial_measure_exponent=result.radial_measure_exponent,
    )


def _branch_spectrum(radial, coefficients, indices, positions):
    knot_indices = [indices[position][1] for position in positions]
    return radial[:, knot_indices] @ coefficients[positions]


def _branch_uncertainty(radial, covariance, indices, positions):
    knot_indices = [indices[position][1] for position in positions]
    design = radial[:, knot_indices]
    branch_covariance = covariance[np.ix_(positions, positions)]
    variance = np.einsum("qi,ij,qj->q", design, branch_covariance, design)
    return np.sqrt(np.maximum(variance, 0.0))


def _quadratic_form_uncertainty(coefficients, covariance, gram):
    variance = 2.0 * np.trace(gram @ covariance @ gram @ covariance)
    variance += 4.0 * coefficients @ gram @ covariance @ gram @ coefficients
    return float(np.sqrt(max(variance, 0.0)))
