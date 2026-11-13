"""Radial spectra associated with fitted angular orders."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import radial_mode
from .indices import validate_indices


@dataclass(frozen=True)
class AngularRadialSpectra:
    """Radial coefficient functions for cosine and sine angular orders."""

    q: NDArray[np.float64]
    orders: NDArray[np.int64]
    cosine: NDArray[np.float64]
    sine: NDArray[np.float64]
    radial_measure_exponent: float
    cosine_uncertainty: NDArray[np.float64] | None = None
    sine_uncertainty: NDArray[np.float64] | None = None


def angular_radial_spectra(
    q: ArrayLike,
    coefficients: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    q_min: float,
    q_max: float,
    radial_measure_exponent: float = 1.0,
    coefficient_covariance: ArrayLike | None = None,
) -> AngularRadialSpectra:
    r"""Evaluate :math:`I_n^{c,s}(q)=\sum_m c_{\pm n,m}R_m^{(s)}(q)`."""
    modes = validate_indices(indices)
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or values.size != len(modes):
        raise ValueError("coefficients must be a one-dimensional array matching indices")
    covariance = None
    if coefficient_covariance is not None:
        covariance = np.asarray(coefficient_covariance, dtype=float)
        if covariance.shape != (len(modes), len(modes)):
            raise ValueError("coefficient_covariance must have shape (K, K)")
    q_values = np.asarray(q, dtype=float)
    if q_values.ndim != 1:
        raise ValueError("q must be one-dimensional")

    n_max = max(abs(n) for n, _ in modes)
    cosine = np.zeros((n_max + 1, q_values.size))
    sine = np.zeros((n_max + 1, q_values.size))
    radial_cache = {
        m: radial_mode(
            m,
            q_values,
            q_min=q_min,
            q_max=q_max,
            radial_measure_exponent=radial_measure_exponent,
        )
        for _, m in modes
    }

    for coefficient, (n, m) in zip(values, modes):
        if n >= 0:
            cosine[n] += coefficient * radial_cache[m]
        else:
            sine[abs(n)] += coefficient * radial_cache[m]

    cosine_uncertainty = None
    sine_uncertainty = None
    if covariance is not None:
        cosine_uncertainty = np.zeros_like(cosine)
        sine_uncertainty = np.zeros_like(sine)
        for order in range(n_max + 1):
            cosine_positions = [
                position for position, (n, _) in enumerate(modes) if n == order
            ]
            sine_positions = [
                position
                for position, (n, _) in enumerate(modes)
                if order > 0 and n == -order
            ]
            cosine_uncertainty[order] = _spectrum_uncertainty(
                modes,
                covariance,
                cosine_positions,
                radial_cache,
                q_values.size,
            )
            sine_uncertainty[order] = _spectrum_uncertainty(
                modes,
                covariance,
                sine_positions,
                radial_cache,
                q_values.size,
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


def _spectrum_uncertainty(
    modes,
    covariance,
    positions,
    radial_cache,
    number_of_q_values,
):
    if not positions:
        return np.zeros(number_of_q_values)
    radial_design = np.column_stack(
        [radial_cache[modes[position][1]] for position in positions]
    )
    selected_covariance = covariance[np.ix_(positions, positions)]
    variance = np.einsum(
        "qi,ij,qj->q",
        radial_design,
        selected_covariance,
        radial_design,
    )
    return np.sqrt(np.maximum(variance, 0.0))
