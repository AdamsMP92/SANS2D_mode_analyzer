"""Parseval powers of fitted angular modes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .indices import validate_indices


@dataclass(frozen=True)
class AngularModePowers:
    """Cosine and sine power summed over all selected radial modes."""

    orders: NDArray[np.int64]
    cosine: NDArray[np.float64]
    sine: NDArray[np.float64]
    cosine_uncertainty: NDArray[np.float64] | None = None
    sine_uncertainty: NDArray[np.float64] | None = None

    @property
    def total(self) -> NDArray[np.float64]:
        return self.cosine + self.sine

    @property
    def total_power(self) -> float:
        return float(np.sum(self.total))


@dataclass(frozen=True)
class ParityPowerFractions:
    """Fractions of fitted coefficient power in even and odd angular orders."""

    even: float
    odd: float
    total_power: float


@dataclass(frozen=True)
class ParityMismatchMeasures:
    """Fit-based measure-averaged and scale-free parity mismatch."""

    area: float
    measure: float
    radial_measure_exponent: float
    mean_square: float
    rms: float
    normalized_mean_square: float
    normalized_rms: float


def angular_mode_powers(
    coefficients: ArrayLike,
    indices: Iterable[Sequence[int]],
    coefficient_covariance: ArrayLike | None = None,
) -> AngularModePowers:
    """Sum squared coefficients over radial modes for each angular order.

    Positive ``n`` contributes to the cosine power, negative ``n`` to the sine
    power at order ``abs(n)``, and ``n == 0`` to the constant/cosine panel.
    """
    modes = validate_indices(indices)
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or values.size != len(modes):
        raise ValueError("coefficients must be a one-dimensional array matching indices")

    covariance = None
    if coefficient_covariance is not None:
        covariance = np.asarray(coefficient_covariance, dtype=float)
        if covariance.shape != (len(modes), len(modes)):
            raise ValueError("coefficient_covariance must have shape (K, K)")

    n_max = max(abs(n) for n, _ in modes)
    cosine = np.zeros(n_max + 1)
    sine = np.zeros(n_max + 1)

    for coefficient, (n, _) in zip(values, modes):
        if n >= 0:
            cosine[n] += coefficient**2
        else:
            sine[abs(n)] += coefficient**2

    cosine_uncertainty = None
    sine_uncertainty = None
    if covariance is not None:
        cosine_uncertainty = np.zeros(n_max + 1)
        sine_uncertainty = np.zeros(n_max + 1)
        for order in range(n_max + 1):
            cosine_positions = [
                position for position, (n, _) in enumerate(modes) if n == order
            ]
            sine_positions = [
                position for position, (n, _) in enumerate(modes) if n == -order and order > 0
            ]
            cosine_uncertainty[order] = _quadratic_power_uncertainty(
                values, covariance, cosine_positions
            )
            sine_uncertainty[order] = _quadratic_power_uncertainty(
                values, covariance, sine_positions
            )

    return AngularModePowers(
        orders=np.arange(n_max + 1, dtype=np.int64),
        cosine=cosine,
        sine=sine,
        cosine_uncertainty=cosine_uncertainty,
        sine_uncertainty=sine_uncertainty,
    )


def parity_power_fractions(
    coefficients: ArrayLike,
    indices: Iterable[Sequence[int]],
) -> ParityPowerFractions:
    """Return even- and odd-angular-order fractions of total fitted power."""
    modes = validate_indices(indices)
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or values.size != len(modes):
        raise ValueError("coefficients must be a one-dimensional array matching indices")

    powers = values**2
    even_orders = np.array([n % 2 == 0 for n, _ in modes])
    even_power = float(np.sum(powers[even_orders]))
    odd_power = float(np.sum(powers[~even_orders]))
    total_power = even_power + odd_power
    if total_power == 0.0:
        raise ValueError("parity power fractions are undefined for zero total power")
    return ParityPowerFractions(
        even=float(even_power / total_power),
        odd=float(odd_power / total_power),
        total_power=total_power,
    )


def parity_mismatch_measures(
    coefficients: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    q_min: float,
    q_max: float,
    radial_measure_exponent: float = 1.0,
) -> ParityMismatchMeasures:
    r"""Evaluate the fitted parity mismatch analytically by Parseval's identity.

    For the orthonormal annulus expansion, every odd angular mode changes sign
    under :math:`\theta\mapsto\theta+\pi`. Therefore

    .. math::

       \int_{\mathcal A}[I(q,\theta)-I(q,\theta+\pi)]^2\,d^2q
       =4\sum_{|n|\ \mathrm{odd},m}c_{nm}^2.

    ``mean_square`` is normalized by the selected annular measure and retains
    units of intensity squared. ``normalized_mean_square`` divides once more
    by the fitted signal integral in the same measure; its square root is a
    dimensionless relative RMS mismatch. All values refer to the truncated
    fitted expansion, not directly paired raw detector pixels.
    """
    lower = float(q_min)
    upper = float(q_max)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("q_min and q_max must be finite")
    if lower < 0.0 or upper <= lower:
        raise ValueError("require 0 <= q_min < q_max")
    exponent = float(radial_measure_exponent)
    if not np.isfinite(exponent) or exponent <= -1.0:
        raise ValueError("radial_measure_exponent must be finite and greater than -1")
    fractions = parity_power_fractions(coefficients, indices)
    area = float(np.pi * (upper**2 - lower**2))
    measure = float(
        2.0
        * np.pi
        * (upper ** (exponent + 1.0) - lower ** (exponent + 1.0))
        / (exponent + 1.0)
    )
    integrated_mismatch = 4.0 * fractions.odd * fractions.total_power
    mean_square = integrated_mismatch / measure
    normalized_mean_square = integrated_mismatch / fractions.total_power
    return ParityMismatchMeasures(
        area=area,
        measure=measure,
        radial_measure_exponent=exponent,
        mean_square=float(mean_square),
        rms=float(np.sqrt(mean_square)),
        normalized_mean_square=float(normalized_mean_square),
        normalized_rms=float(np.sqrt(normalized_mean_square)),
    )


def _quadratic_power_uncertainty(
    coefficients: NDArray[np.float64],
    covariance: NDArray[np.float64],
    positions: list[int],
) -> float:
    """Standard deviation of c.T c for Gaussian coefficient uncertainty."""
    if not positions:
        return 0.0
    selected_coefficients = coefficients[positions]
    selected_covariance = covariance[np.ix_(positions, positions)]
    variance = 2.0 * np.trace(selected_covariance @ selected_covariance)
    variance += 4.0 * (
        selected_coefficients @ selected_covariance @ selected_coefficients
    )
    return float(np.sqrt(max(variance, 0.0)))
