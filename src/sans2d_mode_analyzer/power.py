"""Parseval powers of fitted angular modes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .indices import ModeIndex, validate_indices


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


def power_model_selector(
    coefficients: ArrayLike,
    indices: Iterable[Sequence[int]],
    threshold: float = 0.95,
) -> list[ModeIndex]:
    """Return the smallest power-ranked mode list reaching ``threshold``.

    The threshold is a fraction of the fitted Parseval power
    ``sum(coefficients**2)``.  Returned indices are ordered from largest to
    smallest individual mode power.
    """
    modes = validate_indices(indices)
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or values.size != len(modes):
        raise ValueError("coefficients must be a one-dimensional array matching indices")
    fraction = float(threshold)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("threshold must lie in the interval (0, 1]")

    powers = values**2
    total_power = float(np.sum(powers))
    if total_power == 0.0:
        return []
    ranking = np.argsort(-powers, kind="stable")
    cumulative = np.cumsum(powers[ranking]) / total_power
    number = int(np.searchsorted(cumulative, fraction, side="left") + 1)
    return [modes[position] for position in ranking[:number]]


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
    total_power = float(np.sum(powers))
    if total_power == 0.0:
        raise ValueError("parity power fractions are undefined for zero total power")
    even_power = sum(
        power for power, (n, _) in zip(powers, modes) if abs(n) % 2 == 0
    )
    odd_power = total_power - even_power
    return ParityPowerFractions(
        even=float(even_power / total_power),
        odd=float(odd_power / total_power),
        total_power=total_power,
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
