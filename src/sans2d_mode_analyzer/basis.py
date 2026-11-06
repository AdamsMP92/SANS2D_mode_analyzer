"""Continuous orthonormal basis functions on a circular annulus."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.polynomial.legendre import legval
from numpy.typing import ArrayLike, NDArray

from .indices import ModeIndex, validate_indices


def _annulus(q_min: float, q_max: float) -> tuple[float, float, float]:
    lower = float(q_min)
    upper = float(q_max)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("q_min and q_max must be finite")
    if lower < 0.0 or upper <= lower:
        raise ValueError("require 0 <= q_min < q_max")
    return lower, upper, upper**2 - lower**2


def _integer(value: int, name: str, *, non_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if non_negative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def radial_mode(
    m: int,
    q: ArrayLike,
    *,
    q_min: float,
    q_max: float,
) -> NDArray[np.float64]:
    r"""Evaluate the normalized radial mode :math:`R_m(q)`.

    The Legendre coordinate is affine in :math:`q^2`, so the modes satisfy
    :math:`\int_{q_\min}^{q_\max} R_m R_k q\,dq = \delta_{mk}`.
    """
    degree = _integer(m, "m", non_negative=True)
    lower, upper, delta = _annulus(q_min, q_max)
    q_array = np.asarray(q, dtype=float)
    if not np.all(np.isfinite(q_array)):
        raise ValueError("q must contain only finite values")
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(upper))
    if np.any(q_array < lower - tolerance) or np.any(q_array > upper + tolerance):
        raise ValueError("q values must lie inside [q_min, q_max]")
    x = (2.0 * q_array**2 - upper**2 - lower**2) / delta
    coefficients = np.zeros(degree + 1)
    coefficients[-1] = 1.0
    normalization = np.sqrt(2.0 * (2 * degree + 1) / delta)
    return np.asarray(normalization * legval(x, coefficients), dtype=float)


def angular_mode(n: int, theta: ArrayLike) -> NDArray[np.float64]:
    r"""Evaluate one real orthonormal angular mode on :math:`[0,2\pi)`.

    ``n == 0`` is constant, ``n > 0`` is cosine, and ``n < 0`` is sine.
    The sign therefore encodes the real Fourier branch without a third index.
    """
    order = _integer(n, "n")
    angle = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(angle)):
        raise ValueError("theta must contain only finite values")
    if order == 0:
        return np.full_like(angle, 1.0 / np.sqrt(2.0 * np.pi), dtype=float)
    if order > 0:
        return np.cos(order * angle) / np.sqrt(np.pi)
    return np.sin(abs(order) * angle) / np.sqrt(np.pi)


def basis_function(
    index: Sequence[int],
    q: ArrayLike,
    theta: ArrayLike,
    *,
    q_min: float,
    q_max: float,
) -> NDArray[np.float64]:
    """Evaluate one product mode selected by the two-dimensional index ``(n,m)``."""
    mode = validate_indices([index])[0]
    q_array, theta_array = np.broadcast_arrays(
        np.asarray(q, dtype=float), np.asarray(theta, dtype=float)
    )
    n, m = mode
    return radial_mode(m, q_array, q_min=q_min, q_max=q_max) * angular_mode(
        n, theta_array
    )


def design_matrix(
    q: ArrayLike,
    theta: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    q_min: float,
    q_max: float,
) -> NDArray[np.float64]:
    """Evaluate arbitrary modes and return an array with shape ``q.shape + (K,)``."""
    modes = validate_indices(indices)
    q_array, theta_array = np.broadcast_arrays(
        np.asarray(q, dtype=float), np.asarray(theta, dtype=float)
    )
    columns = [
        basis_function(mode, q_array, theta_array, q_min=q_min, q_max=q_max)
        for mode in modes
    ]
    return np.stack(columns, axis=-1)


def evaluate_expansion(
    q: ArrayLike,
    theta: ArrayLike,
    coefficients: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    q_min: float,
    q_max: float,
) -> NDArray[np.float64]:
    """Evaluate a selected modal sum, including the single-mode case."""
    modes = validate_indices(indices)
    coefficient_array = np.asarray(coefficients, dtype=float)
    if coefficient_array.ndim != 1 or coefficient_array.size != len(modes):
        raise ValueError("coefficients must be a one-dimensional array matching indices")
    if not np.all(np.isfinite(coefficient_array)):
        raise ValueError("coefficients must contain only finite values")
    matrix = design_matrix(q, theta, modes, q_min=q_min, q_max=q_max)
    return np.asarray(matrix @ coefficient_array, dtype=float)
