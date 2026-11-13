"""Piecewise-linear radial basis functions on a circular annulus."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import angular_mode, radial_measure_weight
from .indices import ModeIndex, validate_indices
from .preconditioning import affine_detector_geometry, fractional_annulus_weights


@dataclass(frozen=True)
class PiecewiseLinearIntegratedDesign:
    """Pixel-averaged piecewise-linear design and radial quadrature weights."""

    selected_mask: NDArray[np.bool_]
    coverage: NDArray[np.float64]
    quadrature_weight: NDArray[np.float64]
    indices: tuple[ModeIndex, ...]
    design_matrix: NDArray[np.float64]


def piecewise_linear_knots(
    *,
    q_min: float,
    q_max: float,
    radial_segments: int,
    radial_measure_exponent: float = 1.0,
    spacing: str = "measure",
) -> NDArray[np.float64]:
    """Create ``radial_segments + 1`` knots spanning the analysis annulus.

    ``spacing='measure'`` gives equal intervals in the selected
    :math:`q^s dq` measure. ``spacing='q'`` gives equal increments in q.
    """
    lower, upper = _annulus(q_min, q_max)
    segments = _positive_integer(radial_segments, "radial_segments")
    exponent = _measure_exponent(radial_measure_exponent)
    if spacing == "q":
        return np.linspace(lower, upper, segments + 1)
    if spacing != "measure":
        raise ValueError("spacing must be 'measure' or 'q'")
    transformed = np.linspace(
        lower ** (exponent + 1.0),
        upper ** (exponent + 1.0),
        segments + 1,
    )
    return transformed ** (1.0 / (exponent + 1.0))


def piecewise_linear_indices(
    angular_modes: Iterable[int],
    *,
    radial_segments: int,
) -> tuple[ModeIndex, ...]:
    """Return all angular-mode and knot-index pairs in radial-major order."""
    segments = _positive_integer(radial_segments, "radial_segments")
    orders = []
    seen = set()
    for value in angular_modes:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("angular modes must be integers")
        order = int(value)
        if order in seen:
            raise ValueError("angular modes must not contain duplicates")
        seen.add(order)
        orders.append(order)
    if not orders:
        raise ValueError("at least one angular mode is required")
    return tuple((order, knot) for knot in range(segments + 1) for order in orders)


def piecewise_linear_radial_basis(
    q: ArrayLike,
    knots: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate nodal hat functions with shape ``q.shape + (number_of_knots,)``."""
    knot_values = _validate_knots(knots)
    q_values = np.asarray(q, dtype=float)
    if not np.all(np.isfinite(q_values)):
        raise ValueError("q must contain only finite values")
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(knot_values[-1]))
    if np.any(q_values < knot_values[0] - tolerance) or np.any(
        q_values > knot_values[-1] + tolerance
    ):
        raise ValueError("q values must lie inside the knot interval")
    clipped = np.clip(q_values, knot_values[0], knot_values[-1])
    interval = np.searchsorted(knot_values, clipped, side="right") - 1
    interval = np.clip(interval, 0, knot_values.size - 2)
    left = knot_values[interval]
    right = knot_values[interval + 1]
    fraction = (clipped - left) / (right - left)
    basis = np.zeros(clipped.shape + (knot_values.size,), dtype=float)
    flat_basis = basis.reshape(-1, knot_values.size)
    flat_interval = interval.ravel()
    flat_fraction = fraction.ravel()
    rows = np.arange(flat_interval.size)
    flat_basis[rows, flat_interval] = 1.0 - flat_fraction
    flat_basis[rows, flat_interval + 1] = flat_fraction
    return basis


def piecewise_linear_design_matrix(
    q: ArrayLike,
    theta: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    knots: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate selected products of angular modes and radial hat functions."""
    modes = validate_indices(indices)
    knot_values = _validate_knots(knots)
    if max(knot for _, knot in modes) >= knot_values.size:
        raise ValueError("radial knot index exceeds the supplied knot array")
    q_values, theta_values = np.broadcast_arrays(
        np.asarray(q, dtype=float), np.asarray(theta, dtype=float)
    )
    radial = piecewise_linear_radial_basis(q_values, knot_values)
    columns = [radial[..., knot] * angular_mode(order, theta_values) for order, knot in modes]
    return np.stack(columns, axis=-1)


def piecewise_linear_gram_matrix(
    knots: ArrayLike,
    *,
    radial_measure_exponent: float = 1.0,
) -> NDArray[np.float64]:
    r"""Return :math:`G_{ij}=\int B_i(q)B_j(q)q^s dq` analytically."""
    knot_values = _validate_knots(knots)
    exponent = _measure_exponent(radial_measure_exponent)
    gram = np.zeros((knot_values.size, knot_values.size), dtype=float)
    for interval, (left, right) in enumerate(zip(knot_values[:-1], knot_values[1:])):
        width = right - left
        moment_0 = _power_integral(left, right, exponent)
        moment_1 = _power_integral(left, right, exponent + 1.0)
        moment_2 = _power_integral(left, right, exponent + 2.0)
        left_square = (
            right**2 * moment_0 - 2.0 * right * moment_1 + moment_2
        ) / width**2
        right_square = (
            moment_2 - 2.0 * left * moment_1 + left**2 * moment_0
        ) / width**2
        cross = (
            -moment_2 + (left + right) * moment_1 - left * right * moment_0
        ) / width**2
        gram[interval, interval] += left_square
        gram[interval + 1, interval + 1] += right_square
        gram[interval, interval + 1] += cross
        gram[interval + 1, interval] += cross
    return gram


def piecewise_linear_pixel_integrated_design(
    q_z: ArrayLike,
    q_y: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    knots: ArrayLike,
    radial_measure_exponent: float = 1.0,
    subpixels: int = 7,
    chunk_size: int = 512,
) -> PiecewiseLinearIntegratedDesign:
    """Average the piecewise-linear design over admitted detector subpixels."""
    knot_values = _validate_knots(knots)
    lower = float(knot_values[0])
    upper = float(knot_values[-1])
    modes = validate_indices(indices)
    q_z_values, q_y_values = np.broadcast_arrays(
        np.asarray(q_z, dtype=float), np.asarray(q_y, dtype=float)
    )
    chunk = _positive_integer(chunk_size, "chunk_size")
    if q_z_values.ndim == 1:
        radius = np.hypot(q_z_values, q_y_values)
        selected = (
            np.isfinite(q_z_values)
            & np.isfinite(q_y_values)
            & (radius >= lower)
            & (radius <= upper)
        )
        matrix = piecewise_linear_design_matrix(
            radius[selected],
            np.arctan2(q_y_values[selected], q_z_values[selected]),
            modes,
            knots=knot_values,
        )
        quadrature_weight = np.zeros_like(radius)
        quadrature_weight[selected] = radial_measure_weight(
            radius[selected],
            q_min=lower,
            q_max=upper,
            radial_measure_exponent=radial_measure_exponent,
        )
        return PiecewiseLinearIntegratedDesign(
            selected_mask=selected,
            coverage=selected.astype(float),
            quadrature_weight=quadrature_weight,
            indices=modes,
            design_matrix=matrix,
        )
    if q_z_values.ndim != 2:
        raise ValueError("detector coordinates must be one- or two-dimensional")
    number = _positive_integer(subpixels, "subpixels")
    geometry = affine_detector_geometry(q_z_values, q_y_values)
    coverage = fractional_annulus_weights(
        q_z_values,
        q_y_values,
        q_min=lower,
        q_max=upper,
        subpixels=number,
    )
    selected = coverage > 0.0
    positions = np.flatnonzero(selected)
    integrated = np.zeros((positions.size, len(modes)), dtype=float)
    quadrature_weight = np.zeros(q_z_values.shape, dtype=float)
    offsets = (np.arange(number, dtype=float) + 0.5) / number - 0.5
    flat_q_z = q_z_values.ravel()
    flat_q_y = q_y_values.ravel()
    for start in range(0, positions.size, chunk):
        stop = min(start + chunk, positions.size)
        chunk_positions = positions[start:stop]
        accumulated = np.zeros((chunk_positions.size, len(modes)), dtype=float)
        accumulated_weight = np.zeros(chunk_positions.size, dtype=float)
        counts = np.zeros(chunk_positions.size, dtype=int)
        for row_offset in offsets:
            for column_offset in offsets:
                delta_q = geometry.matrix @ np.array([row_offset, column_offset])
                sample_q_z = flat_q_z[chunk_positions] + delta_q[0]
                sample_q_y = flat_q_y[chunk_positions] + delta_q[1]
                sample_q = np.hypot(sample_q_z, sample_q_y)
                inside = (sample_q >= lower) & (sample_q <= upper)
                if np.any(inside):
                    accumulated[inside] += piecewise_linear_design_matrix(
                        sample_q[inside],
                        np.arctan2(sample_q_y[inside], sample_q_z[inside]),
                        modes,
                        knots=knot_values,
                    )
                    accumulated_weight[inside] += radial_measure_weight(
                        sample_q[inside],
                        q_min=lower,
                        q_max=upper,
                        radial_measure_exponent=radial_measure_exponent,
                    )
                    counts[inside] += 1
        if np.any(counts == 0):
            raise RuntimeError("selected annulus pixel received no subpixel samples")
        integrated[start:stop] = accumulated / counts[:, None]
        quadrature_weight.ravel()[chunk_positions] = accumulated_weight / number**2
    return PiecewiseLinearIntegratedDesign(
        selected_mask=selected,
        coverage=coverage,
        quadrature_weight=quadrature_weight,
        indices=modes,
        design_matrix=integrated,
    )


def _annulus(q_min: float, q_max: float) -> tuple[float, float]:
    lower = float(q_min)
    upper = float(q_max)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("q_min and q_max must be finite")
    if lower < 0.0 or upper <= lower:
        raise ValueError("require 0 <= q_min < q_max")
    return lower, upper


def _measure_exponent(value: float) -> float:
    exponent = float(value)
    if not np.isfinite(exponent) or exponent <= -1.0:
        raise ValueError("radial_measure_exponent must be finite and greater than -1")
    return exponent


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_knots(knots: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(knots, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("knots must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(values)) or values[0] < 0.0:
        raise ValueError("knots must be finite and non-negative")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("knots must be strictly increasing")
    return values


def _power_integral(lower: float, upper: float, exponent: float) -> float:
    return (upper ** (exponent + 1.0) - lower ** (exponent + 1.0)) / (
        exponent + 1.0
    )
