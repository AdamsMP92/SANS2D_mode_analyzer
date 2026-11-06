"""Tests of fitted modes against prescribed angular intensity templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .indices import ModeIndex, validate_indices


@dataclass(frozen=True)
class AngularTemplateTest:
    """Orthogonal projection of fitted modes onto ``A(q) f(theta)``.

    The radial amplitude ``A(q)`` remains arbitrary within the selected radial
    basis.  All reported powers are fractions of the fitted Parseval power.
    """

    name: str
    harmonic_order: int
    fourier_harmonic_to_constant_ratio: float
    orthonormal_harmonic_to_constant_ratio: float
    explained_power_fraction: float
    residual_power_fraction: float
    pair_mismatch_power_fraction: float
    forbidden_mode_power_fraction: float
    radial_orders: NDArray[np.int64]
    radial_scale_coefficients: NDArray[np.float64]
    target_coefficients: NDArray[np.float64]
    residual_coefficients: NDArray[np.float64]
    indices: tuple[ModeIndex, ...]


def angular_template_test(
    coefficients: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    harmonic_order: int,
    fourier_harmonic_to_constant_ratio: float,
    name: str = "angular template",
) -> AngularTemplateTest:
    """Test a constant-plus-cosine angular template with arbitrary ``A(q)``.

    The conventional Fourier template is

    ``f(theta) proportional to 1 + r cos(harmonic_order * theta)``.

    Here ``r`` is ``fourier_harmonic_to_constant_ratio``.  Because this
    package uses ``T_0 = 1/sqrt(2 pi)`` and
    ``T_n = cos(n theta)/sqrt(pi)``, the corresponding fitted coefficient
    condition is ``c_h/c_0 = r/sqrt(2)`` for every radial mode.

    The routine projects each radial coefficient pair ``(c_0m, c_hm)`` onto
    that exact ratio.  Every sine coefficient and every other cosine order is
    orthogonal to the target subspace and therefore counted as forbidden-mode
    power.
    """
    modes = validate_indices(indices)
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or values.size != len(modes):
        raise ValueError("coefficients must be a one-dimensional array matching indices")
    if harmonic_order <= 0:
        raise ValueError("harmonic_order must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be finite")

    constant_radial_orders = {m for n, m in modes if n == 0}
    harmonic_radial_orders = {m for n, m in modes if n == harmonic_order}
    if not constant_radial_orders or constant_radial_orders != harmonic_radial_orders:
        raise ValueError(
            "constant and target cosine modes must contain the same radial orders"
        )

    total_power = float(values @ values)
    if total_power == 0.0:
        raise ValueError("angular template test is undefined for zero fitted power")

    orthonormal_ratio = float(fourier_harmonic_to_constant_ratio) / np.sqrt(2.0)
    template_vector = np.array([1.0, orthonormal_ratio])
    template_norm_squared = float(template_vector @ template_vector)

    position = {mode: index for index, mode in enumerate(modes)}
    radial_orders = np.array(sorted(constant_radial_orders), dtype=np.int64)
    radial_scales = np.empty(radial_orders.size)
    target = np.zeros_like(values)

    for radial_index, radial_order in enumerate(radial_orders):
        constant_position = position[(0, int(radial_order))]
        harmonic_position = position[(harmonic_order, int(radial_order))]
        pair = values[[constant_position, harmonic_position]]
        scale = float(pair @ template_vector / template_norm_squared)
        radial_scales[radial_index] = scale
        target[constant_position] = scale
        target[harmonic_position] = scale * orthonormal_ratio

    residual = values - target
    pair_mask = np.array(
        [n in (0, harmonic_order) for n, _ in modes],
        dtype=bool,
    )
    explained_power = float(target @ target)
    pair_mismatch_power = float(residual[pair_mask] @ residual[pair_mask])
    forbidden_power = float(values[~pair_mask] @ values[~pair_mask])
    residual_power = float(residual @ residual)

    return AngularTemplateTest(
        name=name,
        harmonic_order=harmonic_order,
        fourier_harmonic_to_constant_ratio=float(
            fourier_harmonic_to_constant_ratio
        ),
        orthonormal_harmonic_to_constant_ratio=orthonormal_ratio,
        explained_power_fraction=explained_power / total_power,
        residual_power_fraction=residual_power / total_power,
        pair_mismatch_power_fraction=pair_mismatch_power / total_power,
        forbidden_mode_power_fraction=forbidden_power / total_power,
        radial_orders=radial_orders,
        radial_scale_coefficients=radial_scales,
        target_coefficients=target,
        residual_coefficients=residual,
        indices=modes,
    )


def test_sin_squared(
    coefficients: ArrayLike,
    indices: Iterable[Sequence[int]],
) -> AngularTemplateTest:
    """Test ``I(q, theta) = A(q) sin(theta)^2``.

    In conventional Fourier amplitudes the exact condition is
    ``a_2(q) / a_0(q) = -1``.  In the package's orthonormal basis it is
    ``c_2(q) / c_0(q) = -1/sqrt(2)``.
    """
    return angular_template_test(
        coefficients,
        indices,
        harmonic_order=2,
        fourier_harmonic_to_constant_ratio=-1.0,
        name="sin(theta)^2",
    )


def test_sin_squared_cos_squared(
    coefficients: ArrayLike,
    indices: Iterable[Sequence[int]],
) -> AngularTemplateTest:
    """Test ``I(q, theta) = A(q) sin(theta)^2 cos(theta)^2``.

    Since ``sin(theta)^2 cos(theta)^2 = (1 - cos(4 theta))/8``, the
    conventional Fourier condition is ``a_4(q) / a_0(q) = -1`` and the
    orthonormal-basis condition is ``c_4(q) / c_0(q) = -1/sqrt(2)``.
    """
    return angular_template_test(
        coefficients,
        indices,
        harmonic_order=4,
        fourier_harmonic_to_constant_ratio=-1.0,
        name="sin(theta)^2 cos(theta)^2",
    )
