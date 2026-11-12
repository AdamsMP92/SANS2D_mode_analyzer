"""Joint tests of complete radial coefficient blocks at fixed angular order."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2, f

from .indices import validate_indices


@dataclass(frozen=True)
class AngularBlockTests:
    """Raw block tests and joint-family Holm-adjusted p-values/decisions.

    Holm properties combine both trigonometric branches into one family.
    Missing and rank-zero blocks are excluded. The raw fields and constructor
    remain unchanged; corrected properties are derived from those fields.
    """

    orders: NDArray[np.int64]
    cosine_statistic: NDArray[np.float64]
    sine_statistic: NDArray[np.float64]
    cosine_degrees_of_freedom: NDArray[np.int64]
    sine_degrees_of_freedom: NDArray[np.int64]
    cosine_pvalue: NDArray[np.float64]
    sine_pvalue: NDArray[np.float64]
    cosine_critical_value: NDArray[np.float64]
    sine_critical_value: NDArray[np.float64]
    confidence: float

    @property
    def holm_family_size(self) -> int:
        """Number of positive-rank blocks across both branches."""
        return int(
            np.count_nonzero(self.cosine_degrees_of_freedom > 0)
            + np.count_nonzero(self.sine_degrees_of_freedom > 0)
        )

    def _joint_holm_pvalues(self):
        pvalues = np.concatenate((self.cosine_pvalue, self.sine_pvalue))
        present = np.concatenate((
            self.cosine_degrees_of_freedom > 0,
            self.sine_degrees_of_freedom > 0,
        ))
        return _holm_adjust(pvalues, present)

    @property
    def cosine_holm_pvalue(self) -> NDArray[np.float64]:
        """Cosine p-values adjusted over the joint cosine/sine family."""
        return self._joint_holm_pvalues()[:self.orders.size]

    @property
    def sine_holm_pvalue(self) -> NDArray[np.float64]:
        """Sine p-values adjusted over the joint cosine/sine family."""
        return self._joint_holm_pvalues()[self.orders.size:]

    @property
    def cosine_holm_reject(self) -> NDArray[np.bool_]:
        """Holm rejections at family-wise alpha = 1 - confidence."""
        return self.cosine_holm_pvalue <= 1.0 - self.confidence

    @property
    def sine_holm_reject(self) -> NDArray[np.bool_]:
        """Holm rejections at family-wise alpha = 1 - confidence."""
        return self.sine_holm_pvalue <= 1.0 - self.confidence


def angular_block_tests(
    coefficients: ArrayLike,
    coefficient_covariance: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    residual_degrees_of_freedom: int | None = None,
    confidence: float = 0.95,
) -> AngularBlockTests:
    r"""Test whether all radial coefficients at fixed ``n`` are zero.

    The block statistic is :math:`T_n=c_n^T C_n^+c_n`. The pseudoinverse and
    numerical rank ``r`` retain the same singular values strictly above
    ``k * eps * s_max``, where ``k`` is the block size and ``eps`` is machine
    precision. Both are computed from a single SVD of the covariance block.
    When the residual variance was estimated from the same least-squares
    fit under noise proportional to the inverse combined fit weights,
    ``T_n / r`` is compared with an F distribution with
    ``(r, residual_dof)`` degrees of
    freedom. Without ``residual_degrees_of_freedom``, a chi-squared reference
    with ``r`` degrees of freedom is used. It is calibrated for a Gaussian
    block with known absolute sampling covariance, including a covariance
    propagated through additional geometric fit weights.

    The returned Holm properties adjust both branches jointly, using every
    positive-rank block in this call. They retain undefined p-values as NaN
    and never reject those blocks. A positive-rank block with an unavailable
    p-value still counts toward multiplicity, conservatively as p=1.
    Holm controls family-wise error when the marginal p-values are valid;
    it does not repair a misspecified covariance or account for model selection.
    """
    modes = validate_indices(indices)
    values = np.asarray(coefficients, dtype=float)
    covariance = np.asarray(coefficient_covariance, dtype=float)
    if values.ndim != 1 or values.size != len(modes):
        raise ValueError("coefficients must be a one-dimensional array matching indices")
    if covariance.shape != (len(modes), len(modes)):
        raise ValueError("coefficient_covariance must have shape (K, K)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in the interval (0, 1)")

    n_max = max(abs(n) for n, _ in modes)
    orders = np.arange(n_max + 1, dtype=np.int64)
    cosine = [_block(values, covariance, modes, order) for order in orders]
    sine = [
        _block(values, covariance, modes, -order) if order > 0 else _empty_block()
        for order in orders
    ]

    cosine_arrays = _evaluate_reference_distribution(
        cosine, residual_degrees_of_freedom, confidence
    )
    sine_arrays = _evaluate_reference_distribution(
        sine, residual_degrees_of_freedom, confidence
    )
    return AngularBlockTests(
        orders=orders,
        cosine_statistic=cosine_arrays[0],
        sine_statistic=sine_arrays[0],
        cosine_degrees_of_freedom=cosine_arrays[1],
        sine_degrees_of_freedom=sine_arrays[1],
        cosine_pvalue=cosine_arrays[2],
        sine_pvalue=sine_arrays[2],
        cosine_critical_value=cosine_arrays[3],
        sine_critical_value=sine_arrays[3],
        confidence=float(confidence),
    )


def _block(coefficients, covariance, modes, angular_index):
    positions = [
        position for position, (n, _) in enumerate(modes) if n == angular_index
    ]
    if not positions:
        return _empty_block()
    selected_coefficients = coefficients[positions]
    selected_covariance = covariance[np.ix_(positions, positions)]
    left, singular_values, right_transpose = np.linalg.svd(
        selected_covariance, full_matrices=False
    )
    cutoff = len(positions) * np.finfo(selected_covariance.dtype).eps * singular_values[0]
    # One cutoff and one SVD keep the statistic and its reference rank aligned.
    retained = singular_values > cutoff
    rank = int(np.count_nonzero(retained))
    pseudoinverse = (
        right_transpose[retained].T / singular_values[retained]
    ) @ left[:, retained].T
    statistic = float(
        selected_coefficients
        @ pseudoinverse
        @ selected_coefficients
    )
    return statistic, rank


def _empty_block():
    return float("nan"), 0


def _evaluate_reference_distribution(blocks, residual_dof, confidence):
    statistic = np.asarray([block[0] for block in blocks], dtype=float)
    block_dof = np.asarray([block[1] for block in blocks], dtype=np.int64)
    pvalue = np.full(statistic.shape, np.nan)
    critical = np.full(statistic.shape, np.nan)
    valid = block_dof > 0
    if residual_dof is None:
        pvalue[valid] = chi2.sf(statistic[valid], block_dof[valid])
        critical[valid] = chi2.ppf(confidence, block_dof[valid])
    else:
        pvalue[valid] = f.sf(
            statistic[valid] / block_dof[valid],
            block_dof[valid],
            residual_dof,
        )
        critical[valid] = block_dof[valid] * f.ppf(
            confidence,
            block_dof[valid],
            residual_dof,
        )
    return statistic, block_dof, pvalue, critical


def _holm_adjust(pvalues, present):
    """Joint step-down adjustment; unavailable present tests count as p=1."""
    adjusted = np.full(pvalues.shape, np.nan)
    family_size = np.count_nonzero(present)
    available = present & np.isfinite(pvalues) & (pvalues >= 0) & (pvalues <= 1)
    positions = np.flatnonzero(available)
    if not positions.size:
        return adjusted
    order = positions[np.argsort(pvalues[positions], kind="stable")]
    factors = family_size - np.arange(order.size)
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(factors * pvalues[order])
    )
    return adjusted
