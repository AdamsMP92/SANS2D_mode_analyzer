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
    """Separate cosine and sine block tests for every angular order."""

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


def angular_block_tests(
    coefficients: ArrayLike,
    coefficient_covariance: ArrayLike,
    indices: Iterable[Sequence[int]],
    *,
    residual_degrees_of_freedom: int | None = None,
    confidence: float = 0.95,
) -> AngularBlockTests:
    r"""Test whether all radial coefficients at fixed ``n`` are zero.

    The block statistic is :math:`T_n=c_n^T C_n^{-1}c_n`.  When the residual
    variance was estimated from the same least-squares fit, ``T_n / k`` is
    compared with an F distribution with ``(k, residual_dof)`` degrees of
    freedom.  Without ``residual_degrees_of_freedom``, the asymptotic
    chi-squared distribution with ``k`` degrees of freedom is used.
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
    rank = int(np.linalg.matrix_rank(selected_covariance))
    statistic = float(
        selected_coefficients
        @ np.linalg.pinv(selected_covariance)
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
