"""Automated truncation-convergence tests for the standard mode analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import design_matrix
from .indices import rectangular_indices
from .standard_analyzer import (
    StandardAnalysisResult,
    _resolve_analysis_arguments,
    standard_analyzer,
)


@dataclass(frozen=True)
class ModeConvergenceResult:
    """Truncation sweep and final converged standard-analysis result."""

    n_values: NDArray[np.int64]
    m_values: NDArray[np.int64]
    mse: NDArray[np.float64]
    prediction_change: NDArray[np.float64]
    coefficient_change: NDArray[np.float64]
    relative_mse_excess: NDArray[np.float64]
    stable_mask: NDArray[np.bool_]
    selected_n_max: int
    selected_m_max: int
    stabilized_within_limits: bool
    reference_n_max: int
    reference_m_max: int
    standard_analysis: StandardAnalysisResult


def mode_convergence_standard_analyzer(
    q_z: ArrayLike,
    q_y: ArrayLike,
    intensity: ArrayLike,
    *,
    q_range: tuple[float, float] | list[float] | None = None,
    mode_limits: tuple[int, int] | list[int] | None = None,
    q_min: float | None = None,
    q_max: float | None = None,
    n_max_limit: int | None = None,
    m_max_limit: int | None = None,
    uncertainty: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    absolute_uncertainty: bool = True,
    n_min: int = 0,
    m_min: int = 0,
    prediction_tolerance: float = 0.01,
    coefficient_tolerance: float = 0.05,
    relative_mse_tolerance: float = 0.001,
    output_path: str | Path = "converged_standard_analysis.png",
) -> ModeConvergenceResult:
    """Find the smallest stable rectangular angular/radial truncation.

    ``q_range`` and ``mode_limits`` use the same preferred pair syntax as
    :func:`standard_analyzer`; here ``mode_limits`` contains the largest
    tested ``(n_max, m_max)`` pair. Every integer pair in ``[n_min, n_max_limit]`` and
    ``[m_min, m_max_limit]`` is fitted.  The largest pair is the reference.
    A candidate is stable when all of the following are below tolerance:

    - relative L2 change of its fitted detector signal;
    - relative L2 change of its coefficient vector after zero-padding it into
      the reference basis;
    - its MSE excess over the reference, normalized by the data mean square.

    Among stable candidates, the routine selects the one with the fewest
    product modes.  The final selected pair is then passed once to
    :func:`standard_analyzer`, which creates the requested standard output and
    is returned as ``standard_analysis``.

    ``stabilized_within_limits`` is false when either selected order equals
    its tested upper limit, because stability beyond that boundary has not
    been demonstrated.
    """
    q_min, q_max, n_max_limit, m_max_limit = _resolve_analysis_arguments(
        q_range=q_range,
        mode_limits=mode_limits,
        q_min=q_min,
        q_max=q_max,
        n_max=n_max_limit,
        m_max=m_max_limit,
    )
    lower_n, upper_n = _integer_interval(n_min, n_max_limit, "n")
    lower_m, upper_m = _integer_interval(m_min, m_max_limit, "m")
    prediction_limit = _positive_tolerance(
        prediction_tolerance, "prediction_tolerance"
    )
    coefficient_limit = _positive_tolerance(
        coefficient_tolerance, "coefficient_tolerance"
    )
    mse_limit = _positive_tolerance(
        relative_mse_tolerance, "relative_mse_tolerance"
    )

    q_z_array, q_y_array, intensity_array = np.broadcast_arrays(
        np.asarray(q_z, dtype=float),
        np.asarray(q_y, dtype=float),
        np.asarray(intensity, dtype=float),
    )
    uncertainty_array = None
    if uncertainty is not None:
        uncertainty_array = np.broadcast_to(
            np.asarray(uncertainty, dtype=float), intensity_array.shape
        )
    weight_array = None
    if sample_weight is not None:
        weight_array = np.broadcast_to(
            np.asarray(sample_weight, dtype=float), intensity_array.shape
        )
    if intensity_array.ndim not in (1, 2):
        raise ValueError(
            "mode_convergence_standard_analyzer expects a detector point list "
            "or 2D arrays"
        )

    q = np.hypot(q_z_array, q_y_array)
    theta = np.arctan2(q_y_array, q_z_array)
    selected = (
        np.isfinite(q_z_array)
        & np.isfinite(q_y_array)
        & np.isfinite(intensity_array)
        & (q >= q_min)
        & (q <= q_max)
    )
    if uncertainty_array is not None:
        selected &= np.isfinite(uncertainty_array) & (uncertainty_array > 0.0)
    if weight_array is not None:
        selected &= np.isfinite(weight_array) & (weight_array > 0.0)
    q_selected = q[selected]
    theta_selected = theta[selected]
    values = intensity_array[selected]
    if values.size == 0:
        raise ValueError("no finite detector values lie inside the requested annulus")

    reference_indices = rectangular_indices(upper_n, upper_m)
    reference_matrix = design_matrix(
        q_selected,
        theta_selected,
        reference_indices,
        q_min=q_min,
        q_max=q_max,
    ).reshape(values.size, len(reference_indices))
    if values.size <= len(reference_indices):
        raise ValueError(
            "the reference model must contain fewer modes than detector values"
        )

    whitening = np.ones(values.shape)
    if uncertainty_array is not None:
        whitening /= uncertainty_array[selected]
    if weight_array is not None:
        whitening *= np.sqrt(weight_array[selected])
    weighted_reference_matrix = whitening[:, None] * reference_matrix
    weighted_values = whitening * values

    reference_coefficients, _, reference_rank, _ = np.linalg.lstsq(
        weighted_reference_matrix, weighted_values, rcond=None
    )
    if reference_rank != len(reference_indices):
        raise ValueError("the maximum tested design matrix is rank deficient")
    reference_prediction = reference_matrix @ reference_coefficients
    reference_residual = values - reference_prediction
    reference_mse = float(np.mean(reference_residual**2))

    prediction_scale = max(
        float(np.linalg.norm(reference_prediction)), np.finfo(float).eps
    )
    coefficient_scale = max(
        float(np.linalg.norm(reference_coefficients)), np.finfo(float).eps
    )
    data_mean_square = max(float(np.mean(values**2)), np.finfo(float).eps)
    reference_position = {
        mode: position for position, mode in enumerate(reference_indices)
    }

    n_values = np.arange(lower_n, upper_n + 1, dtype=np.int64)
    m_values = np.arange(lower_m, upper_m + 1, dtype=np.int64)
    shape = (n_values.size, m_values.size)
    mse = np.empty(shape)
    prediction_change = np.empty(shape)
    coefficient_change = np.empty(shape)
    relative_mse_excess = np.empty(shape)

    for n_position, n_max in enumerate(n_values):
        for m_position, m_max in enumerate(m_values):
            candidate_indices = rectangular_indices(int(n_max), int(m_max))
            columns = [reference_position[mode] for mode in candidate_indices]
            candidate_matrix = reference_matrix[:, columns]
            weighted_candidate_matrix = weighted_reference_matrix[:, columns]
            coefficients, _, rank, _ = np.linalg.lstsq(
                weighted_candidate_matrix, weighted_values, rcond=None
            )
            if rank != len(candidate_indices):
                raise ValueError(
                    f"design matrix is rank deficient at n_max={n_max}, "
                    f"m_max={m_max}"
                )

            prediction = candidate_matrix @ coefficients
            residual = values - prediction
            candidate_mse = float(np.mean(residual**2))
            padded_coefficients = np.zeros_like(reference_coefficients)
            padded_coefficients[columns] = coefficients

            mse[n_position, m_position] = candidate_mse
            prediction_change[n_position, m_position] = (
                np.linalg.norm(prediction - reference_prediction) / prediction_scale
            )
            coefficient_change[n_position, m_position] = (
                np.linalg.norm(padded_coefficients - reference_coefficients)
                / coefficient_scale
            )
            relative_mse_excess[n_position, m_position] = max(
                candidate_mse - reference_mse, 0.0
            ) / data_mean_square

    stable_mask = (
        (prediction_change <= prediction_limit)
        & (coefficient_change <= coefficient_limit)
        & (relative_mse_excess <= mse_limit)
    )
    stable_positions = np.argwhere(stable_mask)
    if stable_positions.size == 0:
        raise RuntimeError("no stable truncation found, including the reference model")

    def complexity_key(position: NDArray[np.int64]) -> tuple[int, int, int, int]:
        n_value = int(n_values[position[0]])
        m_value = int(m_values[position[1]])
        number_of_modes = (2 * n_value + 1) * (m_value + 1)
        return number_of_modes, n_value + m_value, n_value, m_value

    chosen_position = min(stable_positions, key=complexity_key)
    chosen_n = int(n_values[chosen_position[0]])
    chosen_m = int(m_values[chosen_position[1]])
    stabilized_within_limits = chosen_n < upper_n and chosen_m < upper_m

    final_analysis = standard_analyzer(
        q_z_array,
        q_y_array,
        intensity_array,
        uncertainty=uncertainty_array,
        sample_weight=weight_array,
        absolute_uncertainty=absolute_uncertainty,
        q_range=(q_min, q_max),
        mode_limits=(chosen_n, chosen_m),
        output_path=output_path,
    )

    return ModeConvergenceResult(
        n_values=n_values,
        m_values=m_values,
        mse=mse,
        prediction_change=prediction_change,
        coefficient_change=coefficient_change,
        relative_mse_excess=relative_mse_excess,
        stable_mask=stable_mask,
        selected_n_max=chosen_n,
        selected_m_max=chosen_m,
        stabilized_within_limits=stabilized_within_limits,
        reference_n_max=upper_n,
        reference_m_max=upper_m,
        standard_analysis=final_analysis,
    )


def _integer_interval(lower, upper, name):
    if isinstance(lower, bool) or not isinstance(lower, (int, np.integer)):
        raise TypeError(f"{name}_min must be an integer")
    if isinstance(upper, bool) or not isinstance(upper, (int, np.integer)):
        raise TypeError(f"{name}_max_limit must be an integer")
    lower_value = int(lower)
    upper_value = int(upper)
    if lower_value < 0 or upper_value < lower_value:
        raise ValueError(f"require 0 <= {name}_min <= {name}_max_limit")
    return lower_value, upper_value


def _positive_tolerance(value, name):
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return tolerance
