"""Explicit, auditable detector-data preconditioners and diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import minimize

from .basis import design_matrix, radial_measure_weight
from .indices import ModeIndex, validate_indices
from .io import DetectorData


@dataclass(frozen=True)
class AffineDetectorGeometry:
    """Least-squares affine map from detector indices to reciprocal space."""

    matrix: NDArray[np.float64]
    offset: NDArray[np.float64]
    rms_residual: float


@dataclass(frozen=True)
class SymmetryDiagnostic:
    """Pixelwise residual and conditional statistic for one symmetry."""

    symmetry: str
    residual: NDArray[np.float64]
    standardized_residual: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    chi_square: float
    degrees_of_freedom: int


@dataclass(frozen=True)
class CenterRefinementResult:
    """Common detector-index shift obtained from antipodal parity."""

    delta_row: float
    delta_column: float
    objective_before: float
    objective_after: float
    pairs_before: int
    pairs_after: int
    success: bool
    message: str


@dataclass(frozen=True)
class OddModeCenterRefinementResult:
    """Reciprocal-space centre shift minimizing fitted odd-mode power.

    The reported shift is relative to the coordinates supplied to
    :func:`refine_center_by_odd_mode_power`. ``objective_before`` and
    ``objective_after`` are fractions of fitted coefficient power in odd
    angular orders.
    """

    delta_q_z: float
    delta_q_y: float
    indices: tuple[ModeIndex, ...]
    max_shift_q_z: float
    max_shift_q_y: float
    objective_before: float
    objective_after: float
    odd_power_before: float
    odd_power_after: float
    total_power_before: float
    total_power_after: float
    selected_before: int
    selected_after: int
    evaluations: int
    success: bool
    message: str


@dataclass(frozen=True)
class MaskSymmetryDiagnostic:
    """First-moment and point-reflection diagnostics of a fractional mask."""

    normalized_centroid: float
    mean_reflection_residual: float
    mean_absolute_reflection_residual: float
    partial_pixel_count: int
    reflection_residual: NDArray[np.float64]


@dataclass(frozen=True)
class PixelIntegratedAnnulus:
    """Fractional annulus mask and pixel-averaged basis design."""

    selected_mask: NDArray[np.bool_]
    coverage: NDArray[np.float64]
    quadrature_weight: NDArray[np.float64]
    indices: tuple[ModeIndex, ...]
    design_matrix: NDArray[np.float64]


@dataclass(frozen=True)
class PointCloudCenteringResult:
    """Parity-centred point list and the auxiliary raster diagnostics."""

    centered_data: DetectorData
    centering_grid: DetectorData
    refinement: CenterRefinementResult
    delta_q_z: float
    delta_q_y: float


@dataclass(frozen=True)
class CartesianBinnedGrid:
    """Area-binned values on an equally spaced Cartesian reciprocal-space grid."""

    q_z: NDArray[np.float64]
    q_y: NDArray[np.float64]
    intensity: NDArray[np.float64]
    source_count: NDArray[np.int64]
    source_weight: NDArray[np.float64]


def cartesian_bin_average(
    q_z: ArrayLike,
    q_y: ArrayLike,
    intensity: ArrayLike,
    *,
    q_extent: float,
    grid_size: int = 161,
    source_weight: ArrayLike | None = None,
) -> CartesianBinnedGrid:
    """Average point samples into equal-area Cartesian square cells.

    ``source_weight`` supplies the quadrature weight of each source sample.
    For samples that are uniform in polar ``(q, theta)`` coordinates, use
    ``source_weight=q`` to account for the area element ``q dq dtheta``.
    """
    q_z_array, q_y_array, intensity_array = np.broadcast_arrays(
        np.asarray(q_z, dtype=float),
        np.asarray(q_y, dtype=float),
        np.asarray(intensity, dtype=float),
    )
    extent = float(q_extent)
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("q_extent must be finite and positive")
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)):
        raise TypeError("grid_size must be an integer")
    size = int(grid_size)
    if size < 2:
        raise ValueError("grid_size must be at least two")

    if source_weight is None:
        weight_array = np.ones(q_z_array.shape, dtype=float)
    else:
        weight_array = np.broadcast_to(
            np.asarray(source_weight, dtype=float), q_z_array.shape
        )
        finite_weight = weight_array[np.isfinite(weight_array)]
        if np.any(finite_weight < 0.0):
            raise ValueError("source_weight must be non-negative")

    axis = np.linspace(-extent, extent, size)
    spacing = float(axis[1] - axis[0])
    lower_edge = float(axis[0] - 0.5 * spacing)
    upper_edge = float(axis[-1] + 0.5 * spacing)

    valid = (
        np.isfinite(q_z_array)
        & np.isfinite(q_y_array)
        & np.isfinite(intensity_array)
        & np.isfinite(weight_array)
        & (weight_array > 0.0)
        & (q_z_array >= lower_edge)
        & (q_z_array < upper_edge)
        & (q_y_array >= lower_edge)
        & (q_y_array < upper_edge)
    )
    if not np.any(valid):
        raise ValueError("no positively weighted source samples lie on the grid")

    column = np.floor((q_z_array[valid] - lower_edge) / spacing).astype(int)
    row = np.floor((q_y_array[valid] - lower_edge) / spacing).astype(int)
    flat_index = row * size + column
    flat_weight = weight_array[valid]
    number_of_cells = size * size
    weight_sum = np.bincount(
        flat_index, weights=flat_weight, minlength=number_of_cells
    )
    weighted_intensity = np.bincount(
        flat_index,
        weights=flat_weight * intensity_array[valid],
        minlength=number_of_cells,
    )
    source_count = np.bincount(flat_index, minlength=number_of_cells)

    binned_intensity = np.full(number_of_cells, np.nan)
    occupied = weight_sum > 0.0
    binned_intensity[occupied] = weighted_intensity[occupied] / weight_sum[occupied]
    grid_q_z, grid_q_y = np.meshgrid(axis, axis)
    return CartesianBinnedGrid(
        q_z=grid_q_z,
        q_y=grid_q_y,
        intensity=binned_intensity.reshape(size, size),
        source_count=source_count.reshape(size, size),
        source_weight=weight_sum.reshape(size, size),
    )


def valid_pixel_mask(data: DetectorData) -> NDArray[np.bool_]:
    """Select finite pixels with positive uncertainty.

    Finite zero and negative intensities are deliberately retained because
    background subtraction can produce them legitimately.
    """
    q_z, q_y, intensity, uncertainty = np.broadcast_arrays(
        data.q_z, data.q_y, data.intensity, data.uncertainty
    )
    return (
        np.isfinite(q_z)
        & np.isfinite(q_y)
        & np.isfinite(intensity)
        & np.isfinite(uncertainty)
        & (uncertainty > 0.0)
    )


def affine_detector_geometry(q_z: ArrayLike, q_y: ArrayLike) -> AffineDetectorGeometry:
    """Fit ``(q_z, q_y) = matrix @ (row, column) + offset``."""
    q_z_array, q_y_array = np.broadcast_arrays(
        np.asarray(q_z, dtype=float), np.asarray(q_y, dtype=float)
    )
    if q_z_array.ndim != 2:
        raise ValueError("affine detector geometry requires two-dimensional arrays")
    rows, columns = np.indices(q_z_array.shape, dtype=float)
    valid = np.isfinite(q_z_array) & np.isfinite(q_y_array)
    if np.count_nonzero(valid) < 3:
        raise ValueError("at least three finite detector coordinates are required")
    coordinates = np.column_stack(
        [rows[valid], columns[valid], np.ones(np.count_nonzero(valid))]
    )
    targets = np.column_stack([q_z_array[valid], q_y_array[valid]])
    coefficients, _, rank, _ = np.linalg.lstsq(coordinates, targets, rcond=None)
    if rank != 3:
        raise ValueError("detector-coordinate affine fit is rank deficient")
    prediction = coordinates @ coefficients
    residual = prediction - targets
    matrix = np.array(
        [
            [coefficients[0, 0], coefficients[1, 0]],
            [coefficients[0, 1], coefficients[1, 1]],
        ]
    )
    if abs(np.linalg.det(matrix)) <= np.finfo(float).eps:
        raise ValueError("detector-coordinate affine map is singular")
    return AffineDetectorGeometry(
        matrix=matrix,
        offset=np.asarray(coefficients[2], dtype=float),
        rms_residual=float(np.sqrt(np.mean(residual**2))),
    )


def apply_center_correction(
    data: DetectorData,
    delta_row: float,
    delta_column: float,
) -> DetectorData:
    """Shift the reciprocal-space origin by a detector-index displacement."""
    geometry = affine_detector_geometry(data.q_z, data.q_y)
    delta_q = geometry.matrix @ np.array([delta_row, delta_column], dtype=float)
    return DetectorData(
        q_z=np.asarray(data.q_z, dtype=float) - delta_q[0],
        q_y=np.asarray(data.q_y, dtype=float) - delta_q[1],
        intensity=np.asarray(data.intensity, dtype=float).copy(),
        uncertainty=np.asarray(data.uncertainty, dtype=float).copy(),
    )


def apply_q_center_correction(
    data: DetectorData,
    delta_q_z: float,
    delta_q_y: float,
) -> DetectorData:
    """Shift the reciprocal-space origin directly, including point lists."""
    delta_z = float(delta_q_z)
    delta_y = float(delta_q_y)
    if not np.isfinite(delta_z) or not np.isfinite(delta_y):
        raise ValueError("reciprocal-space centre correction must be finite")
    return DetectorData(
        q_z=np.asarray(data.q_z, dtype=float) - delta_z,
        q_y=np.asarray(data.q_y, dtype=float) - delta_y,
        intensity=np.asarray(data.intensity, dtype=float).copy(),
        uncertainty=np.asarray(data.uncertainty, dtype=float).copy(),
    )


def rotate_coordinates(data: DetectorData, angle: float) -> DetectorData:
    r"""Rotate the angular origin so that ``theta_new = theta - angle``."""
    cosine = np.cos(float(angle))
    sine = np.sin(float(angle))
    q_z = cosine * data.q_z + sine * data.q_y
    q_y = -sine * data.q_z + cosine * data.q_y
    return DetectorData(
        q_z=np.asarray(q_z, dtype=float),
        q_y=np.asarray(q_y, dtype=float),
        intensity=np.asarray(data.intensity, dtype=float).copy(),
        uncertainty=np.asarray(data.uncertainty, dtype=float).copy(),
    )


def rescale_intensity(data: DetectorData, scale: float) -> DetectorData:
    """Apply a purely numerical common scale to intensity and uncertainty."""
    value = float(scale)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("scale must be finite and positive")
    return DetectorData(
        q_z=np.asarray(data.q_z, dtype=float).copy(),
        q_y=np.asarray(data.q_y, dtype=float).copy(),
        intensity=np.asarray(data.intensity, dtype=float) / value,
        uncertainty=np.asarray(data.uncertainty, dtype=float) / value,
    )


def fractional_annulus_weights(
    q_z: ArrayLike,
    q_y: ArrayLike,
    *,
    q_min: float,
    q_max: float,
    subpixels: int = 7,
) -> NDArray[np.float64]:
    """Return the fractional area of each detector pixel inside an annulus.

    A one-dimensional point list has no pixel geometry and therefore receives
    binary centre-point weights.  Two-dimensional affine detector grids are
    integrated by regular subpixel quadrature.
    """
    lower, upper = _annulus_limits(q_min, q_max)
    q_z_array, q_y_array = np.broadcast_arrays(
        np.asarray(q_z, dtype=float), np.asarray(q_y, dtype=float)
    )
    if q_z_array.ndim == 1:
        q = np.hypot(q_z_array, q_y_array)
        return np.asarray((q >= lower) & (q <= upper), dtype=float)
    if q_z_array.ndim != 2:
        raise ValueError("detector coordinates must be one- or two-dimensional")
    if isinstance(subpixels, bool) or not isinstance(subpixels, (int, np.integer)):
        raise TypeError("subpixels must be an integer")
    number = int(subpixels)
    if number < 1:
        raise ValueError("subpixels must be positive")
    geometry = affine_detector_geometry(q_z_array, q_y_array)
    offsets = (np.arange(number, dtype=float) + 0.5) / number - 0.5
    coverage = np.zeros(q_z_array.shape, dtype=float)
    for row_offset in offsets:
        for column_offset in offsets:
            delta_q = geometry.matrix @ np.array([row_offset, column_offset])
            radius = np.hypot(q_z_array + delta_q[0], q_y_array + delta_q[1])
            coverage += (radius >= lower) & (radius <= upper)
    coverage /= number**2
    coverage[~(np.isfinite(q_z_array) & np.isfinite(q_y_array))] = 0.0
    return coverage


def pixel_integrated_annulus_design(
    q_z: ArrayLike,
    q_y: ArrayLike,
    indices: Sequence[Sequence[int]],
    *,
    q_min: float,
    q_max: float,
    radial_measure_exponent: float = 1.0,
    subpixels: int = 7,
    chunk_size: int = 512,
) -> PixelIntegratedAnnulus:
    """Average annulus basis functions over each admitted detector pixel.

    Only selected pixels are stored in the returned design matrix.  Its row
    order is ``array[selected_mask]`` in NumPy's usual C order.  Chunking keeps
    the temporary matrix bounded even for a large mode set. ``coverage`` is
    the geometric annulus fraction, whereas ``quadrature_weight`` additionally
    applies the selected radial measure relative to equal-area detector pixels.
    """
    lower, upper = _annulus_limits(q_min, q_max)
    modes = validate_indices(indices)
    q_z_array, q_y_array = np.broadcast_arrays(
        np.asarray(q_z, dtype=float), np.asarray(q_y, dtype=float)
    )
    if isinstance(chunk_size, bool) or not isinstance(
        chunk_size, (int, np.integer)
    ):
        raise TypeError("chunk_size must be an integer")
    chunk = int(chunk_size)
    if chunk < 1:
        raise ValueError("chunk_size must be positive")
    if q_z_array.ndim == 1:
        radius = np.hypot(q_z_array, q_y_array)
        selected = (
            np.isfinite(q_z_array)
            & np.isfinite(q_y_array)
            & (radius >= lower)
            & (radius <= upper)
        )
        matrix = design_matrix(
            radius[selected],
            np.arctan2(q_y_array[selected], q_z_array[selected]),
            modes,
            q_min=lower,
            q_max=upper,
            radial_measure_exponent=radial_measure_exponent,
        )
        quadrature_weight = np.zeros_like(radius)
        quadrature_weight[selected] = radial_measure_weight(
            radius[selected],
            q_min=lower,
            q_max=upper,
            radial_measure_exponent=radial_measure_exponent,
        )
        return PixelIntegratedAnnulus(
            selected_mask=selected,
            coverage=selected.astype(float),
            quadrature_weight=quadrature_weight,
            indices=modes,
            design_matrix=matrix,
        )
    if q_z_array.ndim != 2:
        raise ValueError("detector coordinates must be one- or two-dimensional")
    if isinstance(subpixels, bool) or not isinstance(subpixels, (int, np.integer)):
        raise TypeError("subpixels must be an integer")
    number = int(subpixels)
    if number < 1:
        raise ValueError("subpixels must be positive")

    geometry = affine_detector_geometry(q_z_array, q_y_array)
    coverage = fractional_annulus_weights(
        q_z_array,
        q_y_array,
        q_min=lower,
        q_max=upper,
        subpixels=number,
    )
    selected = coverage > 0.0
    flat_positions = np.flatnonzero(selected)
    integrated = np.zeros((flat_positions.size, len(modes)), dtype=float)
    quadrature_weight = np.zeros(q_z_array.shape, dtype=float)
    offsets = (np.arange(number, dtype=float) + 0.5) / number - 0.5
    flat_q_z = q_z_array.ravel()
    flat_q_y = q_y_array.ravel()

    for start in range(0, flat_positions.size, chunk):
        stop = min(start + chunk, flat_positions.size)
        positions = flat_positions[start:stop]
        accumulated = np.zeros((positions.size, len(modes)), dtype=float)
        accumulated_weight = np.zeros(positions.size, dtype=float)
        counts = np.zeros(positions.size, dtype=int)
        for row_offset in offsets:
            for column_offset in offsets:
                delta_q = geometry.matrix @ np.array(
                    [row_offset, column_offset]
                )
                sample_q_z = flat_q_z[positions] + delta_q[0]
                sample_q_y = flat_q_y[positions] + delta_q[1]
                sample_q = np.hypot(sample_q_z, sample_q_y)
                inside = (sample_q >= lower) & (sample_q <= upper)
                if np.any(inside):
                    subpixel_weight = radial_measure_weight(
                        sample_q[inside],
                        q_min=lower,
                        q_max=upper,
                        radial_measure_exponent=radial_measure_exponent,
                    )
                    accumulated[inside] += design_matrix(
                        sample_q[inside],
                        np.arctan2(sample_q_y[inside], sample_q_z[inside]),
                        modes,
                        q_min=lower,
                        q_max=upper,
                        radial_measure_exponent=radial_measure_exponent,
                    )
                    accumulated_weight[inside] += subpixel_weight
                    counts[inside] += 1
        if np.any(counts == 0):
            raise RuntimeError("selected annulus pixel received no subpixel samples")
        integrated[start:stop] = accumulated / counts[:, None]
        quadrature_weight.ravel()[positions] = accumulated_weight / number**2

    return PixelIntegratedAnnulus(
        selected_mask=selected,
        coverage=coverage,
        quadrature_weight=quadrature_weight,
        indices=modes,
        design_matrix=integrated,
    )


def symmetry_diagnostic(
    data: DetectorData,
    *,
    symmetry: str = "parity",
    q_min: float | None = None,
    q_max: float | None = None,
) -> SymmetryDiagnostic:
    """Compare detector pixels with an interpolated symmetry partner."""
    _require_detector_image(data)
    target_q_z, target_q_y = _symmetry_target(data.q_z, data.q_y, symmetry)
    sampled, sampled_variance, sampled_valid = _sample_detector(
        data, target_q_z, target_q_y
    )
    base_valid = valid_pixel_mask(data)
    valid = base_valid & sampled_valid
    radius = np.hypot(data.q_z, data.q_y)
    if q_min is not None:
        valid &= radius >= float(q_min)
    if q_max is not None:
        valid &= radius <= float(q_max)
    residual = np.full(data.intensity.shape, np.nan)
    standardized = np.full(data.intensity.shape, np.nan)
    residual[valid] = data.intensity[valid] - sampled[valid]
    variance = data.uncertainty**2 + sampled_variance
    standardized[valid] = residual[valid] / np.sqrt(variance[valid])
    representative = valid & _representative_half(data.q_z, data.q_y, symmetry)
    chi_square = float(np.sum(standardized[representative] ** 2))
    degrees = int(np.count_nonzero(representative))
    return SymmetryDiagnostic(
        symmetry=symmetry,
        residual=residual,
        standardized_residual=standardized,
        valid_mask=valid,
        chi_square=chi_square,
        degrees_of_freedom=degrees,
    )


def refine_center_by_parity(
    data: DetectorData | Sequence[DetectorData],
    *,
    q_min: float,
    q_max: float,
    initial_shift: tuple[float, float] = (0.0, 0.0),
    max_shift_pixels: float = 1.0,
) -> CenterRefinementResult:
    """Refine one common detector-index centre using antipodal residuals.

    Multiple images can be supplied to avoid allowing one field condition to
    absorb genuine field-dependent structure into its own fitted centre.
    """
    datasets = (data,) if isinstance(data, DetectorData) else tuple(data)
    if not datasets:
        raise ValueError("at least one detector image is required")
    for detector in datasets:
        _require_detector_image(detector)
    lower, upper = _annulus_limits(q_min, q_max)
    maximum = float(max_shift_pixels)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_shift_pixels must be finite and positive")
    initial = np.asarray(initial_shift, dtype=float)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_shift must contain two finite values")

    def objective(shift: NDArray[np.float64]) -> tuple[float, int]:
        total = 0.0
        pairs = 0
        for detector in datasets:
            geometry = affine_detector_geometry(detector.q_z, detector.q_y)
            delta_q = geometry.matrix @ shift
            corrected_q_z = detector.q_z - delta_q[0]
            corrected_q_y = detector.q_y - delta_q[1]
            target_q_z = 2.0 * delta_q[0] - detector.q_z
            target_q_y = 2.0 * delta_q[1] - detector.q_y
            sampled, sampled_variance, sampled_valid = _sample_detector(
                detector, target_q_z, target_q_y
            )
            radius = np.hypot(corrected_q_z, corrected_q_y)
            valid = (
                valid_pixel_mask(detector)
                & sampled_valid
                & (radius >= lower)
                & (radius <= upper)
                & _representative_half(corrected_q_z, corrected_q_y, "parity")
            )
            variance = detector.uncertainty**2 + sampled_variance
            difference = detector.intensity - sampled
            total += float(np.sum(difference[valid] ** 2 / variance[valid]))
            pairs += int(np.count_nonzero(valid))
        if pairs == 0:
            return float("inf"), 0
        return total, pairs

    before, pairs_before = objective(initial)
    bounds = [(-maximum, maximum), (-maximum, maximum)]
    result = minimize(
        lambda shift: objective(np.asarray(shift, dtype=float))[0],
        initial,
        method="Powell",
        bounds=bounds,
        options={"xtol": 1.0e-4, "ftol": 1.0e-6, "maxiter": 120},
    )
    after, pairs_after = objective(np.asarray(result.x, dtype=float))
    return CenterRefinementResult(
        delta_row=float(result.x[0]),
        delta_column=float(result.x[1]),
        objective_before=before,
        objective_after=after,
        pairs_before=pairs_before,
        pairs_after=pairs_after,
        success=bool(result.success),
        message=str(result.message),
    )


def refine_center_by_odd_mode_power(
    data: DetectorData,
    indices: Iterable[Sequence[int]],
    *,
    q_min: float,
    q_max: float,
    max_shift_q: tuple[float, float],
    initial_shift: tuple[float, float] = (0.0, 0.0),
    sample_weight: ArrayLike | None = None,
    radial_measure_exponent: float = 1.0,
) -> OddModeCenterRefinementResult:
    """Refine the centre by minimizing the fitted odd-angular power fraction.

    This is a modal alternative or local supplement to pixelwise parity
    centring.  For every trial shift the intensity is refitted on the fixed
    ``indices`` basis over the requested annulus, and the objective

    ``sum(c_nm**2 for odd abs(n)) / sum(c_nm**2)``

    is minimized.  The search is bounded in reciprocal-space coordinates by
    ``max_shift_q=(max_delta_q_z, max_delta_q_y)``.  A reduced, explicitly
    chosen basis is recommended because the result is conditional on the
    truncation and because every objective evaluation solves a least-squares
    problem.

    The method assumes that odd angular modes are artefacts of miscentring.
    It must therefore not be used to remove odd modes that may be physical.
    """
    modes = validate_indices(indices)
    has_even = any(abs(n) % 2 == 0 for n, _ in modes)
    has_odd = any(abs(n) % 2 == 1 for n, _ in modes)
    if not (has_even and has_odd):
        raise ValueError(
            "odd-mode centering requires a basis containing both even and odd "
            "angular modes"
        )
    lower, upper = _annulus_limits(q_min, q_max)
    exponent = float(radial_measure_exponent)
    if not np.isfinite(exponent) or exponent <= -1.0:
        raise ValueError(
            "radial_measure_exponent must be finite and greater than -1"
        )

    q_z, q_y, intensity, uncertainty = np.broadcast_arrays(
        np.asarray(data.q_z, dtype=float),
        np.asarray(data.q_y, dtype=float),
        np.asarray(data.intensity, dtype=float),
        np.asarray(data.uncertainty, dtype=float),
    )
    if q_z.ndim not in (1, 2):
        raise ValueError(
            "odd-mode centering requires a one-dimensional point list or "
            "two-dimensional detector arrays"
        )
    if sample_weight is None:
        external_weight = np.ones(q_z.shape, dtype=float)
    else:
        external_weight = np.broadcast_to(
            np.asarray(sample_weight, dtype=float), q_z.shape
        )

    maximum = np.asarray(max_shift_q, dtype=float)
    if maximum.shape != (2,) or not np.all(np.isfinite(maximum)):
        raise ValueError("max_shift_q must contain two finite values")
    if np.any(maximum <= 0.0):
        raise ValueError("max_shift_q values must be positive")
    initial = np.asarray(initial_shift, dtype=float)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_shift must contain two finite values")
    if np.any(np.abs(initial) > maximum):
        raise ValueError("initial_shift must lie inside the max_shift_q bounds")

    odd_positions = np.array(
        [abs(n) % 2 == 1 for n, _ in modes], dtype=bool
    )

    def evaluate(shift: NDArray[np.float64]):
        corrected_q_z = q_z - shift[0]
        corrected_q_y = q_y - shift[1]
        radius = np.hypot(corrected_q_z, corrected_q_y)
        angle = np.arctan2(corrected_q_y, corrected_q_z)
        selected = (
            np.isfinite(corrected_q_z)
            & np.isfinite(corrected_q_y)
            & np.isfinite(intensity)
            & np.isfinite(uncertainty)
            & (uncertainty > 0.0)
            & np.isfinite(external_weight)
            & (external_weight > 0.0)
            & (radius >= lower)
            & (radius <= upper)
        )
        count = int(np.count_nonzero(selected))
        if count <= len(modes):
            return float("inf"), float("nan"), float("nan"), count

        radial_weight = radial_measure_weight(
            radius[selected],
            q_min=lower,
            q_max=upper,
            radial_measure_exponent=exponent,
        )
        precision = (
            external_weight[selected]
            * radial_weight
            / uncertainty[selected] ** 2
        )
        positive = np.isfinite(precision) & (precision > 0.0)
        if np.count_nonzero(positive) <= len(modes):
            return float("inf"), float("nan"), float("nan"), count
        matrix = design_matrix(
            radius[selected][positive],
            angle[selected][positive],
            modes,
            q_min=lower,
            q_max=upper,
            radial_measure_exponent=exponent,
        )
        square_root_precision = np.sqrt(precision[positive])
        weighted_matrix = matrix * square_root_precision[:, None]
        weighted_values = intensity[selected][positive] * square_root_precision
        coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_matrix, weighted_values, rcond=None
        )
        if rank != len(modes):
            return float("inf"), float("nan"), float("nan"), count
        coefficient_power = coefficients**2
        total_power = float(np.sum(coefficient_power))
        if not np.isfinite(total_power) or total_power <= 0.0:
            return float("inf"), float("nan"), total_power, count
        odd_power = float(np.sum(coefficient_power[odd_positions]))
        return odd_power / total_power, odd_power, total_power, count

    before = evaluate(initial)
    if not np.isfinite(before[0]):
        raise ValueError(
            "odd-mode centering objective is undefined at the initial shift"
        )

    def objective(normalized_shift: NDArray[np.float64]) -> float:
        return evaluate(np.asarray(normalized_shift, dtype=float) * maximum)[0]

    result = minimize(
        objective,
        initial / maximum,
        method="Powell",
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        options={"xtol": 1.0e-7, "ftol": 1.0e-12, "maxiter": 80},
    )
    shift = np.asarray(result.x, dtype=float) * maximum
    after = evaluate(shift)
    accepted = (
        bool(result.success)
        and np.isfinite(after[0])
        and after[0] <= before[0] * (1.0 + 1.0e-12)
    )
    message = str(result.message)
    if not np.isfinite(after[0]):
        message += "; final odd-power objective is not finite"
    elif after[0] > before[0] * (1.0 + 1.0e-12):
        message += "; final odd-power objective exceeds its initial value"
    return OddModeCenterRefinementResult(
        delta_q_z=float(shift[0]),
        delta_q_y=float(shift[1]),
        indices=modes,
        max_shift_q_z=float(maximum[0]),
        max_shift_q_y=float(maximum[1]),
        objective_before=float(before[0]),
        objective_after=float(after[0]),
        odd_power_before=float(before[1]),
        odd_power_after=float(after[1]),
        total_power_before=float(before[2]),
        total_power_after=float(after[2]),
        selected_before=int(before[3]),
        selected_after=int(after[3]),
        evaluations=int(result.nfev),
        success=accepted,
        message=message,
    )


def center_point_cloud_by_parity(
    data: DetectorData,
    *,
    q_min: float,
    q_max: float,
    grid_size: int = 129,
    source_radius_factor: float = 1.05,
    max_shift_pixels: float = 2.0,
) -> PointCloudCenteringResult:
    """Estimate and apply a parity centre to a detector point list.

    A regular auxiliary grid is used only to estimate the centre. Intensities
    are linearly interpolated and supplied uncertainties are assigned from the
    nearest measured point. The returned ``centered_data`` retains the
    original intensity, uncertainty, and point-list sampling; only its
    reciprocal-space origin is shifted.

    The interpolated uncertainty is a centring weight, not a reconstructed
    covariance model. Consequently, the parity objective is a diagnostic for
    choosing the centre and should not be quoted as a formal goodness-of-fit
    statistic for the original point list.
    """
    if np.asarray(data.intensity).ndim != 1:
        raise ValueError("point-cloud centering requires one-dimensional point lists")
    lower, upper = _annulus_limits(q_min, q_max)
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)):
        raise TypeError("grid_size must be an integer")
    number = int(grid_size)
    if number < 9:
        raise ValueError("grid_size must be at least 9")
    radius_factor = float(source_radius_factor)
    if not np.isfinite(radius_factor) or radius_factor < 1.0:
        raise ValueError("source_radius_factor must be finite and at least one")

    radius = np.hypot(data.q_z, data.q_y)
    selected = valid_pixel_mask(data) & (radius <= radius_factor * upper)
    if np.count_nonzero(selected) < 3:
        raise ValueError("too few valid point-list values for centre interpolation")
    points = np.column_stack([data.q_z[selected], data.q_y[selected]])
    axis = np.linspace(-upper, upper, number)
    grid_q_z, grid_q_y = np.meshgrid(axis, axis)
    grid_points = np.column_stack([grid_q_z.ravel(), grid_q_y.ravel()])
    grid_intensity = LinearNDInterpolator(
        points,
        data.intensity[selected],
        fill_value=np.nan,
    )(grid_points).reshape(grid_q_z.shape)
    grid_uncertainty = NearestNDInterpolator(
        points,
        data.uncertainty[selected],
    )(grid_points).reshape(grid_q_z.shape)
    centering_grid = DetectorData(
        q_z=grid_q_z,
        q_y=grid_q_y,
        intensity=grid_intensity,
        uncertainty=grid_uncertainty,
    )
    refinement = refine_center_by_parity(
        centering_grid,
        q_min=lower,
        q_max=upper,
        max_shift_pixels=max_shift_pixels,
    )
    geometry = affine_detector_geometry(grid_q_z, grid_q_y)
    delta_q = geometry.matrix @ np.array(
        [refinement.delta_row, refinement.delta_column]
    )
    centered_data = apply_q_center_correction(data, delta_q[0], delta_q[1])
    return PointCloudCenteringResult(
        centered_data=centered_data,
        centering_grid=centering_grid,
        refinement=refinement,
        delta_q_z=float(delta_q[0]),
        delta_q_y=float(delta_q[1]),
    )


def symmetry_fold(
    data: DetectorData,
    *,
    symmetries: tuple[str, ...] = ("parity",),
) -> DetectorData:
    """Inverse-variance fold a detector image over a symmetry orbit.

    Supported generators are ``"parity"`` and ``"mirror"``.  The latter is
    reflection across the field-direction ``q_z`` axis.  All orbit members
    must be available; incomplete orbits are returned as NaN.

    Interpolation can correlate neighbouring folded pixels.  The returned
    uncertainty contains the conditional per-pixel variance but not those
    inter-pixel correlations, so folded data are primarily intended for
    visualization and deterministic sensitivity checks.
    """
    _require_detector_image(data)
    requested = tuple(dict.fromkeys(symmetries))
    if not requested or any(item not in {"parity", "mirror"} for item in requested):
        raise ValueError("symmetries must contain 'parity' and/or 'mirror'")
    transforms = [(np.asarray(data.q_z), np.asarray(data.q_y))]
    if "parity" in requested:
        transforms.append((-data.q_z, -data.q_y))
    if "mirror" in requested:
        transforms.append((data.q_z, -data.q_y))
    if set(requested) == {"parity", "mirror"}:
        transforms.append((-data.q_z, data.q_y))

    numerator = np.zeros(data.intensity.shape, dtype=float)
    precision = np.zeros(data.intensity.shape, dtype=float)
    complete = np.ones(data.intensity.shape, dtype=bool)
    for position, (target_q_z, target_q_y) in enumerate(transforms):
        if position == 0:
            sampled = np.asarray(data.intensity, dtype=float)
            variance = np.asarray(data.uncertainty, dtype=float) ** 2
            valid = valid_pixel_mask(data)
        else:
            sampled, variance, valid = _sample_detector(
                data, target_q_z, target_q_y
            )
        complete &= valid
        safe_variance = np.where(valid, variance, 1.0)
        numerator += np.where(valid, sampled / safe_variance, 0.0)
        precision += np.where(valid, 1.0 / safe_variance, 0.0)

    intensity = np.full(data.intensity.shape, np.nan)
    uncertainty = np.full(data.uncertainty.shape, np.nan)
    intensity[complete] = numerator[complete] / precision[complete]
    uncertainty[complete] = np.sqrt(1.0 / precision[complete])
    return DetectorData(
        q_z=np.asarray(data.q_z, dtype=float).copy(),
        q_y=np.asarray(data.q_y, dtype=float).copy(),
        intensity=intensity,
        uncertainty=uncertainty,
    )


def mask_symmetry_diagnostic(
    q_z: ArrayLike,
    q_y: ArrayLike,
    weights: ArrayLike,
) -> MaskSymmetryDiagnostic:
    """Assess directional bias and point-reflection error of a 2D mask."""
    q_z_array, q_y_array, weight_array = np.broadcast_arrays(
        np.asarray(q_z, dtype=float),
        np.asarray(q_y, dtype=float),
        np.asarray(weights, dtype=float),
    )
    if weight_array.ndim != 2:
        raise ValueError("mask symmetry diagnostic requires two-dimensional arrays")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(weight_array))
    if total <= 0.0:
        raise ValueError("weights must contain positive area")
    geometry = affine_detector_geometry(q_z_array, q_y_array)
    reflected, _, valid = _bilinear_sample(
        weight_array,
        np.ones_like(weight_array),
        -q_z_array,
        -q_y_array,
        geometry,
    )
    residual = np.full(weight_array.shape, np.nan)
    residual[valid] = weight_array[valid] - reflected[valid]
    centroid = np.array(
        [np.sum(weight_array * q_z_array), np.sum(weight_array * q_y_array)]
    ) / total
    admitted_radius = np.hypot(q_z_array[weight_array > 0.0], q_y_array[weight_array > 0.0])
    q_scale = max(float(np.nanmax(admitted_radius)), np.finfo(float).eps)
    return MaskSymmetryDiagnostic(
        normalized_centroid=float(np.linalg.norm(centroid) / q_scale),
        mean_reflection_residual=float(np.nanmean(residual)),
        mean_absolute_reflection_residual=float(np.nanmean(np.abs(residual))),
        partial_pixel_count=int(np.count_nonzero((weight_array > 0.0) & (weight_array < 1.0))),
        reflection_residual=residual,
    )


def standardized_residual_outliers(
    residual: ArrayLike,
    uncertainty: ArrayLike,
    *,
    threshold: float = 5.0,
) -> NDArray[np.bool_]:
    """Flag, but never remove, large standardized residuals."""
    residual_array, uncertainty_array = np.broadcast_arrays(
        np.asarray(residual, dtype=float), np.asarray(uncertainty, dtype=float)
    )
    limit = float(threshold)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("threshold must be finite and positive")
    valid = (
        np.isfinite(residual_array)
        & np.isfinite(uncertainty_array)
        & (uncertainty_array > 0.0)
    )
    result = np.zeros(residual_array.shape, dtype=bool)
    result[valid] = np.abs(residual_array[valid] / uncertainty_array[valid]) > limit
    return result


def _annulus_limits(q_min: float, q_max: float) -> tuple[float, float]:
    lower = float(q_min)
    upper = float(q_max)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("q_min and q_max must be finite")
    if lower < 0.0 or upper <= lower:
        raise ValueError("require 0 <= q_min < q_max")
    return lower, upper


def _require_detector_image(data: DetectorData) -> None:
    if np.asarray(data.intensity).ndim != 2:
        raise ValueError("this preconditioner requires a two-dimensional detector image")
    np.broadcast_arrays(data.q_z, data.q_y, data.intensity, data.uncertainty)


def _symmetry_target(q_z, q_y, symmetry):
    if symmetry == "parity":
        return -np.asarray(q_z), -np.asarray(q_y)
    if symmetry == "mirror":
        return np.asarray(q_z), -np.asarray(q_y)
    raise ValueError("symmetry must be 'parity' or 'mirror'")


def _representative_half(q_z, q_y, symmetry):
    tolerance = 32.0 * np.finfo(float).eps * max(
        1.0, float(np.nanmax(np.abs(q_z))), float(np.nanmax(np.abs(q_y)))
    )
    if symmetry == "parity":
        return (q_z > tolerance) | ((np.abs(q_z) <= tolerance) & (q_y >= 0.0))
    if symmetry == "mirror":
        return q_y > tolerance
    raise ValueError("symmetry must be 'parity' or 'mirror'")


def _sample_detector(data, target_q_z, target_q_y):
    geometry = affine_detector_geometry(data.q_z, data.q_y)
    return _bilinear_sample(
        np.asarray(data.intensity, dtype=float),
        np.asarray(data.uncertainty, dtype=float) ** 2,
        np.asarray(target_q_z, dtype=float),
        np.asarray(target_q_y, dtype=float),
        geometry,
    )


def _bilinear_sample(values, variance, target_q_z, target_q_y, geometry):
    target_shape = np.broadcast_shapes(target_q_z.shape, target_q_y.shape)
    q_z = np.broadcast_to(target_q_z, target_shape).reshape(-1)
    q_y = np.broadcast_to(target_q_y, target_shape).reshape(-1)
    target = np.vstack([q_z - geometry.offset[0], q_y - geometry.offset[1]])
    indices = np.linalg.solve(geometry.matrix, target)
    row = indices[0]
    column = indices[1]
    row0 = np.floor(row).astype(int)
    column0 = np.floor(column).astype(int)
    row_fraction = row - row0
    column_fraction = column - column0

    sampled = np.zeros(row.shape, dtype=float)
    sampled_variance = np.zeros(row.shape, dtype=float)
    valid = np.ones(row.shape, dtype=bool)
    source_valid = (
        np.isfinite(values) & np.isfinite(variance) & (variance > 0.0)
    )
    tolerance = 64.0 * np.finfo(float).eps
    for row_step, row_weight in ((0, 1.0 - row_fraction), (1, row_fraction)):
        for column_step, column_weight in (
            (0, 1.0 - column_fraction),
            (1, column_fraction),
        ):
            weight = row_weight * column_weight
            required = weight > tolerance
            row_index = row0 + row_step
            column_index = column0 + column_step
            inside = (
                (row_index >= 0)
                & (row_index < values.shape[0])
                & (column_index >= 0)
                & (column_index < values.shape[1])
            )
            safe_row = np.clip(row_index, 0, values.shape[0] - 1)
            safe_column = np.clip(column_index, 0, values.shape[1] - 1)
            neighbor_valid = inside & source_valid[safe_row, safe_column]
            valid &= (~required) | neighbor_valid
            sampled += np.where(
                neighbor_valid, weight * values[safe_row, safe_column], 0.0
            )
            sampled_variance += np.where(
                neighbor_valid,
                weight**2 * variance[safe_row, safe_column],
                0.0,
            )
    sampled[~valid] = np.nan
    sampled_variance[~valid] = np.nan
    return (
        sampled.reshape(target_shape),
        sampled_variance.reshape(target_shape),
        valid.reshape(target_shape),
    )
