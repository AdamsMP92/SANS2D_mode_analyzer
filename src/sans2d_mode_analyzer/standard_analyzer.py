"""Standard mode fit and compact PNG summary."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import radial_measure_weight
from .block_test import AngularBlockTests, angular_block_tests
from .fit import LinearFitResult, fit_design_matrix, fit_mse, fit_wls
from .indices import even_angular_indices, rectangular_indices, validate_indices
from .io import DetectorData
from .power import (
    AngularModePowers,
    ParityMismatchMeasures,
    ParityPowerFractions,
    angular_mode_powers,
    parity_power_fractions,
    parity_mismatch_measures,
)
from .piecewise_linear_basis import (
    piecewise_linear_indices,
    piecewise_linear_knots,
    piecewise_linear_pixel_integrated_design,
)
from .piecewise_linear_reference import (
    PiecewiseLinearReferenceResult,
    fit_piecewise_linear_reference,
)
from .preconditioning import (
    CenterRefinementResult,
    OddModeCenterRefinementResult,
    affine_detector_geometry,
    apply_center_correction,
    apply_q_center_correction,
    center_point_cloud_by_parity,
    pixel_integrated_annulus_design,
    refine_center_by_odd_mode_power,
    refine_center_by_parity,
)
from .spectra import AngularRadialSpectra, angular_radial_spectra


@dataclass(frozen=True)
class StandardAnalysisResult:
    """Numerical results and detector images from ``standard_analyzer``."""

    fit: LinearFitResult
    powers: AngularModePowers
    spectra: AngularRadialSpectra
    parity_fractions: ParityPowerFractions
    parity_mismatch: ParityMismatchMeasures
    block_tests: AngularBlockTests
    input_image: NDArray[np.float64]
    fit_image: NDArray[np.float64]
    residual_image: NDArray[np.float64]
    selected_mask: NDArray[np.bool_]
    uncertainty_image: NDArray[np.float64] | None
    sample_weight_image: NDArray[np.float64] | None
    beam_centering_method: str | None
    beam_centering_refinement: CenterRefinementResult | None
    odd_mode_centering_refinement: OddModeCenterRefinementResult | None
    center_shift_q_z: float
    center_shift_q_y: float
    radial_measure_exponent: float
    reference: PiecewiseLinearReferenceResult | None
    output_path: Path | None


def standard_analyzer(
    q_z: ArrayLike,
    q_y: ArrayLike,
    intensity: ArrayLike,
    *,
    q_range: tuple[float, float] | list[float],
    mode_limits: tuple[int, int] | list[int] | None = None,
    indices: Iterable[Sequence[int]] | None = None,
    even_only: bool = False,
    uncertainty: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    radial_measure_exponent: float = 1.0,
    absolute_uncertainty: bool = True,
    beam_centering: bool = False,
    beam_centering_grid_size: int = 129,
    beam_centering_source_radius_factor: float = 1.05,
    beam_centering_max_shift_pixels: float = 2.0,
    odd_mode_centering: bool = False,
    odd_mode_centering_mode_limits: tuple[int, int] | list[int] = (5, 8),
    odd_mode_centering_max_shift_pixels: float = 0.25,
    log_intensity_colorbar: bool = False,
    pixel_integration_subpixels: int | None = None,
    pixel_integration_chunk_size: int = 512,
    block_test_confidence: float = 0.95,
    piecewise_linear_reference: bool = False,
    reference_radial_segments: int = 12,
    reference_knot_spacing: str = "measure",
    export_figure: bool = True,
    output_path: str | Path = "standard_analysis.png",
) -> StandardAnalysisResult:
    """Fit the selected mode grid and save a multi-panel PNG summary.

    ``q_range=(q_min, q_max)`` defines the annulus. Supply either
    ``mode_limits=(n_max, m_max)`` for a rectangular truncation or an explicit
    ``indices=[(n,m), ...]`` list. Exactly the listed modes are fitted when
    ``indices`` is used. By default a mode-limit fit uses the full rectangle. With
    ``even_only=True``, only even angular orders are fitted for all radial
    orders up to ``m_max``, enforcing antipodal symmetry in the fitted image;
    this shortcut cannot be combined with explicit ``indices``.
    Supplying ``uncertainty`` activates uncertainty-weighted least squares.
    ``sample_weight`` can additionally contain fractional pixel coverage.
    ``radial_measure_exponent=s`` consistently changes the radial basis and
    detector-fit measure from the default ``q dq`` (``s=1``) to ``q**s dq``.
    Relative to equal-area detector pixels this applies the radial fit weight
    ``q**(s-1)``. Values larger than one emphasize high q; values below one
    emphasize low q. The exponent must be greater than -1.
    With ``piecewise_linear_reference=True``, the same selected data and
    angular modes are fitted a second time using polygonal radial spectra.
    ``reference_radial_segments`` sets the number of ring intervals and
    ``reference_knot_spacing`` is either ``'measure'`` or ``'q'``. The
    alternative spectra and Gram-corrected powers are returned as
    ``result.reference.spectra`` and ``result.reference.powers``.
    With absolute uncertainties, the supplied pixel errors are propagated
    through this weighted fit; coverage does not alter their noise variance.
    Without absolute uncertainties, the combined weights define a relative
    precision model and the residual scatter sets its covariance scale. That
    model does not calibrate geometric-weight noise of unknown variance. Setting
    ``pixel_integration_subpixels`` averages every basis function over the
    admitted subpixel area and automatically uses that fractional area as an
    additional sample weight. With ``beam_centering=True``, a one-dimensional
    point list is centred through :func:`center_point_cloud_by_parity`, whereas
    a two-dimensional detector image uses the native affine-grid parity
    refinement. Supplied uncertainties are required for either path.
    With ``odd_mode_centering=True``, an additional bounded reciprocal-space
    shift minimizes the fitted odd-angular coefficient-power fraction. This
    step is implemented by :func:`refine_center_by_odd_mode_power` and can be
    used either alone or as a local refinement after ``beam_centering``.
    ``odd_mode_centering_mode_limits`` defines its separate, deliberately
    compact full basis; ``odd_mode_centering_max_shift_pixels`` bounds the
    additional search around the current centre. This procedure is justified
    only when odd angular modes are known to be artefacts rather than physical
    signal.
    ``log_intensity_colorbar=True`` changes only the shared display norm of
    the input and fitted intensity panels; nonpositive display values are
    masked but remain included in the numerical fit.
    Holm-adjusted p-values and decisions remain available as numerical
    properties of block_tests, conditional on the chosen noise model.
    The standard overview shows only the T_n values and does not draw critical
    thresholds or rejection markers. The even/odd comparison therefore remains
    descriptive rather than defining an empirical artifact or physical-
    identification limit.
    Set ``export_figure=False`` to skip all Matplotlib rendering while retaining
    the complete numerical result; ``result.output_path`` is then ``None``.
    """
    q_min, q_max, indices, basis_label = _resolve_analysis_arguments(
        q_range=q_range,
        mode_limits=mode_limits,
        indices=indices,
        even_only=even_only,
    )
    radial_measure_exponent = float(radial_measure_exponent)
    if (
        not np.isfinite(radial_measure_exponent)
        or radial_measure_exponent <= -1.0
    ):
        raise ValueError(
            "radial_measure_exponent must be finite and greater than -1"
        )

    q_z, q_y, intensity = np.broadcast_arrays(
        np.asarray(q_z, dtype=float),
        np.asarray(q_y, dtype=float),
        np.asarray(intensity, dtype=float),
    )
    uncertainty_array = None
    if uncertainty is not None:
        uncertainty_array = np.broadcast_to(
            np.asarray(uncertainty, dtype=float), intensity.shape
        )
    weight_array = None
    if sample_weight is not None:
        weight_array = np.broadcast_to(
            np.asarray(sample_weight, dtype=float), intensity.shape
        )
    if intensity.ndim not in (1, 2):
        raise ValueError(
            "standard_analyzer expects a one-dimensional detector point list "
            "or two-dimensional detector arrays"
        )

    beam_centering_method = None
    beam_centering_refinement = None
    odd_mode_centering_refinement = None
    center_shift_q_z = 0.0
    center_shift_q_y = 0.0
    if beam_centering:
        if uncertainty_array is None:
            raise ValueError("beam_centering=True requires supplied uncertainty")
        detector = DetectorData(q_z, q_y, intensity, uncertainty_array)
        if intensity.ndim == 1:
            centering = center_point_cloud_by_parity(
                detector,
                q_min=q_min,
                q_max=q_max,
                grid_size=beam_centering_grid_size,
                source_radius_factor=beam_centering_source_radius_factor,
                max_shift_pixels=beam_centering_max_shift_pixels,
            )
            centered_detector = centering.centered_data
            beam_centering_method = "point-cloud parity"
            beam_centering_refinement = centering.refinement
            center_shift_q_z = centering.delta_q_z
            center_shift_q_y = centering.delta_q_y
        else:
            refinement = refine_center_by_parity(
                detector,
                q_min=q_min,
                q_max=q_max,
                max_shift_pixels=beam_centering_max_shift_pixels,
            )
            geometry = affine_detector_geometry(q_z, q_y)
            delta_q = geometry.matrix @ np.array(
                [refinement.delta_row, refinement.delta_column]
            )
            centered_detector = apply_center_correction(
                detector,
                refinement.delta_row,
                refinement.delta_column,
            )
            beam_centering_method = "affine-grid parity"
            beam_centering_refinement = refinement
            center_shift_q_z = float(delta_q[0])
            center_shift_q_y = float(delta_q[1])
        q_z = np.asarray(centered_detector.q_z, dtype=float)
        q_y = np.asarray(centered_detector.q_y, dtype=float)

    if odd_mode_centering:
        centering_limits = _two_values(
            odd_mode_centering_mode_limits,
            "odd_mode_centering_mode_limits",
        )
        centering_indices = tuple(
            rectangular_indices(
                n_max=centering_limits[0],
                m_max=centering_limits[1],
            )
        )
        maximum_pixels = float(odd_mode_centering_max_shift_pixels)
        if not np.isfinite(maximum_pixels) or maximum_pixels <= 0.0:
            raise ValueError(
                "odd_mode_centering_max_shift_pixels must be finite and positive"
            )
        if q_z.ndim == 2:
            centering_geometry = affine_detector_geometry(q_z, q_y)
            nominal_pixel_size = float(
                np.max(np.linalg.norm(centering_geometry.matrix, axis=0))
            )
        else:
            if (
                isinstance(beam_centering_grid_size, bool)
                or not isinstance(beam_centering_grid_size, (int, np.integer))
                or int(beam_centering_grid_size) < 2
            ):
                raise ValueError(
                    "beam_centering_grid_size must be an integer of at least two"
                )
            nominal_pixel_size = float(
                2.0 * q_max / (int(beam_centering_grid_size) - 1)
            )
        if not np.isfinite(nominal_pixel_size) or nominal_pixel_size <= 0.0:
            raise ValueError("could not determine a positive detector-pixel scale")
        maximum_shift = maximum_pixels * nominal_pixel_size
        modal_uncertainty = (
            np.ones(intensity.shape, dtype=float)
            if uncertainty_array is None
            else uncertainty_array
        )
        modal_detector = DetectorData(
            q_z=q_z,
            q_y=q_y,
            intensity=intensity,
            uncertainty=modal_uncertainty,
        )
        odd_mode_centering_refinement = refine_center_by_odd_mode_power(
            modal_detector,
            centering_indices,
            q_min=q_min,
            q_max=q_max,
            max_shift_q=(maximum_shift, maximum_shift),
            sample_weight=weight_array,
            radial_measure_exponent=radial_measure_exponent,
        )
        if not odd_mode_centering_refinement.success:
            raise RuntimeError(
                "odd-mode center refinement failed: "
                f"{odd_mode_centering_refinement.message}"
            )
        modal_detector = apply_q_center_correction(
            modal_detector,
            odd_mode_centering_refinement.delta_q_z,
            odd_mode_centering_refinement.delta_q_y,
        )
        q_z = np.asarray(modal_detector.q_z, dtype=float)
        q_y = np.asarray(modal_detector.q_y, dtype=float)
        center_shift_q_z += odd_mode_centering_refinement.delta_q_z
        center_shift_q_y += odd_mode_centering_refinement.delta_q_y
        if beam_centering_method is None:
            beam_centering_method = "modal odd-power"
        else:
            beam_centering_method += " + modal odd-power"

    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    selected = (
        np.isfinite(q_z)
        & np.isfinite(q_y)
        & np.isfinite(intensity)
    )
    if uncertainty_array is not None:
        selected &= np.isfinite(uncertainty_array) & (uncertainty_array > 0.0)
    if weight_array is not None:
        selected &= np.isfinite(weight_array) & (weight_array > 0.0)
    integrated_design = None
    selected_design_matrix = None
    effective_weight_array = weight_array
    if pixel_integration_subpixels is None:
        selected &= (q >= q_min) & (q <= q_max)
        radial_weight_array = np.zeros(intensity.shape, dtype=float)
        radial_weight_array[selected] = radial_measure_weight(
            q[selected],
            q_min=q_min,
            q_max=q_max,
            radial_measure_exponent=radial_measure_exponent,
        )
        selected &= radial_weight_array > 0.0
        if radial_measure_exponent != 1.0:
            effective_weight_array = radial_weight_array
            if weight_array is not None:
                effective_weight_array = radial_weight_array * weight_array
    else:
        integrated_design = pixel_integrated_annulus_design(
            q_z,
            q_y,
            indices,
            q_min=q_min,
            q_max=q_max,
            radial_measure_exponent=radial_measure_exponent,
            subpixels=pixel_integration_subpixels,
            chunk_size=pixel_integration_chunk_size,
        )
        selected &= (
            integrated_design.selected_mask
            & (integrated_design.quadrature_weight > 0.0)
        )
        effective_weight_array = integrated_design.quadrature_weight.copy()
        if weight_array is not None:
            effective_weight_array *= weight_array
    if not np.any(selected):
        raise ValueError("no valid detector values lie inside the requested annulus")

    if integrated_design is not None:
        integrated_valid = selected[integrated_design.selected_mask]
        selected_design_matrix = integrated_design.design_matrix[integrated_valid]
        fit = fit_design_matrix(
            intensity[selected],
            indices,
            selected_design_matrix,
            uncertainty=(
                None if uncertainty_array is None else uncertainty_array[selected]
            ),
            sample_weight=effective_weight_array[selected],
            absolute_uncertainty=absolute_uncertainty,
            radial_measure_exponent=radial_measure_exponent,
        )
    elif uncertainty_array is None and effective_weight_array is None:
        fit = fit_mse(
            q[selected],
            theta[selected],
            intensity[selected],
            indices,
            q_min=q_min,
            q_max=q_max,
            radial_measure_exponent=radial_measure_exponent,
        )
    else:
        selected_uncertainty = (
            np.ones(np.count_nonzero(selected))
            if uncertainty_array is None
            else uncertainty_array[selected]
        )
        selected_weight = (
            None
            if effective_weight_array is None
            else effective_weight_array[selected]
        )
        fit = fit_wls(
            q[selected],
            theta[selected],
            intensity[selected],
            indices,
            selected_uncertainty,
            q_min=q_min,
            q_max=q_max,
            radial_measure_exponent=radial_measure_exponent,
            sample_weight=selected_weight,
            absolute_uncertainty=(
                absolute_uncertainty if uncertainty_array is not None else False
            ),
        )
    powers = angular_mode_powers(
        fit.coefficients,
        fit.indices,
        fit.coefficient_covariance,
    )
    parity_fractions = parity_power_fractions(fit.coefficients, fit.indices)
    parity_mismatch = parity_mismatch_measures(
        fit.coefficients,
        fit.indices,
        q_min=q_min,
        q_max=q_max,
        radial_measure_exponent=radial_measure_exponent,
    )
    block_tests = angular_block_tests(
        fit.coefficients,
        fit.coefficient_covariance,
        fit.indices,
        residual_degrees_of_freedom=(
            fit.degrees_of_freedom
            if fit.covariance_scaled_by_residual
            else None
        ),
        confidence=block_test_confidence,
    )
    input_image = np.full(intensity.shape, np.nan)
    fit_image = np.full(intensity.shape, np.nan)
    residual_image = np.full(intensity.shape, np.nan)
    uncertainty_image = None
    sample_weight_image = None
    input_image[selected] = intensity[selected]
    fit_image[selected] = fit.fitted_values
    residual_image[selected] = fit.residuals
    if uncertainty_array is not None:
        uncertainty_image = np.full(intensity.shape, np.nan)
        uncertainty_image[selected] = uncertainty_array[selected]
    if effective_weight_array is not None:
        sample_weight_image = np.zeros(intensity.shape)
        sample_weight_image[selected] = effective_weight_array[selected]

    q_spectrum = np.linspace(q_min, q_max, 300)
    spectra = angular_radial_spectra(
        q_spectrum,
        fit.coefficients,
        fit.indices,
        q_min=q_min,
        q_max=q_max,
        radial_measure_exponent=radial_measure_exponent,
        coefficient_covariance=fit.coefficient_covariance,
    )
    reference = None
    if piecewise_linear_reference:
        angular_modes = tuple(dict.fromkeys(order for order, _ in indices))
        reference_knots = piecewise_linear_knots(
            q_min=q_min,
            q_max=q_max,
            radial_segments=reference_radial_segments,
            radial_measure_exponent=radial_measure_exponent,
            spacing=reference_knot_spacing,
        )
        reference_design = None
        if pixel_integration_subpixels is not None:
            reference_indices = piecewise_linear_indices(
                angular_modes,
                radial_segments=reference_radial_segments,
            )
            integrated_reference = piecewise_linear_pixel_integrated_design(
                q_z,
                q_y,
                reference_indices,
                knots=reference_knots,
                radial_measure_exponent=radial_measure_exponent,
                subpixels=pixel_integration_subpixels,
                chunk_size=pixel_integration_chunk_size,
            )
            reference_valid = selected[integrated_reference.selected_mask]
            reference_design = integrated_reference.design_matrix[reference_valid]
        reference = fit_piecewise_linear_reference(
            q[selected],
            theta[selected],
            intensity[selected],
            angular_modes,
            knots=reference_knots,
            radial_measure_exponent=radial_measure_exponent,
            uncertainty=(
                None if uncertainty_array is None else uncertainty_array[selected]
            ),
            sample_weight=(
                None
                if effective_weight_array is None
                else effective_weight_array[selected]
            ),
            absolute_uncertainty=(
                absolute_uncertainty if uncertainty_array is not None else False
            ),
            design_matrix=reference_design,
            spectrum_q=q_spectrum,
        )
    path = Path(output_path) if export_figure else None
    result = StandardAnalysisResult(
        fit=fit,
        powers=powers,
        spectra=spectra,
        parity_fractions=parity_fractions,
        parity_mismatch=parity_mismatch,
        block_tests=block_tests,
        input_image=input_image,
        fit_image=fit_image,
        residual_image=residual_image,
        selected_mask=selected,
        uncertainty_image=uncertainty_image,
        sample_weight_image=sample_weight_image,
        beam_centering_method=beam_centering_method,
        beam_centering_refinement=beam_centering_refinement,
        odd_mode_centering_refinement=odd_mode_centering_refinement,
        center_shift_q_z=center_shift_q_z,
        center_shift_q_y=center_shift_q_y,
        radial_measure_exponent=radial_measure_exponent,
        reference=reference,
        output_path=path,
    )
    if export_figure:
        from .figure_backend import export_standard_analysis_figure

        export_standard_analysis_figure(
            result,
            q_z,
            q_y,
            q_range=(q_min, q_max),
            basis_label=basis_label,
            log_intensity_colorbar=log_intensity_colorbar,
            output_path=path,
        )
    return result


def _resolve_analysis_arguments(
    *,
    q_range,
    mode_limits,
    indices,
    even_only,
):
    q_values = _two_values(q_range, "q_range")

    lower = float(q_values[0])
    upper = float(q_values[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("q_range values must be finite")
    if lower < 0.0 or upper <= lower:
        raise ValueError("q_range requires 0 <= q_min < q_max")

    if (mode_limits is None) == (indices is None):
        raise ValueError("supply exactly one of mode_limits or indices")

    if indices is not None:
        if even_only:
            raise ValueError("even_only cannot be combined with explicit indices")
        resolved_indices = validate_indices(indices)
        return lower, upper, resolved_indices, "custom basis"

    mode_values = _two_values(mode_limits, "mode_limits")
    resolved_modes = []
    for value, name in zip(mode_values, ("n_max", "m_max")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"{name} must be non-negative")
        resolved_modes.append(integer)
    n_max, m_max = resolved_modes
    index_factory = even_angular_indices if even_only else rectangular_indices
    resolved_indices = tuple(index_factory(n_max=n_max, m_max=m_max))
    basis_label = "even only" if even_only else "full basis"
    return lower, upper, resolved_indices, basis_label


def _two_values(values, name):
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be a two-element tuple or list") from error
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return result
