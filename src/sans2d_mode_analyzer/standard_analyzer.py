"""Standard mode fit and compact PNG summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .block_test import AngularBlockTests, angular_block_tests
from .fit import LinearFitResult, fit_design_matrix, fit_mse, fit_wls
from .indices import even_angular_indices, rectangular_indices
from .io import DetectorData
from .power import (
    AngularModePowers,
    ParityMismatchMeasures,
    ParityPowerFractions,
    angular_mode_powers,
    parity_power_fractions,
    parity_mismatch_measures,
)
from .preconditioning import (
    CenterRefinementResult,
    affine_detector_geometry,
    apply_center_correction,
    center_point_cloud_by_parity,
    pixel_integrated_annulus_design,
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
    center_shift_q_z: float
    center_shift_q_y: float
    output_path: Path


def _orient_detector_image(
    q_z: NDArray[np.float64],
    q_y: NDArray[np.float64],
    image: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return an image with rows along ``q_y`` and columns along ``q_z``.

    ``numpy.meshgrid`` detector arrays normally vary in ``q_z`` along axis 1,
    whereas the ILL NeXus files used in the experimental example vary in
    ``q_z`` along axis 0.  The numerical fit uses the coordinate arrays
    directly and is independent of this storage convention; only ``imshow``
    needs the detector image in a fixed display orientation.
    """
    q_z_axis_0 = float(np.nanmedian(np.abs(np.diff(q_z, axis=0))))
    q_z_axis_1 = float(np.nanmedian(np.abs(np.diff(q_z, axis=1))))
    q_y_axis_0 = float(np.nanmedian(np.abs(np.diff(q_y, axis=0))))
    q_y_axis_1 = float(np.nanmedian(np.abs(np.diff(q_y, axis=1))))

    if q_z_axis_0 > q_z_axis_1 and q_y_axis_1 > q_y_axis_0:
        oriented = image.T
        q_z_line = q_z[:, q_z.shape[1] // 2]
        q_y_line = q_y[q_y.shape[0] // 2, :]
    else:
        oriented = image
        q_z_line = q_z[q_z.shape[0] // 2, :]
        q_y_line = q_y[:, q_y.shape[1] // 2]

    if np.nanmedian(np.diff(q_z_line)) < 0.0:
        oriented = oriented[:, ::-1]
    if np.nanmedian(np.diff(q_y_line)) < 0.0:
        oriented = oriented[::-1, :]
    return oriented


def standard_analyzer(
    q_z: ArrayLike,
    q_y: ArrayLike,
    intensity: ArrayLike,
    *,
    q_range: tuple[float, float] | list[float],
    mode_limits: tuple[int, int] | list[int],
    even_only: bool = False,
    uncertainty: ArrayLike | None = None,
    sample_weight: ArrayLike | None = None,
    absolute_uncertainty: bool = True,
    beam_centering: bool = False,
    beam_centering_grid_size: int = 129,
    beam_centering_source_radius_factor: float = 1.05,
    beam_centering_max_shift_pixels: float = 2.0,
    log_intensity_colorbar: bool = False,
    pixel_integration_subpixels: int | None = None,
    pixel_integration_chunk_size: int = 512,
    block_test_confidence: float = 0.95,
    output_path: str | Path = "standard_analysis.png",
) -> StandardAnalysisResult:
    """Fit the selected mode grid and save a multi-panel PNG summary.

    ``q_range=(q_min, q_max)`` and ``mode_limits=(n_max, m_max)`` define the
    annulus and rectangular basis truncation; two-element lists are accepted
    as well. By default the fit uses the full rectangle. With
    ``even_only=True``, only even angular orders are fitted for all radial
    orders up to ``m_max``, enforcing antipodal symmetry in the fitted image.
    Supplying ``uncertainty`` activates uncertainty-weighted least squares.
    ``sample_weight`` can additionally contain fractional pixel coverage.
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
    ``log_intensity_colorbar=True`` changes only the shared display norm of
    the input and fitted intensity panels; nonpositive display values are
    masked but remain included in the numerical fit.
    Holm-adjusted p-values and decisions remain available as numerical
    properties of block_tests, conditional on the chosen noise model.
    They are not highlighted in the standard overview. The even/odd power
    comparison remains descriptive; nominal T_n thresholds are model-based
    references rather than empirical artifact or physical-identification limits.
    """
    import matplotlib.pyplot as plt

    q_min, q_max, n_max, m_max = _resolve_analysis_arguments(
        q_range=q_range,
        mode_limits=mode_limits,
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

    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    select_indices = even_angular_indices if even_only else rectangular_indices
    indices = select_indices(n_max=n_max, m_max=m_max)
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
    else:
        integrated_design = pixel_integrated_annulus_design(
            q_z,
            q_y,
            indices,
            q_min=q_min,
            q_max=q_max,
            subpixels=pixel_integration_subpixels,
            chunk_size=pixel_integration_chunk_size,
        )
        selected &= integrated_design.selected_mask
        effective_weight_array = integrated_design.coverage.copy()
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
        )
    elif uncertainty_array is None and effective_weight_array is None:
        fit = fit_mse(
            q[selected],
            theta[selected],
            intensity[selected],
            indices,
            q_min=q_min,
            q_max=q_max,
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

    figure = plt.figure(figsize=(16.0, 20.0), constrained_layout=True)
    grid = figure.add_gridspec(5, 6)
    map_axes = [
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
    ]
    cosine_axis = figure.add_subplot(grid[1, 0:3])
    sine_axis = figure.add_subplot(grid[1, 3:6])
    cosine_test_axis = figure.add_subplot(grid[2, 0:3])
    sine_test_axis = figure.add_subplot(
        grid[2, 3:6], sharey=cosine_test_axis
    )
    cosine_map_axis = figure.add_subplot(grid[3, 0:3])
    sine_map_axis = figure.add_subplot(grid[3, 3:6])
    cosine_spectra_axis = figure.add_subplot(grid[4, 0:3])
    sine_spectra_axis = figure.add_subplot(grid[4, 3:6])

    common_min = float(np.nanmin(input_image))
    common_max = float(np.nanmax(input_image))
    intensity_norm = None
    intensity_colorbar_label = "intensity"
    if log_intensity_colorbar:
        from matplotlib.colors import LogNorm

        positive_intensity = np.concatenate(
            [
                input_image[np.isfinite(input_image) & (input_image > 0.0)],
                fit_image[np.isfinite(fit_image) & (fit_image > 0.0)],
            ]
        )
        if positive_intensity.size == 0:
            raise ValueError(
                "log_intensity_colorbar=True requires positive input or fit values"
            )
        log_minimum = float(np.min(positive_intensity))
        log_maximum = float(np.max(positive_intensity))
        if log_maximum <= log_minimum:
            log_maximum = log_minimum * (1.0 + 1.0e-12)
        intensity_norm = LogNorm(vmin=log_minimum, vmax=log_maximum)
        intensity_colorbar_label = "intensity (log color scale)"
    residual_limit = max(float(np.nanmax(np.abs(residual_image))), np.finfo(float).eps)

    if intensity.ndim == 2:
        extent = (
            float(np.nanmin(q_z)),
            float(np.nanmax(q_z)),
            float(np.nanmin(q_y)),
            float(np.nanmax(q_y)),
        )
        display_input = _orient_detector_image(q_z, q_y, input_image)
        display_fit = _orient_detector_image(q_z, q_y, fit_image)
        display_residual = _orient_detector_image(q_z, q_y, residual_image)
        if intensity_norm is not None:
            display_input = np.ma.masked_less_equal(display_input, 0.0)
            display_fit = np.ma.masked_less_equal(display_fit, 0.0)

        input_plot = map_axes[0].imshow(
            display_input,
            origin="lower",
            extent=extent,
            vmin=None if intensity_norm is not None else common_min,
            vmax=None if intensity_norm is not None else common_max,
            norm=intensity_norm,
            cmap="viridis",
            interpolation="nearest",
        )
        map_axes[1].imshow(
            display_fit,
            origin="lower",
            extent=extent,
            vmin=None if intensity_norm is not None else common_min,
            vmax=None if intensity_norm is not None else common_max,
            norm=intensity_norm,
            cmap="viridis",
            interpolation="nearest",
        )
        residual_plot = map_axes[2].imshow(
            display_residual,
            origin="lower",
            extent=extent,
            vmin=-residual_limit,
            vmax=residual_limit,
            cmap="coolwarm",
            interpolation="nearest",
        )
    else:
        scatter_input = input_image[selected]
        scatter_fit = fit_image[selected]
        if intensity_norm is not None:
            scatter_input = np.ma.masked_less_equal(scatter_input, 0.0)
            scatter_fit = np.ma.masked_less_equal(scatter_fit, 0.0)
        input_plot = map_axes[0].scatter(
            q_z[selected],
            q_y[selected],
            c=scatter_input,
            s=5,
            linewidths=0,
            vmin=None if intensity_norm is not None else common_min,
            vmax=None if intensity_norm is not None else common_max,
            norm=intensity_norm,
            cmap="viridis",
        )
        map_axes[1].scatter(
            q_z[selected],
            q_y[selected],
            c=scatter_fit,
            s=5,
            linewidths=0,
            vmin=None if intensity_norm is not None else common_min,
            vmax=None if intensity_norm is not None else common_max,
            norm=intensity_norm,
            cmap="viridis",
        )
        residual_plot = map_axes[2].scatter(
            q_z[selected],
            q_y[selected],
            c=residual_image[selected],
            s=5,
            linewidths=0,
            vmin=-residual_limit,
            vmax=residual_limit,
            cmap="coolwarm",
        )

    for axis, title in zip(map_axes, ["input signal", "mode fit", "residual"]):
        axis.set_title(title)
        axis.set_xlabel(r"$q_z$")
        axis.set_aspect("equal")
        axis.set_xlim(
            max(float(np.nanmin(q_z)), -q_max),
            min(float(np.nanmax(q_z)), q_max),
        )
        axis.set_ylim(
            max(float(np.nanmin(q_y)), -q_max),
            min(float(np.nanmax(q_y)), q_max),
        )
    map_axes[0].set_ylabel(r"$q_y$")
    figure.colorbar(
        input_plot,
        ax=map_axes[:2],
        label=intensity_colorbar_label,
    )
    figure.colorbar(residual_plot, ax=map_axes[2], label="residual")

    power_scale = float(max(np.max(powers.cosine), np.max(powers.sine)))
    cosine_fraction = powers.cosine / power_scale
    sine_fraction = powers.sine / power_scale
    assert powers.cosine_uncertainty is not None
    assert powers.sine_uncertainty is not None
    cosine_uncertainty = powers.cosine_uncertainty / power_scale
    sine_uncertainty = powers.sine_uncertainty / power_scale

    positive = np.concatenate(
        [cosine_fraction[cosine_fraction > 0.0], sine_fraction[sine_fraction > 0.0]]
    )
    lower_limit = max(float(np.min(positive)) / 3.0, 1.0e-14)
    upper_limit = max(
        float(np.max(cosine_fraction + cosine_uncertainty)),
        float(np.max(sine_fraction + sine_uncertainty)),
    ) * 1.5

    cosine_lower_error = np.minimum(
        cosine_uncertainty, np.maximum(cosine_fraction - lower_limit, 0.0)
    )
    sine_lower_error = np.minimum(
        sine_uncertainty, np.maximum(sine_fraction - lower_limit, 0.0)
    )
    angular_orders = powers.orders[::2] if even_only else powers.orders
    sine_orders = angular_orders[1:]
    for axis, orders, fraction, lower_error, power_uncertainty, even_color, title in [
        (
            cosine_axis, angular_orders, cosine_fraction, cosine_lower_error,
            cosine_uncertainty, "tab:blue", "cosine angular powers",
        ),
        (
            sine_axis, sine_orders, sine_fraction, sine_lower_error,
            sine_uncertainty, "tab:orange", "sine angular powers",
        ),
    ]:
        even = orders % 2 == 0
        for plotted_orders, color, label in [
            (orders[even], even_color, "even orders"),
            (orders[~even], "0.75", "odd orders"),
        ]:
            if even_only and plotted_orders.size == 0:
                continue
            axis.errorbar(
                plotted_orders,
                fraction[plotted_orders],
                yerr=np.vstack(
                    [lower_error[plotted_orders], power_uncertainty[plotted_orders]]
                ),
                fmt="o",
                color=color,
                markeredgecolor="0.55" if label == "odd orders" else color,
                capsize=3,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("angular order n")
        axis.set_ylabel(r"relative power $P_n^2/P_{\max}^2$")
        axis.set_xticks(angular_orders)
        axis.set_yscale("log")
        axis.set_ylim(lower_limit, upper_limit)
        axis.grid(axis="y", alpha=0.25)
        if orders.size or not even_only:
            axis.legend(fontsize=8)

    for axis, statistic, critical, valid, even_color, title in [
        (
            cosine_test_axis,
            block_tests.cosine_statistic,
            block_tests.cosine_critical_value,
            block_tests.cosine_degrees_of_freedom > 0,
            "tab:blue",
            "cosine angular block tests",
        ),
        (
            sine_test_axis,
            block_tests.sine_statistic,
            block_tests.sine_critical_value,
            block_tests.sine_degrees_of_freedom > 0,
            "tab:orange",
            "sine angular block tests",
        ),
    ]:
        even = valid & (block_tests.orders % 2 == 0)
        axis.plot(
            block_tests.orders[even],
            statistic[even],
            "o",
            color=even_color,
            label="even orders",
        )
        if not even_only:
            odd = valid & (block_tests.orders % 2 == 1)
            axis.plot(
                block_tests.orders[odd],
                statistic[odd],
                "o",
                color="0.75",
                markeredgecolor="0.55",
                label="odd orders",
            )
        axis.plot(
            block_tests.orders[valid],
            critical[valid],
            "--",
            color="tab:red",
            label=f"nominal {block_test_confidence:.0%} threshold",
        )
        axis.set_title(title)
        axis.set_xlabel("angular order n")
        axis.set_ylabel(r"$T_n$")
        axis.set_xticks(angular_orders)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
        if np.any(valid) or not even_only:
            axis.legend(fontsize=8)

    max_fitted_order = angular_orders[-1]
    cosine_map = np.zeros((m_max + 1, max_fitted_order + 1))
    sine_map = np.zeros_like(cosine_map)
    mode_power_scale = float(np.max(fit.coefficients**2))
    for coefficient, (n, m) in zip(fit.coefficients, fit.indices):
        if n >= 0:
            cosine_map[m, n] = coefficient**2 / mode_power_scale
        else:
            sine_map[m, abs(n)] = coefficient**2 / mode_power_scale

    from matplotlib.colors import LogNorm

    positive_map = np.concatenate(
        [cosine_map[cosine_map > 0.0], sine_map[sine_map > 0.0]]
    )
    map_norm = LogNorm(
        vmin=max(float(np.min(positive_map)) / 2.0, 1.0e-14),
        vmax=1.0,
    )
    cosine_map_plot = cosine_map_axis.imshow(
        np.ma.masked_less_equal(cosine_map, 0.0),
        origin="lower",
        aspect="auto",
        extent=(-0.5, max_fitted_order + 0.5, -0.5, m_max + 0.5),
        norm=map_norm,
        cmap="viridis",
    )
    if sine_orders.size:
        sine_map_axis.imshow(
            np.ma.masked_less_equal(sine_map[:, 1:], 0.0),
            origin="lower",
            aspect="auto",
            extent=(0.5, max_fitted_order + 0.5, -0.5, m_max + 0.5),
            norm=map_norm,
            cmap="viridis",
        )
    for axis, orders, title in [
        (cosine_map_axis, angular_orders, "cosine $(n,m)$ powers"),
        (sine_map_axis, sine_orders, "sine $(n,m)$ powers"),
    ]:
        axis.set_title(title)
        axis.set_xlabel("angular order n")
        axis.set_ylabel("radial order m")
        axis.set_xticks(orders)
        axis.set_yticks(range(m_max + 1))
    figure.colorbar(
        cosine_map_plot,
        ax=[cosine_map_axis, sine_map_axis],
        label=r"relative mode power $c_{n,m}^2/c_{\max}^2$",
    )

    q_spectrum = np.linspace(q_min, q_max, 300)
    spectra = angular_radial_spectra(
        q_spectrum,
        fit.coefficients,
        fit.indices,
        q_min=q_min,
        q_max=q_max,
        coefficient_covariance=fit.coefficient_covariance,
    )
    assert spectra.cosine_uncertainty is not None
    assert spectra.sine_uncertainty is not None
    colors = plt.get_cmap("tab10")
    for order in angular_orders:
        color = colors(int(order) % 10)
        cosine_spectra_axis.plot(
            spectra.q,
            spectra.cosine[order],
            label=f"n={order}",
            color=color,
        )
        cosine_spectra_axis.fill_between(
            spectra.q,
            spectra.cosine[order] - spectra.cosine_uncertainty[order],
            spectra.cosine[order] + spectra.cosine_uncertainty[order],
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
        if order > 0:
            sine_spectra_axis.plot(
                spectra.q,
                spectra.sine[order],
                label=f"n={order}",
                color=color,
            )
            sine_spectra_axis.fill_between(
                spectra.q,
                spectra.sine[order] - spectra.sine_uncertainty[order],
                spectra.sine[order] + spectra.sine_uncertainty[order],
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
    for axis, title in zip(
        [cosine_spectra_axis, sine_spectra_axis],
        [r"cosine radial spectra $I_n^c(q)$", r"sine radial spectra $I_n^s(q)$"],
    ):
        axis.set_title(title)
        axis.set_xlabel("q")
        axis.set_ylabel(r"$I_n(q)$")
        axis.set_xlim(q_min, q_max)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.grid(alpha=0.25)
        if axis is cosine_spectra_axis or sine_orders.size:
            axis.legend(ncol=3, fontsize=7)

    angular_mode_count = len({n for n, _ in fit.indices})
    basis_label = "even only" if even_only else "full basis"
    centering_summary = ""
    if beam_centering_method is not None:
        centering_summary = (
            "\nbeam centre: "
            + beam_centering_method
            + f"   delta q=({center_shift_q_z:.3e}, {center_shift_q_y:.3e})"
        )
    figure.suptitle(
        "Mode analysis "
        f"({basis_label}: {len(fit.indices)} total, "
        f"{angular_mode_count} angular modes)\n"
        + f"even power fraction: {parity_fractions.even:.6f}   "
        + f"odd power fraction: {parity_fractions.odd:.6f}"
        + "\n"
        + r"fit parity $\langle[I(q,\theta)-I(q,\theta+\pi)]^2\rangle_A$: "
        + f"{parity_mismatch.mean_square:.3e}   "
        + f"relative RMS: {parity_mismatch.normalized_rms:.4f}"
        + centering_summary,
        fontsize=12,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return StandardAnalysisResult(
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
        center_shift_q_z=center_shift_q_z,
        center_shift_q_y=center_shift_q_y,
        output_path=path,
    )


def _resolve_analysis_arguments(
    *,
    q_range,
    mode_limits,
):
    q_values = _two_values(q_range, "q_range")
    mode_values = _two_values(mode_limits, "mode_limits")

    lower = float(q_values[0])
    upper = float(q_values[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("q_range values must be finite")
    if lower < 0.0 or upper <= lower:
        raise ValueError("q_range requires 0 <= q_min < q_max")

    resolved_modes = []
    for value, name in zip(mode_values, ("n_max", "m_max")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"{name} must be non-negative")
        resolved_modes.append(integer)
    return lower, upper, resolved_modes[0], resolved_modes[1]


def _two_values(values, name):
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be a two-element tuple or list") from error
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return result
