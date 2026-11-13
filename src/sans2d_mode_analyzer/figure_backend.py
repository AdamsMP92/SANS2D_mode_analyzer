"""Matplotlib backend for the standard-analysis overview figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _orient_detector_image(
    q_z: NDArray[np.float64],
    q_y: NDArray[np.float64],
    image: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return an image with rows along ``q_y`` and columns along ``q_z``."""
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


def export_standard_analysis_figure(
    result,
    q_z: NDArray[np.float64],
    q_y: NDArray[np.float64],
    *,
    q_range: tuple[float, float],
    basis_label: str,
    log_intensity_colorbar: bool,
    output_path: str | Path,
) -> Path:
    """Render and save the standard multi-panel diagnostic overview."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    q_min, q_max = q_range
    fit = result.fit
    powers = result.powers
    spectra = result.spectra
    reference = result.reference
    block_tests = result.block_tests
    input_image = result.input_image
    fit_image = result.fit_image
    residual_image = result.residual_image
    selected = result.selected_mask

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
    sine_test_axis = figure.add_subplot(grid[2, 3:6], sharey=cosine_test_axis)
    cosine_map_axis = figure.add_subplot(grid[3, 0:3])
    sine_map_axis = figure.add_subplot(grid[3, 3:6])
    cosine_spectra_axis = figure.add_subplot(grid[4, 0:3])
    sine_spectra_axis = figure.add_subplot(
        grid[4, 3:6], sharey=cosine_spectra_axis
    )

    common_min = float(np.nanmin(input_image))
    common_max = float(np.nanmax(input_image))
    intensity_norm = None
    intensity_colorbar_label = "intensity"
    if log_intensity_colorbar:
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
    residual_limit = max(
        float(np.nanmax(np.abs(residual_image))), np.finfo(float).eps
    )

    if input_image.ndim == 2:
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
    figure.colorbar(input_plot, ax=map_axes[:2], label=intensity_colorbar_label)
    figure.colorbar(residual_plot, ax=map_axes[2], label="residual")

    power_candidates = [np.max(powers.cosine), np.max(powers.sine)]
    if reference is not None:
        power_candidates.extend(
            [np.max(reference.powers.cosine), np.max(reference.powers.sine)]
        )
    power_scale = float(max(power_candidates))
    cosine_fraction = powers.cosine / power_scale
    sine_fraction = powers.sine / power_scale
    assert powers.cosine_uncertainty is not None
    assert powers.sine_uncertainty is not None
    cosine_uncertainty = powers.cosine_uncertainty / power_scale
    sine_uncertainty = powers.sine_uncertainty / power_scale
    reference_cosine_fraction = None
    reference_sine_fraction = None
    reference_cosine_uncertainty = None
    reference_sine_uncertainty = None
    if reference is not None:
        assert reference.powers.cosine_uncertainty is not None
        assert reference.powers.sine_uncertainty is not None
        reference_cosine_fraction = reference.powers.cosine / power_scale
        reference_sine_fraction = reference.powers.sine / power_scale
        reference_cosine_uncertainty = (
            reference.powers.cosine_uncertainty / power_scale
        )
        reference_sine_uncertainty = (
            reference.powers.sine_uncertainty / power_scale
        )
    positive = np.concatenate(
        [cosine_fraction[cosine_fraction > 0.0], sine_fraction[sine_fraction > 0.0]]
    )
    lower_limit = max(float(np.min(positive)) / 3.0, 1.0e-14)
    upper_limit = max(
        float(np.max(cosine_fraction + cosine_uncertainty)),
        float(np.max(sine_fraction + sine_uncertainty)),
    ) * 1.5
    if reference is not None:
        positive = np.concatenate(
            [
                positive,
                reference_cosine_fraction[reference_cosine_fraction > 0.0],
                reference_sine_fraction[reference_sine_fraction > 0.0],
            ]
        )
        lower_limit = max(float(np.min(positive)) / 3.0, 1.0e-14)
        upper_limit = max(
            upper_limit,
            float(
                np.max(reference_cosine_fraction + reference_cosine_uncertainty)
            )
            * 1.5,
            float(np.max(reference_sine_fraction + reference_sine_uncertainty))
            * 1.5,
        )
    cosine_lower_error = np.minimum(
        cosine_uncertainty, np.maximum(cosine_fraction - lower_limit, 0.0)
    )
    sine_lower_error = np.minimum(
        sine_uncertainty, np.maximum(sine_fraction - lower_limit, 0.0)
    )
    angular_orders = np.asarray(
        sorted({abs(n) for n, _ in fit.indices}), dtype=np.int64
    )
    cosine_orders = np.asarray(
        sorted({n for n, _ in fit.indices if n >= 0}), dtype=np.int64
    )
    sine_orders = np.asarray(
        sorted({abs(n) for n, _ in fit.indices if n < 0}), dtype=np.int64
    )
    for axis, orders, fraction, lower_error, power_error, reference_fraction, reference_error, color, title in [
        (
            cosine_axis,
            cosine_orders,
            cosine_fraction,
            cosine_lower_error,
            cosine_uncertainty,
            reference_cosine_fraction,
            reference_cosine_uncertainty,
            "tab:blue",
            "cosine angular powers",
        ),
        (
            sine_axis,
            sine_orders,
            sine_fraction,
            sine_lower_error,
            sine_uncertainty,
            reference_sine_fraction,
            reference_sine_uncertainty,
            "tab:orange",
            "sine angular powers",
        ),
    ]:
        even = orders % 2 == 0
        for plotted_orders, point_color, label in [
            (orders[even], color, "even orders"),
            (orders[~even], "0.75", "odd orders"),
        ]:
            if plotted_orders.size == 0:
                continue
            axis.errorbar(
                plotted_orders,
                fraction[plotted_orders],
                yerr=np.vstack(
                    [lower_error[plotted_orders], power_error[plotted_orders]]
                ),
                fmt="o",
                color=point_color,
                markeredgecolor="0.55" if label == "odd orders" else point_color,
                capsize=3,
                label=label,
            )
        if reference is not None and orders.size:
            reference_lower_error = np.minimum(
                reference_error,
                np.maximum(reference_fraction - lower_limit, 0.0),
            )
            axis.errorbar(
                orders,
                reference_fraction[orders],
                yerr=np.vstack(
                    [reference_lower_error[orders], reference_error[orders]]
                ),
                fmt="x",
                color="tab:red",
                capsize=3,
                label="piecewise-linear reference",
            )
        axis.set_title(title)
        axis.set_xlabel("angular order n")
        axis.set_ylabel(r"relative power $P_n^2/P_{\max}^2$")
        axis.set_xticks(orders)
        axis.set_yscale("log")
        axis.set_ylim(lower_limit, upper_limit)
        axis.grid(axis="y", alpha=0.25)
        if orders.size:
            axis.legend(fontsize=8)

    for axis, orders, statistic, valid, color, title in [
        (
            cosine_test_axis,
            cosine_orders,
            block_tests.cosine_statistic,
            block_tests.cosine_degrees_of_freedom > 0,
            "tab:blue",
            "cosine angular block statistics",
        ),
        (
            sine_test_axis,
            sine_orders,
            block_tests.sine_statistic,
            block_tests.sine_degrees_of_freedom > 0,
            "tab:orange",
            "sine angular block statistics",
        ),
    ]:
        even = valid & (block_tests.orders % 2 == 0)
        if np.any(even):
            axis.plot(
                block_tests.orders[even],
                statistic[even],
                "o",
                color=color,
                label="even orders",
            )
        odd = valid & (block_tests.orders % 2 == 1)
        if np.any(odd):
            axis.plot(
                block_tests.orders[odd],
                statistic[odd],
                "o",
                color="0.75",
                markeredgecolor="0.55",
                label="odd orders",
            )
        axis.set_title(title)
        axis.set_xlabel("angular order n")
        axis.set_ylabel(r"$T_n$")
        axis.set_xticks(orders)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
        if np.any(valid):
            axis.legend(fontsize=8)

    m_max = max(m for _, m in fit.indices)
    max_fitted_order = angular_orders[-1]
    cosine_map = np.zeros((m_max + 1, max_fitted_order + 1))
    sine_map = np.zeros_like(cosine_map)
    mode_power_scale = float(np.max(fit.coefficients**2))
    for coefficient, (n, m) in zip(fit.coefficients, fit.indices):
        if n >= 0:
            cosine_map[m, n] = coefficient**2 / mode_power_scale
        else:
            sine_map[m, abs(n)] = coefficient**2 / mode_power_scale
    positive_map = np.concatenate(
        [cosine_map[cosine_map > 0.0], sine_map[sine_map > 0.0]]
    )
    map_norm = LogNorm(
        vmin=max(float(np.min(positive_map)) / 2.0, 1.0e-14), vmax=1.0
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
        (cosine_map_axis, cosine_orders, "cosine $(n,m)$ powers"),
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

    assert spectra.cosine_uncertainty is not None
    assert spectra.sine_uncertainty is not None
    colors = plt.get_cmap("tab10")
    for order in cosine_orders:
        color = colors(int(order) % 10)
        cosine_spectra_axis.plot(
            spectra.q,
            spectra.cosine[order],
            label=(f"n={order}" if reference is None else f"n={order} polynomial"),
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
        if reference is not None:
            cosine_spectra_axis.plot(
                reference.spectra.q,
                reference.spectra.cosine[order],
                "--",
                label=f"n={order} polygonal",
                color=color,
            )
            cosine_spectra_axis.fill_between(
                reference.spectra.q,
                reference.spectra.cosine[order]
                - reference.spectra.cosine_uncertainty[order],
                reference.spectra.cosine[order]
                + reference.spectra.cosine_uncertainty[order],
                color=color,
                alpha=0.08,
                linewidth=0.0,
            )
    for order in sine_orders:
        color = colors(int(order) % 10)
        sine_spectra_axis.plot(
            spectra.q,
            spectra.sine[order],
            label=(f"n={order}" if reference is None else f"n={order} polynomial"),
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
        if reference is not None:
            sine_spectra_axis.plot(
                reference.spectra.q,
                reference.spectra.sine[order],
                "--",
                label=f"n={order} polygonal",
                color=color,
            )
            sine_spectra_axis.fill_between(
                reference.spectra.q,
                reference.spectra.sine[order]
                - reference.spectra.sine_uncertainty[order],
                reference.spectra.sine[order]
                + reference.spectra.sine_uncertainty[order],
                color=color,
                alpha=0.08,
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
    centering_summary = ""
    if result.beam_centering_method is not None:
        centering_summary = (
            "\nbeam centre: "
            + result.beam_centering_method
            + f"   delta q=({result.center_shift_q_z:.3e}, "
            + f"{result.center_shift_q_y:.3e})"
        )
    reference_summary = ""
    if reference is not None:
        reference_summary = (
            f"\npiecewise-linear reference: {reference.knots.size - 1} radial intervals"
        )
    figure.suptitle(
        "Mode analysis "
        f"({basis_label}: {len(fit.indices)} total, "
        f"{angular_mode_count} angular modes; "
        f"radial measure q^{result.radial_measure_exponent:g} dq)\n"
        + f"even power fraction: {result.parity_fractions.even:.6f}   "
        + f"odd power fraction: {result.parity_fractions.odd:.6f}\n"
        + r"fit parity $\langle[I(q,\theta)-I(q,\theta+\pi)]^2\rangle$: "
        + f"{result.parity_mismatch.mean_square:.3e}   "
        + f"relative RMS: {result.parity_mismatch.normalized_rms:.4f}"
        + centering_summary
        + reference_summary,
        fontsize=12,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
