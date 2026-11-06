"""Standard full-rectangular fit and compact PNG summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .block_test import AngularBlockTests, angular_block_tests
from .fit import MSEFitResult, fit_mse
from .indices import ModeIndex, rectangular_indices
from .power import (
    AngularModePowers,
    ParityPowerFractions,
    angular_mode_powers,
    parity_power_fractions,
    power_model_selector,
)
from .spectra import angular_radial_spectra


@dataclass(frozen=True)
class StandardAnalysisResult:
    """Numerical results and detector images from ``standard_analyzer``."""

    fit: MSEFitResult
    powers: AngularModePowers
    parity_fractions: ParityPowerFractions
    block_tests: AngularBlockTests
    input_image: NDArray[np.float64]
    fit_image: NDArray[np.float64]
    residual_image: NDArray[np.float64]
    selected_indices_by_threshold: dict[float, tuple[ModeIndex, ...]]
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
    q_min: float,
    q_max: float,
    n_max: int,
    m_max: int,
    power_thresholds: tuple[float, ...] = (0.95, 0.98, 0.99),
    block_test_confidence: float = 0.95,
    output_path: str | Path = "standard_analysis.png",
) -> StandardAnalysisResult:
    """Fit a full rectangular mode grid and save a multi-panel PNG summary."""
    import matplotlib.pyplot as plt

    q_z, q_y, intensity = np.broadcast_arrays(
        np.asarray(q_z, dtype=float),
        np.asarray(q_y, dtype=float),
        np.asarray(intensity, dtype=float),
    )
    if intensity.ndim not in (1, 2):
        raise ValueError(
            "standard_analyzer expects a one-dimensional detector point list "
            "or two-dimensional detector arrays"
        )

    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    selected = np.isfinite(intensity) & (q >= q_min) & (q <= q_max)
    indices = rectangular_indices(n_max=n_max, m_max=m_max)

    fit = fit_mse(
        q[selected],
        theta[selected],
        intensity[selected],
        indices,
        q_min=q_min,
        q_max=q_max,
    )
    powers = angular_mode_powers(
        fit.coefficients,
        fit.indices,
        fit.coefficient_covariance,
    )
    parity_fractions = parity_power_fractions(fit.coefficients, fit.indices)
    block_tests = angular_block_tests(
        fit.coefficients,
        fit.coefficient_covariance,
        fit.indices,
        residual_degrees_of_freedom=fit.degrees_of_freedom,
        confidence=block_test_confidence,
    )
    selected_indices_by_threshold = {
        threshold: tuple(
            power_model_selector(fit.coefficients, fit.indices, threshold)
        )
        for threshold in power_thresholds
    }

    input_image = np.full(intensity.shape, np.nan)
    fit_image = np.full(intensity.shape, np.nan)
    residual_image = np.full(intensity.shape, np.nan)
    input_image[selected] = intensity[selected]
    fit_image[selected] = fit.fitted_values
    residual_image[selected] = fit.residuals

    figure = plt.figure(figsize=(14.0, 18.5), constrained_layout=True)
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

        input_plot = map_axes[0].imshow(
            display_input,
            origin="lower",
            extent=extent,
            vmin=common_min,
            vmax=common_max,
            cmap="viridis",
            interpolation="nearest",
        )
        map_axes[1].imshow(
            display_fit,
            origin="lower",
            extent=extent,
            vmin=common_min,
            vmax=common_max,
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
        input_plot = map_axes[0].scatter(
            q_z[selected],
            q_y[selected],
            c=input_image[selected],
            s=5,
            linewidths=0,
            vmin=common_min,
            vmax=common_max,
            cmap="viridis",
        )
        map_axes[1].scatter(
            q_z[selected],
            q_y[selected],
            c=fit_image[selected],
            s=5,
            linewidths=0,
            vmin=common_min,
            vmax=common_max,
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
    figure.colorbar(input_plot, ax=map_axes[:2], label="intensity")
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
    cosine_even = powers.orders % 2 == 0
    cosine_odd = ~cosine_even
    sine_orders = powers.orders[1:]
    sine_even = sine_orders % 2 == 0
    sine_odd = ~sine_even

    for mask, color, label in [
        (cosine_even, "tab:blue", "even orders"),
        (cosine_odd, "0.75", "odd orders"),
    ]:
        cosine_axis.errorbar(
            powers.orders[mask],
            cosine_fraction[mask],
            yerr=np.vstack(
                [cosine_lower_error[mask], cosine_uncertainty[mask]]
            ),
            fmt="o",
            color=color,
            markeredgecolor="0.55" if label == "odd orders" else color,
            capsize=3,
            label=label,
        )
    for mask, color, label in [
        (sine_even, "tab:orange", "even orders"),
        (sine_odd, "0.75", "odd orders"),
    ]:
        sine_axis.errorbar(
            sine_orders[mask],
            sine_fraction[1:][mask],
            yerr=np.vstack(
                [sine_lower_error[1:][mask], sine_uncertainty[1:][mask]]
            ),
            fmt="o",
            color=color,
            markeredgecolor="0.55" if label == "odd orders" else color,
            capsize=3,
            label=label,
        )

    for axis, title in zip(
        [cosine_axis, sine_axis], ["cosine angular powers", "sine angular powers"]
    ):
        axis.set_title(title)
        axis.set_xlabel("angular order n")
        axis.set_ylabel(r"relative power $P_n^2/P_{\max}^2$")
        axis.set_xticks(powers.orders)
        axis.set_yscale("log")
        axis.set_ylim(lower_limit, upper_limit)
        axis.grid(axis="y", alpha=0.25)
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
        odd = valid & (block_tests.orders % 2 == 1)
        axis.plot(
            block_tests.orders[even],
            statistic[even],
            "o",
            color=even_color,
            label="even orders",
        )
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
            label=f"{block_test_confidence:.0%} threshold",
        )
        axis.set_title(title)
        axis.set_xlabel("angular order n")
        axis.set_ylabel(r"$T_n$")
        axis.set_xticks(block_tests.orders)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)

    cosine_map = np.zeros((m_max + 1, n_max + 1))
    sine_map = np.zeros((m_max + 1, n_max + 1))
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
        extent=(-0.5, n_max + 0.5, -0.5, m_max + 0.5),
        norm=map_norm,
        cmap="viridis",
    )
    sine_map_axis.imshow(
        np.ma.masked_less_equal(sine_map[:, 1:], 0.0),
        origin="lower",
        aspect="auto",
        extent=(0.5, n_max + 0.5, -0.5, m_max + 0.5),
        norm=map_norm,
        cmap="viridis",
    )
    for axis, title in zip(
        [cosine_map_axis, sine_map_axis],
        ["cosine $(n,m)$ powers", "sine $(n,m)$ powers"],
    ):
        axis.set_title(title)
        axis.set_xlabel("angular order n")
        axis.set_ylabel("radial order m")
        axis.set_xticks(range(0 if axis is cosine_map_axis else 1, n_max + 1))
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
    for order in spectra.orders:
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
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=7)

    threshold_summary = "   ".join(
        f"[{threshold:.0%}: "
        f"{len(selected_indices_by_threshold[threshold])} total, "
        f"{len({n for n, _ in selected_indices_by_threshold[threshold]})} angular]"
        for threshold in power_thresholds
    )
    full_angular_mode_count = len({n for n, _ in fit.indices})
    figure.suptitle(
        "Power-ranked model reduction "
        f"(full basis: {len(fit.indices)} total, "
        f"{full_angular_mode_count} angular modes)\n"
        + threshold_summary
        + "\n"
        + f"even power fraction: {parity_fractions.even:.6f}   "
        + f"odd power fraction: {parity_fractions.odd:.6f}",
        fontsize=13,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return StandardAnalysisResult(
        fit=fit,
        powers=powers,
        parity_fractions=parity_fractions,
        block_tests=block_tests,
        input_image=input_image,
        fit_image=fit_image,
        residual_image=residual_image,
        selected_indices_by_threshold=selected_indices_by_threshold,
        output_path=path,
    )
