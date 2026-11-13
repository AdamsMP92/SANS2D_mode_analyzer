import numpy as np
import pytest

from sans2d_mode_analyzer import (
    design_matrix,
    evaluate_expansion,
    pixel_integrated_annulus_design,
    standard_analyzer,
)


def test_standard_analyzer_fits_and_writes_summary(tmp_path):
    q_min = 0.2
    q_max = 1.0
    axis = np.linspace(-q_max, q_max, 41)
    q_z, q_y = np.meshgrid(axis, axis)
    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    inside = (q >= q_min) & (q <= q_max)

    intensity = np.full(q.shape, np.nan)
    intensity[inside] = evaluate_expansion(
        q[inside],
        theta[inside],
        coefficients=[3.0, 0.5],
        indices=[(0, 0), (2, 1)],
        q_min=q_min,
        q_max=q_max,
    )

    output = tmp_path / "summary.png"
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(q_min, q_max),
        mode_limits=(2, 1),
        output_path=output,
    )

    assert output.exists()
    assert result.fit.mse < 1.0e-24
    assert result.fit_image.shape == intensity.shape
    assert result.powers.orders.tolist() == [0, 1, 2]
    assert (1, 1) in result.fit.indices
    assert (-1, 1) in result.fit.indices
    assert result.powers.cosine_uncertainty is not None
    assert result.spectra.q.shape == (300,)
    assert result.spectra.cosine.shape == (3, 300)
    assert result.spectra.cosine_uncertainty is not None
    assert result.block_tests.cosine_statistic.shape == (3,)
    assert result.parity_mismatch.mean_square >= 0.0
    np.testing.assert_allclose(
        result.parity_fractions.even + result.parity_fractions.odd,
        1.0,
    )


def test_standard_analyzer_can_skip_figure_export(tmp_path):
    q_z = np.array([0.2, -0.2, 0.3, -0.3])
    q_y = np.array([0.1, -0.1, -0.1, 0.1])
    output = tmp_path / "not-created.png"
    result = standard_analyzer(
        q_z,
        q_y,
        np.ones(4),
        q_range=(0.1, 0.5),
        mode_limits=(0, 0),
        export_figure=False,
        output_path=output,
    )

    assert result.output_path is None
    assert not output.exists()
    assert result.spectra.q.shape == (300,)
    assert result.fit_image.shape == q_z.shape


def test_standard_analyzer_radial_measure_exponent_changes_q_weighting():
    q_min, q_max = 0.2, 1.0
    axis = np.linspace(-q_max, q_max, 81)
    q_z, q_y = np.meshgrid(axis, axis)
    q = np.hypot(q_z, q_y)
    intensity = np.where((q >= q_min) & (q <= q_max), q, np.nan)

    default = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(q_min, q_max),
        mode_limits=(0, 0),
        export_figure=False,
    )
    high_q_weighted = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(q_min, q_max),
        mode_limits=(0, 0),
        radial_measure_exponent=3.0,
        export_figure=False,
    )

    assert default.radial_measure_exponent == 1.0
    assert default.fit.radial_measure_exponent == 1.0
    assert default.sample_weight_image is None
    assert high_q_weighted.radial_measure_exponent == 3.0
    assert high_q_weighted.fit.radial_measure_exponent == 3.0
    assert high_q_weighted.spectra.radial_measure_exponent == 3.0
    assert high_q_weighted.sample_weight_image is not None
    assert np.nanmean(high_q_weighted.fit_image) > np.nanmean(default.fit_image)


def test_standard_analyzer_accepts_explicit_sparse_indices(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    figures = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda figure, *args, **kwargs: figures.append(figure),
    )
    q_min, q_max = 0.2, 1.0
    axis = np.linspace(-q_max, q_max, 41)
    q_z, q_y = np.meshgrid(axis, axis)
    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    inside = (q >= q_min) & (q <= q_max)
    selected_indices = [(0, 0), (3, 1), (-2, 2)]
    coefficients = [3.0, -0.4, 0.7]
    intensity = np.full(q.shape, np.nan)
    intensity[inside] = evaluate_expansion(
        q[inside],
        theta[inside],
        coefficients,
        selected_indices,
        q_min=q_min,
        q_max=q_max,
    )

    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(q_min, q_max),
        indices=selected_indices,
        output_path=tmp_path / "custom.png",
    )

    assert result.fit.indices == tuple(selected_indices)
    np.testing.assert_allclose(result.fit.coefficients, coefficients, atol=1e-12)
    assert any("custom basis" in title.get_text() for title in figures[0].texts)
    axes = {axis.get_title(): axis for axis in figures[0].axes}
    assert axes["cosine radial spectra $I_n^c(q)$"].get_legend_handles_labels()[1] == [
        "n=0",
        "n=3",
    ]
    assert axes["sine radial spectra $I_n^s(q)$"].get_legend_handles_labels()[1] == [
        "n=2"
    ]
    assert axes["cosine radial spectra $I_n^c(q)$"].get_ylim() == axes[
        "sine radial spectra $I_n^s(q)$"
    ].get_ylim()


def test_standard_analyzer_requires_one_mode_specification(tmp_path):
    coordinates = np.array([0.2, -0.2, 0.3, -0.3])
    transverse = np.array([0.1, -0.1, -0.1, 0.1])
    intensity = np.ones(4)
    with pytest.raises(ValueError, match="exactly one"):
        standard_analyzer(
            coordinates,
            transverse,
            intensity,
            q_range=(0.1, 0.5),
            output_path=tmp_path / "neither.png",
        )
    with pytest.raises(ValueError, match="exactly one"):
        standard_analyzer(
            coordinates,
            transverse,
            intensity,
            q_range=(0.1, 0.5),
            mode_limits=(0, 0),
            indices=[(0, 0)],
            output_path=tmp_path / "both.png",
        )
    with pytest.raises(ValueError, match="even_only"):
        standard_analyzer(
            coordinates,
            transverse,
            intensity,
            q_range=(0.1, 0.5),
            indices=[(0, 0)],
            even_only=True,
            output_path=tmp_path / "ambiguous.png",
        )


@pytest.mark.parametrize(("n_max", "signed_orders"), [(3, (0, 2, -2)), (1, (0,)), (0, (0,))])
def test_standard_analyzer_even_only_refits_and_plots_selected_orders(
    n_max, signed_orders, monkeypatch, tmp_path
):
    from matplotlib.figure import Figure

    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: figures.append(figure))
    q_min, q_max = 0.2, 1.0
    q, theta = np.meshgrid(np.linspace(0.3, 0.9, 7), np.linspace(-2.3, 2.4, 19))
    q, theta = q.ravel(), theta.ravel()
    intensity = evaluate_expansion(
        q, theta, [2.0, -0.25, 0.6, 0.4], [(0, 0), (2, 1), (-2, 1), (1, 0)],
        q_min=q_min, q_max=q_max,
    )
    uncertainty = 0.2 + q / 10.0
    result = standard_analyzer(
        q * np.cos(theta),
        q * np.sin(theta),
        intensity,
        q_range=(q_min, q_max),
        mode_limits=(n_max, 1),
        even_only=True,
        uncertainty=uncertainty,
        output_path=tmp_path / "even-only.png",
    )

    expected_indices = tuple((n, m) for m in range(2) for n in signed_orders)
    assert result.fit.indices == expected_indices
    design = design_matrix(q, theta, expected_indices, q_min=q_min, q_max=q_max)
    expected_coefficients, *_ = np.linalg.lstsq(
        design / uncertainty[:, None], intensity / uncertainty, rcond=None
    )
    np.testing.assert_allclose(result.fit.coefficients, expected_coefficients, atol=1e-13)
    assert result.parity_fractions.odd == 0.0
    assert result.parity_mismatch.normalized_rms == 0.0

    figure = figures[0]
    assert any("even only" in title.get_text() for title in figure.texts)
    assert tuple(figure.get_size_inches()) == (16.0, 20.0)
    assert len(figure.axes) == 14
    assert figure.axes[0].get_subplotspec().get_gridspec().nrows == 5
    axes = {axis.get_title(): axis for axis in figure.axes}
    for branch, orders in (
        ("cosine", [n for n in signed_orders if n >= 0]),
        ("sine", [-n for n in signed_orders if n < 0]),
    ):
        power_axis = axes[f"{branch} angular powers"]
        plotted_orders = [
            order
            for container in power_axis.containers
            for order in container.lines[0].get_xdata()
        ]
        assert plotted_orders == orders
        block_axis = axes[f"{branch} angular block statistics"]
        assert "odd orders" not in block_axis.get_legend_handles_labels()[1]
        spectrum_axis = next(
            axis for title, axis in axes.items() if title.startswith(f"{branch} radial spectra")
        )
        assert spectrum_axis.get_legend_handles_labels()[1] == [f"n={n}" for n in orders]


def test_standard_analyzer_accepts_flat_detector_point_list(tmp_path):
    q_min = 0.2
    q_max = 1.0
    q_axis = np.linspace(q_min, q_max, 12)
    theta_axis = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    q, theta = np.meshgrid(q_axis, theta_axis, indexing="ij")
    q_z = (q * np.cos(theta)).ravel()
    q_y = (q * np.sin(theta)).ravel()
    intensity = evaluate_expansion(
        q.ravel(),
        theta.ravel(),
        coefficients=[2.0, 0.25],
        indices=[(0, 0), (-2, 1)],
        q_min=q_min,
        q_max=q_max,
    )

    output = tmp_path / "point-list-summary.png"
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=[q_min, q_max],
        mode_limits=[2, 1],
        output_path=output,
    )

    assert output.exists()
    assert result.fit.mse < 1.0e-24
    assert result.fit_image.shape == intensity.shape


def test_standard_analyzer_accepts_uncertainty_and_pixel_integration(tmp_path):
    q_min = 0.2
    q_max = 1.0
    axis = np.linspace(-q_max, q_max, 25)
    q_z, q_y = np.meshgrid(axis, axis)
    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    intensity = np.full(q.shape, np.nan)
    inside = (q >= q_min) & (q <= q_max)
    intensity[inside] = evaluate_expansion(
        q[inside],
        theta[inside],
        coefficients=[3.0],
        indices=[(0, 0)],
        q_min=q_min,
        q_max=q_max,
    )
    uncertainty = np.full(q.shape, 0.2)
    output = tmp_path / "weighted-integrated.png"
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        uncertainty=uncertainty,
        q_range=(q_min, q_max),
        mode_limits=(1, 0),
        pixel_integration_subpixels=3,
        output_path=output,
    )
    assert output.exists()
    assert result.sample_weight_image is not None
    assert not result.fit.covariance_scaled_by_residual
    assert np.any(
        (result.sample_weight_image > 0.0)
        & (result.sample_weight_image < 1.0)
    )
    # Check propagation through the actual high-level pixel-integration path
    # using an independent SVD response, not just the lower-level fit helper.
    integrated = pixel_integrated_annulus_design(
        q_z, q_y, result.fit.indices, q_min=q_min, q_max=q_max, subpixels=3,
    )
    np.testing.assert_allclose(integrated.quadrature_weight, integrated.coverage)
    matrix = integrated.design_matrix[
        result.selected_mask[integrated.selected_mask]
    ]
    sigma = uncertainty[result.selected_mask]
    weight = result.sample_weight_image[result.selected_mask]
    weighted_matrix = np.sqrt(weight)[:, None] * matrix / sigma[:, None]
    noise_response = np.linalg.pinv(weighted_matrix) * np.sqrt(weight)
    expected_covariance = noise_response @ noise_response.T
    np.testing.assert_allclose(
        result.fit.coefficient_covariance, expected_covariance, rtol=2e-12, atol=1e-17,
    )
    constant = result.fit.indices.index((0, 0))
    np.testing.assert_allclose(
        result.block_tests.cosine_statistic[0],
        result.fit.coefficients[constant]**2 / expected_covariance[constant, constant],
        rtol=2e-12,
    )
    # m=0 has a constant radial function sqrt(2 / (q_max**2-q_min**2)).
    np.testing.assert_allclose(
        result.spectra.cosine_uncertainty[0]**2,
        2 * expected_covariance[constant, constant] / (q_max**2 - q_min**2),
        rtol=2e-12,
    )


def test_standard_analyzer_automatically_centers_point_cloud(tmp_path):
    axis = np.linspace(-1.0, 1.0, 25)
    q_z, q_y = np.meshgrid(axis, axis)
    step = axis[1] - axis[0]
    expected_q_z = -0.22 * step
    expected_q_y = 0.17 * step
    intensity = np.exp(
        -2.0 * (q_z - expected_q_z) ** 2
        - 4.0 * (q_y - expected_q_y) ** 2
    )
    uncertainty = np.full(intensity.shape, 0.05)
    result = standard_analyzer(
        q_z.ravel(),
        q_y.ravel(),
        intensity.ravel(),
        uncertainty=uncertainty.ravel(),
        beam_centering=True,
        beam_centering_grid_size=31,
        beam_centering_max_shift_pixels=1.0,
        q_range=(0.2, 0.8),
        mode_limits=(1, 1),
        output_path=tmp_path / "auto-centered-point-cloud.png",
    )
    assert result.beam_centering_method == "point-cloud parity"
    assert result.beam_centering_refinement is not None
    np.testing.assert_allclose(result.center_shift_q_z, expected_q_z, atol=0.007)
    np.testing.assert_allclose(result.center_shift_q_y, expected_q_y, atol=0.007)


def test_standard_analyzer_requires_uncertainty_for_beam_centering(tmp_path):
    with pytest.raises(ValueError, match="requires supplied uncertainty"):
        standard_analyzer(
            np.array([0.2, -0.2, 0.3, -0.3]),
            np.array([0.1, -0.1, -0.1, 0.1]),
            np.ones(4),
            beam_centering=True,
            q_range=(0.1, 0.5),
            mode_limits=(0, 0),
            output_path=tmp_path / "unused.png",
        )


def test_standard_analyzer_automatically_centers_detector_grid(tmp_path):
    axis = np.linspace(-1.0, 1.0, 21)
    q_z, q_y = np.meshgrid(axis, axis)
    step = axis[1] - axis[0]
    expected_q_z = 0.16 * step
    expected_q_y = -0.21 * step
    intensity = np.exp(
        -2.0 * (q_z - expected_q_z) ** 2
        - 4.0 * (q_y - expected_q_y) ** 2
    )
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        uncertainty=np.full_like(intensity, 0.05),
        beam_centering=True,
        beam_centering_max_shift_pixels=1.0,
        q_range=(0.2, 0.8),
        mode_limits=(1, 1),
        output_path=tmp_path / "auto-centered-grid.png",
    )
    assert result.beam_centering_method == "affine-grid parity"
    np.testing.assert_allclose(result.center_shift_q_z, expected_q_z, atol=0.007)
    np.testing.assert_allclose(result.center_shift_q_y, expected_q_y, atol=0.007)


def test_log_intensity_colorbar_masks_only_the_display(tmp_path):
    axis = np.linspace(-1.0, 1.0, 21)
    q_z, q_y = np.meshgrid(axis, axis)
    q = np.hypot(q_z, q_y)
    intensity = 2.0 + q_z
    inside = (q >= 0.2) & (q <= 0.8)
    intensity = np.where(inside, intensity, np.nan)
    negative_position = np.argwhere(inside)[0]
    intensity[tuple(negative_position)] = -0.25
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(0.2, 0.8),
        mode_limits=(1, 1),
        log_intensity_colorbar=True,
        output_path=tmp_path / "log-intensity.png",
    )
    assert result.output_path.exists()
    assert result.selected_mask[tuple(negative_position)]
    assert result.input_image[tuple(negative_position)] == -0.25
