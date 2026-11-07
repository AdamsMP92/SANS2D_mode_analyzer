import numpy as np
import pytest

from sans2d_mode_analyzer import evaluate_expansion, standard_analyzer


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
    assert result.powers.cosine_uncertainty is not None
    assert result.spectra.q.shape == (300,)
    assert result.spectra.cosine.shape == (3, 300)
    assert result.spectra.cosine_uncertainty is not None
    assert result.block_tests.cosine_statistic.shape == (3,)
    assert result.parity_mismatch.mean_square >= 0.0
    assert set(result.selected_indices_by_threshold) == {0.95, 0.98, 0.99}
    assert all(result.selected_indices_by_threshold.values())
    assert result.reconstruction_threshold == 0.95
    assert result.reconstruction_indices == result.selected_indices_by_threshold[0.95]
    assert result.reconstruction_image.shape == intensity.shape
    assert result.reconstruction_spectra.q.shape == (300,)
    np.testing.assert_allclose(
        result.parity_fractions.even + result.parity_fractions.odd,
        1.0,
    )


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


def test_standard_analyzer_accepts_custom_reconstruction_threshold(tmp_path):
    axis = np.linspace(-1.0, 1.0, 25)
    q_z, q_y = np.meshgrid(axis, axis)
    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    inside = (q >= 0.2) & (q <= 0.8)
    intensity = np.full(q.shape, np.nan)
    intensity[inside] = evaluate_expansion(
        q[inside],
        theta[inside],
        coefficients=[3.0, 1.0],
        indices=[(0, 0), (2, 0)],
        q_min=0.2,
        q_max=0.8,
    )
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(0.2, 0.8),
        mode_limits=(2, 0),
        reconstruction_threshold=0.80,
        output_path=tmp_path / "custom-reconstruction.png",
    )
    assert result.reconstruction_threshold == 0.80
    assert result.reconstruction_indices == ((0, 0),)


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


def test_standard_analyzer_rejects_mixed_pair_and_legacy_arguments(tmp_path):
    with pytest.raises(ValueError, match="either q_range"):
        standard_analyzer(
            np.array([0.2, -0.2, 0.3, -0.3]),
            np.array([0.1, -0.1, -0.1, 0.1]),
            np.ones(4),
            q_range=(0.1, 0.5),
            q_min=0.1,
            q_max=0.5,
            mode_limits=(0, 0),
            output_path=tmp_path / "unused.png",
        )


def test_standard_analyzer_keeps_legacy_separate_keywords(tmp_path):
    axis = np.linspace(-1.0, 1.0, 21)
    q_z, q_y = np.meshgrid(axis, axis)
    q = np.hypot(q_z, q_y)
    intensity = np.where((q >= 0.2) & (q <= 0.8), 2.0 + 0.2 * q_z, np.nan)
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_min=0.2,
        q_max=0.8,
        n_max=1,
        m_max=1,
        output_path=tmp_path / "legacy.png",
    )
    assert result.output_path.exists()
