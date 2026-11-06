import numpy as np

from sans2d_mode_analyzer import (
    DetectorData,
    apply_center_correction,
    apply_q_center_correction,
    center_point_cloud_by_parity,
    fractional_annulus_weights,
    mask_symmetry_diagnostic,
    pixel_integrated_annulus_design,
    refine_center_by_parity,
    rotate_coordinates,
    symmetry_diagnostic,
    symmetry_fold,
    valid_pixel_mask,
)


def detector_grid(size=41, limit=1.0):
    axis = np.linspace(-limit, limit, size)
    q_z, q_y = np.meshgrid(axis, axis)
    uncertainty = np.full(q_z.shape, 0.05)
    return q_z, q_y, uncertainty


def test_valid_mask_retains_finite_nonpositive_intensity():
    q_z, q_y, uncertainty = detector_grid(size=3)
    intensity = np.arange(9, dtype=float).reshape(3, 3) - 5.0
    data = DetectorData(q_z, q_y, intensity, uncertainty)
    assert np.all(valid_pixel_mask(data))


def test_fractional_annulus_mask_has_partial_pixels_and_no_directional_bias():
    q_z, q_y, _ = detector_grid(size=41)
    weights = fractional_annulus_weights(
        q_z, q_y, q_min=0.25, q_max=0.8, subpixels=7
    )
    assert np.any((weights > 0.0) & (weights < 1.0))
    diagnostic = mask_symmetry_diagnostic(q_z, q_y, weights)
    assert diagnostic.normalized_centroid < 2.0e-5
    assert abs(diagnostic.mean_reflection_residual) < 1.0e-12


def test_pixel_integrated_design_tracks_fractional_annulus_coverage():
    q_z, q_y, _ = detector_grid(size=21)
    integrated = pixel_integrated_annulus_design(
        q_z,
        q_y,
        [(0, 0), (2, 1)],
        q_min=0.25,
        q_max=0.8,
        subpixels=3,
        chunk_size=32,
    )
    assert integrated.design_matrix.shape == (
        np.count_nonzero(integrated.selected_mask),
        2,
    )
    assert np.any(
        (integrated.coverage > 0.0) & (integrated.coverage < 1.0)
    )
    assert np.all(np.isfinite(integrated.design_matrix))


def test_coordinate_rotation_changes_only_coordinate_convention():
    q_z, q_y, uncertainty = detector_grid(size=5)
    intensity = q_z + 2.0 * q_y
    data = DetectorData(q_z, q_y, intensity, uncertainty)
    rotated = rotate_coordinates(data, np.pi / 2.0)
    np.testing.assert_allclose(rotated.q_z, q_y, atol=1e-15)
    np.testing.assert_allclose(rotated.q_y, -q_z, atol=1e-15)
    np.testing.assert_allclose(rotated.intensity, intensity)


def test_direct_q_center_correction_supports_point_lists():
    data = DetectorData(
        q_z=np.array([-0.2, 0.3]),
        q_y=np.array([0.1, -0.4]),
        intensity=np.array([-2.0, 3.0]),
        uncertainty=np.array([0.2, 0.3]),
    )
    corrected = apply_q_center_correction(data, 0.01, -0.02)
    np.testing.assert_allclose(corrected.q_z, [-0.21, 0.29])
    np.testing.assert_allclose(corrected.q_y, [0.12, -0.38])
    np.testing.assert_allclose(corrected.intensity, data.intensity)


def test_parity_fold_removes_an_odd_angular_component():
    q_z, q_y, uncertainty = detector_grid(size=41)
    theta = np.arctan2(q_y, q_z)
    intensity = 4.0 + 0.6 * np.cos(2.0 * theta) + 0.8 * np.cos(theta)
    data = DetectorData(q_z, q_y, intensity, uncertainty)
    before = symmetry_diagnostic(data, symmetry="parity", q_min=0.2, q_max=0.9)
    folded = symmetry_fold(data, symmetries=("parity",))
    after = symmetry_diagnostic(folded, symmetry="parity", q_min=0.2, q_max=0.9)
    assert before.chi_square > 1.0e3
    assert after.chi_square < 1.0e-18


def test_parity_center_refinement_recovers_subpixel_shift():
    q_z, q_y, uncertainty = detector_grid(size=41)
    step = q_z[0, 1] - q_z[0, 0]
    expected_row = 0.22
    expected_column = -0.31
    delta_q_z = expected_column * step
    delta_q_y = expected_row * step
    corrected_q_z = q_z - delta_q_z
    corrected_q_y = q_y - delta_q_y
    intensity = np.exp(
        -2.0 * corrected_q_z**2 - 4.0 * corrected_q_y**2
    )
    data = DetectorData(q_z, q_y, intensity, uncertainty)
    refinement = refine_center_by_parity(
        data,
        q_min=0.2,
        q_max=0.8,
        max_shift_pixels=0.8,
    )
    assert refinement.objective_after < refinement.objective_before
    np.testing.assert_allclose(refinement.delta_row, expected_row, atol=0.04)
    np.testing.assert_allclose(refinement.delta_column, expected_column, atol=0.04)
    corrected = apply_center_correction(
        data, refinement.delta_row, refinement.delta_column
    )
    diagnostic = symmetry_diagnostic(
        corrected, symmetry="parity", q_min=0.2, q_max=0.8
    )
    assert diagnostic.chi_square < refinement.objective_before


def test_point_cloud_centering_wrapper_preserves_original_samples():
    q_z, q_y, uncertainty = detector_grid(size=31)
    step = q_z[0, 1] - q_z[0, 0]
    delta_q_z = -0.24 * step
    delta_q_y = 0.18 * step
    intensity = np.exp(
        -2.0 * (q_z - delta_q_z) ** 2 - 4.0 * (q_y - delta_q_y) ** 2
    )
    point_data = DetectorData(
        q_z=q_z.ravel(),
        q_y=q_y.ravel(),
        intensity=intensity.ravel(),
        uncertainty=uncertainty.ravel(),
    )
    result = center_point_cloud_by_parity(
        point_data,
        q_min=0.2,
        q_max=0.8,
        grid_size=31,
        max_shift_pixels=1.0,
    )
    np.testing.assert_allclose(result.delta_q_z, delta_q_z, atol=0.006)
    np.testing.assert_allclose(result.delta_q_y, delta_q_y, atol=0.006)
    np.testing.assert_allclose(result.centered_data.intensity, point_data.intensity)
    np.testing.assert_allclose(
        result.centered_data.uncertainty, point_data.uncertainty
    )
