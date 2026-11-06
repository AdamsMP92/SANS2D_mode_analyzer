import numpy as np
import pytest

from sans2d_mode_analyzer import (
    angular_mode_powers,
    basis_function,
    design_matrix,
    evaluate_expansion,
    fit_mse,
    fit_design_matrix,
    fit_wls,
    parity_power_fractions,
    parity_mismatch_measures,
    power_model_selector,
    rectangular_indices,
)

Q_MIN = 0.015
Q_MAX = 0.050


def test_continuous_basis_is_numerically_orthonormal():
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(48)
    angular_nodes = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    q_squared = 0.5 * (
        (Q_MAX**2 - Q_MIN**2) * radial_nodes + Q_MAX**2 + Q_MIN**2
    )
    q = np.sqrt(q_squared)
    q_grid, theta_grid = np.meshgrid(q, angular_nodes, indexing="ij")
    indices = rectangular_indices(n_max=3, m_max=2)
    matrix = design_matrix(
        q_grid, theta_grid, indices, q_min=Q_MIN, q_max=Q_MAX
    ).reshape(-1, len(indices))
    weights = np.repeat(
        radial_weights * (Q_MAX**2 - Q_MIN**2) / 4.0,
        angular_nodes.size,
    )
    weights *= 2.0 * np.pi / angular_nodes.size
    gram = matrix.T @ (weights[:, None] * matrix)
    np.testing.assert_allclose(gram, np.eye(len(indices)), atol=2e-13)


def test_single_mode_expansion_matches_basis_function():
    q = np.linspace(Q_MIN, Q_MAX, 9)
    theta = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    expected = 2.75 * basis_function((-2, 1), q, theta, q_min=Q_MIN, q_max=Q_MAX)
    actual = evaluate_expansion(
        q,
        theta,
        [2.75],
        [(-2, 1)],
        q_min=Q_MIN,
        q_max=Q_MAX,
    )
    np.testing.assert_allclose(actual, expected)


def test_two_column_numpy_index_list_is_accepted_directly():
    indices = np.array([[0, 0], [2, 1], [-4, 2]], dtype=int)
    matrix = design_matrix(
        [0.02, 0.03],
        [0.0, 0.5],
        indices,
        q_min=Q_MIN,
        q_max=Q_MAX,
    )
    assert matrix.shape == (2, 3)


def test_mse_fit_recovers_arbitrary_selected_modes_on_2d_grid():
    q_values = np.linspace(Q_MIN, Q_MAX, 18)
    theta_values = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    q, theta = np.meshgrid(q_values, theta_values, indexing="ij")
    indices = [(0, 0), (4, 0), (-2, 1), (6, 2)]
    coefficients = np.array([4.0, -0.7, 0.35, 1.2])
    values = evaluate_expansion(
        q, theta, coefficients, indices, q_min=Q_MIN, q_max=Q_MAX
    )
    result = fit_mse(
        q, theta, values, indices, q_min=Q_MIN, q_max=Q_MAX
    )
    assert result.indices == tuple(indices)
    np.testing.assert_allclose(result.coefficients, coefficients, atol=2e-13)
    np.testing.assert_allclose(result.fitted_values, values, atol=2e-12)
    assert result.mse < 1e-24


def test_wls_fit_uses_uncertainty_and_fractional_sample_weight():
    q_values = np.linspace(Q_MIN, Q_MAX, 12)
    theta_values = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    q, theta = np.meshgrid(q_values, theta_values, indexing="ij")
    indices = [(0, 0), (2, 1)]
    coefficients = np.array([2.0, -0.4])
    values = evaluate_expansion(
        q, theta, coefficients, indices, q_min=Q_MIN, q_max=Q_MAX
    )
    uncertainty = 0.1 + 0.2 * (q - Q_MIN) / (Q_MAX - Q_MIN)
    sample_weight = np.linspace(0.2, 1.0, values.size).reshape(values.shape)
    result = fit_wls(
        q,
        theta,
        values,
        indices,
        uncertainty,
        sample_weight=sample_weight,
        q_min=Q_MIN,
        q_max=Q_MAX,
    )
    np.testing.assert_allclose(result.coefficients, coefficients, atol=2e-13)
    assert result.weighted_sum_squares < 1e-23
    assert result.coefficient_covariance.shape == (2, 2)
    assert not result.covariance_scaled_by_residual


def test_precomputed_design_matrix_uses_same_linear_solver():
    q = np.linspace(Q_MIN, Q_MAX, 20)
    theta = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    indices = [(0, 0), (2, 1)]
    matrix = design_matrix(q, theta, indices, q_min=Q_MIN, q_max=Q_MAX)
    coefficients = np.array([1.5, -0.25])
    values = matrix @ coefficients
    result = fit_design_matrix(values, indices, matrix)
    np.testing.assert_allclose(result.coefficients, coefficients, atol=1e-13)


def test_indices_and_coefficients_must_have_matching_length():
    with pytest.raises(ValueError, match="matching indices"):
        evaluate_expansion(
            [0.02],
            [0.0],
            [1.0, 2.0],
            [(0, 0)],
            q_min=Q_MIN,
            q_max=Q_MAX,
        )


def test_duplicate_indices_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        design_matrix(
            [0.02],
            [0.0],
            [(0, 0), (0, 0)],
            q_min=Q_MIN,
            q_max=Q_MAX,
        )


def test_angular_mode_powers_separate_cosine_and_sine_branches():
    indices = [(0, 0), (0, 1), (2, 0), (-2, 0), (4, 1), (-1, 2)]
    coefficients = [3.0, 4.0, 2.0, 5.0, 6.0, 7.0]
    powers = angular_mode_powers(coefficients, indices)
    np.testing.assert_allclose(powers.cosine, [25.0, 0.0, 4.0, 0.0, 36.0])
    np.testing.assert_allclose(powers.sine, [0.0, 49.0, 25.0, 0.0, 0.0])
    assert powers.total_power == 139.0


def test_angular_power_uncertainty_includes_quadratic_gaussian_term():
    indices = [(0, 0), (0, 1)]
    coefficients = np.array([3.0, 4.0])
    covariance = np.diag([0.1, 0.2])
    powers = angular_mode_powers(coefficients, indices, covariance)
    expected_variance = 2.0 * (0.1**2 + 0.2**2)
    expected_variance += 4.0 * (3.0**2 * 0.1 + 4.0**2 * 0.2)
    np.testing.assert_allclose(
        powers.cosine_uncertainty,
        [np.sqrt(expected_variance)],
    )


def test_power_model_selector_returns_smallest_power_ranked_index_list():
    indices = [(0, 0), (2, 0), (-2, 0), (0, 1)]
    coefficients = np.sqrt([5.0, 1.0, 3.0, 1.0])
    selected = power_model_selector(coefficients, indices, threshold=0.8)
    assert selected == [(0, 0), (-2, 0)]


def test_parity_power_fractions_partition_total_power():
    indices = [(0, 0), (1, 0), (-1, 1), (2, 0)]
    coefficients = [2.0, 1.0, np.sqrt(3.0), 2.0]
    fractions = parity_power_fractions(coefficients, indices)
    assert fractions.total_power == 12.0
    np.testing.assert_allclose(fractions.even, 8.0 / 12.0)
    np.testing.assert_allclose(fractions.odd, 4.0 / 12.0)
    np.testing.assert_allclose(fractions.even + fractions.odd, 1.0)


def test_parity_mismatch_is_four_times_the_odd_parseval_power():
    indices = [(0, 0), (1, 0), (-1, 1), (2, 0)]
    coefficients = [2.0, 1.0, np.sqrt(3.0), 2.0]
    mismatch = parity_mismatch_measures(
        coefficients,
        indices,
        q_min=Q_MIN,
        q_max=Q_MAX,
    )
    expected_integral = 4.0 * 4.0
    np.testing.assert_allclose(mismatch.mean_square * mismatch.area, expected_integral)
    np.testing.assert_allclose(mismatch.normalized_mean_square, 4.0 / 3.0)
    np.testing.assert_allclose(
        mismatch.normalized_rms,
        np.sqrt(mismatch.normalized_mean_square),
    )
