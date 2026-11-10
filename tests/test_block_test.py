import numpy as np
import pytest

from sans2d_mode_analyzer import angular_block_tests


def test_angular_block_tests_group_radial_coefficients_by_branch():
    indices = [(0, 0), (0, 1), (2, 0), (-1, 0)]
    coefficients = np.array([3.0, 4.0, 2.0, 1.0])
    covariance = np.diag([0.25, 1.0, 0.25, 0.25])
    tests = angular_block_tests(
        coefficients,
        covariance,
        indices,
        residual_degrees_of_freedom=1000,
    )

    np.testing.assert_allclose(tests.cosine_statistic, [52.0, np.nan, 16.0])
    np.testing.assert_allclose(tests.sine_statistic, [np.nan, 4.0, np.nan])
    assert tests.cosine_degrees_of_freedom.tolist() == [2, 0, 1]
    assert tests.sine_degrees_of_freedom.tolist() == [0, 1, 0]
    assert tests.cosine_pvalue[0] < 1.0e-8
    assert tests.cosine_statistic[2] > tests.cosine_critical_value[2]


@pytest.mark.parametrize("scale", [2.0**-40, 1.0, 2.0**40])
@pytest.mark.parametrize("relative_to_cutoff,expected_rank", [(0.5, 1), (1.0, 1), (2.0, 2)])
def test_block_cutoff_is_relative_strict_and_scale_invariant(scale, relative_to_cutoff, expected_rank):
    eigenvalues = scale * np.array([1.0, relative_to_cutoff * 2 * np.finfo(float).eps])
    result = angular_block_tests(np.sqrt(eigenvalues), np.diag(eigenvalues), [(2, 0), (2, 1)])
    assert result.cosine_degrees_of_freedom[2] == expected_rank
    np.testing.assert_allclose(result.cosine_statistic[2], expected_rank, rtol=2e-14)


def test_block_truncation_uses_covariance_directions_for_correlated_coefficients():
    rotation, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    eigenvalues = np.array([1.0, 1e-5, 1e-18])
    covariance = (rotation * eigenvalues) @ rotation.T
    coefficients = rotation @ np.sqrt(eigenvalues)
    result = angular_block_tests(coefficients, covariance, [(-2, 0), (-2, 1), (-2, 2)])
    assert result.sine_degrees_of_freedom[2] == 2
    np.testing.assert_allclose(result.sine_statistic[2], 2.0, rtol=1e-10)


@pytest.mark.parametrize("residual_dof", [None, 37])
def test_zero_covariance_has_no_calibrated_block_test(residual_dof):
    result = angular_block_tests(
        [1.0, 2.0], np.zeros((2, 2)), [(2, 0), (2, 1)],
        residual_degrees_of_freedom=residual_dof,
    )
    assert result.cosine_statistic[2] == 0.0
    assert result.cosine_degrees_of_freedom[2] == 0
    assert np.isnan(result.cosine_pvalue[2])
    assert np.isnan(result.cosine_critical_value[2])
