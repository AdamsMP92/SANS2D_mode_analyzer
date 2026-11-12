from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2, f

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
    assert result.holm_family_size == 0
    assert np.all(np.isnan(result.cosine_holm_pvalue))
    assert np.all(np.isnan(result.sine_holm_pvalue))
    assert not np.any(result.cosine_holm_reject)
    assert not np.any(result.sine_holm_reject)


@pytest.mark.parametrize("residual_dof", [None, 23])
@pytest.mark.parametrize("confidence", [0.95, 0.90])
def test_holm_corrects_cosine_and_sine_as_one_family(residual_dof, confidence):
    probabilities = np.array([0.01, 0.04, 0.03])
    statistic = (
        chi2.isf(probabilities, 1) if residual_dof is None
        else f.isf(probabilities, 1, residual_dof)
    )
    result = angular_block_tests(
        np.sqrt(statistic), np.eye(3), [(0, 0), (2, 0), (-1, 0)],
        residual_degrees_of_freedom=residual_dof, confidence=confidence,
    )
    np.testing.assert_allclose(result.cosine_pvalue[[0, 2]], probabilities[:2])
    np.testing.assert_allclose(result.sine_pvalue[1], probabilities[2])
    assert result.holm_family_size == 3
    np.testing.assert_allclose(result.cosine_holm_pvalue[[0, 2]], [0.03, 0.06])
    np.testing.assert_allclose(result.sine_holm_pvalue[1], 0.06)
    expected = np.array([0.03, 0.06, 0.06]) <= 1 - confidence
    np.testing.assert_array_equal(result.cosine_holm_reject[[0, 2]], expected[:2])
    assert result.sine_holm_reject[1] == expected[2]


def test_holm_excludes_missing_and_rank_zero_blocks():
    probabilities = np.array([0.015, 0.2])
    result = angular_block_tests(
        np.r_[np.sqrt(chi2.isf(probabilities, 1)), 0.0],
        np.diag([1.0, 1.0, 0.0]), [(2, 0), (-4, 0), (0, 0)],
    )
    assert result.holm_family_size == 2
    np.testing.assert_allclose(result.cosine_holm_pvalue[2], 0.03)
    np.testing.assert_allclose(result.sine_holm_pvalue[4], 0.2)
    assert result.cosine_holm_reject[2]
    assert not result.sine_holm_reject[4]
    assert np.all(np.isnan(result.cosine_holm_pvalue[[0, 1, 3, 4]]))
    assert np.all(np.isnan(result.sine_holm_pvalue[:4]))
    assert not np.any(result.cosine_holm_reject[[0, 1, 3, 4]])


@pytest.mark.parametrize(
    "probabilities,expected",
    [
        ([0.01, 0.01, 0.04], [0.03, 0.03, 0.04]),
        ([1.0, 0.0, 0.4], [1.0, 0.0, 0.8]),
        ([np.nan, 0.01, 0.2], [np.nan, 0.03, 0.4]),
        ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]),
    ],
)
def test_holm_ties_extreme_and_unavailable_probabilities(probabilities, expected):
    # Replacing raw fields also checks compatibility of the original result
    # constructor: all new information is available as derived properties.
    base = angular_block_tests([1.0, 1.0, 1.0], np.eye(3), [(0, 0), (2, 0), (-1, 0)])
    cosine, sine = base.cosine_pvalue.copy(), base.sine_pvalue.copy()
    cosine[[0, 2]], sine[1] = probabilities[:2], probabilities[2]
    result = replace(base, cosine_pvalue=cosine, sine_pvalue=sine)
    assert result.holm_family_size == 3
    np.testing.assert_allclose(result.cosine_holm_pvalue[[0, 2]], expected[:2])
    np.testing.assert_allclose(result.sine_holm_pvalue[1], expected[2])
    np.testing.assert_array_equal(
        result.cosine_holm_reject[[0, 2]], np.array(expected[:2]) <= 1 - result.confidence
    )
    assert result.sine_holm_reject[1] == (expected[2] <= 1 - result.confidence)


def test_holm_singleton_rejection_includes_alpha_boundary():
    base = angular_block_tests([1.0], [[1.0]], [(0, 0)])
    alpha = 1 - base.confidence
    boundary = replace(base, cosine_pvalue=np.array([alpha]))
    above = replace(base, cosine_pvalue=np.array([np.nextafter(alpha, 1.0)]))
    assert boundary.holm_family_size == 1
    assert boundary.cosine_holm_reject[0]
    assert not above.cosine_holm_reject[0]
