import numpy as np

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
