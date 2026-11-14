"""Stage-4 regression test for correlated Gaussian detector noise."""

import numpy as np
from scipy.stats import chi2

from sans2d_mode_analyzer import (
    design_matrix,
    fit_design_matrix,
    rectangular_indices,
)
from synthetic_models import generate_random_modal_detector_data


def test_correlated_noise_requires_the_full_sampling_covariance():
    """True covariance calibrates powers; marginal errors alone do not."""
    q_min, q_max = 0.05, 0.30
    n_max, m_max = 2, 2
    number_of_realizations = 1024

    q_z, q_y, exact_image, generated_indices, generated_coefficients = (
        generate_random_modal_detector_data(
            q_min,
            q_max,
            n_max,
            m_max,
            n_coeffs=3,
            angular_parity="even",
            number_of_pixels=25,
            index_seed=71,
            coefficient_seed=31,
            return_model=True,
        )
    )
    selected = np.isfinite(exact_image)
    q = np.hypot(q_z[selected], q_y[selected])
    theta = np.arctan2(q_y[selected], q_z[selected])
    exact_values = exact_image[selected]

    fit_indices = rectangular_indices(n_max=n_max, m_max=m_max)
    matrix = design_matrix(
        q,
        theta,
        fit_indices,
        q_min=q_min,
        q_max=q_max,
    )
    true_coefficients = np.zeros(len(fit_indices))
    positions = {index: position for position, index in enumerate(fit_indices)}
    for index, coefficient in zip(
        map(tuple, generated_indices), generated_coefficients
    ):
        true_coefficients[positions[index]] = coefficient
    np.testing.assert_allclose(
        exact_values,
        matrix @ true_coefficients,
        atol=2.0e-14,
    )

    intensity_scale = np.max(np.abs(exact_values))
    independent_sigma = np.full(exact_values.shape, 0.020 * intensity_scale)
    pattern_matrix = np.column_stack(
        [
            np.full(exact_values.shape, 0.010 * intensity_scale),
            0.030 * intensity_scale * q_y[selected] / q_max,
        ]
    )
    marginal_sigma = np.sqrt(
        independent_sigma**2 + np.sum(pattern_matrix**2, axis=1)
    )

    # The public fit sees only marginal errors and therefore applies a
    # diagonal working covariance to this correlated data-generating model.
    weighted_matrix = matrix / marginal_sigma[:, None]
    orthogonal, triangular = np.linalg.qr(weighted_matrix, mode="reduced")
    response = np.linalg.solve(
        triangular,
        orthogonal.T / marginal_sigma[None, :],
    )
    diagonal_response = response * marginal_sigma[None, :]
    reported_covariance = diagonal_response @ diagonal_response.T

    independent_response = response * independent_sigma[None, :]
    correlated_response = response @ pattern_matrix
    true_covariance = (
        independent_response @ independent_response.T
        + correlated_response @ correlated_response.T
    )

    random = np.random.default_rng(7000)
    fitted_coefficients = np.empty(
        (number_of_realizations, len(fit_indices))
    )
    first_fit = None
    for realization in range(number_of_realizations):
        noise = (
            random.normal(size=exact_values.size) * independent_sigma
            + pattern_matrix @ random.normal(size=pattern_matrix.shape[1])
        )
        fit = fit_design_matrix(
            exact_values + noise,
            fit_indices,
            matrix,
            uncertainty=marginal_sigma,
            absolute_uncertainty=True,
        )
        fitted_coefficients[realization] = fit.coefficients
        if first_fit is None:
            first_fit = fit

    assert first_fit is not None
    np.testing.assert_allclose(
        first_fit.coefficient_covariance,
        reported_covariance,
        rtol=2.0e-12,
        atol=1.0e-15,
    )

    zero_mode = true_coefficients == 0.0
    coefficient_errors = (
        fitted_coefficients[:, zero_mode] - true_coefficients[zero_mode]
    )
    true_variance = np.diag(true_covariance)[zero_mode]
    reported_variance = np.diag(reported_covariance)[zero_mode]
    true_normalized_power = coefficient_errors**2 / true_variance

    # With the full sampling covariance, every zero-mode marginal again has
    # the chi-square(1) moments and coverage derived in Stages 2 and 3.
    mean_tolerance = 5.0 / np.sqrt(number_of_realizations)
    power_mean_tolerance = 5.0 * np.sqrt(
        2.0 / number_of_realizations
    )
    standardized_errors = coefficient_errors / np.sqrt(true_variance)
    assert np.all(np.abs(np.mean(standardized_errors, axis=0)) < mean_tolerance)
    assert np.all(
        np.abs(np.mean(true_normalized_power, axis=0) - 1.0)
        < power_mean_tolerance
    )

    power_variance_tolerance = 5.0 * np.sqrt(
        56.0 / number_of_realizations
    )
    assert np.all(
        np.abs(np.var(true_normalized_power, axis=0, ddof=1) - 2.0)
        < power_variance_tolerance
    )

    probability = 0.95
    true_coverage = np.mean(
        true_normalized_power <= chi2.ppf(probability, df=1),
        axis=0,
    )
    coverage_tolerance = 5.0 * np.sqrt(
        probability * (1.0 - probability) / number_of_realizations
    ) + 1.0 / number_of_realizations
    assert np.all(np.abs(true_coverage - probability) < coverage_tolerance)

    # The smooth q_y background is coherent across pixels and strongly
    # excites one parity-odd mode.  Treating only its marginal pixel variance
    # as independent noise must remain detectably miscalibrated.
    variance_ratio = true_variance / reported_variance
    target = int(np.argmax(variance_ratio))
    assert variance_ratio[target] > 20.0
    reported_normalized_power = (
        coefficient_errors[:, target] ** 2 / reported_variance[target]
    )
    reported_coverage = np.mean(
        reported_normalized_power <= chi2.ppf(probability, df=1)
    )
    assert reported_coverage < 0.5
