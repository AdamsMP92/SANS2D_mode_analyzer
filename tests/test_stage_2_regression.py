"""Stage-2 regression test for white-noise transfer into modal powers."""

import numpy as np
from scipy.stats import chi2

from sans2d_mode_analyzer import (
    design_matrix,
    fit_design_matrix,
    rectangular_indices,
)
from synthetic_models import generate_random_modal_detector_data


def test_white_noise_zero_mode_powers_follow_analytic_prediction():
    """Repeated fits reproduce the Gaussian predictions for zero-mode power.

    The random seed makes this a deterministic regression test.  Its
    tolerances are nevertheless set by five Monte-Carlo standard errors, not
    by the particular numerical values produced by that seed.
    """
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

    noise_sigma = 0.03 * np.max(np.abs(exact_values))
    uncertainty = np.full(exact_values.shape, noise_sigma)

    # Independent analytic response of the fixed linear design.
    orthogonal, triangular = np.linalg.qr(matrix, mode="reduced")
    response = np.linalg.solve(triangular, orthogonal.T)
    analytic_covariance = noise_sigma**2 * (response @ response.T)

    random = np.random.default_rng(5000)
    fitted_coefficients = np.empty(
        (number_of_realizations, len(fit_indices))
    )
    first_fit = None
    for realization in range(number_of_realizations):
        noisy_values = exact_values + random.normal(
            0.0,
            noise_sigma,
            exact_values.size,
        )
        fit = fit_design_matrix(
            noisy_values,
            fit_indices,
            matrix,
            uncertainty=uncertainty,
            absolute_uncertainty=True,
        )
        fitted_coefficients[realization] = fit.coefficients
        if first_fit is None:
            first_fit = fit

    assert first_fit is not None
    np.testing.assert_allclose(
        first_fit.coefficient_covariance,
        analytic_covariance,
        rtol=2.0e-12,
        atol=1.0e-15,
    )

    zero_mode = true_coefficients == 0.0
    coefficient_variance = np.diag(analytic_covariance)[zero_mode]
    coefficient_errors = (
        fitted_coefficients[:, zero_mode] - true_coefficients[zero_mode]
    )
    standardized_errors = coefficient_errors / np.sqrt(coefficient_variance)
    normalized_powers = standardized_errors**2

    # E[z] = 0 and E[z^2] = 1 for z ~ N(0, 1).
    mean_tolerance = 5.0 / np.sqrt(number_of_realizations)
    power_mean_tolerance = 5.0 * np.sqrt(
        2.0 / number_of_realizations
    )
    assert np.all(np.abs(np.mean(standardized_errors, axis=0)) < mean_tolerance)
    assert np.all(
        np.abs(np.mean(normalized_powers, axis=0) - 1.0)
        < power_mean_tolerance
    )

    # For z^2 ~ chi-square(1), Var(z^2) = 2.  Its fourth central moment is
    # 60, giving approximately sqrt((60 - 2^2) / R) = sqrt(56 / R) as
    # the standard error of the empirical variance.
    power_variance_tolerance = 5.0 * np.sqrt(
        56.0 / number_of_realizations
    )
    assert np.all(
        np.abs(np.var(normalized_powers, axis=0, ddof=1) - 2.0)
        < power_variance_tolerance
    )

    # Each unoccupied mode should fall below the 95% chi-square(1) quantile
    # in about 95% of independent detector-noise realizations.
    probability = 0.95
    observed_coverage = np.mean(
        normalized_powers <= chi2.ppf(probability, df=1),
        axis=0,
    )
    coverage_tolerance = 5.0 * np.sqrt(
        probability * (1.0 - probability) / number_of_realizations
    ) + 1.0 / number_of_realizations
    assert np.all(
        np.abs(observed_coverage - probability) < coverage_tolerance
    )
