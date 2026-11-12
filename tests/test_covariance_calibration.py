"""Repeated-noise checks of fit covariance and angular-block calibration.

Small fixed designs, 4096 Gaussian realizations and one BLAS worker suffice;
no experimental files or plots are needed.  The signal lies exactly in the
fitted model, so approximation bias cannot explain the covariance differences.

Absolute measurement errors are propagated through the geometrically weighted
fit. Precision factors are encoded in those supplied measurement errors.
Strict xfails remain only for interpreting the residual-scaled precision
model as a geometric-weight noise model, which that mode does not implement.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.stats import norm, t

from sans2d_mode_analyzer import (
    angular_block_tests,
    design_matrix,
    fit_design_matrix,
    fit_wls,
    pixel_integrated_annulus_design,
)

N_REALIZATIONS = 4096
INDICES = [(0, 0), (2, 0), (2, 1), (-1, 0), (-1, 1)]
TRUE_COEFFICIENTS = np.array([1.3, 0.0, 0.0, 0.0, 0.0])


@dataclass
class NoiseEnsemble:
    coefficients: np.ndarray
    reported_covariance: np.ndarray
    sampling_covariance: np.ndarray
    pvalues: np.ndarray
    statistics: np.ndarray
    critical_values: np.ndarray
    sampling_pvalues: np.ndarray | None
    residual_dof: int
    absolute: bool


def _sampling_covariance(matrix, uncertainty, weight, noise_std):
    """Propagate independent measurement noise through an independent QR solve."""
    whitening = np.sqrt(weight) / uncertainty
    orthogonal, triangular = np.linalg.qr(
        whitening[:, None] * matrix, mode="reduced"
    )
    response = np.linalg.solve(triangular, orthogonal.T * whitening)
    noise_response = response * noise_std
    return noise_response @ noise_response.T


def _run_noise_ensemble(noise_model, absolute, entry_point):
    rng = np.random.default_rng(913)
    q = np.sqrt(rng.uniform(0.2**2, 1.0, 48))
    # Missing angular coverage makes the fitted coefficient blocks correlated.
    theta = rng.uniform(-2.3, 1.8, q.size)
    matrix = design_matrix(q, theta, INDICES, q_min=0.2, q_max=1.0)
    uncertainty = 0.12 + 0.13 * q
    weight = (
        np.ones(q.size)
        if noise_model == "measurement"
        else 0.03 + 0.97 * (0.5 + 0.5 * np.cos(2 * theta)) ** 4
    )
    # Rule out ill-conditioning as an explanation of the statistical mismatch.
    assert np.linalg.cond(np.sqrt(weight)[:, None] * matrix / uncertainty[:, None]) < 10
    # A precision weight changes measurement variance; an area weight does not.
    noise_std = (
        uncertainty / np.sqrt(weight)
        if noise_model == "precision"
        else uncertainty.copy()
    )
    if not absolute:
        # The common scale is deliberately unknown to the fitter.
        noise_std *= 2.4
    # With absolute errors, uncertainty is the actual measurement standard
    # deviation. Encode precision in sigma/sqrt(w), preserving the same WLS
    # objective as the historical combined-precision interpretation.
    fit_uncertainty = noise_std if absolute and noise_model == "precision" else uncertainty
    fit_weight = None if absolute and noise_model == "precision" else weight
    sampling_covariance = _sampling_covariance(
        matrix, uncertainty, weight, noise_std
    )
    noise = np.random.default_rng(272).normal(size=(N_REALIZATIONS, q.size))
    coefficients = np.empty((N_REALIZATIONS, len(INDICES)))
    covariance = np.empty((N_REALIZATIONS, len(INDICES), len(INDICES)))
    pvalues = np.empty((N_REALIZATIONS, 2))
    statistics = np.empty_like(pvalues)
    critical_values = np.empty_like(pvalues)
    sampling_pvalues = (
        np.empty_like(pvalues) if noise_model == "geometric" else None
    )
    residual_dof = q.size - len(INDICES)
    for i, values in enumerate(matrix @ TRUE_COEFFICIENTS + noise * noise_std):
        kwargs = dict(
            sample_weight=fit_weight,
            absolute_uncertainty=absolute,
        )
        if entry_point == "coordinates":
            fit = fit_wls(
                q, theta, values, INDICES, fit_uncertainty,
                q_min=0.2, q_max=1.0, **kwargs,
            )
        else:
            fit = fit_design_matrix(
                values, INDICES, matrix, uncertainty=fit_uncertainty, **kwargs,
            )
        coefficients[i] = fit.coefficients
        covariance[i] = fit.coefficient_covariance
        assert fit.degrees_of_freedom == residual_dof
        assert fit.covariance_scaled_by_residual == (not absolute)
        block = angular_block_tests(
            fit.coefficients, fit.coefficient_covariance, INDICES,
            residual_degrees_of_freedom=None if absolute else residual_dof,
        )
        assert block.cosine_degrees_of_freedom[2] == 2
        assert block.sine_degrees_of_freedom[1] == 2
        pvalues[i] = [block.cosine_pvalue[2], block.sine_pvalue[1]]
        statistics[i] = [block.cosine_statistic[2], block.sine_statistic[1]]
        critical_values[i] = [
            block.cosine_critical_value[2], block.sine_critical_value[1]
        ]
        if sampling_pvalues is not None:
            # The generating noise covariance is known in this experiment,
            # including its true scale.  No residual-scale F claim is made.
            reference = angular_block_tests(
                fit.coefficients, sampling_covariance, INDICES
            )
            sampling_pvalues[i] = [
                reference.cosine_pvalue[2], reference.sine_pvalue[1]
            ]

    ratio = np.diag(np.cov(coefficients, rowvar=False)) / np.diag(
        covariance.mean(axis=0)
    )
    print(
        f"\n{noise_model}, {entry_point}, absolute={absolute}: "
        f"empirical/reported variances={np.array2string(ratio, precision=3)}, "
        f"T_n rejection at 5%={np.mean(pvalues < 0.05, axis=0)}"
    )
    return NoiseEnsemble(
        coefficients, covariance, sampling_covariance, pvalues, statistics,
        critical_values, sampling_pvalues, residual_dof, absolute,
    )


@pytest.fixture(scope="module", params=["coordinates", "design_matrix"])
def entry_point(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False], ids=["absolute", "residual"])
def absolute(request):
    return request.param


@pytest.fixture(scope="module", params=["measurement", "precision"])
def calibrated_ensemble(request, absolute, entry_point):
    return _run_noise_ensemble(request.param, absolute, entry_point)


@pytest.fixture(scope="module")
def geometric_ensemble(absolute, entry_point):
    return _run_noise_ensemble("geometric", absolute, entry_point)


def _assert_uniform_null_pvalues(pvalues):
    # Five binomial standard errors plus one count. Fixed seeds make the test
    # reproducible; the tolerance reflects Monte Carlo precision, not rounding.
    for probability in (0.01, 0.05, 0.5, 0.95):
        observed = np.mean(pvalues < probability, axis=0)
        tolerance = 5 * np.sqrt(
            probability * (1 - probability) / N_REALIZATIONS
        ) + 1 / N_REALIZATIONS
        np.testing.assert_allclose(observed, probability, atol=tolerance, rtol=0)


def _assert_sampling_scatter(ensemble):
    error = ensemble.coefficients - TRUE_COEFFICIENTS
    standardized = np.linalg.solve(
        np.linalg.cholesky(ensemble.sampling_covariance), error.T
    ).T
    np.testing.assert_allclose(
        standardized.mean(axis=0), 0, atol=5 / np.sqrt(N_REALIZATIONS), rtol=0
    )
    # Wishart standard errors: sqrt(2/(M-1)) on diagonal, 1/sqrt(M-1)
    # off diagonal. This checks correlations as well as marginal variances.
    covariance = np.cov(standardized, rowvar=False)
    expected = np.eye(len(INDICES))
    tolerance = 5 * np.sqrt((1 + expected) / (N_REALIZATIONS - 1))
    assert np.all(np.abs(covariance - expected) < tolerance)


def _assert_reported_covariance(ensemble):
    factor = np.linalg.cholesky(ensemble.sampling_covariance)
    mean_covariance = ensemble.reported_covariance.mean(axis=0)
    whitened = np.linalg.solve(factor, mean_covariance)
    whitened = np.linalg.solve(factor, whitened.T).T
    if ensemble.absolute:
        np.testing.assert_allclose(
            whitened, np.eye(len(INDICES)), atol=2e-12, rtol=0
        )
    else:
        # For a valid noise model the mean residual variance has relative
        # standard error sqrt(2/(M * residual_dof)).
        tolerance = 5 * np.sqrt(
            2 / (N_REALIZATIONS * ensemble.residual_dof)
        )
        np.testing.assert_allclose(
            whitened, np.eye(len(INDICES)), atol=tolerance, rtol=0
        )


def test_measurement_and_precision_covariance_predict_repeated_fits(calibrated_ensemble):
    _assert_sampling_scatter(calibrated_ensemble)
    _assert_reported_covariance(calibrated_ensemble)


def test_measurement_and_precision_block_nulls_are_calibrated(calibrated_ensemble):
    _assert_uniform_null_pvalues(calibrated_ensemble.pvalues)
    np.testing.assert_array_equal(
        calibrated_ensemble.pvalues < 0.05,
        calibrated_ensemble.statistics > calibrated_ensemble.critical_values,
    )


def _assert_coefficient_intervals(ensemble):
    threshold = (
        norm.ppf(0.975) if ensemble.absolute else t.ppf(0.975, ensemble.residual_dof)
    )
    standard_error = np.sqrt(
        np.diagonal(ensemble.reported_covariance, axis1=1, axis2=2)
    )
    missed = np.mean(
        np.abs(ensemble.coefficients - TRUE_COEFFICIENTS)
        > threshold * standard_error,
        axis=0,
    )
    tolerance = 5 * np.sqrt(0.05 * 0.95 / N_REALIZATIONS) + 1 / N_REALIZATIONS
    np.testing.assert_allclose(missed, 0.05, atol=tolerance, rtol=0)


def test_measurement_and_precision_coefficient_intervals_cover_truth(calibrated_ensemble):
    _assert_coefficient_intervals(calibrated_ensemble)


def test_geometric_sampling_covariance_predicts_scatter_and_block_nulls(geometric_ensemble):
    """The known-noise sampling covariance works on the very same fitted data."""
    _assert_sampling_scatter(geometric_ensemble)
    _assert_uniform_null_pvalues(geometric_ensemble.sampling_pvalues)


def test_geometric_reported_covariance_predicts_measurement_scatter(geometric_ensemble, request):
    if not geometric_ensemble.absolute:
        request.applymarker(pytest.mark.xfail(
            strict=True, raises=AssertionError,
            reason="Relative precision mode does not model geometrically weighted measurement noise",
        ))
    _assert_reported_covariance(geometric_ensemble)
    if geometric_ensemble.absolute:
        _assert_coefficient_intervals(geometric_ensemble)


def test_geometric_reported_block_nulls_are_calibrated(geometric_ensemble, request):
    if not geometric_ensemble.absolute:
        request.applymarker(pytest.mark.xfail(
            strict=True, raises=AssertionError,
            reason="Relative precision F test is not calibrated for geometric-weight noise",
        ))
    _assert_uniform_null_pvalues(geometric_ensemble.pvalues)


def test_pixel_integrated_area_weights_propagate_known_measurement_noise():
    coordinates = np.linspace(-1.05, 1.05, 13)
    q_z, q_y = np.meshgrid(coordinates, coordinates, indexing="ij")
    integrated = pixel_integrated_annulus_design(
        q_z, q_y, INDICES, q_min=0.2, q_max=1.0, subpixels=7,
    )
    matrix = integrated.design_matrix
    weight = integrated.coverage[integrated.selected_mask]
    assert np.any((weight > 0) & (weight < 1))
    uncertainty = 0.12 + 0.13 * np.hypot(q_z, q_y)[integrated.selected_mask]
    fit = fit_design_matrix(
        matrix @ TRUE_COEFFICIENTS, INDICES, matrix,
        uncertainty=uncertainty, sample_weight=weight,
    )
    sampling = _sampling_covariance(matrix, uncertainty, weight, uncertainty)
    np.testing.assert_allclose(fit.coefficient_covariance, sampling, rtol=1e-10, atol=0)


@pytest.mark.parametrize("weight_scale", [0.01, 10.0])
@pytest.mark.parametrize("entry_point", ["coordinates", "design_matrix"])
def test_absolute_covariance_is_invariant_to_global_geometric_weight_scale(
    entry_point, weight_scale,
):
    rng = np.random.default_rng(64)
    q = rng.uniform(0.2, 1.0, 35)
    theta = rng.uniform(-2.0, 2.0, q.size)
    matrix = design_matrix(q, theta, INDICES, q_min=0.2, q_max=1.0)
    sigma = 0.1 + 0.2 * q
    weight = rng.uniform(0.1, 1.0, q.size)
    values = matrix @ TRUE_COEFFICIENTS + rng.normal(size=q.size) * sigma

    def fit_with_weight(weights):
        if entry_point == "coordinates":
            return fit_wls(
                q, theta, values, INDICES, sigma,
                q_min=0.2, q_max=1.0, sample_weight=weights,
            )
        return fit_design_matrix(
            values, INDICES, matrix, uncertainty=sigma, sample_weight=weights,
        )

    original = fit_with_weight(weight)
    rescaled = fit_with_weight(weight_scale * weight)
    np.testing.assert_allclose(rescaled.coefficients, original.coefficients, atol=2e-14)
    np.testing.assert_allclose(
        rescaled.coefficient_covariance, original.coefficient_covariance, rtol=2e-12,
    )


@pytest.mark.parametrize("angular_index,branch", [(2, "cosine"), (-2, "sine")])
@pytest.mark.parametrize("eigenvalues,expected_rank", [([1, 0.2, 0.05], 3), ([1, 0.2, 0], 2)])
@pytest.mark.parametrize("residual_dof", [None, 19])
def test_correlated_and_singular_block_nulls_use_retained_rank(
    angular_index, branch, eigenvalues, expected_rank, residual_dof,
):
    """Generate a Gaussian null, with an independent scale only for F tests."""
    rng = np.random.default_rng(518)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    factor = rotation * np.sqrt(eigenvalues)
    covariance = factor @ factor.T
    coefficients = np.einsum(
        "si,ji->sj", rng.normal(size=(N_REALIZATIONS, 3)), factor
    )
    assert np.all(np.isfinite(coefficients))
    scales = (
        np.ones(N_REALIZATIONS)
        if residual_dof is None
        else rng.chisquare(residual_dof, size=N_REALIZATIONS) / residual_dof
    )
    pvalues = np.empty(N_REALIZATIONS)
    for i, values in enumerate(coefficients):
        result = angular_block_tests(
            values, scales[i] * covariance, [(angular_index, m) for m in range(3)],
            residual_degrees_of_freedom=residual_dof,
        )
        assert getattr(result, f"{branch}_degrees_of_freedom")[2] == expected_rank
        pvalues[i] = getattr(result, f"{branch}_pvalue")[2]
    _assert_uniform_null_pvalues(pvalues)


@pytest.mark.parametrize("residual_dof", [None, 37])
@pytest.mark.parametrize("signal_present", [False, True])
def test_joint_holm_controls_familywise_error_with_dependent_blocks(
    residual_dof, signal_present,
):
    """Check family-wise error for complete and partial Gaussian nulls."""
    rng = np.random.default_rng(884)
    indices = [(0, 0)] + [(n, 0) for k in range(1, 5) for n in (k, -k)]
    size = len(indices)
    correlation = 0.35
    coefficients = (
        np.sqrt(1 - correlation) * rng.normal(size=(N_REALIZATIONS, size))
        + np.sqrt(correlation) * rng.normal(size=(N_REALIZATIONS, 1))
    )
    covariance = (1 - correlation) * np.eye(size) + correlation * np.ones((size, size))
    if signal_present:
        coefficients[:, 0] += 6
    scale = (
        np.ones(N_REALIZATIONS) if residual_dof is None
        else rng.chisquare(residual_dof, size=N_REALIZATIONS) / residual_dof
    )
    true_null = np.ones(size, dtype=bool)
    true_null[0] = not signal_present
    corrected_errors = raw_errors = signal_detections = 0
    for values, variance_scale in zip(coefficients, scale):
        result = angular_block_tests(
            values, variance_scale * covariance, indices,
            residual_degrees_of_freedom=residual_dof,
        )
        assert result.holm_family_size == size
        cosine_rejected = result.cosine_holm_reject
        sine_rejected = result.sine_holm_reject
        rejected = np.array([
            (cosine_rejected if n >= 0 else sine_rejected)[abs(n)]
            for n, _ in indices
        ])
        raw = np.array([
            getattr(result, f"{'cosine' if n >= 0 else 'sine'}_pvalue")[abs(n)]
            for n, _ in indices
        ])
        corrected_errors += bool(np.any(rejected[true_null]))
        raw_errors += bool(np.any(raw[true_null] <= 0.05))
        signal_detections += bool(rejected[0])
    familywise_error = corrected_errors / N_REALIZATIONS
    tolerance = 5 * np.sqrt(0.05 * 0.95 / N_REALIZATIONS) + 1 / N_REALIZATIONS
    assert familywise_error <= 0.05 + tolerance
    assert raw_errors / N_REALIZATIONS > 0.12
    if signal_present:
        assert signal_detections / N_REALIZATIONS > 0.95
    print(
        f"Holm FWER={familywise_error:.4f}, raw FWER={raw_errors/N_REALIZATIONS:.4f}, "
        f"residual_dof={residual_dof}, signal_present={signal_present}"
    )

