"""Independent checks of the supplementary-methods equations.

References use direct-q quadrature, analytic Fourier amplitudes, Gaussian
quadrature, QR whitening, and closed-form null tails. They do not generate
all expected values by calling the basis functions under test. Near-singular
blocks check that T_n and its reference rank retain the same directions.
"""

import numpy as np
import pytest
from scipy.special import betainc, eval_legendre, gammaincc
from scipy.stats import chi2, f

from sans2d_mode_analyzer import (
    angular_block_tests,
    angular_mode_powers,
    angular_radial_spectra,
    design_matrix,
    evaluate_expansion,
    fit_wls,
    parity_mismatch_measures,
    power_model_selector,
    rectangular_indices,
    standard_analyzer,
)


def annular_quadrature(lower, upper):
    # Integrate directly in q, independently of the implementation's q^2 map.
    nodes, weights = np.polynomial.legendre.leggauss(40)
    q = lower + (upper - lower) * (nodes + 1.0) / 2.0
    radial_weight = weights * (upper - lower) * q / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    return q, theta, radial_weight, 2.0 * np.pi / theta.size


def reference_radial(q, m, lower, upper):
    delta = upper**2 - lower**2
    x = (2.0 * q**2 - upper**2 - lower**2) / delta
    return np.sqrt(2.0 * (2 * m + 1) / delta) * eval_legendre(m, x)


def analytic_amplitudes(q):
    return 1.2 + 0.6 * q**2 - 0.2 * q**4, -0.7 + 0.3 * q**2, 0.25 + 0.1 * q**4


@pytest.mark.parametrize("lower,upper", [(0.0, 1.0), (0.015, 0.05), (1.1, 1.3)])
def test_direct_area_quadrature_confirms_basis_parseval_and_linear_integral(lower, upper):
    """Equations S2--S8 and S14: no missing angular or area factor."""
    q, theta, radial_weight, angular_weight = annular_quadrature(lower, upper)
    indices = rectangular_indices(5, 5)
    matrix = design_matrix(q[:, None], theta[None, :], indices, q_min=lower, q_max=upper)
    flat_matrix = matrix.reshape(-1, len(indices))
    weights = np.repeat(radial_weight, theta.size) * angular_weight
    gram = flat_matrix.T @ (weights[:, None] * flat_matrix)
    np.testing.assert_allclose(gram, np.eye(len(indices)), atol=2e-12)
    coefficients = np.random.default_rng(17).normal(size=len(indices))
    signal = flat_matrix @ coefficients
    np.testing.assert_allclose(weights @ signal**2, coefficients @ coefficients, rtol=2e-13)
    area = np.pi * (upper**2 - lower**2)
    np.testing.assert_allclose(weights @ signal, np.sqrt(area) * coefficients[0], atol=2e-12)


def test_analytic_fourier_signal_recovers_spectra_and_absolute_powers_on_masked_samples():
    """S15, S17--S20: factors sqrt(2 pi), sqrt(pi), 2 pi and pi."""
    lower, upper = 0.15, 1.3
    rng = np.random.default_rng(42)
    q = rng.uniform(lower, upper, 650)
    theta = rng.uniform(-2.5, 2.5, q.size)  # Incomplete angular coverage.
    a0, a2, b3 = analytic_amplitudes(q)
    intensity = a0 + a2 * np.cos(2 * theta) + b3 * np.sin(3 * theta)
    indices = rectangular_indices(3, 2)
    fit = fit_wls(
        q, theta, intensity, indices, 0.1 + 0.2 * q,
        sample_weight=rng.uniform(0.2, 1.0, q.size), q_min=lower, q_max=upper,
    )
    qr, angles, radial_weight, angular_weight = annular_quadrature(lower, upper)
    a0, a2, b3 = analytic_amplitudes(qr)
    spectra = angular_radial_spectra(qr, fit.coefficients, indices, q_min=lower, q_max=upper)
    expected_cosine = np.zeros_like(spectra.cosine)
    expected_sine = np.zeros_like(spectra.sine)
    expected_cosine[0] = np.sqrt(2 * np.pi) * a0
    expected_cosine[2] = np.sqrt(np.pi) * a2
    expected_sine[3] = np.sqrt(np.pi) * b3
    np.testing.assert_allclose(spectra.cosine, expected_cosine, atol=5e-13)
    np.testing.assert_allclose(spectra.sine, expected_sine, atol=5e-13)
    powers = angular_mode_powers(fit.coefficients, indices)
    expected_cosine_power = [2 * np.pi * (radial_weight @ a0**2), 0, np.pi * (radial_weight @ a2**2), 0]
    expected_sine_power = [0, 0, 0, np.pi * (radial_weight @ b3**2)]
    np.testing.assert_allclose(powers.cosine, expected_cosine_power, atol=5e-13)
    np.testing.assert_allclose(powers.sine, expected_sine_power, atol=5e-13)
    signal = a0[:, None] + a2[:, None] * np.cos(2 * angles) + b3[:, None] * np.sin(3 * angles)
    np.testing.assert_allclose(
        powers.total_power, radial_weight @ np.sum(signal**2, axis=1) * angular_weight, rtol=2e-13,
    )


def test_spectral_uncertainties_include_correlations_between_radial_coefficients():
    """S21: marginal block covariance including off-diagonal entries."""
    lower, upper = 0.2, 1.1
    q = np.linspace(lower, upper, 23)
    indices = [(0, 2), (-2, 0), (0, 0), (-2, 3)]
    factor = np.array([[0.3, 0, 0, 0], [0.1, 0.4, 0, 0], [0.2, -0.1, 0.5, 0], [0, 0.2, 0.1, 0.6]])
    covariance = factor @ factor.T
    spectra = angular_radial_spectra(
        q, [0.5, -0.2, 1.3, 0.7], indices, q_min=lower, q_max=upper,
        coefficient_covariance=covariance,
    )
    for positions, returned in [([0, 2], spectra.cosine_uncertainty[0]), ([1, 3], spectra.sine_uncertainty[2])]:
        radial = np.column_stack([reference_radial(q, indices[p][1], lower, upper) for p in positions])
        full_q_covariance = radial @ covariance[np.ix_(positions, positions)] @ radial.T
        np.testing.assert_allclose(returned**2, np.diag(full_q_covariance), rtol=2e-13)


def test_power_uncertainty_matches_exact_gaussian_quadrature():
    """S16: check fourth moments independently, including covariance terms."""
    coefficients = np.array([1.2, -0.4])
    covariance = np.array([[0.15, 0.06], [0.06, 0.09]])
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    z1, z2 = np.meshgrid(nodes, nodes)
    gaussian_samples = coefficients + np.sqrt(2) * np.column_stack([z1.ravel(), z2.ravel()]) @ np.linalg.cholesky(covariance).T
    probability_weight = np.outer(weights, weights).ravel() / np.pi
    squared_norm = np.sum(gaussian_samples**2, axis=1)
    mean = probability_weight @ squared_norm
    variance = probability_weight @ (squared_norm - mean)**2
    powers = angular_mode_powers(coefficients, [(0, 0), (0, 2)], covariance)
    np.testing.assert_allclose(powers.cosine_uncertainty[0]**2, variance, rtol=2e-14)
    np.testing.assert_allclose(mean - powers.cosine[0], np.trace(covariance), atol=2e-15)


def test_overview_uses_common_angular_power_scale_distinct_from_product_mode_scale(monkeypatch, tmp_path):
    """Check the actual plotted values, including unscaled T_n statistics."""
    from matplotlib.figure import Figure

    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, *args, **kwargs: saved.append(figure))
    lower, upper = 0.2, 1.0
    q, theta = np.meshgrid(np.linspace(lower, upper, 12), np.linspace(0, 2 * np.pi, 32, endpoint=False))
    r0 = reference_radial(q, 0, lower, upper)
    r1 = reference_radial(q, 1, lower, upper)
    intensity = (2 * r0 / np.sqrt(2 * np.pi) + 0.5 * r0 * np.cos(2 * theta) / np.sqrt(np.pi)
                 + 1.5 * (r0+r1) * np.sin(2 * theta) / np.sqrt(np.pi))
    analysis = standard_analyzer(q * np.cos(theta), q * np.sin(theta), intensity,
                                 uncertainty=np.full(q.shape, 0.1), q_range=(lower, upper),
                                 mode_limits=(2, 1), output_path=tmp_path / "overview.png")
    figure = saved[0]
    assert figure.axes[0].get_subplotspec().get_gridspec().nrows == 5
    axes = {axis.get_title(): axis for axis in figure.axes}
    for title, expected in [("cosine angular powers", {0: 4/4.5, 2: 0.25/4.5}),
                            ("sine angular powers", {2: 1.0})]:
        points = {}
        for container in axes[title].containers:
            line = container.lines[0]
            points.update(zip(line.get_xdata(), line.get_ydata()))
        for order, value in expected.items():
            np.testing.assert_allclose(points[order], value, rtol=2e-13)
    mode_map = axes["cosine $(n,m)$ powers"].images[0].get_array()
    np.testing.assert_allclose(mode_map[0, 0], 1.0, rtol=2e-13)
    np.testing.assert_allclose(mode_map[0, 2], 0.25/4.0, rtol=2e-13)
    for branch in ("cosine", "sine"):
        axis = axes[f"{branch} angular block tests"]
        even_line, odd_line, threshold_line = axis.lines
        statistic = getattr(analysis.block_tests, f"{branch}_statistic")
        critical = getattr(analysis.block_tests, f"{branch}_critical_value")
        for line in (even_line, odd_line):
            order = np.asarray(line.get_xdata(), dtype=int)
            np.testing.assert_allclose(line.get_ydata(), statistic[order])
        order = np.asarray(threshold_line.get_xdata(), dtype=int)
        np.testing.assert_allclose(threshold_line.get_ydata(), critical[order])
        assert threshold_line.get_label().startswith("nominal")
        assert not any(
            collection.get_label().startswith("Holm")
            for collection in axis.collections
        )
    assert analysis.block_tests.holm_family_size == 5


@pytest.mark.parametrize("residual_dof", [None, 37])
def test_correlated_block_statistics_and_null_tails_have_correct_rank_factors(residual_dof):
    """S22--S23: whitening and analytic chi2(2)/F(2,nu) tails."""
    factor = np.array([[1, 0, 0, 0], [0.6, 1.2, 0, 0], [0.8, 0.1, 1, 0], [-0.3, 0.5, 0.1, 1]])
    covariance = factor @ factor.T
    whitened_coefficients = np.array([1.3, -0.7])
    coefficients = np.r_[factor[:2, :2] @ whitened_coefficients, 2.0, -0.3]
    result = angular_block_tests(
        coefficients, covariance, [(2, 0), (2, 3), (0, 0), (-1, 2)],
        residual_degrees_of_freedom=residual_dof,
    )
    statistic = whitened_coefficients @ whitened_coefficients
    if residual_dof is None:
        tail = np.exp(-statistic / 2)
        threshold = -2 * np.log(0.05)
    else:
        tail = (1 + statistic / residual_dof)**(-residual_dof / 2)
        threshold = residual_dof * (0.05**(-2 / residual_dof) - 1)
    assert result.cosine_degrees_of_freedom[2] == 2
    np.testing.assert_allclose(result.cosine_statistic[2], statistic, rtol=2e-14)
    np.testing.assert_allclose(result.cosine_pvalue[2], tail, rtol=2e-14)
    np.testing.assert_allclose(result.cosine_critical_value[2], threshold, rtol=2e-14)


@pytest.mark.parametrize("absolute", [True, False])
def test_weighted_fit_covariance_matches_independent_qr_and_distinguishes_area_weights(absolute):
    """S10--S13: propagated absolute errors versus relative precision model."""
    lower, upper = 0.2, 1.0
    rng = np.random.default_rng(23)
    q, theta = rng.uniform(lower, upper, 91), rng.uniform(-2.4, 2.1, 91)
    indices = [(0, 0), (2, 1), (-1, 2)]
    matrix = np.column_stack([
        reference_radial(q, 0, lower, upper) / np.sqrt(2 * np.pi),
        reference_radial(q, 1, lower, upper) * np.cos(2 * theta) / np.sqrt(np.pi),
        reference_radial(q, 2, lower, upper) * np.sin(theta) / np.sqrt(np.pi),
    ])
    intensity = matrix @ [0.7, -0.2, 0.1] + 0.03 * np.cos(5 * theta)
    sigma = 0.1 + 0.2 * q
    area = rng.uniform(0.2, 1.0, q.size)
    whitening = np.sqrt(area) / sigma
    orthogonal, triangular = np.linalg.qr(whitening[:, None] * matrix, mode="reduced")
    expected_coefficients = np.linalg.solve(triangular, orthogonal.T @ (whitening * intensity))
    inverse = np.linalg.solve(triangular, np.eye(len(indices)))
    base_covariance = inverse @ inverse.T
    residual = intensity - matrix @ expected_coefficients
    q_statistic = np.sum((whitening * residual)**2)
    scale = 1.0 if absolute else q_statistic / (q.size - len(indices))
    result = fit_wls(q, theta, intensity, indices, sigma, sample_weight=area,
                     absolute_uncertainty=absolute, q_min=lower, q_max=upper)
    actual_sampling_covariance = base_covariance @ (matrix.T @ ((area**2 / sigma**2)[:, None] * matrix)) @ base_covariance
    np.testing.assert_allclose(result.coefficients, expected_coefficients, rtol=2e-13)
    expected_covariance = actual_sampling_covariance if absolute else scale * base_covariance
    np.testing.assert_allclose(result.coefficient_covariance, expected_covariance, rtol=2e-13, atol=2e-16)
    np.testing.assert_allclose(result.mse, np.mean(residual**2), rtol=2e-13)
    np.testing.assert_allclose(result.reduced_chi_square, q_statistic / (q.size-len(indices)), rtol=2e-13)
    assert not np.allclose(base_covariance, actual_sampling_covariance, rtol=0.01, atol=0)


def test_parity_and_compression_are_continuous_norm_identities():
    """S26--S27 and S33: verify norms without coefficient-only expectations."""
    lower, upper = 0.1, 1.2
    q, theta, radial_weight, angular_weight = annular_quadrature(lower, upper)
    indices = [(0, 0), (2, 1), (-3, 2), (1, 0)]
    coefficients = np.array([3.0, -0.7, 0.4, 0.3])
    signal = evaluate_expansion(q[:, None], theta[None, :], coefficients, indices, q_min=lower, q_max=upper)
    reflected = evaluate_expansion(q[:, None], theta[None, :] + np.pi, coefficients, indices, q_min=lower, q_max=upper)
    integrated_square = radial_weight @ np.sum(signal**2, axis=1) * angular_weight
    integrated_difference = radial_weight @ np.sum((signal-reflected)**2, axis=1) * angular_weight
    mismatch = parity_mismatch_measures(coefficients, indices, q_min=lower, q_max=upper)
    np.testing.assert_allclose(mismatch.mean_square * mismatch.area, integrated_difference, rtol=2e-13)
    np.testing.assert_allclose(mismatch.normalized_rms**2, integrated_difference / integrated_square, rtol=2e-13)
    threshold = 0.97
    selected = power_model_selector(coefficients, indices, threshold)
    selected_coefficients = [coefficients[indices.index(mode)] for mode in selected]
    reduced = evaluate_expansion(q[:, None], theta[None, :], selected_coefficients, selected, q_min=lower, q_max=upper)
    lost_power = radial_weight @ np.sum((signal-reduced)**2, axis=1) * angular_weight
    np.testing.assert_allclose(lost_power, sum(c*c for c, mode in zip(coefficients, indices) if mode not in selected), rtol=2e-13)
    assert lost_power <= (1-threshold) * integrated_square


@pytest.mark.parametrize("eigenvalues,expected_rank", [([1.0, 5e-16], 2), ([1.0] + [2e-15] * 9, 1)])
@pytest.mark.parametrize("angular_index,branch", [(2, "cosine"), (-2, "sine")])
@pytest.mark.parametrize("residual_dof", [None, 37])
def test_near_singular_block_reference_rank_matches_evaluated_dimension(
    eigenvalues, expected_rank, angular_index, branch, residual_dof,
):
    """Each retained standardized component contributes one to T and one df."""
    result = angular_block_tests(
        np.sqrt(eigenvalues), np.diag(eigenvalues),
        [(angular_index, m) for m in range(len(eigenvalues))],
        residual_degrees_of_freedom=residual_dof,
    )
    statistic = getattr(result, f"{branch}_statistic")[2]
    rank = getattr(result, f"{branch}_degrees_of_freedom")[2]
    assert rank == expected_rank
    np.testing.assert_allclose(statistic, expected_rank, rtol=2e-14)

    def reference_tail(value):
        if residual_dof is None:
            return gammaincc(expected_rank / 2, value / 2)
        return betainc(residual_dof / 2, expected_rank / 2,
                       residual_dof / (residual_dof + value))

    np.testing.assert_allclose(getattr(result, f"{branch}_pvalue")[2],
                               reference_tail(statistic), rtol=2e-14)
    np.testing.assert_allclose(reference_tail(getattr(result, f"{branch}_critical_value")[2]),
                               0.05, rtol=2e-13)


@pytest.mark.parametrize("residual_dof", [None, 37])
@pytest.mark.parametrize("permute", [False, True])
def test_holm_equation_matches_closed_bonferroni_tests(residual_dof, permute):
    """S24: independent closed-testing reference for correlated, unequal-rank blocks."""
    indices = [(0, 0), (0, 1), (2, 0), (-1, 0), (-1, 2), (-1, 3), (-3, 0)]
    groups = [(0, [0, 1]), (2, [2]), (-1, [3, 4, 5]), (-3, [6])]
    factor = np.random.default_rng(412).normal(size=(7, 7)) + 3 * np.eye(7)
    covariance = factor @ factor.T
    coefficients = np.empty(7)
    for (_, positions), probability in zip(groups, [0.0001, 0.013, 0.03, 0.2]):
        rank = len(positions)
        statistic = (
            -2 * np.log(probability) if rank == 2 and residual_dof is None else None
        )
        if statistic is None:
            statistic = (
                chi2.isf(probability, rank) if residual_dof is None
                else rank * f.isf(probability, rank, residual_dof)
            )
        standardized = np.zeros(rank)
        standardized[0] = np.sqrt(statistic)
        coefficients[positions] = np.linalg.cholesky(
            covariance[np.ix_(positions, positions)]
        ) @ standardized
    if permute:
        order = np.array([5, 0, 3, 2, 6, 1, 4])
        indices = [indices[i] for i in order]
        coefficients = coefficients[order]
        covariance = covariance[np.ix_(order, order)]
    result = angular_block_tests(
        coefficients, covariance, indices, residual_degrees_of_freedom=residual_dof,
    )
    raw = np.array([
        getattr(result, f"{'cosine' if n >= 0 else 'sine'}_pvalue")[abs(n)]
        for n, _ in groups
    ])
    # Holm is equivalent to closed Bonferroni testing: enumerate all
    # intersections containing a given null, rather than reproducing sorting.
    expected = np.zeros(4)
    for mask in range(1, 1 << 4):
        subset = np.array([j for j in range(4) if mask & (1 << j)])
        intersection_pvalue = min(1.0, len(subset) * raw[subset].min())
        expected[subset] = np.maximum(expected[subset], intersection_pvalue)
    for (n, _), probability in zip(groups, expected):
        branch = "cosine" if n >= 0 else "sine"
        np.testing.assert_allclose(
            getattr(result, f"{branch}_holm_pvalue")[abs(n)], probability, rtol=2e-13,
        )
        assert getattr(result, f"{branch}_holm_reject")[abs(n)] == (probability <= 0.05)
    assert result.holm_family_size == 4
