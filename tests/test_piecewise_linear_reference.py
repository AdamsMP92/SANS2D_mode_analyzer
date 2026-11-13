import numpy as np

from sans2d_mode_analyzer import (
    fit_piecewise_linear_reference,
    piecewise_linear_design_matrix,
    piecewise_linear_gram_matrix,
    piecewise_linear_indices,
    piecewise_linear_knots,
    piecewise_linear_radial_basis,
    standard_analyzer,
)


def test_piecewise_linear_knots_and_hat_basis_follow_measure_spacing():
    knots = piecewise_linear_knots(
        q_min=0.2,
        q_max=1.0,
        radial_segments=4,
        radial_measure_exponent=1.0,
        spacing="measure",
    )
    np.testing.assert_allclose(np.diff(knots**2), np.diff(knots**2)[0])
    basis_at_knots = piecewise_linear_radial_basis(knots, knots)
    np.testing.assert_allclose(basis_at_knots, np.eye(knots.size), atol=1e-15)
    q = np.linspace(knots[0], knots[-1], 101)
    np.testing.assert_allclose(
        np.sum(piecewise_linear_radial_basis(q, knots), axis=-1),
        1.0,
    )


def test_piecewise_linear_gram_matrix_matches_numerical_measure_integral():
    exponent = 2.5
    knots = np.array([0.2, 0.4, 0.7, 1.0])
    nodes, weights = np.polynomial.legendre.leggauss(80)
    numerical = np.zeros((knots.size, knots.size))
    for lower, upper in zip(knots[:-1], knots[1:]):
        q = 0.5 * ((upper - lower) * nodes + upper + lower)
        quadrature = 0.5 * (upper - lower) * weights * q**exponent
        basis = piecewise_linear_radial_basis(q, knots)
        numerical += basis.T @ (quadrature[:, None] * basis)
    analytical = piecewise_linear_gram_matrix(
        knots,
        radial_measure_exponent=exponent,
    )
    np.testing.assert_allclose(analytical, numerical, rtol=2e-12, atol=2e-14)


def test_piecewise_linear_reference_recovers_polygonal_angular_spectra():
    q_min, q_max = 0.2, 1.0
    exponent = 2.0
    knots = piecewise_linear_knots(
        q_min=q_min,
        q_max=q_max,
        radial_segments=3,
        radial_measure_exponent=exponent,
    )
    angular_modes = [0, 2, -2]
    indices = piecewise_linear_indices(angular_modes, radial_segments=3)
    q_axis = np.linspace(q_min, q_max, 18)
    theta_axis = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    q, theta = np.meshgrid(q_axis, theta_axis, indexing="ij")
    coefficients = np.linspace(0.5, 2.0, len(indices))
    matrix = piecewise_linear_design_matrix(q, theta, indices, knots=knots)
    intensity = matrix @ coefficients
    result = fit_piecewise_linear_reference(
        q,
        theta,
        intensity,
        angular_modes,
        knots=knots,
        radial_measure_exponent=exponent,
    )
    np.testing.assert_allclose(result.fit.coefficients, coefficients, atol=2e-13)
    np.testing.assert_allclose(
        result.fit.predict(q, theta),
        intensity,
        atol=2e-12,
    )
    assert result.spectra.radial_measure_exponent == exponent
    zero_positions = [position for position, (n, _) in enumerate(indices) if n == 0]
    zero_coefficients = coefficients[zero_positions]
    expected_power = zero_coefficients @ result.gram_matrix @ zero_coefficients
    np.testing.assert_allclose(result.powers.cosine[0], expected_power)


def test_standard_analyzer_returns_piecewise_linear_reference_with_pixel_integration(
    tmp_path,
):
    q_min, q_max = 0.2, 1.0
    axis = np.linspace(-q_max, q_max, 25)
    q_z, q_y = np.meshgrid(axis, axis)
    q = np.hypot(q_z, q_y)
    intensity = np.where((q >= q_min) & (q <= q_max), 3.0, np.nan)
    uncertainty = np.full(q.shape, 0.2)
    output = tmp_path / "piecewise-reference.png"
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(q_min, q_max),
        mode_limits=(0, 0),
        uncertainty=uncertainty,
        pixel_integration_subpixels=3,
        piecewise_linear_reference=True,
        reference_radial_segments=3,
        output_path=output,
    )
    assert output.exists()
    assert result.reference is not None
    assert result.reference.knots.size == 4
    assert result.reference.spectra.cosine.shape == (1, 300)
    assert result.reference.powers.cosine.shape == (1,)
    np.testing.assert_allclose(
        result.reference.fit.fitted_values,
        3.0,
        atol=1e-12,
    )
