import numpy as np

from synthetic_models import (
    add_correlated_gaussian_noise,
    add_heteroskedastic_gaussian_noise,
    add_white_noise,
    create_detector_grid,
    generate_modal_detector_data,
    generate_random_modal_detector_data,
    put_on_detector_grid,
    random_mode_coefficients,
    sphere_sans_signal,
)


def test_synthetic_model_building_blocks_are_reproducible():
    q_z, q_y, q, theta, inside = create_detector_grid(0.1, 1.0, 31)
    signal = sphere_sans_signal(q[inside], theta[inside], radius=2.0)
    noisy_a, sigma_a = add_white_noise(signal, relative_noise=0.02, seed=4)
    noisy_b, sigma_b = add_white_noise(signal, relative_noise=0.02, seed=4)
    image = put_on_detector_grid(noisy_a, inside)

    assert q_z.shape == q_y.shape == q.shape == theta.shape == inside.shape
    np.testing.assert_allclose(noisy_a, noisy_b)
    assert sigma_a == sigma_b
    assert np.count_nonzero(np.isfinite(image)) == np.count_nonzero(inside)


def test_white_noise_preserves_a_nan_detector_mask():
    signal = np.array([[np.nan, -2.0], [1.0, np.nan]])
    noisy, noise_sigma = add_white_noise(signal, relative_noise=0.1, seed=5)

    np.testing.assert_array_equal(np.isnan(noisy), np.isnan(signal))
    assert noise_sigma == 0.2


def test_heteroskedastic_noise_uses_the_uncertainty_map_and_preserves_mask():
    signal = np.array([[np.nan, 2.0], [-1.0, 0.5]])
    uncertainty = np.array([[np.nan, 0.1], [0.2, 0.4]])
    first = add_heteroskedastic_gaussian_noise(signal, uncertainty, seed=12)
    second = add_heteroskedastic_gaussian_noise(signal, uncertainty, seed=12)

    np.testing.assert_array_equal(np.isnan(first), np.isnan(signal))
    np.testing.assert_array_equal(first, second)


def test_heteroskedastic_noise_rejects_invalid_selected_uncertainties():
    signal = np.array([1.0, np.nan])

    for invalid in (0.0, -0.1, np.nan):
        uncertainty = np.array([invalid, np.nan])
        with np.testing.assert_raises(ValueError):
            add_heteroskedastic_gaussian_noise(signal, uncertainty)


def test_correlated_noise_returns_reproducible_marginal_uncertainties():
    signal = np.array([[np.nan, 2.0], [-1.0, 0.5]])
    independent = np.array([[np.nan, 0.2], [0.2, 0.2]])
    patterns = np.array(
        [
            [[np.nan, 0.1], [0.1, 0.1]],
            [[np.nan, -0.3], [0.0, 0.3]],
        ]
    )

    first, first_marginal = add_correlated_gaussian_noise(
        signal, independent, patterns, seed=19
    )
    second, second_marginal = add_correlated_gaussian_noise(
        signal, independent, patterns, seed=19
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_marginal, second_marginal)
    np.testing.assert_array_equal(np.isnan(first), np.isnan(signal))
    expected = np.sqrt(independent**2 + np.sum(patterns**2, axis=0))
    np.testing.assert_allclose(first_marginal, expected, equal_nan=True)


def test_random_mode_coefficients_match_indices_and_seed():
    indices = [(0, 0), (2, 1), (-3, 2)]
    first = random_mode_coefficients(indices, seed=11)
    second = random_mode_coefficients(indices, seed=11)

    assert first.shape == (len(indices),)
    assert np.all((-1.0 <= first) & (first <= 1.0))
    np.testing.assert_array_equal(first, second)


def test_modal_detector_data_are_directly_recovered_by_standard_analyzer():
    from sans2d_mode_analyzer import standard_analyzer

    indices = [(0, 0), (2, 1), (-1, 2)]
    coefficients = np.array([2.0, -0.4, 0.25])
    q_z, q_y, intensity = generate_modal_detector_data(
        q_min=0.2,
        q_max=1.0,
        indices=indices,
        coefficients=coefficients,
        number_of_pixels=51,
    )
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(0.2, 1.0),
        indices=indices,
        export_figure=False,
    )

    assert q_z.shape == q_y.shape == intensity.shape == (51, 51)
    assert np.any(np.isnan(intensity))
    np.testing.assert_allclose(result.fit.coefficients, coefficients, atol=1e-12)
    assert result.fit.mse < 1.0e-24


def test_modal_detector_data_can_be_generated_about_a_shifted_center():
    from sans2d_mode_analyzer import evaluate_expansion

    q_min, q_max = 0.2, 1.0
    center_shift = (0.04, -0.03)
    indices = [(0, 0), (2, 1)]
    coefficients = np.array([2.0, -0.4])
    q_z, q_y, intensity = generate_modal_detector_data(
        q_min,
        q_max,
        indices,
        coefficients,
        number_of_pixels=51,
        center_shift=center_shift,
    )

    relative_q_z = q_z - center_shift[0]
    relative_q_y = q_y - center_shift[1]
    q = np.hypot(relative_q_z, relative_q_y)
    theta = np.arctan2(relative_q_y, relative_q_z)
    selected = np.isfinite(intensity)
    expected = evaluate_expansion(
        q[selected],
        theta[selected],
        coefficients,
        indices,
        q_min=q_min,
        q_max=q_max,
    )

    np.testing.assert_allclose(intensity[selected], expected)
    assert np.all((q[selected] >= q_min) & (q[selected] <= q_max))


def test_random_modal_detector_wrapper_is_reproducible():
    arguments = dict(
        q_min=0.2,
        q_max=1.0,
        n_max=4,
        m_max=2,
        n_coeffs=5,
        angular_parity="even",
        number_of_pixels=31,
        index_seed=2,
        coefficient_seed=3,
    )
    first = generate_random_modal_detector_data(**arguments)
    second = generate_random_modal_detector_data(**arguments)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_random_modal_detector_wrapper_can_return_generating_model():
    result = generate_random_modal_detector_data(
        q_min=0.2,
        q_max=1.0,
        n_max=4,
        m_max=2,
        n_coeffs=5,
        angular_parity="odd",
        number_of_pixels=31,
        index_seed=2,
        coefficient_seed=3,
        return_model=True,
    )
    q_z, q_y, intensity, indices, coefficients = result

    assert q_z.shape == q_y.shape == intensity.shape == (31, 31)
    assert indices.shape == (5, 2)
    assert coefficients.shape == (5,)
    assert np.all(np.abs(indices[:, 0]) % 2 == 1)
