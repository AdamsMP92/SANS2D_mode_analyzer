import numpy as np

from synthetic_models import (
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
