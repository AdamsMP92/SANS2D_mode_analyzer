import numpy as np

from synthetic_models import (
    add_white_noise,
    create_detector_grid,
    put_on_detector_grid,
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
