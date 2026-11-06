import numpy as np

from sans2d_mode_analyzer import (
    rectangular_indices,
    test_sin_squared as run_sin_squared_test,
    test_sin_squared_cos_squared as run_sin_squared_cos_squared_test,
)


def coefficient_vector(indices, values_by_mode):
    return np.array([values_by_mode.get(mode, 0.0) for mode in indices])


def test_exact_sin_squared_template_has_unit_explained_power():
    indices = rectangular_indices(n_max=4, m_max=2)
    values = {}
    for radial_order, constant in enumerate([2.0, -0.5, 1.25]):
        values[(0, radial_order)] = constant
        values[(2, radial_order)] = -constant / np.sqrt(2.0)

    result = run_sin_squared_test(coefficient_vector(indices, values), indices)

    np.testing.assert_allclose(result.explained_power_fraction, 1.0)
    np.testing.assert_allclose(result.residual_power_fraction, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(result.pair_mismatch_power_fraction, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(result.forbidden_mode_power_fraction, 0.0)


def test_exact_spin_flip_fourfold_template_uses_fourth_cosine_order():
    indices = rectangular_indices(n_max=4, m_max=1)
    values = {
        (0, 0): 3.0,
        (4, 0): -3.0 / np.sqrt(2.0),
        (0, 1): -1.0,
        (4, 1): 1.0 / np.sqrt(2.0),
    }

    result = run_sin_squared_cos_squared_test(
        coefficient_vector(indices, values), indices
    )

    assert result.harmonic_order == 4
    np.testing.assert_allclose(result.explained_power_fraction, 1.0)
    np.testing.assert_allclose(result.residual_power_fraction, 0.0, atol=1.0e-15)


def test_template_test_separates_pair_mismatch_and_forbidden_modes():
    indices = rectangular_indices(n_max=4, m_max=0)
    values = {
        (0, 0): 2.0,
        (2, 0): 0.0,
        (-1, 0): 1.0,
    }

    result = run_sin_squared_test(coefficient_vector(indices, values), indices)

    assert result.pair_mismatch_power_fraction > 0.0
    np.testing.assert_allclose(result.forbidden_mode_power_fraction, 0.2)
    np.testing.assert_allclose(
        result.residual_power_fraction,
        result.pair_mismatch_power_fraction + result.forbidden_mode_power_fraction,
    )
