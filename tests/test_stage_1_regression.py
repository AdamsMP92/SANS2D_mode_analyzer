"""Stage 1: exact recovery of deterministic modal signals."""

import numpy as np

from sans2d_mode_analyzer import standard_analyzer
from synthetic_models import generate_random_modal_detector_data


def _true_vector(fitted_indices, generated_indices, generated_coefficients):
    expected = np.zeros(len(fitted_indices))
    fitted_positions = {
        index: position for position, index in enumerate(fitted_indices)
    }
    for index, coefficient in zip(
        map(tuple, generated_indices), generated_coefficients
    ):
        expected[fitted_positions[index]] = coefficient
    return expected


def test_seeded_modal_signal_is_recovered_in_its_full_even_subspace():
    q_min, q_max = 0.05, 0.30
    n_max, m_max = 8, 8
    generating_indices = [(-4, 5), (0, 6), (0, 8), (2, 8)]
    generating_coefficients = np.array(
        [
            -0.34605544678887856,
            0.97455368667585107,
            -0.36257832302896653,
            0.57709787164005788,
        ]
    )

    q_z, q_y, intensity = generate_random_modal_detector_data(
        q_min,
        q_max,
        n_max,
        m_max,
        n_coeffs=4,
        angular_parity="even",
        number_of_pixels=61,
        index_seed=7,
        coefficient_seed=8,
    )
    analysis = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(q_min, q_max),
        mode_limits=(n_max, m_max),
        even_only=True,
        export_figure=False,
    )

    expected = _true_vector(
        analysis.fit.indices,
        generating_indices,
        generating_coefficients,
    )

    np.testing.assert_allclose(analysis.fit.coefficients, expected, atol=2e-12)
    assert analysis.fit.mse < 1.0e-24
    assert analysis.parity_fractions.odd == 0.0
    assert analysis.output_path is None


def test_noiseless_random_ensemble_has_only_roundoff_coefficient_error():
    q_min, q_max = 0.05, 0.30
    n_max, m_max = 4, 4
    errors = []

    for sample in range(8):
        q_z, q_y, intensity, indices, coefficients = (
            generate_random_modal_detector_data(
                q_min,
                q_max,
                n_max,
                m_max,
                n_coeffs=5,
                angular_parity="all",
                number_of_pixels=41,
                index_seed=100 + sample,
                coefficient_seed=200 + sample,
                return_model=True,
            )
        )
        analysis = standard_analyzer(
            q_z,
            q_y,
            intensity,
            q_range=(q_min, q_max),
            mode_limits=(n_max, m_max),
            export_figure=False,
        )
        expected = _true_vector(analysis.fit.indices, indices, coefficients)
        errors.append(analysis.fit.coefficients - expected)
        assert analysis.fit.mse < 1.0e-24

    errors = np.asarray(errors)
    coefficient_rmse = np.sqrt(np.mean(errors**2))

    assert coefficient_rmse < 1.0e-12
    assert np.max(np.abs(errors)) < 1.0e-11
