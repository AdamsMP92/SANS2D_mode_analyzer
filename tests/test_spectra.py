import numpy as np

from sans2d_mode_analyzer import angular_radial_spectra, radial_mode


def test_angular_radial_spectra_reconstruct_each_angular_branch():
    q_min = 0.1
    q_max = 1.0
    q = np.linspace(q_min, q_max, 20)
    indices = [(0, 0), (2, 1), (-1, 0)]
    coefficients = [3.0, 4.0, 5.0]
    spectra = angular_radial_spectra(
        q,
        coefficients,
        indices,
        q_min=q_min,
        q_max=q_max,
    )
    np.testing.assert_allclose(
        spectra.cosine[0],
        3.0 * radial_mode(0, q, q_min=q_min, q_max=q_max),
    )
    np.testing.assert_allclose(
        spectra.cosine[2],
        4.0 * radial_mode(1, q, q_min=q_min, q_max=q_max),
    )
    np.testing.assert_allclose(
        spectra.sine[1],
        5.0 * radial_mode(0, q, q_min=q_min, q_max=q_max),
    )


def test_angular_radial_spectrum_uncertainty_propagates_covariance():
    q_min = 0.1
    q_max = 1.0
    q = np.linspace(q_min, q_max, 20)
    indices = [(0, 0), (0, 1)]
    covariance = np.diag([0.04, 0.09])
    spectra = angular_radial_spectra(
        q,
        coefficients=[3.0, 4.0],
        indices=indices,
        q_min=q_min,
        q_max=q_max,
        coefficient_covariance=covariance,
    )
    radial_0 = radial_mode(0, q, q_min=q_min, q_max=q_max)
    radial_1 = radial_mode(1, q, q_min=q_min, q_max=q_max)
    expected = np.sqrt(0.04 * radial_0**2 + 0.09 * radial_1**2)
    np.testing.assert_allclose(spectra.cosine_uncertainty[0], expected)
    np.testing.assert_allclose(spectra.sine_uncertainty, 0.0)
