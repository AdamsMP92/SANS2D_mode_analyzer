"""Stage-5 regression test for translation-induced odd angular modes."""

import numpy as np

from sans2d_mode_analyzer import standard_analyzer
from synthetic_models import generate_modal_detector_data


def test_parity_centering_removes_translation_induced_odd_modes():
    """A displaced even signal produces n=1,3 modes that centering removes."""
    q_min, q_max = 0.05, 0.30
    number_of_pixels = 81
    nominal_pixel_size = 2.0 * q_max / (number_of_pixels - 1)
    imposed_shift = (0.0, 0.35 * nominal_pixel_size)
    generating_indices = [(0, 0), (0, 2), (2, 1)]
    generating_coefficients = np.array([3.0, -0.7, 1.1])

    q_z, q_y, shifted_intensity = generate_modal_detector_data(
        q_min,
        q_max,
        generating_indices,
        generating_coefficients,
        number_of_pixels=number_of_pixels,
        center_shift=imposed_shift,
    )
    uncertainty = np.full(shifted_intensity.shape, np.nan)
    uncertainty[np.isfinite(shifted_intensity)] = 1.0

    raw = standard_analyzer(
        q_z,
        q_y,
        shifted_intensity,
        uncertainty=uncertainty,
        q_range=(q_min, q_max),
        mode_limits=(8, 8),
        export_figure=False,
    )
    centered = standard_analyzer(
        q_z,
        q_y,
        shifted_intensity,
        uncertainty=uncertainty,
        beam_centering=True,
        q_range=(q_min, q_max),
        mode_limits=(8, 8),
        export_figure=False,
    )
    modal_centered = standard_analyzer(
        q_z,
        q_y,
        shifted_intensity,
        uncertainty=uncertainty,
        beam_centering=True,
        odd_mode_centering=True,
        odd_mode_centering_mode_limits=(5, 8),
        odd_mode_centering_max_shift_pixels=0.25,
        q_range=(q_min, q_max),
        mode_limits=(8, 8),
        export_figure=False,
    )

    # The fitted displacement agrees with the imposed physical centre to a
    # small fraction of one nominal detector pixel.
    assert abs(centered.center_shift_q_z - imposed_shift[0]) < (
        0.02 * nominal_pixel_size
    )
    assert abs(centered.center_shift_q_y - imposed_shift[1]) < (
        0.02 * nominal_pixel_size
    )
    assert abs(modal_centered.center_shift_q_z - imposed_shift[0]) < 1.0e-8
    assert abs(modal_centered.center_shift_q_y - imposed_shift[1]) < 1.0e-8
    assert modal_centered.odd_mode_centering_refinement is not None
    assert modal_centered.beam_centering_method == (
        "affine-grid parity + modal odd-power"
    )

    # A q_y translation of an n=0,2 cosine signal couples at first order to
    # the n=1,3 sine branches.  They must be the two strongest raw odd-sine
    # orders in this controlled signal.
    odd_orders = np.arange(1, 9, 2)
    odd_sine_powers = raw.powers.sine[odd_orders]
    strongest_orders = set(odd_orders[np.argsort(odd_sine_powers)[-2:]])
    assert strongest_orders == {1, 3}

    # Their complete radial spectra follow the first-order translation
    # coupling derived from the corrected n=0,2 cosine spectra.
    q = centered.spectra.q
    physical_i0 = centered.spectra.cosine[0] / np.sqrt(2.0 * np.pi)
    physical_i2 = centered.spectra.cosine[2] / np.sqrt(np.pi)
    derivative_i0 = np.gradient(physical_i0, q, edge_order=2)
    derivative_i2 = np.gradient(physical_i2, q, edge_order=2)
    predictions = {
        1: -np.sqrt(np.pi)
        * imposed_shift[1]
        * (derivative_i0 - 0.5 * derivative_i2 - physical_i2 / q),
        3: -np.sqrt(np.pi)
        * imposed_shift[1]
        * (0.5 * derivative_i2 - physical_i2 / q),
    }
    for order, prediction in predictions.items():
        fitted_spectrum = raw.spectra.sine[order]
        relative_rms_error = np.sqrt(
            np.mean((fitted_spectrum - prediction) ** 2)
            / np.mean(fitted_spectrum**2)
        )
        assert relative_rms_error < 0.03

    # Removing the recovered displacement suppresses the artificial odd
    # sector by more than three orders of magnitude in power.
    assert raw.parity_fractions.odd > 1.0e-4
    assert centered.parity_fractions.odd < 1.0e-6
    assert centered.parity_fractions.odd < (
        raw.parity_fractions.odd / 1000.0
    )
    assert modal_centered.parity_fractions.odd < 1.0e-16
    assert modal_centered.odd_mode_centering_refinement.objective_after < (
        modal_centered.odd_mode_centering_refinement.objective_before / 1.0e8
    )

    # The corrected even-mode coefficients return to the exact generating
    # model.  Modes absent from that model remain numerically negligible.
    positions = {
        index: position
        for position, index in enumerate(centered.fit.indices)
    }
    for index, coefficient in zip(
        generating_indices,
        generating_coefficients,
    ):
        assert np.isclose(
            centered.fit.coefficients[positions[index]],
            coefficient,
            atol=2.0e-3,
        )
    occupied = set(generating_indices)
    absent_coefficients = [
        coefficient
        for index, coefficient in zip(
            centered.fit.indices,
            centered.fit.coefficients,
        )
        if index not in occupied
    ]
    assert np.max(np.abs(absent_coefficients)) < 1.0e-3

