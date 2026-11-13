"""Simple functions for constructing synthetic two-dimensional SANS data."""

import numpy as np

from sans2d_mode_analyzer.basis import evaluate_expansion
from sans2d_mode_analyzer.indices import validate_indices, random_mode_indices


def random_mode_coefficients(indices, seed=None):
    """Return one reproducible uniform random coefficient per mode index."""
    modes = validate_indices(indices)
    random = np.random.default_rng(seed)
    return random.uniform(-1.0, 1.0, len(modes))


def create_detector_grid(q_min, q_max, number_of_pixels=161):
    """Create Cartesian detector coordinates, polar coordinates, and an annulus mask."""
    q_axis = np.linspace(-q_max, q_max, number_of_pixels)
    q_z, q_y = np.meshgrid(q_axis, q_axis)
    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    inside = (q >= q_min) & (q <= q_max)
    return q_z, q_y, q, theta, inside


def generate_modal_detector_data(
    q_min,
    q_max,
    indices,
    coefficients,
    number_of_pixels=161,
    radial_measure_exponent=1.0,
):
    """Evaluate a modal expansion on a Cartesian detector annulus."""
    q_z, q_y, q, theta, inside = create_detector_grid(
        q_min,
        q_max,
        number_of_pixels=number_of_pixels,
    )
    values = evaluate_expansion(
        q[inside],
        theta[inside],
        coefficients,
        indices,
        q_min=q_min,
        q_max=q_max,
        radial_measure_exponent=radial_measure_exponent,
    )
    intensity = put_on_detector_grid(values, inside)
    return q_z, q_y, intensity


def generate_random_modal_detector_data(
    q_min,
    q_max,
    n_max,
    m_max,
    n_coeffs=8,
    angular_parity="even",
    number_of_pixels=161,
    radial_measure_exponent=1.0,
    index_seed=7,
    coefficient_seed=8,
):
    """Generate a reproducible random modal signal on a Cartesian detector."""
    indices = random_mode_indices(
        n_max=n_max,
        m_max=m_max,
        n_coeffs=n_coeffs,
        angular_parity=angular_parity,
        seed=index_seed,
    )

    coefficients = random_mode_coefficients(indices, seed=coefficient_seed)

    return generate_modal_detector_data(
        q_min,
        q_max,
        indices,
        coefficients,
        number_of_pixels=number_of_pixels,
        radial_measure_exponent=radial_measure_exponent,
    )


def sphere_sans_signal(q, theta, radius):
    """Return (F(q)^2 + 6 F'(q)^2) (1 + cos(theta)^2)."""
    x = q * radius
    form_factor = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    derivative = (
        3.0
        * radius
        * (x**2 * np.sin(x) - 3.0 * (np.sin(x) - x * np.cos(x)))
        / x**4
    )
    return (form_factor**2 + 6.0 * derivative**2) * (1.0 + np.cos(theta) ** 2)


def add_white_noise(signal, relative_noise=0.03, seed=7):
    """Add reproducible white Gaussian noise relative to the signal maximum."""
    noise_sigma = relative_noise * signal.max()
    random = np.random.default_rng(seed)
    noisy_signal = signal + random.normal(0.0, noise_sigma, signal.shape)
    return noisy_signal, noise_sigma


def put_on_detector_grid(values, inside):
    """Put values from the annulus back into a NaN-masked detector image."""
    image = np.full(inside.shape, np.nan)
    image[inside] = values
    return image


def powder_average_affine_noise_signal(q_min, q_max, radius, relative_noise=0.03):
    q_z, q_y, q, theta, inside = create_detector_grid(q_min, q_max)
    signal = sphere_sans_signal(q[inside], theta[inside], radius)
    noisy_signal, noise_sigma = add_white_noise(signal, relative_noise=relative_noise)
    noisy_image = put_on_detector_grid(noisy_signal, inside)
    return q_z, q_y, noisy_image
