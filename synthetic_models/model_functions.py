"""Simple functions for constructing synthetic two-dimensional SANS data."""

import numpy as np


def create_detector_grid(q_min, q_max, number_of_pixels=161):
    """Create Cartesian detector coordinates, polar coordinates, and an annulus mask."""
    q_axis = np.linspace(-q_max, q_max, number_of_pixels)
    q_z, q_y = np.meshgrid(q_axis, q_axis)
    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    inside = (q >= q_min) & (q <= q_max)
    return q_z, q_y, q, theta, inside


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


def powder_average_affine_noise_signal(q_min, q_max, radius):
    q_z, q_y, q, theta, inside = create_detector_grid(q_min, q_max)
    signal = sphere_sans_signal(q[inside], theta[inside], radius)
    noisy_signal, noise_sigma = add_white_noise(signal)
    noisy_image = put_on_detector_grid(noisy_signal, inside)
    return q_z, q_y, noisy_image