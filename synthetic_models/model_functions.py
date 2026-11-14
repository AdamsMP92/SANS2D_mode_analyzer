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
    center_shift=(0.0, 0.0),
):
    """Evaluate a modal expansion with an optional displaced annulus centre.

    ``center_shift=(delta_q_z, delta_q_y)`` places the physical signal centre
    at those detector coordinates.  The returned coordinates remain laboratory
    detector coordinates, while the modal expansion is evaluated relative to
    the displaced centre.
    """
    shift = np.asarray(center_shift, dtype=float)
    if shift.shape != (2,) or not np.all(np.isfinite(shift)):
        raise ValueError("center_shift must contain two finite values")
    detector_limit = q_max + float(np.max(np.abs(shift)))
    q_axis = np.linspace(-detector_limit, detector_limit, number_of_pixels)
    q_z, q_y = np.meshgrid(q_axis, q_axis)
    relative_q_z = q_z - shift[0]
    relative_q_y = q_y - shift[1]
    q = np.hypot(relative_q_z, relative_q_y)
    theta = np.arctan2(relative_q_y, relative_q_z)
    inside = (q >= q_min) & (q <= q_max)
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
    return_model=False,
    center_shift=(0.0, 0.0),
):
    """Generate a reproducible random modal signal on a Cartesian detector.

    With ``return_model=True``, append the generating ``(n, m)`` index array
    and coefficient vector to the returned ``q_z, q_y, intensity`` arrays.
    """
    indices = random_mode_indices(
        n_max=n_max,
        m_max=m_max,
        n_coeffs=n_coeffs,
        angular_parity=angular_parity,
        seed=index_seed,
    )

    coefficients = random_mode_coefficients(indices, seed=coefficient_seed)

    detector_data = generate_modal_detector_data(
        q_min,
        q_max,
        indices,
        coefficients,
        number_of_pixels=number_of_pixels,
        radial_measure_exponent=radial_measure_exponent,
        center_shift=center_shift,
    )
    if return_model:
        return (
            *detector_data,
            np.asarray(indices, dtype=int),
            np.asarray(coefficients, dtype=float),
        )
    return detector_data


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
    """Add white Gaussian noise relative to the largest absolute finite value."""
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(signal)
    if not np.any(finite):
        raise ValueError("signal must contain at least one finite value")
    noise_sigma = relative_noise * np.max(np.abs(signal[finite]))
    random = np.random.default_rng(seed)
    noisy_signal = signal.copy()
    noisy_signal[finite] += random.normal(0.0, noise_sigma, np.count_nonzero(finite))
    return noisy_signal, noise_sigma


def add_heteroskedastic_gaussian_noise(signal, uncertainty, seed=7):
    """Add independent Gaussian noise with a supplied standard deviation map."""
    signal, uncertainty = np.broadcast_arrays(
        np.asarray(signal, dtype=float),
        np.asarray(uncertainty, dtype=float),
    )
    finite = np.isfinite(signal)
    if not np.any(finite):
        raise ValueError("signal must contain at least one finite value")
    if not np.all(np.isfinite(uncertainty[finite])) or np.any(
        uncertainty[finite] <= 0.0
    ):
        raise ValueError(
            "uncertainty must be finite and positive wherever signal is finite"
        )
    random = np.random.default_rng(seed)
    noisy_signal = signal.copy()
    noisy_signal[finite] += random.normal(
        0.0,
        uncertainty[finite],
        np.count_nonzero(finite),
    )
    return noisy_signal


def add_correlated_gaussian_noise(
    signal,
    independent_uncertainty,
    correlated_patterns,
    seed=7,
):
    """Add independent noise plus Gaussian amplitudes of fixed spatial patterns.

    If the pattern images are the columns of ``H``, the selected-pixel noise
    covariance is ``diag(independent_uncertainty**2) + H @ H.T``.  The returned
    uncertainty map contains only the marginal standard deviation at each
    pixel; it does not encode the off-diagonal correlations.
    """
    signal, independent_uncertainty = np.broadcast_arrays(
        np.asarray(signal, dtype=float),
        np.asarray(independent_uncertainty, dtype=float),
    )
    patterns = np.asarray(correlated_patterns, dtype=float)
    if patterns.ndim == signal.ndim:
        patterns = patterns[None, ...]
    if patterns.ndim != signal.ndim + 1 or patterns.shape[1:] != signal.shape:
        raise ValueError(
            "correlated_patterns must have shape (R,) + signal.shape"
        )

    finite = np.isfinite(signal)
    if not np.any(finite):
        raise ValueError("signal must contain at least one finite value")
    if not np.all(np.isfinite(independent_uncertainty[finite])) or np.any(
        independent_uncertainty[finite] <= 0.0
    ):
        raise ValueError(
            "independent_uncertainty must be finite and positive wherever "
            "signal is finite"
        )
    if not np.all(np.isfinite(patterns[:, finite])):
        raise ValueError(
            "correlated_patterns must be finite wherever signal is finite"
        )

    random = np.random.default_rng(seed)
    latent_amplitudes = random.normal(size=patterns.shape[0])
    noisy_signal = signal.copy()
    noisy_signal[finite] += random.normal(
        0.0,
        independent_uncertainty[finite],
        np.count_nonzero(finite),
    )
    noisy_signal[finite] += patterns[:, finite].T @ latent_amplitudes

    marginal_uncertainty = np.full(signal.shape, np.nan)
    marginal_uncertainty[finite] = np.sqrt(
        independent_uncertainty[finite] ** 2
        + np.sum(patterns[:, finite] ** 2, axis=0)
    )
    return noisy_signal, marginal_uncertainty


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
