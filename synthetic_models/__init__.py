"""Small synthetic data models used by the examples."""

from .model_functions import (
    add_correlated_gaussian_noise,
    add_heteroskedastic_gaussian_noise,
    add_white_noise,
    create_detector_grid,
    generate_modal_detector_data,
    generate_random_modal_detector_data,
    put_on_detector_grid,
    random_mode_coefficients,
    sphere_sans_signal,
    powder_average_affine_noise_signal,
)

__all__ = [
    "add_correlated_gaussian_noise",
    "add_heteroskedastic_gaussian_noise",
    "add_white_noise",
    "create_detector_grid",
    "generate_modal_detector_data",
    "generate_random_modal_detector_data",
    "put_on_detector_grid",
    "random_mode_coefficients",
    "sphere_sans_signal",
    "powder_average_affine_noise_signal",
]
