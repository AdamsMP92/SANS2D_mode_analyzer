"""Small synthetic data models used by the examples."""

from .model_functions import (
    add_white_noise,
    create_detector_grid,
    put_on_detector_grid,
    sphere_sans_signal,
    powder_average_affine_noise_signal,
)

__all__ = [
    "add_white_noise",
    "create_detector_grid",
    "put_on_detector_grid",
    "sphere_sans_signal",
    "powder_average_affine_noise_signal",
]
