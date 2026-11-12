"""Minimal fit of a noisy synthetic two-dimensional SANS signal."""

from pathlib import Path
import numpy as np

from sans2d_mode_analyzer import standard_analyzer
from synthetic_models import powder_average_affine_noise_signal

repository = Path(__file__).resolve().parents[1]
output_directory = repository / "examples" / "output"
output_directory.mkdir(parents=True, exist_ok=True)

# Generate synthetic data.
radius = 30.0
x_min = 0.5
x_max = 10.0
q_range = (x_min / radius, x_max / radius)
q_min, q_max = q_range
q_z, q_y, noisy_image = powder_average_affine_noise_signal(q_min, q_max, radius)

# Standard mode analyzer full fit.
mode_limits = (25, 25)
analysis = standard_analyzer(
    q_z,
    q_y,
    noisy_image,
    q_range=q_range,
    mode_limits=mode_limits,
    even_only=False,
    output_path=output_directory / "synthetic_sphere_fit1.png",
)


# Standard mode analyzer even only fit.
mode_limits = (25, 25)
analysis = standard_analyzer(
    q_z,
    q_y,
    noisy_image,
    q_range=q_range,
    mode_limits=mode_limits,
    even_only=True,
    output_path=output_directory / "synthetic_sphere_fit2.png",
)
