"""Minimal fit of a noisy synthetic two-dimensional SANS signal."""

import numpy as np

from sans2d_mode_analyzer import standard_analyzer
from synthetic_models import powder_average_affine_noise_signal

# Generate synthetic data.
radius = 30.0
x_min = 0.5
x_max = 5.0
q_min = x_min / radius
q_max = x_max / radius
q_z, q_y, noisy_image = powder_average_affine_noise_signal(q_min, q_max, radius)

# Standard mode analyzer.
n_max = 8
m_max = 8
analysis = standard_analyzer(
    q_z,
    q_y,
    noisy_image,
    q_min=q_min,
    q_max=q_max,
    n_max=n_max,
    m_max=m_max,
    output_path="examples/synthetic_sphere_fit.png",
)
