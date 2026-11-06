"""Minimal fit of a noisy synthetic two-dimensional SANS signal."""

from pathlib import Path
import numpy as np

from sans2d_mode_analyzer import standard_analyzer
from synthetic_models import powder_average_affine_noise_signal

repository = Path(__file__).resolve().parents[1]

# Generate synthetic data.
radius = 30.0
x_min = 0.5
x_max = 10.0
q_min = x_min / radius
q_max = x_max / radius
q_z, q_y, noisy_image = powder_average_affine_noise_signal(q_min, q_max, radius)

# Standard mode analyzer.
n_max = 15
m_max = 15
analysis = standard_analyzer(
    q_z,
    q_y,
    noisy_image,
    q_min=q_min,
    q_max=q_max,
    n_max=n_max,
    m_max=m_max,
    output_path=repository / "examples/synthetic_sphere_fit.png",
)
