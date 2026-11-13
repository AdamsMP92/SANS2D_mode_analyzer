"""Minimal fit of a noisy synthetic two-dimensional SANS signal."""

from pathlib import Path
import sys
import numpy as np

repository = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository / "src"))
sys.path.insert(0, str(repository))

from sans2d_mode_analyzer import standard_analyzer
from synthetic_models import powder_average_affine_noise_signal
from sans2d_mode_analyzer import angular_mode_indices

output_directory = repository / "examples" / "output"
output_directory.mkdir(parents=True, exist_ok=True)

# Generate synthetic data.
radius = 30.0
x_min = 0.5
x_max = 10.0
q_range = (x_min / radius, x_max / radius)
q_min, q_max = q_range
q_z, q_y, noisy_image = powder_average_affine_noise_signal(q_min, q_max, radius, relative_noise=0.01)

# Fit a modest full rectangular basis and write one diagnostic overview.
mode_limits = (4, 10)
standard_analyzer(
    q_z,
    q_y,
    noisy_image,
    q_range=q_range,
    mode_limits=mode_limits,
    output_path=output_directory / "synthetic_sphere_fit.png",
)


exponent = 1.0
indices = angular_mode_indices(
    angular_modes=[-2, -1, 0, 1, 2, 3],
    m_max=35,
)

standard_analyzer(
    q_z,
    q_y,
    noisy_image,
    q_range=q_range,
    indices=indices,
    radial_measure_exponent=exponent,
    piecewise_linear_reference=True,
    reference_radial_segments=35,
    reference_knot_spacing="q",
    output_path=output_directory / "synthetic_sphere_fit2.png",
)
