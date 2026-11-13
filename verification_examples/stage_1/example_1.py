"""Exact recovery of a deterministic random modal detector signal."""

from pathlib import Path
import sys

repository = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "src"))
sys.path.insert(0, str(repository))

from sans2d_mode_analyzer import standard_analyzer
from synthetic_models import generate_random_modal_detector_data

# Define one reproducible synthetic test case.
q_min = 0.05
q_max = 0.3
n_max = 8
m_max = 8
n_coeffs = 4
angular_parity = "even"

q_z, q_y, intensity = generate_random_modal_detector_data(
    q_min,
    q_max,
    n_max,
    m_max,
    n_coeffs=n_coeffs,
    angular_parity=angular_parity,
    number_of_pixels=161,
    radial_measure_exponent=1.0,
    index_seed=71,
    coefficient_seed=31,
)

# Fit the full even truncation from which the occupied modes were drawn.
output_directory = Path(__file__).resolve().parent / "output"
output_directory.mkdir(parents=True, exist_ok=True)
analysis = standard_analyzer(
    q_z,
    q_y,
    intensity,
    q_range=(q_min, q_max),
    mode_limits=(n_max, m_max),
    even_only=False,
    output_path=output_directory / "stage_1_example_1.png",
)

print(f"fitted modes: {len(analysis.fit.indices)}")
print(f"occupied generating modes: {n_coeffs}")
print(f"fit MSE: {analysis.fit.mse:.3e}")
print(f"output: {analysis.output_path}")
