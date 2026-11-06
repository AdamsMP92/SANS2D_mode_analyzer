"""Apply the standard mode analyzer to one Ivan experimental data set."""

from pathlib import Path

import numpy as np

from sans2d_mode_analyzer import (
    load_ascii_detector_data,
    mode_convergence_standard_analyzer,
    standard_analyzer,
    test_sin_squared,
)


repository = Path(__file__).resolve().parents[1]
data_file = repository / "experimental_data/ivan_orthodecomp/000000.dat"
data = load_ascii_detector_data(data_file)

intensity = np.array(data.intensity, copy=True)
invalid = ~np.isfinite(data.uncertainty) | (data.uncertainty <= 0.0)
intensity[invalid] = np.nan

output = repository / "examples/experimental_ivan_000000_fit.png"


n_max = 30
m_max = 30
analysis = standard_analyzer(
    data.q_z,
    data.q_y,
    intensity,
    q_min=0.005,
    q_max=0.030,
    n_max=n_max,
    m_max=m_max,
    output_path=output,
)

sin_squared = test_sin_squared(
    analysis.fit.coefficients,
    analysis.fit.indices,
)

print("sin(theta)^2 mode test")
print(f"  explained power fraction = {sin_squared.explained_power_fraction:.6f}")
print(f"  residual power fraction = {sin_squared.residual_power_fraction:.6f}")
print(
    "  c0/c2 pair-mismatch fraction = "
    f"{sin_squared.pair_mismatch_power_fraction:.6f}"
)
print(
    "  forbidden-mode power fraction = "
    f"{sin_squared.forbidden_mode_power_fraction:.6f}"
)
