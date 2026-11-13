"""Repeated exact coefficient recovery for noiseless modal detector data."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repository = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "src"))
sys.path.insert(0, str(repository))

from sans2d_mode_analyzer import standard_analyzer
from synthetic_models import generate_random_modal_detector_data


q_min = 0.05
q_max = 0.30
n_max = 6
m_max = 6
n_coeffs = 8
number_of_samples = 25

sample_errors = []
sample_rmse = []
occupied_errors = []
unoccupied_errors = []

for sample in range(number_of_samples):
    q_z, q_y, intensity, generated_indices, generated_coefficients = (
        generate_random_modal_detector_data(
            q_min,
            q_max,
            n_max,
            m_max,
            n_coeffs=n_coeffs,
            angular_parity="all",
            number_of_pixels=61,
            index_seed=sample,
            coefficient_seed=1000 + sample,
            return_model=True,
        )
    )
    analysis = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_range=(q_min, q_max),
        mode_limits=(n_max, m_max),
        export_figure=False,
    )

    true_coefficients = np.zeros(len(analysis.fit.indices))
    fitted_positions = {
        index: position for position, index in enumerate(analysis.fit.indices)
    }
    occupied = np.zeros(len(analysis.fit.indices), dtype=bool)
    for index, coefficient in zip(
        map(tuple, generated_indices), generated_coefficients
    ):
        position = fitted_positions[index]
        true_coefficients[position] = coefficient
        occupied[position] = True

    error = analysis.fit.coefficients - true_coefficients
    sample_errors.append(error)
    sample_rmse.append(np.sqrt(np.mean(error**2)))
    occupied_errors.extend(error[occupied])
    unoccupied_errors.extend(error[~occupied])

sample_errors = np.asarray(sample_errors)
sample_rmse = np.asarray(sample_rmse)
occupied_errors = np.asarray(occupied_errors)
unoccupied_errors = np.asarray(unoccupied_errors)

overall_rmse = np.sqrt(np.mean(sample_errors**2))
occupied_rmse = np.sqrt(np.mean(occupied_errors**2))
unoccupied_rmse = np.sqrt(np.mean(unoccupied_errors**2))
maximum_error = np.max(np.abs(sample_errors))

print(f"samples: {number_of_samples}")
print(f"coefficients per fit: {sample_errors.shape[1]}")
print(f"occupied coefficients per sample: {n_coeffs}")
print(f"overall coefficient RMSE: {overall_rmse:.3e}")
print(f"occupied coefficient RMSE: {occupied_rmse:.3e}")
print(f"unoccupied coefficient RMSE: {unoccupied_rmse:.3e}")
print(f"maximum absolute coefficient error: {maximum_error:.3e}")

output_directory = Path(__file__).resolve().parent / "output"
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / "stage_1_example_2_coefficient_errors.png"

figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
axes[0].plot(sample_errors.ravel(), ".", markersize=2)
axes[0].axhline(0.0, color="black", linewidth=0.7)
axes[0].set_xlabel("coefficient instance")
axes[0].set_ylabel("fitted minus true coefficient")
axes[0].set_title("coefficient errors")
axes[0].grid(alpha=0.25)

axes[1].plot(np.arange(number_of_samples), sample_rmse, "o")
axes[1].set_yscale("log")
axes[1].set_xlabel("sample")
axes[1].set_ylabel("coefficient RMSE")
axes[1].set_title("RMSE per noiseless realization")
axes[1].grid(alpha=0.25)

figure.savefig(output_path, dpi=150)
plt.close(figure)
print(f"output: {output_path}")
