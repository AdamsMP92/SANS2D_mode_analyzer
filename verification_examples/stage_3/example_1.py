"""Heteroskedastic Gaussian noise transfer into fitted modal powers."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

repository = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "src"))
sys.path.insert(0, str(repository))

from sans2d_mode_analyzer import design_matrix, rectangular_indices, standard_analyzer
from synthetic_models import (
    add_heteroskedastic_gaussian_noise,
    generate_random_modal_detector_data,
)


# Reuse the deterministic modal model of Stages 1 and 2.
q_min = 0.05
q_max = 0.30
n_max = 8
m_max = 8
n_coeffs = 4
number_of_pixels = 61
number_of_samples = 1000
batch_size = 50

q_z, q_y, exact_intensity, generated_indices, generated_coefficients = (
    generate_random_modal_detector_data(
        q_min,
        q_max,
        n_max,
        m_max,
        n_coeffs=n_coeffs,
        angular_parity="even",
        number_of_pixels=number_of_pixels,
        index_seed=71,
        coefficient_seed=31,
        return_model=True,
    )
)

valid = np.isfinite(exact_intensity)
q = np.hypot(q_z, q_y)
theta = np.arctan2(q_y, q_z)
fit_indices = rectangular_indices(n_max=n_max, m_max=m_max)
matrix = design_matrix(
    q[valid],
    theta[valid],
    fit_indices,
    q_min=q_min,
    q_max=q_max,
)

true_coefficients = np.zeros(len(fit_indices))
fit_positions = {index: position for position, index in enumerate(fit_indices)}
for index, coefficient in zip(map(tuple, generated_indices), generated_coefficients):
    true_coefficients[fit_positions[index]] = coefficient
np.testing.assert_allclose(
    exact_intensity[valid],
    matrix @ true_coefficients,
    atol=2.0e-14,
)

# A known radial uncertainty profile: the standard deviation rises from
# 1.5% to 6% of the maximum exact intensity across the fitted annulus.
intensity_scale = np.max(np.abs(exact_intensity[valid]))
radial_position = (q[valid] - q_min) / (q_max - q_min)
pixel_uncertainty = 0.015 * intensity_scale * (1.0 + 3.0 * radial_position)
uncertainty_image = np.full(exact_intensity.shape, np.nan)
uncertainty_image[valid] = pixel_uncertainty

# One representative detector fit verifies the public weighted-fit path.
noisy_intensity = add_heteroskedastic_gaussian_noise(
    exact_intensity,
    uncertainty_image,
    seed=4000,
)
analysis = standard_analyzer(
    q_z,
    q_y,
    noisy_intensity,
    uncertainty=uncertainty_image,
    q_range=(q_min, q_max),
    mode_limits=(n_max, m_max),
    absolute_uncertainty=True,
    export_figure=False,
)

# Weighted least squares with W = diag(1 / sigma_p^2).  The response maps
# detector noise to coefficient noise: c_hat - c = response @ epsilon.
weighted_matrix = matrix / pixel_uncertainty[:, None]
orthogonal, triangular = np.linalg.qr(weighted_matrix, mode="reduced")
response = np.linalg.solve(
    triangular,
    orthogonal.T / pixel_uncertainty[None, :],
)
noise_response = response * pixel_uncertainty[None, :]
coefficient_covariance = noise_response @ noise_response.T
np.testing.assert_allclose(
    analysis.fit.coefficient_covariance,
    coefficient_covariance,
    rtol=2.0e-12,
    atol=1.0e-15,
)

# Repeated heteroskedastic-noise realizations in memory-bounded batches.
random = np.random.default_rng(5000)
coefficient_errors = np.empty((number_of_samples, len(fit_indices)))
for start in range(0, number_of_samples, batch_size):
    stop = min(start + batch_size, number_of_samples)
    pixel_noise = random.normal(
        0.0,
        pixel_uncertainty,
        size=(stop - start, matrix.shape[0]),
    )
    coefficient_errors[start:stop] = pixel_noise @ response.T

zero = true_coefficients == 0.0
zero_errors = coefficient_errors[:, zero]
predicted_power = np.diag(coefficient_covariance)[zero]
empirical_power = np.mean(zero_errors**2, axis=0)
normalized_zero_power = zero_errors**2 / predicted_power

mean_floor = np.mean(predicted_power)
floor_variation = np.std(predicted_power) / mean_floor
mean_power_ratio = np.mean(empirical_power / predicted_power)
power_variance_ratio = np.mean(
    np.var(zero_errors**2, axis=0, ddof=1) / (2.0 * predicted_power**2)
)

print(
    "pixel standard-deviation range: "
    f"{pixel_uncertainty.min():.6e} ... {pixel_uncertainty.max():.6e}"
)
print(f"true zero coefficients: {np.count_nonzero(zero)}")
print(f"relative variation of predicted power floor: {floor_variation:.3%}")
print(f"empirical/predicted mean power: {mean_power_ratio:.4f}")
print(f"empirical/predicted power variance: {power_variance_ratio:.4f}")

output_directory = Path(__file__).resolve().parent / "output"
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / "stage_3_example_1_heteroskedastic_noise.png"

# Put the unoccupied-mode powers on their signed angular/radial index grid.
map_shape = (m_max + 1, 2 * n_max + 1)
analytic_map = np.full(map_shape, np.nan)
monte_carlo_map = np.full(map_shape, np.nan)
zero_counter = 0
for is_zero, (n, m) in zip(zero, fit_indices):
    if not is_zero:
        continue
    column = n + n_max
    analytic_map[m, column] = predicted_power[zero_counter] / mean_floor
    monte_carlo_map[m, column] = empirical_power[zero_counter] / mean_floor
    zero_counter += 1

figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
figure.suptitle(
    "Heteroskedastic Gaussian pixel noise with known standard deviations",
    fontsize=11,
)

uncertainty_plot = axes[0, 0].imshow(
    uncertainty_image,
    origin="lower",
    extent=(-q_max, q_max, -q_max, q_max),
    cmap="magma",
)
axes[0, 0].set_title(r"prescribed pixel uncertainty $\sigma_p$")
axes[0, 0].set_xlabel(r"$q_z$")
axes[0, 0].set_ylabel(r"$q_y$")
figure.colorbar(uncertainty_plot, ax=axes[0, 0], label=r"$\sigma_p$")

map_extent = (-n_max - 0.5, n_max + 0.5, -0.5, m_max + 0.5)
map_min = min(np.nanmin(analytic_map), np.nanmin(monte_carlo_map))
map_max = max(np.nanmax(analytic_map), np.nanmax(monte_carlo_map))
padding = 0.05 * (map_max - map_min)
power_maps = []
for axis, values, title in [
    (axes[0, 1], analytic_map, r"analytic mean $E[\widehat c_{nm}^2]=C_{kk}$"),
    (axes[1, 0], monte_carlo_map, r"mean of 1000 simulated $\widehat c_{nm}^2$"),
]:
    power_maps.append(
        axis.imshow(
            values,
            origin="lower",
            extent=map_extent,
            aspect="auto",
            vmin=map_min - padding,
            vmax=map_max + padding,
            cmap="viridis",
        )
    )
    axis.set_title(title)
    axis.set_xlabel("signed angular index n")
    axis.set_ylabel("radial index m")
    axis.set_xticks(range(-n_max, n_max + 1, 2))
    axis.set_yticks(range(m_max + 1))
figure.colorbar(
    power_maps[0],
    ax=[axes[0, 1], axes[1, 0]],
    label="zero-mode power / mean analytic floor",
)

histogram_edges = np.linspace(0.0, 8.0, 65)
axes[1, 1].hist(
    normalized_zero_power.ravel(),
    bins=histogram_edges,
    density=True,
    alpha=0.65,
    label="Monte Carlo",
)
x = np.linspace(0.01, histogram_edges[-1], 500)
axes[1, 1].plot(x, chi2.pdf(x, df=1), color="black", label=r"$\chi_1^2$")
axes[1, 1].set_xlim(0.0, histogram_edges[-1])
axes[1, 1].set_xlabel(r"normalized zero-mode power $\widehat c_k^2/C_{kk}$")
axes[1, 1].set_ylabel("probability density")
axes[1, 1].set_title("analytic normalized-power distribution")
axes[1, 1].grid(alpha=0.25)
axes[1, 1].legend(fontsize=8)

figure.savefig(output_path, dpi=150)
plt.close(figure)
print(f"output: {output_path}")
