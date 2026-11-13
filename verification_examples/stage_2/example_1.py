"""White-noise transfer into fitted powers of true zero coefficients."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import chi2

repository = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "src"))
sys.path.insert(0, str(repository))

from sans2d_mode_analyzer import design_matrix, rectangular_indices, standard_analyzer
from synthetic_models import add_white_noise, generate_random_modal_detector_data


# Reuse the deterministic modal model of Stage 1.
q_min = 0.05
q_max = 0.30
n_max = 8
m_max = 8
n_coeffs = 4
number_of_pixels = 61
relative_noise = 0.03
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

# One representative noisy detector fit verifies the package covariance.
noisy_intensity, noise_sigma = add_white_noise(
    exact_intensity,
    relative_noise=relative_noise,
    seed=4000,
)
uncertainty = np.full(exact_intensity.shape, np.nan)
uncertainty[valid] = noise_sigma
analysis = standard_analyzer(
    q_z,
    q_y,
    noisy_intensity,
    uncertainty=uncertainty,
    q_range=(q_min, q_max),
    mode_limits=(n_max, m_max),
    absolute_uncertainty=True,
    export_figure=False,
)

# Exact linear response and coefficient covariance for white pixel noise:
# c_hat - c = A epsilon and C = sigma^2 A A^T.
orthogonal, triangular = np.linalg.qr(matrix, mode="reduced")
response = np.linalg.solve(triangular, orthogonal.T)
coefficient_covariance = noise_sigma**2 * (response @ response.T)
np.testing.assert_allclose(
    analysis.fit.coefficient_covariance,
    coefficient_covariance,
    rtol=2.0e-12,
    atol=1.0e-15,
)

# Repeated pixel-noise realizations, evaluated in small memory-bounded batches.
random = np.random.default_rng(5000)
coefficient_errors = np.empty((number_of_samples, len(fit_indices)))
for start in range(0, number_of_samples, batch_size):
    stop = min(start + batch_size, number_of_samples)
    pixel_noise = random.normal(
        0.0,
        noise_sigma,
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

print(f"white-noise standard deviation: {noise_sigma:.6e}")
print(f"true zero coefficients: {np.count_nonzero(zero)}")
print(f"mean predicted zero-mode power: {mean_floor:.6e}")
print(f"relative variation of predicted carpet: {floor_variation:.3%}")
print(f"empirical/predicted mean power: {mean_power_ratio:.4f}")
print(f"empirical/predicted power variance: {power_variance_ratio:.4f}")

output_directory = Path(__file__).resolve().parent / "output"
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / "stage_2_example_1_white_noise_carpet.png"

# Put the zero-mode powers back on their signed angular/radial index grid.
map_shape = (m_max + 1, 2 * n_max + 1)
analytic_map = np.full(map_shape, np.nan)
monte_carlo_map = np.full(map_shape, np.nan)
single_sample_map = np.full(map_shape, np.nan)
zero_counter = 0
for is_zero, (n, m) in zip(zero, fit_indices):
    if not is_zero:
        continue
    column = n + n_max
    analytic_map[m, column] = predicted_power[zero_counter] / mean_floor
    monte_carlo_map[m, column] = empirical_power[zero_counter] / mean_floor
    single_sample_map[m, column] = normalized_zero_power[0, zero_counter]
    zero_counter += 1

figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
figure.suptitle(
    "White cells are the four occupied generating modes; colored cells are true zero modes",
    fontsize=11,
)
map_extent = (-n_max - 0.5, n_max + 0.5, -0.5, m_max + 0.5)
mean_maps = []
for axis, values, title in [
    (axes[0, 0], analytic_map, r"analytic mean $E[\widehat c_{nm}^2]=C_{kk}$"),
    (axes[0, 1], monte_carlo_map, r"mean of 1000 simulated $\widehat c_{nm}^2$"),
]:
    mean_maps.append(
        axis.imshow(
            values,
            origin="lower",
            extent=map_extent,
            aspect="auto",
            vmin=0.8,
            vmax=1.2,
            cmap="viridis",
        )
    )
    axis.set_title(title)
    axis.set_xlabel("signed angular index n")
    axis.set_ylabel("radial index m")
    axis.set_xticks(range(-n_max, n_max + 1, 2))
    axis.set_yticks(range(m_max + 1))
figure.colorbar(
    mean_maps[0],
    ax=axes[0],
    label="mean zero-mode power / mean analytic floor",
)

single_map = axes[1, 0].imshow(
    single_sample_map,
    origin="lower",
    extent=map_extent,
    aspect="auto",
    norm=LogNorm(vmin=1.0e-3, vmax=10.0),
    cmap="viridis",
)
axes[1, 0].set_title(r"one realization: $\widehat c_{nm}^2/C_{kk}$")
axes[1, 0].set_xlabel("signed angular index n")
axes[1, 0].set_ylabel("radial index m")
axes[1, 0].set_xticks(range(-n_max, n_max + 1, 2))
axes[1, 0].set_yticks(range(m_max + 1))
figure.colorbar(single_map, ax=axes[1, 0], label="normalized zero-mode power")

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
axes[1, 1].set_title("analytic power distribution")
axes[1, 1].grid(alpha=0.25)
axes[1, 1].legend(fontsize=8)

figure.savefig(output_path, dpi=150)
plt.close(figure)
print(f"output: {output_path}")
