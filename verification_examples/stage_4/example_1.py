"""Correlated Gaussian background and diagonal-covariance miscalibration."""

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
from synthetic_models import (
    add_correlated_gaussian_noise,
    generate_random_modal_detector_data,
)


# Reuse the deterministic modal model of the preceding verification stages.
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

# Correlated-noise model epsilon = D z + H a.  The first column of H is a
# common detector offset; the second is a smooth q_y gradient and therefore
# couples particularly strongly to parity-odd sine modes.
intensity_scale = np.max(np.abs(exact_intensity[valid]))
independent_uncertainty = np.full(exact_intensity.shape, np.nan)
independent_uncertainty[valid] = 0.020 * intensity_scale
common_pattern = np.full(exact_intensity.shape, np.nan)
common_pattern[valid] = 0.010 * intensity_scale
gradient_pattern = np.full(exact_intensity.shape, np.nan)
gradient_pattern[valid] = 0.030 * intensity_scale * q_y[valid] / q_max
patterns = np.stack([common_pattern, gradient_pattern])

noisy_intensity, marginal_uncertainty = add_correlated_gaussian_noise(
    exact_intensity,
    independent_uncertainty,
    patterns,
    seed=4000,
)
analysis = standard_analyzer(
    q_z,
    q_y,
    noisy_intensity,
    uncertainty=marginal_uncertainty,
    q_range=(q_min, q_max),
    mode_limits=(n_max, m_max),
    absolute_uncertainty=True,
    export_figure=True,
)

# The current public fit receives only the marginal standard deviations and
# therefore uses a diagonal working covariance.  Its linear response L is
# nevertheless known exactly for this fixed weighted design.
pixel_uncertainty = marginal_uncertainty[valid]
weighted_matrix = matrix / pixel_uncertainty[:, None]
orthogonal, triangular = np.linalg.qr(weighted_matrix, mode="reduced")
response = np.linalg.solve(
    triangular,
    orthogonal.T / pixel_uncertainty[None, :],
)

# Covariance reported under the diagonal working model.
diagonal_noise_response = response * pixel_uncertainty[None, :]
reported_covariance = diagonal_noise_response @ diagonal_noise_response.T
np.testing.assert_allclose(
    analysis.fit.coefficient_covariance,
    reported_covariance,
    rtol=2.0e-12,
    atol=1.0e-15,
)

# Actual covariance for epsilon = D z + H a, including off-diagonal pixels.
independent_sigma = independent_uncertainty[valid]
pattern_matrix = patterns[:, valid].T
independent_response = response * independent_sigma[None, :]
correlated_response = response @ pattern_matrix
true_covariance = (
    independent_response @ independent_response.T
    + correlated_response @ correlated_response.T
)

# Monte Carlo simulation directly in coefficient space, avoiding an N x N
# covariance matrix while preserving the exact low-rank pixel correlations.
random = np.random.default_rng(5000)
coefficient_errors = np.empty((number_of_samples, len(fit_indices)))
for start in range(0, number_of_samples, batch_size):
    stop = min(start + batch_size, number_of_samples)
    independent_draws = random.normal(
        size=(stop - start, matrix.shape[0])
    )
    latent_draws = random.normal(
        size=(stop - start, pattern_matrix.shape[1])
    )
    coefficient_errors[start:stop] = (
        independent_draws @ independent_response.T
        + latent_draws @ correlated_response.T
    )

zero = true_coefficients == 0.0
zero_positions = np.flatnonzero(zero)
zero_errors = coefficient_errors[:, zero]
true_power = np.diag(true_covariance)[zero]
reported_power = np.diag(reported_covariance)[zero]
empirical_power = np.mean(zero_errors**2, axis=0)
true_normalized_power = zero_errors**2 / true_power
reported_normalized_power = zero_errors**2 / reported_power
variance_ratio = true_power / reported_power

most_miscalibrated_zero = int(np.argmax(variance_ratio))
most_miscalibrated_position = zero_positions[most_miscalibrated_zero]
most_miscalibrated_index = fit_indices[most_miscalibrated_position]
coverage_limit = chi2.ppf(0.95, df=1)
true_coverage = np.mean(
    true_normalized_power[:, most_miscalibrated_zero] <= coverage_limit
)
reported_coverage = np.mean(
    reported_normalized_power[:, most_miscalibrated_zero] <= coverage_limit
)
mean_power_ratio = np.mean(empirical_power / true_power)
power_variance_ratio = np.mean(
    np.var(zero_errors**2, axis=0, ddof=1) / (2.0 * true_power**2)
)

print(f"true zero coefficients: {np.count_nonzero(zero)}")
print(f"empirical/true mean power: {mean_power_ratio:.4f}")
print(f"empirical/true power variance: {power_variance_ratio:.4f}")
print(
    "largest true/reported coefficient-variance ratio: "
    f"{variance_ratio[most_miscalibrated_zero]:.3f} "
    f"at mode {most_miscalibrated_index}"
)
print(f"95% coverage using true covariance: {true_coverage:.3%}")
print(f"95% coverage using diagonal reported covariance: {reported_coverage:.3%}")

output_directory = Path(__file__).resolve().parent / "output"
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / "stage_4_example_1_correlated_gaussian_noise.png"

map_shape = (m_max + 1, 2 * n_max + 1)
variance_ratio_map = np.full(map_shape, np.nan)
empirical_true_map = np.full(map_shape, np.nan)
zero_counter = 0
for is_zero, (n, m) in zip(zero, fit_indices):
    if not is_zero:
        continue
    column = n + n_max
    variance_ratio_map[m, column] = variance_ratio[zero_counter]
    empirical_true_map[m, column] = empirical_power[zero_counter] / true_power[zero_counter]
    zero_counter += 1

figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
figure.suptitle(
    "Correlated Gaussian background: true versus diagonal covariance",
    fontsize=11,
)

pattern_plot = axes[0, 0].imshow(
    gradient_pattern,
    origin="lower",
    extent=(-q_max, q_max, -q_max, q_max),
    cmap="coolwarm",
)
axes[0, 0].set_title(r"correlated gradient pattern $h_2(q_z,q_y)$")
axes[0, 0].set_xlabel(r"$q_z$")
axes[0, 0].set_ylabel(r"$q_y$")
figure.colorbar(pattern_plot, ax=axes[0, 0], label=r"$h_2$")

map_extent = (-n_max - 0.5, n_max + 0.5, -0.5, m_max + 0.5)
ratio_plot = axes[0, 1].imshow(
    variance_ratio_map,
    origin="lower",
    extent=map_extent,
    aspect="auto",
    norm=LogNorm(
        vmin=float(np.nanmin(variance_ratio_map)),
        vmax=float(np.nanmax(variance_ratio_map)),
    ),
    cmap="magma",
)
axes[0, 1].set_title(r"true variance / diagonal reported variance")
axes[0, 1].set_xlabel("signed angular index n")
axes[0, 1].set_ylabel("radial index m")
figure.colorbar(ratio_plot, ax=axes[0, 1], label="variance ratio")

mean_plot = axes[1, 0].imshow(
    empirical_true_map,
    origin="lower",
    extent=map_extent,
    aspect="auto",
    vmin=0.8,
    vmax=1.2,
    cmap="viridis",
)
axes[1, 0].set_title(r"Monte Carlo mean power / true analytic variance")
axes[1, 0].set_xlabel("signed angular index n")
axes[1, 0].set_ylabel("radial index m")
figure.colorbar(mean_plot, ax=axes[1, 0], label="normalized mean power")

histogram_edges = np.linspace(0.0, 20.0, 81)
axes[1, 1].hist(
    true_normalized_power[:, most_miscalibrated_zero],
    bins=histogram_edges,
    density=True,
    alpha=0.55,
    label="normalized with true covariance",
)
axes[1, 1].hist(
    reported_normalized_power[:, most_miscalibrated_zero],
    bins=histogram_edges,
    density=True,
    alpha=0.45,
    label="normalized with diagonal covariance",
)
x = np.linspace(0.01, histogram_edges[-1], 600)
axes[1, 1].plot(x, chi2.pdf(x, df=1), color="black", label=r"$\chi_1^2$")
axes[1, 1].set_xlim(0.0, histogram_edges[-1])
axes[1, 1].set_xlabel("normalized power of mode " + str(most_miscalibrated_index))
axes[1, 1].set_ylabel("probability density")
axes[1, 1].set_title("effect of ignoring pixel correlations")
axes[1, 1].grid(alpha=0.25)
axes[1, 1].legend(fontsize=8)

figure.savefig(output_path, dpi=150)
plt.close(figure)
print(f"output: {output_path}")
