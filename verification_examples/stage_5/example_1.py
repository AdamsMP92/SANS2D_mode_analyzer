"""Mode coupling caused by a displaced centre of an even synthetic signal."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repository = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "src"))
sys.path.insert(0, str(repository))

from sans2d_mode_analyzer import standard_analyzer
from synthetic_models import generate_modal_detector_data


q_min = 0.05
q_max = 0.30
number_of_pixels = 81
n_max = 8
m_max = 8

# The exact signal contains only the even cosine orders n=0 and n=2.
generating_indices = [(0, 0), (0, 2), (2, 1)]
generating_coefficients = np.array([3.0, -0.7, 1.1])

# Impose a 0.35-pixel displacement in q_y and no q_z displacement.
nominal_pixel_size = 2.0 * q_max / (number_of_pixels - 1)
imposed_shift = (0.0, 0.35 * nominal_pixel_size)
q_z, q_y, shifted_intensity = generate_modal_detector_data(
    q_min,
    q_max,
    generating_indices,
    generating_coefficients,
    number_of_pixels=number_of_pixels,
    center_shift=imposed_shift,
)

# Unit uncertainties are numerical weights only: this stage contains no
# stochastic noise and makes no statistical-coverage claim.
uncertainty = np.full(shifted_intensity.shape, np.nan)
uncertainty[np.isfinite(shifted_intensity)] = 1.0

output_directory = Path(__file__).resolve().parent / "output"
output_directory.mkdir(parents=True, exist_ok=True)
raw_output = output_directory / "stage_5_example_1_uncentered.png"
centered_output = output_directory / "stage_5_example_1_centered.png"
modal_centered_output = (
    output_directory / "stage_5_example_1_modal_centered.png"
)
comparison_output = output_directory / "stage_5_example_1_translation_coupling.png"

raw = standard_analyzer(
    q_z,
    q_y,
    shifted_intensity,
    uncertainty=uncertainty,
    q_range=(q_min, q_max),
    mode_limits=(n_max, m_max),
    output_path=raw_output,
)
centered = standard_analyzer(
    q_z,
    q_y,
    shifted_intensity,
    uncertainty=uncertainty,
    beam_centering=True,
    q_range=(q_min, q_max),
    mode_limits=(n_max, m_max),
    output_path=centered_output,
)
modal_centered = standard_analyzer(
    q_z,
    q_y,
    shifted_intensity,
    uncertainty=uncertainty,
    beam_centering=True,
    odd_mode_centering=True,
    odd_mode_centering_mode_limits=(5, 8),
    odd_mode_centering_max_shift_pixels=0.25,
    q_range=(q_min, q_max),
    mode_limits=(n_max, m_max),
    output_path=modal_centered_output,
)

recovered_shift = np.array(
    [centered.center_shift_q_z, centered.center_shift_q_y]
)
print(f"imposed centre shift: {imposed_shift}")
print(f"recovered centre shift: {tuple(recovered_shift)}")
print(f"raw odd power fraction: {raw.parity_fractions.odd:.6e}")
print(f"centered odd power fraction: {centered.parity_fractions.odd:.6e}")
print(
    "modal-refined centre shift: "
    f"{(modal_centered.center_shift_q_z, modal_centered.center_shift_q_y)}"
)
print(
    "modal-refined odd power fraction: "
    f"{modal_centered.parity_fractions.odd:.6e}"
)

# First-order translation prediction from the corrected even spectra.
q = modal_centered.spectra.q
physical_i0 = modal_centered.spectra.cosine[0] / np.sqrt(2.0 * np.pi)
physical_i2 = modal_centered.spectra.cosine[2] / np.sqrt(np.pi)
physical_i0_derivative = np.gradient(physical_i0, q, edge_order=2)
physical_i2_derivative = np.gradient(physical_i2, q, edge_order=2)
delta_q_y = imposed_shift[1]
predicted_i1_sine = -np.sqrt(np.pi) * delta_q_y * (
    physical_i0_derivative
    - 0.5 * physical_i2_derivative
    - physical_i2 / q
)
predicted_i3_sine = -np.sqrt(np.pi) * delta_q_y * (
    0.5 * physical_i2_derivative - physical_i2 / q
)

figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
figure.suptitle(
    "Translation-induced odd modes from an exact n=0,2 signal",
    fontsize=11,
)

image = axes[0, 0].imshow(
    shifted_intensity,
    origin="lower",
    extent=(q_z.min(), q_z.max(), q_y.min(), q_y.max()),
    cmap="viridis",
)
axes[0, 0].plot(0.0, 0.0, "+", color="white", markersize=10, label="nominal centre")
axes[0, 0].plot(
    imposed_shift[0],
    imposed_shift[1],
    "x",
    color="red",
    markersize=8,
    label="imposed centre",
)
axes[0, 0].set_title("shifted synthetic detector signal")
axes[0, 0].set_xlabel(r"$q_z$")
axes[0, 0].set_ylabel(r"$q_y$")
axes[0, 0].legend(fontsize=8)
figure.colorbar(image, ax=axes[0, 0], label="intensity")

orders = raw.powers.orders
odd_orders = orders[(orders > 0) & (orders % 2 == 1)]
for result, color, label in [
    (raw, "tab:red", "uncentered"),
    (centered, "tab:orange", "parity-centred"),
    (modal_centered, "tab:blue", "modal-refined"),
]:
    axes[0, 1].semilogy(
        odd_orders,
        result.powers.cosine[odd_orders],
        "o-",
        color=color,
        label=f"{label} cosine",
    )
    axes[0, 1].semilogy(
        odd_orders,
        result.powers.sine[odd_orders],
        "x--",
        color=color,
        label=f"{label} sine",
    )
axes[0, 1].set_title("odd angular powers")
axes[0, 1].set_xlabel("angular order n")
axes[0, 1].set_ylabel("power")
axes[0, 1].set_xticks(odd_orders)
axes[0, 1].grid(alpha=0.25)
axes[0, 1].legend(fontsize=8)

for axis, order, prediction in [
    (axes[1, 0], 1, predicted_i1_sine),
    (axes[1, 1], 3, predicted_i3_sine),
]:
    axis.plot(q, raw.spectra.sine[order], label=f"uncentered $I_{order}^s(q)$")
    axis.plot(q, prediction, "--", label="first-order shift prediction")
    axis.plot(
        q,
        modal_centered.spectra.sine[order],
        ":",
        linewidth=2.0,
        label=f"modal-refined $I_{order}^s(q)$",
    )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_title(f"translation coupling into sine order n={order}")
    axis.set_xlabel("q")
    axis.set_ylabel(r"$I_n^s(q)$")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

figure.savefig(comparison_output, dpi=150)
plt.close(figure)
print(f"output: {raw_output}")
print(f"output: {centered_output}")
print(f"output: {modal_centered_output}")
print(f"output: {comparison_output}")
