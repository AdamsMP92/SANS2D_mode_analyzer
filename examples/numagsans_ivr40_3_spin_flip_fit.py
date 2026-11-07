"""Compare the IVR40_1 and IVR40_2 NuMagSANS spin-flip cross sections."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sans2d_mode_analyzer import (
    fractional_annulus_weights,
    standard_analyzer,
)


def load_cartesian_spin_flip(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a square Cartesian NuMagSANS map written by this example."""
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 4))
    grid_size = int(round(np.sqrt(values.shape[0])))
    shape = (grid_size, grid_size)
    return (
        values[:, 0].reshape(shape),
        values[:, 1].reshape(shape),
        values[:, 2].reshape(shape),
    )


q_min, q_max = 0.05, 0.50
#cartesian_grid_size = 161
repository = Path(__file__).resolve().parents[1]
spectra_by_index = {}

for index in (1, 2):
    data_set = f"NuMagSANS_Output_IVR40_{index}"

    cartesian_file = (
        repository
        / "experimental_data/NuMagSANS_IVR40"
        / data_set
        / "SANS2D_cartesian.csv"
    )

    q_z, q_y, spin_flip = load_cartesian_spin_flip(cartesian_file)
    annulus_weight = fractional_annulus_weights(
        q_z, q_y, q_min=q_min, q_max=q_max, subpixels=7
    )

    analysis = standard_analyzer(
        q_z,
        q_y,
        spin_flip,
        q_range=(q_min, q_max),
        mode_limits=(2, 10),
        sample_weight=annulus_weight,
        log_intensity_colorbar=False,
        output_path=repository / "examples" / f"{data_set}.png",
        reconstruction_threshold=0.999,
    )

    spectra_by_index[index] = analysis.spectra

    print(f"{data_set}: fitted samples = {analysis.fit.fitted_values.size}")
    print(f"{data_set}: fitted modes = {len(analysis.fit.indices)}")
    print(
        f"{data_set}: odd angular power fraction = "
        f"{analysis.parity_fractions.odd:.3e}"
    )


# Compare the radial coefficient functions that multiply the orthonormal
# constant and cos(2 theta) angular basis functions.  In a conventional
# Fourier expansion their coefficients are I_0^c/sqrt(2*pi) and
# I_2^c/sqrt(pi), respectively.
figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
labels = {
    1: "IVR40_1: dilute",
    2: "IVR40_2: StructData05",
}

for index, spectra in spectra_by_index.items():
    axes[0].plot(spectra.q, spectra.cosine[0], label=labels[index])
    axes[1].plot(spectra.q, spectra.cosine[2], label=labels[index])

for axis, order in zip(axes, (0, 2)):
    axis.set_title(rf"cosine radial spectrum $I_{{{order}}}^c(q)$")
    axis.set_xlabel(r"$q$")
    axis.set_ylabel(rf"$I_{{{order}}}^c(q)$")
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.grid(alpha=0.25)
    axis.legend()

comparison_path = repository / "examples/NuMagSANS_IVR40_1_vs_2_I0_I2.png"
figure.savefig(comparison_path, dpi=150)
plt.close(figure)
print(f"radial-spectrum comparison: {comparison_path}")
