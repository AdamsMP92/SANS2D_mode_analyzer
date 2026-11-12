"""Run the standard analysis for the retained NuMagSANS IVR40 maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sans2d_mode_analyzer import fractional_annulus_weights, standard_analyzer


def load_cartesian_spin_flip(path: Path):
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 4))
    grid_size = int(round(np.sqrt(values.shape[0])))
    if grid_size**2 != values.shape[0]:
        raise ValueError(f"{path} does not contain a square Cartesian grid")
    shape = (grid_size, grid_size)
    return tuple(values[:, column].reshape(shape) for column in range(3))


dataset_directory = Path(__file__).resolve().parent
data_directory = dataset_directory / "data"
results_directory = dataset_directory / "analysis_results"
results_directory.mkdir(parents=True, exist_ok=True)

q_range = (0.05, 0.50)
q_min, q_max = q_range
spectra_by_index = {}

for index in (1, 2):
    stem = f"NuMagSANS_Output_IVR40_{index}"
    data_file = data_directory / stem / "SANS2D_cartesian.csv"
    q_z, q_y, spin_flip = load_cartesian_spin_flip(data_file)
    annulus_weight = fractional_annulus_weights(
        q_z, q_y, q_min=q_min, q_max=q_max, subpixels=7
    )
    result = standard_analyzer(
        q_z,
        q_y,
        spin_flip,
        sample_weight=annulus_weight,
        q_range=q_range,
        mode_limits=(2, 10),
        output_path=results_directory / f"{stem}_standard_analysis.png",
    )
    spectra_by_index[index] = result.spectra

figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
labels = {1: "IVR40_1: dilute", 2: "IVR40_2: StructData05"}
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
figure.savefig(results_directory / "IVR40_1_vs_2_I0_I2.png", dpi=150)
plt.close(figure)

