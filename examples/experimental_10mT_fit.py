"""Apply the standard mode analyzer to the lowest-field experimental data."""

from pathlib import Path
import sys

repository = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository / "src"))

from sans2d_mode_analyzer import (
    fractional_annulus_weights,
    load_ill_nexus,
    standard_analyzer,
    angular_mode_indices,
)

output_directory = repository / "examples" / "output"
output_directory.mkdir(parents=True, exist_ok=True)
data_file = (
    repository
    / "experimental_data/doi_10.48550_arXiv.2606.18791/data/10mT_S3.nxs"
)
data = load_ill_nexus(data_file)

q_range = (0.015, 0.10)
q_min, q_max = q_range
annulus_weight = fractional_annulus_weights(
    data.q_z, data.q_y, q_min=q_min, q_max=q_max, subpixels=7
)

exponent = 1.0
indices = angular_mode_indices(
    angular_modes=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    m_max=35,
)

analysis = standard_analyzer(
    data.q_z,
    data.q_y,
    data.intensity,
    uncertainty=data.uncertainty,
    sample_weight=annulus_weight,
    beam_centering=True,
    q_range=q_range,
    indices=indices,
    radial_measure_exponent=exponent,
    piecewise_linear_reference=True,
    reference_radial_segments=35,
    reference_knot_spacing="q",
    output_path=output_directory / "experimental_10mT_fit.png",
)
