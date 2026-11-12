"""Apply the standard mode analyzer to the lowest-field experimental data."""

from pathlib import Path

from sans2d_mode_analyzer import (
    fractional_annulus_weights,
    load_ill_nexus,
    mode_convergence_standard_analyzer,
    standard_analyzer,
)


repository = Path(__file__).resolve().parents[1]
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


convergence_analysis = False
if convergence_analysis:
    convergence = mode_convergence_standard_analyzer(
        data.q_z,
        data.q_y,
        data.intensity,
        uncertainty=data.uncertainty,
        sample_weight=annulus_weight,
        q_range=q_range,
        mode_limits=(22, 22),
        output_path=output_directory / "experimental_10mT_fit.png",
    )

    print(f"selected n_max = {convergence.selected_n_max}")
    print(f"selected m_max = {convergence.selected_m_max}")
    analysis = convergence.standard_analysis
else:
    mode_limits = (25, 25)
    analysis = standard_analyzer(
        data.q_z,
        data.q_y,
        data.intensity,
        uncertainty=data.uncertainty,
        sample_weight=annulus_weight,
        beam_centering=True,
        q_range=q_range,
        mode_limits=mode_limits,
        output_path=output_directory / "experimental_10mT_fit.png",
    )
