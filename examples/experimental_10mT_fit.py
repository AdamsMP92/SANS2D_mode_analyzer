"""Apply the standard mode analyzer to the lowest-field experimental data."""

from pathlib import Path

from sans2d_mode_analyzer import (
    load_ill_nexus,
    mode_convergence_standard_analyzer,
    standard_analyzer,
)


repository = Path(__file__).resolve().parents[1]
data_file = repository / "experimental_data/220nm_nanoflower/10mT_S3.nxs"
data = load_ill_nexus(data_file)


convergence_analysis = False
if convergence_analysis:
    convergence = mode_convergence_standard_analyzer(
        data.q_z,
        data.q_y,
        data.intensity,
        q_min=0.015,
        q_max=0.10,
        n_max_limit=22,
        m_max_limit=22,
        output_path=repository / "examples/experimental_10mT_fit.png",
    )

    print(f"selected n_max = {convergence.selected_n_max}")
    print(f"selected m_max = {convergence.selected_m_max}")
    analysis = convergence.standard_analysis
else:
    n_max = 25
    m_max = 25
    analysis = standard_analyzer(
        data.q_z,
        data.q_y,
        data.intensity,
        q_min=0.015,
        q_max=0.10,
        n_max=n_max,
        m_max=m_max,
        output_path=repository / "examples/experimental_10mT_fit.png",
    )
