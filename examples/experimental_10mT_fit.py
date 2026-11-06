"""Apply the standard mode analyzer to the lowest-field experimental data."""

from pathlib import Path

from sans2d_mode_analyzer import load_ill_nexus, standard_analyzer


repository = Path(__file__).resolve().parents[1]
data_file = repository / "experimental_data/220nm_nanoflower/10mT_S3.nxs"
data = load_ill_nexus(data_file)

analysis = standard_analyzer(
    data.q_z,
    data.q_y,
    data.intensity,
    q_min=0.015,
    q_max=0.050,
    n_max=8,
    m_max=8,
    output_path=repository / "examples/experimental_10mT_fit.png",
)
