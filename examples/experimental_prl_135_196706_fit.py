"""Analyze the low-field nanoporous-Fe map associated with PRL 135."""

from pathlib import Path
import sys

repository = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository / "src"))

from sans2d_mode_analyzer import (
    load_ascii_detector_data,
    standard_analyzer,
)


output_directory = repository / "examples" / "output"
output_directory.mkdir(parents=True, exist_ok=True)
data_file = (
    repository
    / "experimental_data/doi_10.1103_5yc2-pv4y/data/igc_nanoporous_Fe_0p1T.dat"
)
data = load_ascii_detector_data(data_file)

output1 = output_directory / "experimental_prl_135_196706_raw_fit.png"
output2 = output_directory / "experimental_prl_135_196706_centered_fit.png"

q_range = (0.005, 0.030)
mode_limits = (30, 30)

# Analysis without beam centering correction.
analysis = standard_analyzer(
    data.q_z,
    data.q_y,
    data.intensity,
    uncertainty=data.uncertainty,
    q_range=q_range,
    mode_limits=mode_limits,
    log_intensity_colorbar=False,
    output_path=output1,
)

# Repeat the same analysis with automatic point-cloud beam centering.
centered_analysis = standard_analyzer(
    data.q_z,
    data.q_y,
    data.intensity,
    uncertainty=data.uncertainty,
    beam_centering=True,
    q_range=q_range,
    mode_limits=mode_limits,
    log_intensity_colorbar=False,
    output_path=output2,
)
