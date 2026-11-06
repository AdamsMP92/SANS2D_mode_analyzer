"""Run the standard mode analysis for all Ivan detector point lists."""

from pathlib import Path

import numpy as np

from sans2d_mode_analyzer import load_ascii_detector_data, standard_analyzer


dataset_directory = Path(__file__).resolve().parent
results_directory = dataset_directory / "analysis_results"
data_files = sorted(dataset_directory.glob("*.dat"))

for data_file in data_files:
    data = load_ascii_detector_data(data_file)
    intensity = np.array(data.intensity, copy=True)
    invalid = ~np.isfinite(data.uncertainty) | (data.uncertainty <= 0.0)
    intensity[invalid] = np.nan

    output = results_directory / f"{data_file.stem}_standard_analysis.png"
    standard_analyzer(
        data.q_z,
        data.q_y,
        intensity,
        q_min=0.005,
        q_max=0.030,
        n_max=25,
        m_max=25,
        output_path=output,
    )
    print(f"wrote {output.relative_to(dataset_directory)}")
