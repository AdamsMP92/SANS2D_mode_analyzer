"""Run the standard mode analysis for all nanoflower detector files."""

from pathlib import Path

import numpy as np

from sans2d_mode_analyzer import load_ill_nexus, standard_analyzer


dataset_directory = Path(__file__).resolve().parent
results_directory = dataset_directory / "analysis_results"

data_files = [
    "10mT_S3.nxs",
    "30mT_S3.nxs",
    "100mT_S3.nxs",
    "1T_S3.nxs",
]

for filename in data_files:
    data = load_ill_nexus(dataset_directory / filename)
    intensity = np.array(data.intensity, copy=True)
    invalid = ~np.isfinite(data.uncertainty) | (data.uncertainty <= 0.0)
    intensity[invalid] = np.nan

    output = results_directory / f"{Path(filename).stem}_standard_analysis.png"
    standard_analyzer(
        data.q_z,
        data.q_y,
        intensity,
        q_min=0.015,
        q_max=0.050,
        n_max=8,
        m_max=8,
        output_path=output,
    )
    print(f"wrote {output.relative_to(dataset_directory)}")
