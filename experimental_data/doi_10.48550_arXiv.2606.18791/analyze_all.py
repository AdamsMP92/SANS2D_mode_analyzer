"""Run the standard mode analysis for all nanoflower detector files."""

from pathlib import Path

from sans2d_mode_analyzer import (
    fractional_annulus_weights,
    load_ill_nexus,
    standard_analyzer,
)


dataset_directory = Path(__file__).resolve().parent
data_directory = dataset_directory / "data"
results_directory = dataset_directory / "analysis_results"

data_files = [
    "10mT_S3.nxs",
    "30mT_S3.nxs",
    "100mT_S3.nxs",
    "1T_S3.nxs",
]

for filename in data_files:
    data = load_ill_nexus(data_directory / filename)

    output = results_directory / f"{Path(filename).stem}_standard_analysis.png"
    annulus_weight = fractional_annulus_weights(
        data.q_z, data.q_y, q_min=0.015, q_max=0.10, subpixels=7
    )
    standard_analyzer(
        data.q_z,
        data.q_y,
        data.intensity,
        uncertainty=data.uncertainty,
        sample_weight=annulus_weight,
        q_range=(0.015, 0.1),
        mode_limits=(25, 25),
        output_path=output,
    )
    print(f"wrote {output.relative_to(dataset_directory)}")
