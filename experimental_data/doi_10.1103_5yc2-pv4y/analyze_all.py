"""Run the centred standard analysis for the retained PRL detector maps."""

from pathlib import Path

from sans2d_mode_analyzer import load_ascii_detector_data, standard_analyzer


dataset_directory = Path(__file__).resolve().parent
data_directory = dataset_directory / "data"
results_directory = dataset_directory / "analysis_results"
data_files = sorted(data_directory.glob("*.dat"))

for data_file in data_files:
    data = load_ascii_detector_data(data_file)

    output = results_directory / f"{data_file.stem}_standard_analysis.png"
    standard_analyzer(
        data.q_z,
        data.q_y,
        data.intensity,
        uncertainty=data.uncertainty,
        beam_centering=True,
        q_range=(0.005, 0.030),
        mode_limits=(25, 25),
        output_path=output,
    )
    print(f"wrote {output.relative_to(dataset_directory)}")
