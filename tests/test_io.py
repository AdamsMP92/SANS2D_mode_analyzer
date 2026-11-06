import h5py
import numpy as np

from sans2d_mode_analyzer import load_ascii_detector_data, load_ill_nexus


def test_load_ill_nexus_reads_and_scales_detector_arrays(tmp_path):
    path = tmp_path / "small.nxs"
    with h5py.File(path, "w") as nexus:
        data = nexus.create_group("entry0/data1")
        data["MultiDetector1_data"] = np.arange(6).reshape(2, 3, 1)
        data["error1"] = np.ones((2, 3, 1))
        data["qx1"] = np.full((2, 3, 1), 0.01)
        data["qy1"] = np.full((2, 3, 1), -0.02)

    detector = load_ill_nexus(path)
    assert detector.intensity.shape == (2, 3)
    np.testing.assert_allclose(detector.q_z, 0.1)
    np.testing.assert_allclose(detector.q_y, -0.2)


def test_load_ascii_detector_data_skips_text_header(tmp_path):
    path = tmp_path / "detector.dat"
    path.write_text(
        "ASCII data\n\n"
        "-0.1 0.2 3.0 0.4\n"
        "0.1 -0.2 5.0 0.6\n"
    )

    detector = load_ascii_detector_data(path)
    np.testing.assert_allclose(detector.q_z, [-0.1, 0.1])
    np.testing.assert_allclose(detector.q_y, [0.2, -0.2])
    np.testing.assert_allclose(detector.intensity, [3.0, 5.0])
    np.testing.assert_allclose(detector.uncertainty, [0.4, 0.6])
