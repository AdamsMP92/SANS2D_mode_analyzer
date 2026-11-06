import numpy as np

from sans2d_mode_analyzer import evaluate_expansion, standard_analyzer


def test_standard_analyzer_fits_and_writes_summary(tmp_path):
    q_min = 0.2
    q_max = 1.0
    axis = np.linspace(-q_max, q_max, 41)
    q_z, q_y = np.meshgrid(axis, axis)
    q = np.hypot(q_z, q_y)
    theta = np.arctan2(q_y, q_z)
    inside = (q >= q_min) & (q <= q_max)

    intensity = np.full(q.shape, np.nan)
    intensity[inside] = evaluate_expansion(
        q[inside],
        theta[inside],
        coefficients=[3.0, 0.5],
        indices=[(0, 0), (2, 1)],
        q_min=q_min,
        q_max=q_max,
    )

    output = tmp_path / "summary.png"
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_min=q_min,
        q_max=q_max,
        n_max=2,
        m_max=1,
        output_path=output,
    )

    assert output.exists()
    assert result.fit.mse < 1.0e-24
    assert result.fit_image.shape == intensity.shape
    assert result.powers.orders.tolist() == [0, 1, 2]
    assert result.powers.cosine_uncertainty is not None
    assert result.block_tests.cosine_statistic.shape == (3,)
    assert set(result.selected_indices_by_threshold) == {0.95, 0.98, 0.99}
    assert all(result.selected_indices_by_threshold.values())
    np.testing.assert_allclose(
        result.parity_fractions.even + result.parity_fractions.odd,
        1.0,
    )


def test_standard_analyzer_accepts_flat_detector_point_list(tmp_path):
    q_min = 0.2
    q_max = 1.0
    q_axis = np.linspace(q_min, q_max, 12)
    theta_axis = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    q, theta = np.meshgrid(q_axis, theta_axis, indexing="ij")
    q_z = (q * np.cos(theta)).ravel()
    q_y = (q * np.sin(theta)).ravel()
    intensity = evaluate_expansion(
        q.ravel(),
        theta.ravel(),
        coefficients=[2.0, 0.25],
        indices=[(0, 0), (-2, 1)],
        q_min=q_min,
        q_max=q_max,
    )

    output = tmp_path / "point-list-summary.png"
    result = standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_min=q_min,
        q_max=q_max,
        n_max=2,
        m_max=1,
        output_path=output,
    )

    assert output.exists()
    assert result.fit.mse < 1.0e-24
    assert result.fit_image.shape == intensity.shape
