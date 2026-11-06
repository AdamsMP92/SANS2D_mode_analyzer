import numpy as np

from sans2d_mode_analyzer import (
    evaluate_expansion,
    mode_convergence_standard_analyzer,
)


def test_mode_convergence_standard_analyzer_finds_required_orders(tmp_path):
    q_min = 0.2
    q_max = 1.0
    q_axis = np.linspace(q_min, q_max, 14)
    theta_axis = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
    q, theta = np.meshgrid(q_axis, theta_axis, indexing="ij")
    q_z = q * np.cos(theta)
    q_y = q * np.sin(theta)
    intensity = evaluate_expansion(
        q,
        theta,
        coefficients=[3.0, -0.75],
        indices=[(0, 0), (4, 2)],
        q_min=q_min,
        q_max=q_max,
    )

    result = mode_convergence_standard_analyzer(
        q_z,
        q_y,
        intensity,
        q_min=q_min,
        q_max=q_max,
        n_max_limit=6,
        m_max_limit=4,
        prediction_tolerance=1.0e-10,
        coefficient_tolerance=1.0e-10,
        relative_mse_tolerance=1.0e-20,
        output_path=tmp_path / "converged.png",
    )

    assert result.selected_n_max == 4
    assert result.selected_m_max == 2
    assert result.stabilized_within_limits
    assert result.standard_analysis.output_path.exists()
