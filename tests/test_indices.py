import pytest

from sans2d_mode_analyzer import even_angular_indices, rectangular_indices


def test_rectangular_indices_have_deterministic_real_fourier_order():
    assert rectangular_indices(n_max=2, m_max=1) == [
        (0, 0),
        (1, 0),
        (-1, 0),
        (2, 0),
        (-2, 0),
        (0, 1),
        (1, 1),
        (-1, 1),
        (2, 1),
        (-2, 1),
    ]


def test_even_indices_remove_all_odd_angular_orders():
    indices = even_angular_indices(n_max=5, m_max=2)
    assert indices
    assert all(abs(n) % 2 == 0 for n, _ in indices)
    assert len(indices) == 3 * (1 + 2 * 2)


@pytest.mark.parametrize("n_max,m_max", [(-1, 1), (1, -1)])
def test_negative_truncation_limits_are_rejected(n_max, m_max):
    with pytest.raises(ValueError):
        rectangular_indices(n_max=n_max, m_max=m_max)
