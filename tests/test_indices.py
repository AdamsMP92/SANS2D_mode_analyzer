import numpy as np
import pytest

from sans2d_mode_analyzer import (
    angular_mode_indices,
    even_angular_indices,
    rectangular_indices,
)


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


def test_selected_angular_modes_are_combined_with_every_radial_order():
    assert angular_mode_indices([0, 4, -2], m_max=2) == [
        (0, 0),
        (4, 0),
        (-2, 0),
        (0, 1),
        (4, 1),
        (-2, 1),
        (0, 2),
        (4, 2),
        (-2, 2),
    ]


def test_selected_angular_modes_accept_numpy_integers():
    assert angular_mode_indices(np.array([0, 2], dtype=np.int64), m_max=0) == [
        (0, 0),
        (2, 0),
    ]


@pytest.mark.parametrize(
    ("angular_modes", "error"),
    [([], ValueError), ([0, 0], ValueError), ([0, 1.5], TypeError)],
)
def test_selected_angular_modes_reject_invalid_lists(angular_modes, error):
    with pytest.raises(error):
        angular_mode_indices(angular_modes, m_max=1)


@pytest.mark.parametrize("n_max,m_max", [(-1, 1), (1, -1)])
def test_negative_truncation_limits_are_rejected(n_max, m_max):
    with pytest.raises(ValueError):
        rectangular_indices(n_max=n_max, m_max=m_max)
