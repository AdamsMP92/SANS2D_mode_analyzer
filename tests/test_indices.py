import numpy as np
import pytest

from sans2d_mode_analyzer import (
    angular_mode_indices,
    even_angular_indices,
    random_mode_indices,
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


def test_random_indices_are_reproducible_unique_and_canonically_ordered():
    first = random_mode_indices(5, 3, 11, seed=17)
    second = random_mode_indices(5, 3, 11, seed=17)
    candidates = rectangular_indices(5, 3)
    positions = [candidates.index(index) for index in first]

    assert first == second
    assert len(first) == len(set(first)) == 11
    assert positions == sorted(positions)


@pytest.mark.parametrize(("parity", "remainder"), [("even", 0), ("odd", 1)])
def test_random_indices_respect_angular_parity_filter(parity, remainder):
    indices = random_mode_indices(
        n_max=6,
        m_max=2,
        n_coeffs=8,
        angular_parity=parity,
        seed=4,
    )

    assert all(abs(n) % 2 == remainder for n, _ in indices)
    if parity == "odd":
        assert all(n != 0 for n, _ in indices)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n_coeffs": 0}, ValueError),
        ({"n_coeffs": 2, "angular_parity": "neither"}, ValueError),
        ({"n_coeffs": 3, "n_max": 0, "angular_parity": "odd"}, ValueError),
        ({"n_coeffs": 2, "seed": -1}, ValueError),
        ({"n_coeffs": 2, "seed": 1.5}, TypeError),
    ],
)
def test_random_indices_reject_invalid_scenarios(kwargs, error):
    arguments = {"n_max": 2, "m_max": 1, **kwargs}
    with pytest.raises(error):
        random_mode_indices(**arguments)


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
