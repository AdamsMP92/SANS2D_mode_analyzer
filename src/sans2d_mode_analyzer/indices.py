"""Mode-index validation and common truncation schemes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ModeIndex = tuple[int, int]


def _validate_limit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _signed_angular_orders(n_max: int) -> list[int]:
    """Return 0, +1, -1, +2, -2, ... in deterministic basis order."""
    return [0] + [signed for n in range(1, n_max + 1) for signed in (n, -n)]


def rectangular_indices(n_max: int, m_max: int) -> list[ModeIndex]:
    """Return all real Fourier/Legendre modes in a rectangular truncation.

    The returned order is radial-major.  For each ``m`` it is
    ``n = 0, +1, -1, +2, -2, ...``.  Positive angular indices denote cosine
    modes and negative indices denote sine modes.
    """
    n_limit = _validate_limit(n_max, "n_max")
    m_limit = _validate_limit(m_max, "m_max")
    angular = _signed_angular_orders(n_limit)
    return [(n, m) for m in range(m_limit + 1) for n in angular]


def even_angular_indices(n_max: int, m_max: int) -> list[ModeIndex]:
    """Return a rectangular truncation containing only even angular orders."""
    return [
        (n, m)
        for n, m in rectangular_indices(n_max=n_max, m_max=m_max)
        if abs(n) % 2 == 0
    ]


def random_mode_indices(
    n_max: int,
    m_max: int,
    n_coeffs: int,
    *,
    angular_parity: Literal["all", "even", "odd"] = "all",
    seed: int | None = None,
) -> list[ModeIndex]:
    """Select a reproducible random subset of a rectangular truncation.

    ``n_coeffs`` distinct modes are sampled without replacement.  The optional
    ``angular_parity`` filter restricts the candidate set according to
    ``abs(n)`` before sampling; the constant mode therefore belongs to the
    even set.  Although the subset is random, the returned modes retain the
    canonical radial-major order used by :func:`rectangular_indices`.

    This helper constructs controlled synthetic test scenarios.  It does not
    infer or select modes from measured data.
    """
    count = _validate_limit(n_coeffs, "n_coeffs")
    if count == 0:
        raise ValueError("n_coeffs must be a positive integer")
    if angular_parity not in {"all", "even", "odd"}:
        raise ValueError("angular_parity must be 'all', 'even', or 'odd'")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be a non-negative integer or None")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be a non-negative integer or None")

    candidates = rectangular_indices(n_max=n_max, m_max=m_max)
    if angular_parity != "all":
        requested_remainder = 0 if angular_parity == "even" else 1
        candidates = [
            index for index in candidates if abs(index[0]) % 2 == requested_remainder
        ]
    if count > len(candidates):
        raise ValueError(
            f"n_coeffs={count} exceeds the {len(candidates)} modes available "
            f"for angular_parity={angular_parity!r}"
        )

    random = np.random.default_rng(seed)
    selected_positions = np.sort(
        random.choice(len(candidates), size=count, replace=False)
    )
    return [candidates[int(position)] for position in selected_positions]


def angular_mode_indices(
    angular_modes: Iterable[int],
    m_max: int,
) -> list[ModeIndex]:
    """Combine selected signed angular modes with all radial modes to ``m_max``.

    The input order of ``angular_modes`` is retained within every radial block.
    For example, ``angular_mode_indices([0, 2, -2], 1)`` returns
    ``[(0, 0), (2, 0), (-2, 0), (0, 1), (2, 1), (-2, 1)]``.
    """
    m_limit = _validate_limit(m_max, "m_max")
    angular: list[int] = []
    for position, n in enumerate(angular_modes):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise TypeError(
                f"angular mode at position {position} must be an integer"
            )
        angular.append(int(n))
    if not angular:
        raise ValueError("at least one angular mode is required")
    if len(set(angular)) != len(angular):
        raise ValueError("angular modes must not contain duplicates")
    return [(n, m) for m in range(m_limit + 1) for n in angular]


def validate_indices(indices: Iterable[Sequence[int]]) -> tuple[ModeIndex, ...]:
    """Normalize an arbitrary two-column index iterable.

    Duplicate modes are rejected because they make the coefficient mapping
    ambiguous and the design matrix rank deficient by construction.
    """
    normalized: list[ModeIndex] = []
    for position, index in enumerate(indices):
        try:
            values = tuple(index)
        except TypeError as error:
            raise TypeError(f"mode index {position} is not a two-item sequence") from error
        if len(values) != 2:
            raise ValueError(f"mode index {position} must contain exactly (n, m)")
        n, m = values
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise TypeError(f"angular index n at position {position} must be an integer")
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
            raise TypeError(f"radial index m at position {position} must be an integer")
        item = (int(n), int(m))
        if item[1] < 0:
            raise ValueError(f"radial index m at position {position} must be non-negative")
        normalized.append(item)

    if not normalized:
        raise ValueError("at least one mode index is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("mode indices must not contain duplicates")
    return tuple(normalized)


def indices_array(indices: Iterable[Sequence[int]]) -> NDArray[np.int64]:
    """Return validated indices as an integer array with shape ``(K, 2)``."""
    return np.asarray(validate_indices(indices), dtype=np.int64)
