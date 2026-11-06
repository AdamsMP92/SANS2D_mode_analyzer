"""Import helpers for experimental detector data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DetectorData:
    """Two-dimensional detector coordinates, intensity, and uncertainty."""

    q_z: NDArray[np.float64]
    q_y: NDArray[np.float64]
    intensity: NDArray[np.float64]
    uncertainty: NDArray[np.float64]


def load_ill_nexus(path: str | Path, q_scale: float = 10.0) -> DetectorData:
    """Load the ILL NeXus layout used by the bundled nanoflower data.

    The default scale converts the stored inverse-Angstrom coordinates to
    inverse nanometres.  ``qx1`` is returned as the field-direction ``q_z``.
    """
    with h5py.File(path, "r") as nexus:
        data = nexus["entry0/data1"]
        intensity = np.squeeze(np.asarray(data["MultiDetector1_data"], dtype=float))
        uncertainty = np.squeeze(np.asarray(data["error1"], dtype=float))
        q_z = np.squeeze(np.asarray(data["qx1"], dtype=float)) * q_scale
        q_y = np.squeeze(np.asarray(data["qy1"], dtype=float)) * q_scale

    if intensity.shape != q_z.shape:
        intensity = intensity.T
        uncertainty = uncertainty.T
    return DetectorData(q_z, q_y, intensity, uncertainty)


def load_ascii_detector_data(path: str | Path) -> DetectorData:
    """Load an ASCII point list with columns ``q_z q_y I uncertainty``.

    Nonnumeric lines, including the ``ASCII data`` header in one of the
    bundled Ivan data files, are skipped.
    """
    rows: list[tuple[float, float, float, float]] = []
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 4:
                continue
            try:
                rows.append(tuple(float(field) for field in fields))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"no four-column numeric detector data found in {path}")
    values = np.asarray(rows, dtype=float)
    return DetectorData(
        q_z=values[:, 0],
        q_y=values[:, 1],
        intensity=values[:, 2],
        uncertainty=values[:, 3],
    )
