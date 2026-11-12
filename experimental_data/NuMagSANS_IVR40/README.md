# NuMagSANS IVR40 spin-flip simulations

This collection contains two retained Cartesian maps of the simulated average
spin-flip cross section `S_sf` from the NuMagSANS IVR40 outputs labelled
`IVR40_1` and `IVR40_2`. They are simulation benchmarks, not experimental
detector counts.

## Stored format

Each `SANS2D_cartesian.csv` file has the columns

```text
qz,qy,q,theta,S_sf
```

The maps are equally spaced `161 x 161` Cartesian grids. They were derived
before inclusion in this repository from fine polar NuMagSANS output by area
averaging onto the Cartesian cells. The original polar output and conversion
script are not retained here, so the CSV files are the reproducible starting
point for the packaged examples.

## Standard analysis

Run both maps from the repository root with

```text
pixi run analyze-numagsans-ivr40
```

The script [`analyze_all.py`](analyze_all.py) fits the full rectangular basis
with `mode_limits=(2, 10)` over `q_range=(0.05, 0.50)`. Fractional annulus
coverage is evaluated with a `7 x 7` subpixel grid. No measurement
uncertainties are supplied; consequently, the displayed coefficient
uncertainties and block-test references are conditional on the residual-scaled
finite-expansion model and must not be interpreted as experimental counting
statistics.

Results are written to [`analysis_results`](analysis_results/README.md).

