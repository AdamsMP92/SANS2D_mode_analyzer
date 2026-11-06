# Detector-map benchmarks associated with DOI 10.1103/5yc2-pv4y

This directory contains a curated two-file subset of four-column detector maps
copied from the earlier MATLAB `OrthoDecomp` directory. Project context
associates the files with

> I. Titov et al., "Spin-Disorder-Induced Angular Anisotropy in Polarized
> Magnetic Neutron Scattering," *Physical Review Letters* **135**, 196706
> (2025). [https://doi.org/10.1103/5yc2-pv4y](https://doi.org/10.1103/5yc2-pv4y)

The source files themselves contain no sample, instrument, observable, or unit
metadata. They should therefore be treated as analysis benchmarks rather than
as a self-contained archival reproduction of a particular paper figure.

## Format and retained files

Each numeric row contains

```text
q_z    q_y    intensity    uncertainty
```

| Current file | Original file | Numeric rows | Observed numerical q range |
|---|---|---:|---:|
| `igc_nanoporous_Fe_0p1T.dat` | `000000.dat` | 62105 | 0.003863--0.594523 |
| `igc_nanoporous_Fe_9T_15m_5K.dat` | `9T_15m_5K.dat` | 16251 | 0.003765--0.057577 |

Both files are identified with the inert-gas-condensed nanoporous-Fe sample:
the first with the low-field measurement shown in the paper at `0.1 T`, and
the second with a high-field condition at `9 T`. These sample assignments are
based on the project context and are not encoded in the source files
themselves. The `15m` and `5K` components are inherited from the original
high-field filename and have not been independently verified here.
Finite negative corrected intensities are retained and must not be interpreted
as raw detector counts.

## Curation audit

The original directory contained seven detector maps. No byte-identical or
numerically identical intensity data sets were found. Two pairs use exactly
the same reciprocal-coordinate grids but have distinct intensities:

- `0p05T_15m_5K.dat` and `9T_15m_5K.dat` have an intensity correlation of
  approximately 0.752 over their common valid pixels;
- `0p1T_8m_300K.dat` and `9T_8m_300K.dat` have an intensity correlation of
  approximately 0.859 over their common valid pixels.

They are therefore different measurements, not duplicates. The repository was
nevertheless reduced to a compact benchmark subset: `000000.dat` was retained
because it is the file explicitly loaded by the supplied legacy MATLAB
analysis and is identified with the paper's low-field nanoporous-Fe data;
`9T_15m_5K.dat` was retained as a second, explicitly named high-field
condition. The other five files were removed
because their provenance is incomplete and their diagnostic mode analyses were
less suitable as compact examples. This is a curation decision, not a claim of
physical redundancy; the removed files remain recoverable from Git history.

The original MATLAB directory also contained large helper exports named
`aaaaa.dat`, `aaaaa.csv`, and `SANSData251.csv`. These were never included
because their surrounding code identifies them as intermediate or synthetic
analysis data rather than independent experimental conditions.

## Coordinate-unit caveat

The legacy MATLAB script labels the reciprocal coordinates as `nm^-1` and uses
the numerical interval `0.005 <= q <= 0.030`. The paper reports an experimental
range of approximately `0.04 <= q <= 1.0 nm^-1`. This discrepancy is consistent
with the stored coordinates instead being in inverse Angstroms, for which the
legacy numerical interval would correspond to `0.05 <= q <= 0.30 nm^-1`.
Because the files contain no unit metadata, the analyzer does not apply this
factor-of-ten conversion and the documentation refers to a numerical q range.

## Standard analysis

Run both retained maps from the repository root with

```text
pixi run analyze-prl-135-196706
```

The script [`analyze_all.py`](analyze_all.py) applies the full rectangular mode
basis with `mode_limits=(25, 25)` over the numerical interval
`q_range=(0.005, 0.030)`. The supplied uncertainties are used in weighted least
squares. Nonfinite intensities and nonpositive or nonfinite uncertainties are
excluded internally by `standard_analyzer`; finite negative intensities remain
included.

Parity-based point-cloud centering is enabled for every map. A temporary
regular grid is used only to estimate the reciprocal-space centre shift; the
subsequent modal fit retains the original point samples, intensities, and
uncertainties. The centre is estimated independently for each retained map.
Results are written to [`analysis_results`](analysis_results/README.md).

The compact
[`examples/experimental_prl_135_196706_fit.py`](../../examples/experimental_prl_135_196706_fit.py)
compares the uncentered and parity-centered analyses of
`igc_nanoporous_Fe_0p1T.dat`.

## SHA-256 checksums

```text
91b9a0ea9361fab1c4b0b82d553aa85b6a822014463e416439d148b6b5547bca  igc_nanoporous_Fe_0p1T.dat
621af78e4cb6d84cacd41c8ae146babb42d3b081a797248258d4abd20089270a  igc_nanoporous_Fe_9T_15m_5K.dat
```
