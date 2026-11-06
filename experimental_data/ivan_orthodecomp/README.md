# Ivan OrthoDecomp data

This directory contains seven four-column ASCII detector files copied from
the earlier MATLAB `OrthoDecomp` analysis directory. The accompanying legacy
`Test_IvanData.m` script identifies the columns as

```text
q_z    q_y    intensity    uncertainty
```

and labels the reciprocal-space coordinates in `nm^-1`. The condition names
are retained verbatim because no separate experimental metadata file was
present in the source directory.

The data are associated with the following article:

> I. Titov et al., "Spin-Disorder-Induced Angular Anisotropy in Polarized
> Magnetic Neutron Scattering," *Physical Review Letters* **135**, 196706
> (2025). [https://doi.org/10.1103/5yc2-pv4y](https://doi.org/10.1103/5yc2-pv4y)

## Files

| File | Numeric rows | Observed q range (nm^-1) |
|---|---:|---:|
| `000000.dat` | 62105 | 0.003863--0.594523 |
| `0p05T_15m_5K.dat` | 16251 | 0.003765--0.057577 |
| `0p1T_8m_300K.dat` | 16214 | 0.007430--0.107651 |
| `119495.dat` | 16228 | 0.007611--0.107967 |
| `5A.dat` | 15876 | 0.000194--0.042803 |
| `9T_15m_5K.dat` | 16251 | 0.003765--0.057577 |
| `9T_8m_300K.dat` | 16214 | 0.007430--0.107651 |

`5A.dat` begins with the nonnumeric line `ASCII data`. Several files contain
zero or negative corrected intensities. These values are preserved exactly
and must not be interpreted as raw counts.

The source directory also contained old MATLAB programs and three
million-row helper exports (`aaaaa.dat`, `aaaaa.csv`, and `SANSData251.csv`).
They were not copied because their surrounding code indicates that they are
intermediate or synthetic analysis data rather than independent experimental
conditions.

## Standard analysis

Run all seven detector files from the repository root with

```text
pixi run analyze-ivan
```

The script [`analyze_all.py`](analyze_all.py) applies the full rectangular
mode basis with `n_max = 8` and `m_max = 8` over the legacy MATLAB interval
`0.005 <= q <= 0.030 nm^-1`. Rows with nonfinite intensity or
nonpositive/nonfinite supplied uncertainty are omitted; finite negative
corrected intensities remain included. The fit is currently unweighted.

The ASCII files are analyzed directly as detector point lists without
interpolation onto a new grid. Results are written to
[`analysis_results`](analysis_results/README.md).

## SHA-256 checksums

```text
91b9a0ea9361fab1c4b0b82d553aa85b6a822014463e416439d148b6b5547bca  000000.dat
ebd0c07c964a1a38613eb59cf93828f8e512c4befc6848bdc226b454995f718d  0p05T_15m_5K.dat
2a97ebbf1607a104648cfe203d28a3d2d97ccc16fb811f0b8f3a3590e18fa370  0p1T_8m_300K.dat
6a37ea249dd874db855d1f88c03eb57255d29166194317ab0226fdee333f9b81  119495.dat
8a21570340bbaffbc0b61fb40a645c0631a40a1e69f3327b9214dca12ca38bf9  5A.dat
621af78e4cb6d84cacd41c8ae146babb42d3b081a797248258d4abd20089270a  9T_15m_5K.dat
a69c62ba752300f526470c49b53b80f311f68b33143df1db1523445d3dfde0e8  9T_8m_300K.dat
```
