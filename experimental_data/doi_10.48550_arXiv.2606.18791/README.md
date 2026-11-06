# 220 nm iron-oxide nanoflower data

This directory contains the four supplied two-dimensional spin-flip SANS
NeXus files for nominal fields of 10 mT, 30 mT, 100 mT, and 1 T. They are
byte-for-byte copies of the files used in the associated local manuscript
analysis.

The data are associated with the following preprint:

> V. Rai et al., "Polarized neutron scattering as a probe for vortex-type
> spin correlations in iron oxide multicore assemblies," arXiv:2606.18791
> (2026). [https://doi.org/10.48550/arXiv.2606.18791](https://doi.org/10.48550/arXiv.2606.18791)

## Data interpretation

- `10mT_S3.nxs` and `30mT_S3.nxs` contain
  `0.5 * (I^(+-) + I^(-+))`;
- `100mT_S3.nxs` and `1T_S3.nxs` contain only the `+-` channel because the
  polarization-analysis correction of the corresponding `-+` data was not
  considered reliable;
- `qx1` is the applied-field direction and is denoted `q_z` by the analyzer;
- `qy1` is the transverse detector direction `q_y`;
- reciprocal coordinates are converted from inverse Angstroms to inverse
  nanometres by multiplication with 10.

## Standard analysis

Run all four fields from the repository root with

```text
pixi run analyze-arxiv-2606-18791
```

The script [`analyze_all.py`](analyze_all.py) applies the full rectangular
mode basis with `mode_limits=(25, 25)` over
`q_range=(0.015, 0.100)` in `nm^-1`. Detector entries with nonfinite intensity or
nonpositive/nonfinite supplied uncertainty are omitted. The supplied
uncertainties are used in a weighted least-squares fit, and the circular
annulus boundary is represented by `7 x 7` subpixel fractional-area weights.

No refined scattering-centre correction, parity folding, or
polarization-channel reconstruction is applied. The resulting PNG summaries
are written to [`analysis_results`](analysis_results/README.md). They are a
common descriptive baseline, not a replacement for the documented
experiment-specific reduction.

## SHA-256 checksums

```text
f79966cef3caefb7227230741574f2ecd35ba785cdcae36fcdb5f8fd1d9a9bb1  10mT_S3.nxs
8d906725b511a719cc820483290643b7573ad30ed4649fca51e7f3313dc3b9d9  30mT_S3.nxs
7ca33ec80248ad6316503894a762b0cd272a9b54337254397502fc0e424f8d55  100mT_S3.nxs
fb2f183e8bf14d6618515996c68b366890f4b514498dd60ca4e0ec351b1578a8  1T_S3.nxs
```
