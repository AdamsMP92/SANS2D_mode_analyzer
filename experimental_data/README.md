# Experimental data

Each experimental data set is self-contained and documents its own
provenance, format, analysis assumptions, and generated results:

- [`doi_10.48550_arXiv.2606.18791`](doi_10.48550_arXiv.2606.18791/README.md):
  polarized spin-flip SANS from the iron-oxide nanoflower study;
- [`doi_10.1103_5yc2-pv4y`](doi_10.1103_5yc2-pv4y/README.md): a curated
  two-file benchmark subset associated with the polarized
  magnetic-neutron-scattering study;
- [`NuMagSANS_IVR40`](NuMagSANS_IVR40/README.md): Cartesian spin-flip maps for
  the IVR40_1 and IVR40_2 simulation outputs.

Input files are stored in the `data/` directory of each collection. The
NuMagSANS collection preserves its individual run directories inside `data/`.
Analysis scripts, documentation, and generated `analysis_results/` remain at
the collection level.

The standard analyses can be regenerated from the repository root with
`pixi run analyze-numagsans-ivr40`, `pixi run analyze-arxiv-2606-18791`, and
`pixi run analyze-prl-135-196706`.
