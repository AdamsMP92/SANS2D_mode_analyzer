# Experimental data

Each experimental data set is self-contained and documents its own
provenance, format, analysis assumptions, and generated results:

- [`220nm_nanoflower`](220nm_nanoflower/README.md): polarized spin-flip SANS
  from the iron-oxide nanoflower study;
- [`ivan_orthodecomp`](ivan_orthodecomp/README.md): four-column detector data
  associated with the polarized magnetic-neutron-scattering study.

The standard analyses can be regenerated from the repository root with
`pixi run analyze-nanoflower` and `pixi run analyze-ivan`.
