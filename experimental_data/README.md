# Experimental data

Each experimental data set is self-contained and documents its own
provenance, format, analysis assumptions, and generated results:

- [`doi_10.48550_arXiv.2606.18791`](doi_10.48550_arXiv.2606.18791/README.md):
  polarized spin-flip SANS from the iron-oxide nanoflower study;
- [`doi_10.1103_5yc2-pv4y`](doi_10.1103_5yc2-pv4y/README.md): a curated
  two-file benchmark subset associated with the polarized
  magnetic-neutron-scattering study;
- [`NuMagSANS_IVR40/data`](NuMagSANS_IVR40/data/): Cartesian spin-flip
  maps for the IVR40_1 and IVR40_2 simulation outputs.

Input files are stored in the `data/` directory of each collection. The
NuMagSANS collection preserves its individual run directories inside `data/`.
Analysis scripts, documentation, and generated `analysis_results/` remain at
the collection level.

The standard analyses can be regenerated from the repository root with
`pixi run analyze-arxiv-2606-18791` and
`pixi run analyze-prl-135-196706`.

<!-- covariance-comparison:start -->
## Covariance-correction comparison

All six standard analyses were rerun after the absolute-noise covariance
correction. Their existing q ranges, mode limits (25, 25), sample selection,
area weights, and centering settings were retained. The standard PNGs above
were regenerated. The comparison evaluates both covariances at exactly the
same fitted coefficients; intensities, raw powers, and radial spectra are fixed.

Reproduce both the detector analyses and the lightweight summary:

~~~sh
PYTHONPATH=src python experimental_data/compare_covariance.py
python experimental_data/summarize_covariance.py
~~~

Both scripts use one BLAS worker and a noninteractive plotting backend.
The analysis also checks memory pressure before each data set on macOS.

The following entries are median percentage changes. Power errors cover
51 angular blocks; spectral bands cover all 51 branches/orders on the
same 300-point q grid. Rejection counts are formal per-data-set calculations
under the assumed noise model at alpha=0.05, not classifications of
physically identified contributions.

| Data set | Δ σ(P_n) [%] | Δ σ(I_n) [%] | Δ T_n [%] | Nominal model rejections, old → new | Holm model rejections, old → new |
|---|---:|---:|---:|---:|---:|
| [10 mT](doi_10.48550_arXiv.2606.18791/analysis_results/covariance_comparison/10mT_S3_comparison.png) | -0.981 | -0.021 | +2.282 | 3 → 4 | 3 → 3 |
| [30 mT](doi_10.48550_arXiv.2606.18791/analysis_results/covariance_comparison/30mT_S3_comparison.png) | -1.087 | -0.021 | +1.420 | 4 → 4 | 3 → 3 |
| [100 mT](doi_10.48550_arXiv.2606.18791/analysis_results/covariance_comparison/100mT_S3_comparison.png) | -0.920 | -0.020 | +2.347 | 2 → 2 | 1 → 1 |
| [1 T](doi_10.48550_arXiv.2606.18791/analysis_results/covariance_comparison/1T_S3_comparison.png) | -1.194 | -0.021 | +2.296 | 2 → 2 | 0 → 0 |
| [PRL 0.1 T](doi_10.1103_5yc2-pv4y/analysis_results/covariance_comparison/igc_nanoporous_Fe_0p1T_comparison.png) | ≈0 | ≈0 | ≈0 | 47 → 47 | 46 → 46 |
| [PRL 9 T](doi_10.1103_5yc2-pv4y/analysis_results/covariance_comparison/igc_nanoporous_Fe_9T_15m_5K_comparison.png) | ≈0 | ≈0 | ≈0 | 10 → 10 | 4 → 4 |

Nominal decision changes: 10mT_S3: cos 5 (p=0.070697 → 0.048392).
Holm correction uses the complete family of 51 existing block tests per
data set. Its significant block sets remain unchanged; no reduced-model selection or refit is introduced.

The strongest local reduction of a spectral band is 16.79%.
These larger effects occur near annulus boundaries; the median over the
whole radial grid is much smaller. The two PRL benchmarks have unit sample
weights and remain unchanged to numerical precision.

Three covariances are distinguished: legacy_normal uses the previous
inverse normal matrix; precision_qr evaluates the same precision-model
formula with QR factors; corrected propagates the actual pixel errors.
The maximum relative Frobenius difference between the first two is
1.82e-11; numerical rank changes are absent in all six cases.
Thus the observed changes are attributable to geometric weights rather
than the switch to a more stable matrix factorization.

The comparisons retain the package's diagonal-noise, fixed-design model.
They do not propagate uncertainty of fitted centers or reduction-induced
pixel correlations. The pre-existing PRL coordinate-unit caveat still applies.

The p-values and Holm counts are supplementary, model-conditional
calculations. Their noise assumptions have not been established by
the synthetic calibration tests for these experimental maps. They do
not measure the frequency of experimental artifacts or physical
misidentifications. The standard plots do not highlight Holm decisions.
For a parity-even observable, odd powers can instead provide a
descriptive empirical reference level; this is not assigned a formal
significance threshold, and physical odd contributions must first
be excluded.

Detailed per-block CSV files, metadata JSON files, and comparison PNGs
are in each collection's analysis_results/covariance_comparison directory.
The heat maps use a symmetric scale at the 99th percentile of absolute
changes; larger values are color-saturated. Full numerical ranges are
retained in the tables and CSV/JSON summaries.

- [Summary CSV](analysis_results/covariance_comparison/summary.csv)
- [Combined metadata and results](analysis_results/covariance_comparison/summary.json)
- [Raw and Holm-adjusted p-values](analysis_results/covariance_comparison/holm_blocks.csv)

![Covariance comparison](analysis_results/covariance_comparison/overview.png)
<!-- covariance-comparison:end -->
