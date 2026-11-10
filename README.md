# SANS2D mode analyzer

`sans2d-mode-analyzer` provides a small linear-analysis core for real-valued
two-dimensional SANS patterns on a circular annulus

\[
q_{\min} \le q \le q_{\max}, \qquad 0 \le \theta < 2\pi.
\]

The radial modes are normalized Legendre polynomials in a coordinate affine in
`q**2`, and are therefore orthonormal for the detector-plane measure `q dq`.
The angular modes form a real orthonormal Fourier basis.

More explicitly,

```text
x(q) = (2 q^2 - q_max^2 - q_min^2) / (q_max^2 - q_min^2)
R_m(q) = sqrt(2 (2m + 1) / (q_max^2 - q_min^2)) P_m(x(q))
T_0(theta) = 1 / sqrt(2 pi)
T_n(theta) = cos(n theta) / sqrt(pi)       for n > 0
T_n(theta) = sin(abs(n) theta) / sqrt(pi)  for n < 0
Phi_(n,m)(q, theta) = R_m(q) T_n(theta).
```

Consequently, the product modes are orthonormal for `q dq dtheta` over the
chosen annulus. By default the fitted coefficients minimize
`mean((data - design_matrix @ coefficients)**2)` over exactly the selected
columns. Experimental uncertainties and fractional detector-pixel weights can
instead activate weighted least squares as described below.

The standalone [supplementary methods](documentation/supplementary_methods.tex)
document describes the implemented expansion, fitting and covariance,
Parseval powers, angular block statistics, radial spectra, and detector
preprocessing in REVTeX 4.2 format. A compiled
[PDF](documentation/supplementary_methods.pdf) is included for reading.

## Mode-index convention

Every mode is selected by a two-dimensional index `(n, m)`:

- `m >= 0` is the radial Legendre degree;
- `n == 0` is the constant angular mode;
- `n > 0` is `cos(n theta)`;
- `n < 0` is `sin(abs(n) theta)`.

The sign of `n` keeps cosine and sine independently selectable without adding
a third index.

## Environment and tests

```text
pixi install
pixi run test
pixi run example
pixi run experimental-example
pixi run prl-example
pixi run numagsans-example
pixi run analyze-arxiv-2606-18791
pixi run analyze-prl-135-196706
```

The example creates a noisy synthetic detector pattern proportional to

```text
(F(q)^2 + 6 F'(q)^2) (1 + cos(theta)^2),
```

where `F` is the normalized spherical form-factor amplitude.  It then fits a
rectangular mode grid over `0.5 <= qR <= 5.0` and writes
`examples/synthetic_sphere_fit.png`.  The PNG contains the input, fit, and
residual in its upper row and the cosine and sine angular powers in its lower
row.  The complete beginner-oriented script is in
[`examples/synthetic_sphere_fit.py`](examples/synthetic_sphere_fit.py).
The detector construction, sphere model, white-noise addition, and insertion
onto the masked detector grid are separate short functions in
[`synthetic_models/model_functions.py`](synthetic_models/model_functions.py).

The repository also contains the supplied experimental NeXus files under
[`experimental_data/doi_10.48550_arXiv.2606.18791`](experimental_data/doi_10.48550_arXiv.2606.18791).
`examples/experimental_10mT_fit.py` applies the same standard analyzer to the
lowest-field data in the interval `0.015 <= q <= 0.100 nm^-1`.
Two retained four-column ASCII SANS benchmark maps from the earlier MATLAB
OrthoDecomp work are stored under
[`experimental_data/doi_10.1103_5yc2-pv4y`](experimental_data/doi_10.1103_5yc2-pv4y).
Each data set documents its own format, provenance, caveats, analysis script,
and generated results in its local README.
The compact `examples/experimental_prl_135_196706_fit.py` example applies the
standard analyzer to `igc_nanoporous_Fe_0p1T.dat` over the legacy numerical interval
`0.005 <= q <= 0.030`. It writes a raw fit and a second fit after a
parity-based centre refinement. Because the supplied data are a point list,
the centre is estimated on a small interpolation grid, but the corrected fit
uses the original intensities, uncertainties, and point coordinates with only
their reciprocal-space origin shifted.

`examples/numagsans_ivr40_3_spin_flip_fit.py` analyzes and compares the
simulated average spin-flip cross section `S_sf` from the IVR40 output indices
1 and 2. The original NuMagSANS output is a very fine `1000 x 1000` polar
product grid. The retained CSV files contain only `qz`, `qy`, `q`, `theta`,
and `S_sf`. The example removes the duplicated `theta = 2 pi` endpoint and
area-averages the polar samples onto an equally spaced `161 x 161` Cartesian
grid. The polar source samples are weighted by `q`, which accounts for the
Jacobian of `q dq dtheta`; the resulting square cells can then be analyzed as
ordinary equal-area detector pixels. The derived maps are stored as
`SANS2D_cartesian.csv`. The example writes one standard-analysis overview per
data set and a direct comparison of the fitted radial spectra `I_0^c(q)` and
`I_2^c(q)`. Since these deterministic simulation data have no experimental
uncertainty column, fit-uncertainty diagnostics are conditional on the
residual of the finite mode truncation.

## Standard analyzer

`standard_analyzer(...)` accepts either two-dimensional detector arrays or
one-dimensional detector point lists `q_z`, `q_y`, and `intensity`. It selects
the requested `q_range=(q_min, q_max)`, creates the complete mode grid from
`mode_limits=(n_max, m_max)`, performs the fit, and writes the multi-panel
summary PNG. The returned result exposes the plotted radial coefficient
functions and their propagated uncertainties directly as `result.spectra`.
For example, `result.spectra.cosine[0]` and
`result.spectra.cosine[2]` contain `I_0^c(q)` and `I_2^c(q)` on
`result.spectra.q`. The final summary row shows a reduced two-dimensional
reconstruction and its selected cosine and sine radial spectra. It retains the
smallest power-ranked mode set carrying `reconstruction_threshold=0.95` of the
fitted modal Parseval power by default; the threshold is configurable. The
same data are returned as `result.reconstruction_image`,
`result.reconstruction_indices`, and `result.reconstruction_spectra`. Optional
`uncertainty` and `sample_weight` arrays change the fit metric to

```text
sum_p sample_weight[p] * ((data[p] - fit[p]) / uncertainty[p])**2.
```

The returned object contains the numerical fit, reconstructed and residual
data, selection mask, effective weights, and angular powers.

Set `beam_centering=True` to run parity-based centering immediately before the
standard fit. The analyzer dispatches from the input shape: a one-dimensional
detector point list uses `center_point_cloud_by_parity(...)`, while a
two-dimensional affine detector grid uses `refine_center_by_parity(...)`
directly. Experimental uncertainties are required. The result records the
method, reciprocal-space shift, and complete refinement diagnostic as
`beam_centering_method`, `center_shift_q_z`, `center_shift_q_y`, and
`beam_centering_refinement`.

```python
centered = standard_analyzer(
    data.q_z,
    data.q_y,
    data.intensity,
    uncertainty=data.uncertainty,
    beam_centering=True,
    q_range=(0.005, 0.030),
    mode_limits=(12, 8),
)
```

If externally supplied `sample_weight` values depend on the annulus geometry,
they should be recomputed after centering or replaced by the internal
`pixel_integration_subpixels` path. Detector-efficiency weights that are tied
to detector entries rather than reciprocal coordinates remain applicable.

Set `log_intensity_colorbar=True` to display the input signal and mode fit with
one shared logarithmic color scale. This option affects only these two panels.
Finite nonpositive corrected intensities remain selected and fitted, but are
masked in the logarithmic display because they have no real logarithm. The
signed residual panel retains its linear diverging scale.

For a fitted coefficient vector, `angular_mode_powers(coefficients, indices)`
computes

```text
cosine_power[n] = sum_m coefficient[(+n, m)]^2
sine_power[n]   = sum_m coefficient[(-n, m)]^2.
```

The standard overview shows the powers as data points normalized by the
largest cosine or sine power in the fit.  Both panels use the same global
maximum.  The error bars are conditional one-standard-deviation uncertainties
propagated from the ordinary least-squares coefficient covariance; the fitted
maximum used for normalization is treated as a fixed scale.

`power_model_selector(coefficients, indices, threshold=0.95)` ranks individual
`(n, m)` modes by `coefficient**2` and returns the shortest descending-power
index list whose cumulative Parseval power reaches the requested fraction.
The standard analyzer applies this selector at 95%, 98%, and 99% by default
and reports both the retained `(n, m)` product-mode count and the number of
distinct retained angular indices `n` for all three thresholds in the PNG
title. Positive and negative `n` count as different angular modes because
they represent the cosine and sine branches, respectively. The returned
`selected_indices_by_threshold` dictionary contains the corresponding three
index lists.  The two additional rows show the individual `(n, m)`
power maps and the reconstructed radial spectra
`I_n^c(q) = sum_m c_(+n,m) R_m(q)` and
`I_n^s(q) = sum_m c_(-n,m) R_m(q)`.  Their shaded one-standard-deviation
bands propagate the fitted coefficient covariance according to
`variance[I_n(q)] = R(q).T @ covariance_n @ R(q)`.

`parity_power_fractions(coefficients, indices)` partitions the fitted Parseval
power without assigning a quality label:

```text
even_power_fraction = sum_(abs(n) even, m) c_(n,m)^2 / sum_(n,m) c_(n,m)^2
odd_power_fraction  = sum_(abs(n) odd,  m) c_(n,m)^2 / sum_(n,m) c_(n,m)^2
```

The two fractions add to one.  The standard PNG reports both values.  Under an
assumed `I(q, theta) = I(q, theta + pi)` symmetry, odd orders are forbidden and
the odd fraction can serve as a neutral symmetry-violation diagnostic.  It is
not universally a signal-quality measure, for example when a physical chiral
contribution is allowed in an individual polarized channel.

The header additionally reports the fit-based area-averaged parity mismatch

```text
(1 / A) integral_A [I(q, theta) - I(q, theta + pi)]^2 q dq dtheta
```

and its dimensionless relative RMS. Parseval's identity gives the exact
relations `mean_square = 4 * odd_power / A` and
`relative_rms = 2 * sqrt(odd_power_fraction)` for the fitted expansion. The
first value retains intensity-squared units and is therefore not comparable
between arbitrary intensity normalizations. The relative RMS is scale-free,
but is not statistically independent of the already reported odd-power
fraction and remains conditional on the selected basis truncation.

`angular_block_tests(...)` jointly tests all radial coefficients belonging to
one angular order and one trigonometric branch.  For a coefficient block
`c_n` with covariance `C_n`, it evaluates
`T_n = c_n.T @ pseudoinverse(C_n) @ c_n`. One SVD determines both the
pseudoinverse and numerical rank `r`, retaining singular values strictly
above `k * eps * s_max`, where `k` is the block size. The same retained
directions determine the statistic and its reference degrees of freedom.
When absolute experimental uncertainties are supplied, the statistic is
compared with its chi-squared threshold with `r` degrees of freedom. When
the covariance scale is estimated from the residuals, the corresponding
finite-residual-degree-of-freedom F threshold is used. A point above its
threshold rejects the joint null hypothesis that all radial coefficients of
that angular contribution are zero, conditional on the selected least-squares
noise model.

## Detector-data preconditioning

Preconditioners are explicit functions rather than hidden steps inside the
analyzer. Finite negative and zero background-corrected intensities are never
removed automatically. The public functions include:

- `valid_pixel_mask(...)` for finite coordinates/intensities and positive
  uncertainties;
- `fractional_annulus_weights(...)` for subpixel coverage of circular annulus
  boundaries;
- `pixel_integrated_annulus_design(...)` for a memory-bounded, subpixel-averaged
  basis matrix;
- `refine_center_by_parity(...)` and `apply_center_correction(...)` for a common
  detector-index centre correction;
- `center_point_cloud_by_parity(...)` for the standard auxiliary-grid centre
  refinement of detector point lists while preserving their original samples;
- `symmetry_diagnostic(...)` for parity or field-axis mirror residuals;
- `symmetry_fold(...)` for optional inverse-variance orbit averaging;
- `mask_symmetry_diagnostic(...)` for mask centroid and point-reflection checks;
- `rotate_coordinates(...)`, `rescale_intensity(...)`, and
  `standardized_residual_outliers(...)` for explicit coordinate, numerical,
  and diagnostic operations.

For a weighted detector fit with fractional annulus boundaries:

```python
from sans2d_mode_analyzer import fractional_annulus_weights, standard_analyzer

area = fractional_annulus_weights(
    data.q_z,
    data.q_y,
    q_min=0.015,
    q_max=0.050,
    subpixels=7,
)
result = standard_analyzer(
    data.q_z,
    data.q_y,
    data.intensity,
    uncertainty=data.uncertainty,
    sample_weight=area,
    q_range=(0.015, 0.050),
    mode_limits=(8, 4),
)
```

Set `pixel_integration_subpixels=7` to average each basis function over the
admitted part of every pixel. This is more precise but intentionally not the
default because its cost scales with the number of subpixels and fitted modes.

Parity folding assumes `I(q, theta) = I(q, theta + pi)`. Mirror folding adds
`I(q, theta) = I(q, -theta)`. These operations therefore impose physical
assumptions and are never applied automatically. Interpolation also creates
neighbouring-pixel correlations not contained in the returned diagonal
uncertainties; folded data are primarily intended for visualization and
sensitivity checks. For formal inference, selecting the corresponding
symmetry-constrained mode indices while retaining the original pixels is
usually preferable.

## Mode-convergence test

`mode_convergence_standard_analyzer(...)` automates the selection of the rectangular
truncation `(n_max, m_max)`. It fits every integer pair up to a supplied
reference limit and compares each candidate with the largest tested model:

```python
from sans2d_mode_analyzer import mode_convergence_standard_analyzer

convergence = mode_convergence_standard_analyzer(
    q_z,
    q_y,
    intensity,
    q_range=(0.015, 0.050),
    mode_limits=(10, 10),
    prediction_tolerance=0.01,
    coefficient_tolerance=0.05,
    relative_mse_tolerance=0.001,
    output_path="converged_standard_analysis.png",
)

print(convergence.selected_n_max, convergence.selected_m_max)
analysis = convergence.standard_analysis
```

A candidate is stable only when its reconstructed detector signal, its
zero-padded coefficient vector, and its MSE are all sufficiently close to the
largest tested reference model. Among stable candidates, the pair containing
the fewest product modes is selected. The convergence sweep reuses one
maximum-order design matrix; only the selected pair is passed to the complete
`standard_analyzer(...)` for plotting and the returned final analysis.

The result also contains the complete MSE, prediction-change,
coefficient-change, relative-MSE-excess, and stability grids. If
`stabilized_within_limits` is false, at least one selected order lies on the
tested upper boundary, so convergence beyond that boundary has not been
demonstrated. Agreement with a finite reference truncation is a numerical
stability diagnostic, not proof that the reference itself is free of
overfitting.

## Angular template tests

The template tests are deliberately separate from `standard_analyzer(...)`
and can be applied to its fitted coefficients as a second analysis step:

```python
from sans2d_mode_analyzer import (
    test_sin_squared,
    test_sin_squared_cos_squared,
)

sin2 = test_sin_squared(result.fit.coefficients, result.fit.indices)
spin_flip = test_sin_squared_cos_squared(
    result.fit.coefficients,
    result.fit.indices,
)

print(sin2.explained_power_fraction)
print(sin2.pair_mismatch_power_fraction)
print(sin2.forbidden_mode_power_fraction)
```

Both tests leave the radial amplitude `A(q)` unrestricted within the fitted
radial basis and project the complete coefficient vector onto the requested
angular-template subspace. The residual is separated into

- a mismatch between the constant and target cosine amplitudes; and
- power in all other cosine and sine modes, which are forbidden by the exact
  template.

For the conventional Fourier expansion `a_0(q) + a_n(q) cos(n theta)`, both
`sin(theta)^2` and `sin(theta)^2 cos(theta)^2` require
`a_n(q) / a_0(q) = -1`, with `n = 2` and `n = 4`, respectively. The package
uses an orthonormal angular basis, so the equivalent internal coefficient
condition is `c_n(q) / c_0(q) = -1/sqrt(2)`, not `-1`.

These fractions diagnose agreement with an ideal angular form conditional on
the fitted expansion. They do not yet propagate coefficient uncertainty or
replace checks of centering, masking, reduction, and the selected `q` range.

## Example

```python
import numpy as np

from sans2d_mode_analyzer import (
    even_angular_indices,
    evaluate_expansion,
    fit_mse,
    rectangular_indices,
)

q_min, q_max = 0.015, 0.050
q_axis = np.linspace(q_min, q_max, 30)
theta_axis = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
q, theta = np.meshgrid(q_axis, theta_axis, indexing="ij")

# All radial modes m <= 3 and real angular modes |n| <= 8.
full_indices = rectangular_indices(n_max=8, m_max=3)

# The same truncation, but with every odd angular order removed.
parity_indices = even_angular_indices(n_max=8, m_max=3)

# Fit exactly the modes listed here; no other modes are introduced.
selected_indices = [(0, 0), (4, 0), (-2, 1), (6, 2)]
true_coefficients = np.array([4.0, -0.7, 0.35, 1.2])
measured_intensity = evaluate_expansion(
    q,
    theta,
    true_coefficients,
    selected_indices,
    q_min=q_min,
    q_max=q_max,
)
result = fit_mse(
    q,
    theta,
    measured_intensity,
    selected_indices,
    q_min=q_min,
    q_max=q_max,
)
print(result.coefficients, result.mse)

# Reconstruct the fitted expansion.
reconstruction = evaluate_expansion(
    q,
    theta,
    result.coefficients,
    result.indices,
    q_min=q_min,
    q_max=q_max,
)

# A single selected mode is the same API.
one_mode = evaluate_expansion(
    q,
    theta,
    coefficients=[2.5],
    indices=[(4, 0)],
    q_min=q_min,
    q_max=q_max,
)
```

`fit_mse` currently performs an unweighted linear least-squares fit, which is
equivalent to minimizing the ordinary sample MSE.  Experimental uncertainty
weights, pixel-area integration, masks, and symmetry-specific selectors can be
added as separate layers without changing the explicit mode-index interface.
