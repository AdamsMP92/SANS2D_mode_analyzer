# SANS2D mode analyzer

`sans2d-mode-analyzer` provides a small linear-analysis core for real-valued
two-dimensional SANS patterns on a circular annulus

\[
q_{\min} \le q \le q_{\max}, \qquad 0 \le \theta < 2\pi.
\]

The radial modes are normalized Legendre polynomials in a coordinate affine in
`q**(s + 1)`, where `s = radial_measure_exponent`. They are therefore
orthonormal for the radial measure `q**s dq`. The default `s=1` is the
detector-plane measure `q dq`. The angular modes form a real orthonormal
Fourier basis.

More explicitly,

```text
x_s(q) = (2 q^(s+1) - q_max^(s+1) - q_min^(s+1))
         / (q_max^(s+1) - q_min^(s+1))
R_m^(s)(q) = sqrt((2m + 1)(s + 1)
                  / (q_max^(s+1) - q_min^(s+1))) P_m(x_s(q))
T_0(theta) = 1 / sqrt(2 pi)
T_n(theta) = cos(n theta) / sqrt(pi)       for n > 0
T_n(theta) = sin(abs(n) theta) / sqrt(pi)  for n < 0
Phi_(n,m)(q, theta) = R_m(q) T_n(theta).
```

Consequently, the product modes are orthonormal for `q**s dq dtheta` over the
chosen annulus. By default (`s=1`) the fitted coefficients minimize
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

For the same radial truncation applied to a selected set of angular modes,
use

```python
from sans2d_mode_analyzer import angular_mode_indices

indices = angular_mode_indices([0, 2, -2, 4], m_max=5)
```

This creates every pair `(n, m)` with the listed signed angular modes and
`0 <= m <= 5`; the result can be passed directly as `indices=indices` to
`standard_analyzer` or the lower-level fit functions.

For reproducible synthetic verification scenarios, select a random subset of
a rectangular truncation with

```python
from sans2d_mode_analyzer import random_mode_indices
from synthetic_models import random_mode_coefficients

indices = random_mode_indices(
    n_max=8,
    m_max=3,
    n_coeffs=6,
    angular_parity="even",
    seed=7,
)
coefficients = random_mode_coefficients(indices, seed=8)
```

Sampling is without replacement. `angular_parity` may be `"all"`, `"even"`,
or `"odd"`; the constant angular mode belongs to the even set. The selected
subset is random, but it is returned in the package's canonical radial-major
order. This helper generates controlled test cases and is not a data-driven
mode-selection method.
`random_mode_coefficients` independently samples one coefficient in `[-1, 1]`
for every selected index. Its output order matches the supplied index order.

The resulting deterministic modal signal can be placed directly on a Cartesian
detector annulus:

```python
from synthetic_models import generate_modal_detector_data

q_z, q_y, intensity = generate_modal_detector_data(
    q_min=0.015,
    q_max=0.050,
    indices=indices,
    coefficients=coefficients,
    number_of_pixels=161,
)
```

Values outside the selected annulus are `NaN`. The three returned arrays can
be passed directly to `standard_analyzer`; the detector convention remains
`(q_z, q_y)` throughout the package.
For centering verification, `center_shift=(delta_q_z, delta_q_y)` places the
physical centre of the synthetic annulus at a prescribed detector coordinate
while leaving the returned coordinates in the laboratory frame.  The
noise-free Stage-5 example in
[`verification_examples/stage_5/example_1.py`](verification_examples/stage_5/example_1.py)
uses this option to show how an imposed translation creates odd angular modes
and how `beam_centering=True` removes that geometrical contribution under the
assumed antipodal symmetry.
Set `return_model=True` on `generate_random_modal_detector_data` to append the
generating `(n, m)` index array and coefficient vector to these three detector
arrays. This is intended for coefficient-recovery and regression tests.

## Environment and tests

```text
pixi install
pixi run test
pixi run example
pixi run experimental-example
pixi run prl-example
pixi run numagsans-example
pixi run analyze-numagsans-ivr40
pixi run analyze-arxiv-2606-18791
pixi run analyze-prl-135-196706
```

The scripts in `examples/` write their PNG figures to `examples/output/`.
The output directory is created automatically and ignored by Git.

The example creates a noisy synthetic detector pattern proportional to

```text
(F(q)^2 + 6 F'(q)^2) (1 + cos(theta)^2),
```

where `F` is the normalized spherical form-factor amplitude. Over
`0.5 <= qR <= 10.0`, the example performs two complementary analyses:

1. A full rectangular basis with `mode_limits=(4, 10)` is written to
   `examples/output/synthetic_sphere_fit.png`.
2. A custom basis generated by
   `angular_mode_indices(angular_modes=[0, 2], m_max=14)` is written to
   `examples/output/synthetic_sphere_fit2.png`.

The second fit uses the exact angular orders expected from
`1 + cos(theta)^2`, while retaining a more flexible radial expansion. It
therefore demonstrates how a physically motivated angular subspace can be
passed directly to `standard_analyzer` without imposing a rectangular angular
truncation. Both PNGs contain the standard six-row overview of the input,
fit, residual, angular powers, block statistics, mode-power maps, and radial
spectra. The final row repeats the magnitudes of the radial spectra on common
log-log axes. The complete beginner-oriented script is in
[`examples/synthetic_sphere_fit.py`](examples/synthetic_sphere_fit.py).
The detector construction, sphere model, white-noise, supplied-map
heteroskedastic Gaussian-noise, low-rank correlated Gaussian-noise addition,
and insertion onto the masked detector grid are separate short functions in
[`synthetic_models/model_functions.py`](synthetic_models/model_functions.py).

The repository also contains the supplied experimental NeXus files under
[`experimental_data/doi_10.48550_arXiv.2606.18791`](experimental_data/doi_10.48550_arXiv.2606.18791).
`examples/experimental_10mT_fit.py` applies the same standard analyzer to the
lowest-field data in the interval `0.015 <= q <= 0.100 nm^-1`.
Two retained four-column ASCII SANS benchmark maps from the earlier MATLAB
OrthoDecomp work are stored under
[`experimental_data/doi_10.1103_5yc2-pv4y`](experimental_data/doi_10.1103_5yc2-pv4y).
Each data set documents its own format, provenance, caveats, analysis script,
and generated results in its local README. Input files are stored in each
collection's `data/` subdirectory.
The compact `examples/experimental_prl_135_196706_fit.py` example applies the
standard analyzer to `igc_nanoporous_Fe_0p1T.dat` over the legacy numerical interval
`0.005 <= q <= 0.030`. It writes a raw fit and a second fit after a
parity-based centre refinement. Because the supplied data are a point list,
the centre is estimated on a small interpolation grid, but the corrected fit
uses the original intensities, uncertainties, and point coordinates with only
their reciprocal-space origin shifted.

`examples/numagsans_ivr40_3_spin_flip_fit.py` analyzes and compares the
simulated average spin-flip cross section `S_sf` from the IVR40 output indices
1 and 2. The retained `161 x 161` Cartesian CSV files contain `qz`, `qy`, `q`,
`theta`, and `S_sf`; the original polar output and conversion script are not
part of this repository. The example loads these prepared maps, writes one
standard-analysis overview per data set, and directly compares the fitted
radial spectra `I_0^c(q)` and `I_2^c(q)`. Since these deterministic simulation
data have no experimental uncertainty column, fit-uncertainty diagnostics are
conditional on the residual of the finite mode truncation. The data-local
[`README`](experimental_data/NuMagSANS_IVR40/README.md) documents the retained
format and reproducible analysis task.

## Standard analyzer

`standard_analyzer(...)` accepts either two-dimensional detector arrays or
one-dimensional detector point lists `q_z`, `q_y`, and `intensity`. It selects
the requested `q_range=(q_min, q_max)`, performs the fit, and writes the
multi-panel summary PNG. Supply `mode_limits=(n_max, m_max)` to generate a
rectangular mode grid, or pass an explicit `indices=[(n,m), ...]` list to fit
exactly those modes. The two forms are mutually exclusive. By default a
mode-limit fit uses the full rectangle (`even_only=False`).
Set `even_only=True` to fit only even angular orders up to `n_max`, with
all radial orders through `m_max`. The overview title marks this choice.
The resulting fit has antipodal symmetry by construction. For an arbitrary
symmetry or sparse basis, construct the desired index list explicitly instead.

The optional `radial_measure_exponent` changes how the radial interval is
weighted:

```python
result = standard_analyzer(
    q_z,
    q_y,
    intensity,
    q_range=(q_min, q_max),
    mode_limits=(8, 4),
    radial_measure_exponent=2.0,
)
```

For equal-area Cartesian detector pixels, the standard analyzer combines the
corresponding radial basis with the relative residual weight
`q**(radial_measure_exponent - 1)`. Thus values above one emphasize the
high-`q` part of the annulus and values below one emphasize low `q`; the
default value `1.0` leaves the established analysis unchanged. Explicit
measurement uncertainties, user-supplied `sample_weight`, and fractional
pixel coverage are multiplied with this radial weight. The exponent must be
greater than `-1`. If the annulus includes `q=0`, exponents below one require
special cell integration and are therefore rejected by the point-sampled
weight helper.

The lower-level basis and fit functions accept the same
`radial_measure_exponent`. Since arbitrary point lists do not imply a sampling
measure, low-level callers who want the same residual weighting should pass
`radial_measure_weight(...)` as the `sample_weight` of `fit_wls`.
Changing the exponent changes the orthonormal norm and hence the numerical
coefficients and modal powers. Compare those quantities across analyses only
when the exponent, annulus, and coordinate units are the same; reconstructed
intensities remain directly interpretable in the original intensity units.

### Piecewise-linear radial reference

The standard analyzer can optionally repeat the fit with the same angular
modes but replace the global radial polynomials by local linear hat functions:

```python
result = standard_analyzer(
    q_z,
    q_y,
    intensity,
    q_range=(q_min, q_max),
    mode_limits=(8, 4),
    radial_measure_exponent=1.0,
    piecewise_linear_reference=True,
    reference_radial_segments=12,
    reference_knot_spacing="measure",
)

polynomial_spectra = result.spectra
polynomial_powers = result.powers
polygonal_spectra = result.reference.spectra
polygonal_powers = result.reference.powers
```

`reference_radial_segments=K` creates `K` non-overlapping radial intervals
and `K + 1` knot amplitudes. With `reference_knot_spacing="measure"`, the
knots delimit equal intervals in `q**radial_measure_exponent dq`; use
`"q"` for uniform q increments. The reference fit uses the same selected
pixels, angular orders, uncertainties, effective weights, radial measure, and
optional subpixel integration as the principal analysis.

The polygonal coefficients are the fitted values of each `I_n(q)` at the
knots. Since neighboring hat functions overlap, their squared coefficients
are not additive powers. The implementation instead returns the exact
Gram-corrected angular power `a_n.T @ G @ a_n`, where
`G_ij = integral B_i(q) B_j(q) q**s dq`. Consequently, the reference powers
and the orthonormal-basis Parseval powers use the same continuous norm.
When figure export is enabled, the overview overlays the polygonal spectra as
dashed curves and the corresponding powers as red cross markers. The
individual `(n,m)` power maps remain specific to the orthonormal polynomial
basis because individual overlapping hat coefficients are not additive powers.

The returned result exposes the plotted radial coefficient
functions and their propagated uncertainties directly as `result.spectra`.
For example, `result.spectra.cosine[0]` and
`result.spectra.cosine[2]` contain `I_0^c(q)` and `I_2^c(q)` on
`result.spectra.q`. The overview ends with the signed cosine and sine radial
spectra of the full fit followed by their absolute values on common log-log
axes. Optional `uncertainty` and `sample_weight` arrays change the
fit metric to

```text
sum_p sample_weight[p] * ((data[p] - fit[p]) / uncertainty[p])**2.
```

With the default `absolute_uncertainty=True`, `uncertainty[p]` is the actual
standard deviation of an independent pixel measurement. The covariance
propagates these errors through the weighted fit. Additional `sample_weight`
values, including fractional annulus coverage, change the fitting metric;
they do not change the supplied pixel noise. Writing
`A = (X.T @ W @ X)^-1 @ X.T @ W`, the returned covariance is
`A @ diag(uncertainty**2) @ A.T`, evaluated using QR factors rather than an
inverse normal matrix. With unit sample weights this reduces to the usual
inverse precision matrix. Statistical precision factors should be included
in `uncertainty` itself (for example `sigma / sqrt(precision_weight)`).

With `absolute_uncertainty=False`, or without supplied uncertainties, the
combined weights instead specify the relative precision model
`Var(data[p]) = scale * uncertainty[p]**2 / sample_weight[p]` (unit
uncertainties if absent). The residuals estimate its common scale. Its
covariance and F tests are conditional on that model; they are not calibrated
for geometric coverage weights with unknown pixel-noise variance. For
experimental coverage-weighted fits, supply absolute measurement errors.

The returned object contains the numerical linear fit, reconstructed and
residual data, selection mask, effective weights, angular powers, block tests,
and radial spectra.

Figure rendering is isolated in
[`src/sans2d_mode_analyzer/figure_backend.py`](src/sans2d_mode_analyzer/figure_backend.py).
By default `standard_analyzer` calls this backend and writes the requested PNG.
For numerical sweeps or scripted postprocessing, disable rendering without
changing any numerical calculation:

```python
result = standard_analyzer(
    q_z,
    q_y,
    intensity,
    q_range=(q_min, q_max),
    indices=indices,
    export_figure=False,
)
```

In this case `result.output_path` is `None`; the fit, powers, block tests,
detector-space arrays, parity diagnostics, and radial spectra remain available.
Matplotlib is imported only when figure export is enabled.

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

For a parity-even observable, an optional second step can minimize the fitted
fraction of coefficient power in odd angular orders. The numerical operation
is implemented independently as `refine_center_by_odd_mode_power(...)`; the
standard analyzer only orchestrates it:

```python
modal_centered = standard_analyzer(
    data.q_z,
    data.q_y,
    data.intensity,
    uncertainty=data.uncertainty,
    beam_centering=True,
    odd_mode_centering=True,
    odd_mode_centering_mode_limits=(5, 8),
    odd_mode_centering_max_shift_pixels=0.25,
    q_range=(0.005, 0.030),
    mode_limits=(12, 8),
)
```

The compact centering basis is separate from the final analysis basis. The
additional reciprocal-space shift and complete before/after diagnostics are
returned as `odd_mode_centering_refinement`; `center_shift_q_z` and
`center_shift_q_y` contain the total applied correction. This optimization is
conditional on the parity assumption, centering basis, annulus, weights, and
search bound. It must not be used to suppress odd modes that may be physical.

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
propagated from the returned coefficient covariance; the fitted
maximum used for normalization is treated as a fixed scale.

The individual `(n,m)` power maps and the radial spectra
`I_n^c(q) = sum_m c_(+n,m) R_m(q)` and
`I_n^s(q) = sum_m c_(-n,m) R_m(q)` are shown below the angular summaries.
Their shaded one-standard-deviation bands propagate the fitted coefficient
covariance according to
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
the covariance scale is estimated from the residuals under the relative
precision model described above, the corresponding
finite-residual-degree-of-freedom F threshold is used. A point above its
nominal threshold rejects the joint null hypothesis that all radial coefficients of
that angular contribution are zero, conditional on the selected least-squares
noise model.

Holm correction is available on request through numerical properties of the
same result. It is not highlighted in the standard overview. One family
contains all positive-rank cosine and sine blocks in a single analysis,
including the isotropic block. Missing blocks, the zero-order sine block,
and rank-zero blocks do not inflate its size. A full rectangle with
`n_max=25` normally has 51 tests, regardless of the number of radial modes.

~~~python
tests = result.block_tests
tests.cosine_pvalue        # original, uncorrected p-values
tests.cosine_holm_pvalue   # corrected over both branches jointly
tests.sine_holm_pvalue
tests.cosine_holm_reject   # boolean decisions at 1 - block_test_confidence
tests.sine_holm_reject
tests.holm_family_size
~~~

These are optional numerical diagnostics exposed as derived properties; existing
function arguments, result constructors, and raw fields remain available.
The family-wise alpha defaults to 0.05 and follows `block_test_confidence`.
Holm is valid under arbitrary dependence when the individual p-values are
valid under the chosen covariance model. It does not calibrate an invalid
noise model or correct for selecting a basis or q range using the same data.
The calibration simulations check the implementation in specified Gaussian
models; they do not establish that these assumptions hold for reduced
experimental data or that systematic errors are represented.

The T_n panels show the block statistics without critical-value curves or
rejection markers. Nominal critical values, raw p-values, and Holm-adjusted
p-values remain available numerically through `result.block_tests`, but the
standard overview deliberately leaves the even/odd comparison descriptive.

For an observable expected to be parity even, a broad background of odd
angular powers can serve as an empirical reference for noise, asymmetry, and
processing artifacts. An even power below that background is not clearly
resolved above this empirical level. This is a descriptive robustness
criterion, not a calibrated p-value threshold. The comparison uses the same
power normalization, but mode uncertainties and sensitivities can differ.
Physical odd contributions must be excluded before treating them as a
background; in particular, individual spin-flip channels may contain a
chiral contribution. The analyzer does not automatically derive a cutoff
from the odd powers.

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
- `refine_center_by_odd_mode_power(...)` for a bounded modal refinement that
  minimizes the fitted odd-angular power fraction;
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

`fit_mse` performs an unweighted linear least-squares fit, equivalent to
minimizing the ordinary sample MSE. Use `fit_wls` for supplied measurement
uncertainties and optional sample weights. The standard analyzer additionally
supports detector masks, fractional annulus coverage, pixel-integrated basis
functions, and optional parity-based centering without changing the explicit
mode-index convention.
