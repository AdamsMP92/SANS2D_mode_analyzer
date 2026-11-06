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
chosen annulus.  The fitted coefficients minimize
`mean((data - design_matrix @ coefficients)**2)` over exactly the selected
columns.

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
pixi run ivan-example
pixi run analyze-nanoflower
pixi run analyze-ivan
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
[`experimental_data/220nm_nanoflower`](experimental_data/220nm_nanoflower).
`examples/experimental_10mT_fit.py` applies the same standard analyzer to the
lowest-field data in the interval `0.015 <= q <= 0.050 nm^-1`.
An additional set of four-column ASCII SANS detector data from the earlier
MATLAB OrthoDecomp work is stored under
[`experimental_data/ivan_orthodecomp`](experimental_data/ivan_orthodecomp).
Each data set documents its own format, provenance, caveats, analysis script,
and generated results in its local README.
The compact `examples/experimental_ivan_000000_fit.py` example applies the
same standard/convergence switch to `000000.dat` over the legacy interval
`0.005 <= q <= 0.030 nm^-1`.

## Standard analyzer

`standard_analyzer(...)` accepts either two-dimensional detector arrays or
one-dimensional detector point lists `q_z`, `q_y`, and `intensity`. It selects
the requested annulus, creates the complete `rectangular_indices(n_max, m_max)`
grid, performs the MSE fit, and writes the multi-panel summary PNG. The
returned object contains the numerical fit, reconstructed and residual data,
and the angular powers.

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

`angular_block_tests(...)` jointly tests all radial coefficients belonging to
one angular order and one trigonometric branch.  For a coefficient block
`c_n` with covariance `C_n`, it evaluates
`T_n = c_n.T @ inverse(C_n) @ c_n`.  The standard analyzer compares the
cosine and sine statistics with their 95% F-test thresholds in a separate plot
row.  A point above its threshold rejects the joint null hypothesis that all
radial coefficients of that angular contribution are zero, conditional on the
ordinary least-squares noise model.

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
    q_min=0.015,
    q_max=0.050,
    n_max_limit=10,
    m_max_limit=10,
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
