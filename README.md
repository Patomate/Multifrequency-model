# Multi-frequency closure fit

Python scripts that fit a model of circular Gaussian components to VLBI observations of the
same source at several frequencies at once. The fit uses only closure quantities and needs no
prior model of the source. Its main product is the **core shift**: the position of the core at
each frequency relative to the core at the reference frequency.

| File | Purpose |
|---|---|
| `multifreq_fit_blind_full_v2.py` | Fit with a number of components chosen by the user (the same in every map). |
| `multifreq_fit_auto_components.py` | Fit with 2, 3, 4 ... components and choose the number automatically (BIC). |
| `multifreq_plots.py` | Figures, used by both scripts. |

## Requirements

Python 3 with `numpy`, `scipy`, `matplotlib` and [`ehtim`](https://github.com/achael/eht-imaging)
(used only to read UVFITS files and form the closures). Tested with Python 3.14, numpy 2.4,
scipy 1.17 and matplotlib 3.10.

## The model

Every map (one per UVFITS file) is described by the same number of circular Gaussians,
N = 1 core + (N - 1) jet components.

* **Jet components are shared between frequencies.** Each jet component has a single position
  and FWHM, common to all maps; only its flux is free in each map. This assumes the jet
  components are optically thin features whose position does not depend on frequency.
* **Each map has its own core**, with its own FWHM and its own position. The core of the
  reference file (by default the highest frequency) is at the origin; the position of every
  other core relative to it is the core shift of that file.
* **Fluxes are fractions.** In each map the flux fractions of all components add up to 1.

Several files may share a frequency (e.g. two IFs of the same band); each one is treated as a
separate map with its own core.

## What changes with respect to a visibility fit

A usual model fit (e.g. Difmap `modelfit`) compares the model with the calibrated
visibilities. Here the model is compared with **closure phases** and **log closure amplitudes**
instead: the model visibilities are computed on the same baselines as the data and combined into
the same closures. The consequences are:

* **No starting model are needed**, because closures do not depend on
  the station gains.
* **No absolute position.** Closures do not change if the whole source is shifted, so each core
  is placed at the origin of its own map. The core shift can only be measured relative to
  features shared between frequencies, which is the role of the shared jet components.
* **No total flux.** Closures do not change if the source is scaled, so the fit gives flux
  fractions. Fluxes in Jy are estimated afterwards from the visibility amplitudes and are
  approximate.
* Visibilities are still used, but only after the fit: for the Jy scale, the self-calibrated
  residual maps and the figures.

## How the fit works (`multifreq_fit_blind_full_v2.py`)

**Loading.** Each UVFITS file is read with ehtim, and a minimal independent set of closure
phases and debiased log closure amplitudes is formed at each time. The resolution of each file
is taken as 1/(longest baseline) and is used as the scale for every check that depends on size.
One time stamp out of every five is set aside as **validation** data: the fit never uses it,
and it gives the final quality metric.

**Step 1. Each file on its own.** Every file is fitted independently with N components
(differential evolution followed by least squares). In each map the brightest component is taken
as the core. The jet components of all files are then matched to those of one file (the lowest
frequency) by position and size, and the median offset of the matched components gives a first
estimate of each core shift. This builds the starting point of the joint fit. Closures with
residuals far outside the noise with respect to this starting point (more than 8 sigma and more
than 6 robust standard deviations above the median) are removed as bad data.

**Step 2. Global search.** The joint model of all files is searched with differential evolution,
started from the step-1 solution, plus several hundred random starts.

**Step 3. Refinement.** The best candidates are refined with least squares using an analytic
Jacobian and a robust loss (residuals above ~2 sigma have reduced weight). If a jet component
ends with almost no flux in every map, it is moved to the brightest peak of the residual map and
kept there only if the fit improves by more than 0.5% and it stays a separate component (more
than half a beam from every core and fainter than it).

**Step 4. Significance of each jet component in each map.** For every jet component and every
map, the fit is redone without that component in that map only, and the increase of chi2 is
recorded (Δχ², in units of the reduced chi2 of the best fit).

* Δχ² below `--dchi2-min` (default 9, about 3 sigma for one parameter) marks the component as
  not significant in that map (`n.s.`). It stays in the model: every map keeps the same
  components.
* If the fit without a component is *better* (Δχ² < -1), the best fit was not the true minimum.
  The fit is restarted from that solution with the component present but faint, and the result
  is adopted if it improves the fit and no jet component has taken the place of a core (closer
  than half a beam to it, or brighter than it). The test is then repeated.

**Errors.** Uncertainties are formal 1-sigma errors from the covariance matrix of the fit,
scaled by the reduced chi2. Errors of derived quantities (distances, position angles, core shift
magnitude) are obtained by error propagation. Real uncertainties are likely larger.

**Quality metric.** The fit is judged by a single number: the reduced chi2 of the validation
closures, sum(r²)/N, where r are the normalised residuals. It is about 1 when the model
describes the data down to the noise, and unlike the chi2 of the fitted closures it does not
decrease just because more components are added.

## How the number of components is chosen (`multifreq_fit_auto_components.py`)

The same fit (steps 1 to 4, without the restart in step 4) is run with N = 2, 3, 4 ...
components per map. A model with one more jet component is **kept only if its chi2 drops by
more than the BIC penalty**:

    Δχ² / χ²_red(larger model)  >  Δk · ln(N_closures)

where Δk is the number of added parameters (each extra jet component adds x, y, FWHM and one
flux per map) and N_closures is the number of training closures. Dividing by the reduced chi2 of
the larger model keeps systematic residuals, which no Gaussian can describe, from rewarding
extra components. Each model is compared with the last kept one, on the same closures. The
search stops after `--patience` models in a row that are not kept (default 2), or at
`--max-comp`.

## Usage

Interactive (the scripts ask for the files, the search radius of each one and, for
`full_v2`, the number of components):

    python multifreq_fit_blind_full_v2.py
    python multifreq_fit_auto_components.py

Direct, with each file given as `file[:fov_uas]`, where `fov_uas` is the search radius around the
core in microarcseconds:

    python multifreq_fit_blind_full_v2.py --ncomp 5 c1.uvfits:20000 x1.uvfits:20000 u1.uvfits:20000
    python multifreq_fit_auto_components.py --max-comp 6 c1.uvfits:20000 x1.uvfits:20000 u1.uvfits:20000

Main options:

| Option | Default | Meaning |
|---|---|---|
| `--ncomp` | asked | Components per map (`full_v2` only). |
| `--min-comp`, `--max-comp` | 2, 8 | Range of components tried (`auto_components` only). |
| `--patience` | 2 | Models in a row not kept before stopping (`auto_components` only). |
| `--preset` | `serious` | Search effort: `quick` for tests, `serious` for results. |
| `--ref` | highest frequency | Index (0-based, input order) of the reference file. |
| `--snrcut` | 3 | Minimum SNR of the visibilities used in the closures. |
| `--avg` | 0 | Coherent time averaging in seconds before forming closures (0 = none). |
| `--max-core-shift` | 1000 | Bound on each core shift component (µas). |
| `--max-core-fwhm`, `--max-fwhm` | 1000, 8000 | Bounds on core and jet FWHM (µas). |
| `--dchi2-min` | 9 | Δχ² needed for a jet component to be significant in a map. |
| `--workers` | all CPUs | Parallel processes. |
| `--seed` | 8101 | Random seed (results are reproducible). |
| `--out` | timestamped folder | Output folder. |

With the `serious` preset, a fit of five files with five components takes about two minutes on
a 10-core laptop; `auto_components` takes that time for each number of components tried.

## Output

`multifreq_fit_blind_full_v2.py` writes to `fit_results_<timestamp>/`:

* `fit_summary.txt`, the main result, in this order:
  * core shifts of each file relative to the reference core (dx, dy, magnitude and position
    angle, with errors);
  * a table per map with flux, flux fraction, position, distance and position angle from the
    core, FWHM and Δχ² of every component;
  * the distance from the core of each jet component in every map (its differences between maps
    are the core shifts projected on that component);
  * fit quality (validation reduced chi2 per file and in total, dynamic range of the
    self-calibrated residual maps) and a short description of the search;
  * for reference, the fit in the native format of each map, the shared jet components in the
    reference frame, pairwise core shifts, the other final candidates and every parameter;
  * the log printed during the run.
* `best_params.json`: all parameters with errors and the per-file results, for further analysis.
* Figures:
  * `core_shifts.png`: core positions relative to the reference core (1-sigma error bars);
  * `overlay_aligned.png`: all maps superposed in the reference frame, with a zoom on the cores;
  * `components.png`: the components of each map;
  * `validation_closures.png`: observed vs model validation closures;
  * `amplitude_vs_uvdist.png`, `closure_phase_vs_uvdist.png`: data and model against uv distance.

`multifreq_fit_auto_components.py` writes to `auto_fit_<timestamp>/` one folder `ncomp_<N>/`
per model tried, with the same contents, plus `auto_summary.txt` (comparison of all models, the
chosen one and the significance of its components) and `auto_chi2.png` (validation reduced chi2
against the number of components).

## Limitations

* The core shift is measured relative to the shared jet components. If a component near the core
  is not the same feature at all frequencies, or moves with frequency, the core shift absorbs
  that difference. Components well separated from the core (several beams at the lowest
  frequency) are the most reliable reference.
* Fluxes in Jy are approximate, since closures do not constrain the total flux.
* Errors are formal and do not include the dependence on the chosen number of components.
