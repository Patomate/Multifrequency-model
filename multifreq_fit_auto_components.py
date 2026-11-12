#!/usr/bin/env python3
"""
Automatic choice of the number of components for the blind multi-frequency closure fit.

The fit itself is the same as in multifreq_fit_blind_full_v2.py: circular Gaussians (1 core +
N-1 shared jets, same N in every map) fitted to closure phases and log closure amplitudes of
several UVFITS files at once, with no prior model. Jets share position and FWHM across files
and only their flux changes per map; every map has its own core FWHM and core shift relative to
the core of the reference file (by default the highest-frequency one).

Here the fit is run with N = 2, 3, 4 ... components per map, and N is chosen automatically:
  * A model with one more jet is kept only if its closure chi2 drops by more than the BIC
    penalty, ln(N_closures) per added parameter (each extra jet adds x, y, FWHM and one flux
    per map). The chi2 drop is measured in units of the reduced chi2 of the larger model, so
    systematic residuals that no Gaussian can describe do not reward extra components.
  * The comparison is always against the last kept model, on the same training closures
    (before the outlier filter), so every model is judged on exactly the same data.
  * The search stops after --patience models in a row that are not kept, or at --max-comp.

What changes with respect to a visibility fit (e.g. Difmap modelfit)
  * The model is compared with closures, not visibilities: model visibilities are computed on
    the same baselines and combined into closure phases and log closure amplitudes exactly as
    the data (JointModel._dataset_terms). No self-calibration and no starting model are needed.
  * No absolute position: closures do not change if the whole source is shifted, so each core
    is placed at the origin of its own map and the core shift is measured only relative to the
    shared jets.
  * No total flux: closures do not change if the source is scaled, so the fit gives flux
    fractions; Jy values are estimated afterwards from the visibility amplitudes (flux_scale)
    and are approximate.
  * Fewer and noisier data: per time there are (N-1)(N-2)/2 independent closure phases and
    N(N-3)/2 closure amplitudes instead of N(N-1)/2 visibilities, and each combines 3 or 4
    visibilities. Only visibilities with SNR > snrcut are used.
  * Visibilities are still used, but only after the fit: Jy scale, self-calibrated residual
    maps (revive_dead_jets, dynamic range) and plots.

Usage
  Interactive (asks for the files and the search FOV of each one):
      python multifreq_fit_auto_components.py

  Direct, "file[:fov_uas]" (fov = search radius around the core, uas):
      python multifreq_fit_auto_components.py c1.uvfits:20000 x1.uvfits:15000 u1.uvfits:8000 --max-comp 8

Output (in --out, default auto_fit_<timestamp>/): auto_summary.txt and auto_chi2.png with the
comparison of all models, and one folder ncomp_<N>/ per model with its complete fit
(fit_summary.txt, figures, best_params.json).
"""
import os

# One BLAS thread per process: parallelism comes from the process pool.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import contextlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares, linear_sum_assignment

from multifreq_plots import (plot_components, plot_core_shifts, plot_overlay,
                             plot_validation, plot_uv)

RADPERUAS = np.pi / (180.0 * 3600.0 * 1e6)
FWHM_TO_SIGMA_RAD = RADPERUAS / (2.0 * np.sqrt(2.0 * np.log(2.0)))
TWO_PI_I_UAS = 2j * np.pi * RADPERUAS
# |V| below VIS_FLOOR is treated as zero (avoids log(0) in the closure amplitudes)
VIS_FLOOR = 1e-12
# smallest closure error allowed (rad for phases, ln units for amplitudes), so a closure with an
# unrealistically small error cannot dominate the fit
SIGMA_FLOOR = 1e-4

# Search effort. band_* = independent per-file fits (step 1); de_* = joint differential
# evolution (step 2); n_random/n_polish = random starts and how many of them are polished;
# max_nfev = max least-squares evaluations; max_points/polish_points = closure thinning;
# n_final = how many solutions are polished on all the training closures.
PRESETS = {
    "quick": dict(
        band_de_maxiter=30, band_de_popsize=6, band_perturb=3,
        de_maxiter=40, de_popsize=6, n_random=60, n_polish=8,
        max_nfev=300, max_points=400, polish_points=200, n_final=3,
    ),
    "serious": dict(
        band_de_maxiter=150, band_de_popsize=10, band_perturb=8,
        de_maxiter=200, de_popsize=12, n_random=500, n_polish=40,
        max_nfev=1000, max_points=None, polish_points=1500, n_final=10,
    ),
}


class ClosureBlock:
    """Closure phases and log closure amplitudes of one file, ready to evaluate the model on.

    Each closure comes with the (u, v) of its 3 or 4 baselines. Many closures share
    baselines, so only the unique baselines are kept: the model computes each visibility once,
    and cp_idx (3 x n_cp) and ca_idx (4 x n_ca) say which of them form each closure.
    cp_obs/ca_obs are the measured values and cp_sig/ca_sig their errors from ehtim, with a
    floor (SIGMA_FLOOR) so no closure gets infinite weight.
    Used by JointModel._dataset_terms for residuals and Jacobian; built with make_blocks every
    time the set of closures changes (train, validation, thinned).
    """

    def __init__(self, cp, ca):
        n_cp, n_ca = len(cp), len(ca)
        u = np.concatenate([np.asarray(cp[f"u{i}"], float) for i in (1, 2, 3)]
                           + [np.asarray(ca[f"u{i}"], float) for i in (1, 2, 3, 4)])
        v = np.concatenate([np.asarray(cp[f"v{i}"], float) for i in (1, 2, 3)]
                           + [np.asarray(ca[f"v{i}"], float) for i in (1, 2, 3, 4)])
        if len(u):
            uniq, inverse = np.unique(np.stack([u, v], axis=1), axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
        else:
            uniq, inverse = np.zeros((0, 2)), np.zeros(0, dtype=int)
        self.u, self.v = uniq[:, 0].copy(), uniq[:, 1].copy()
        self.q2 = self.u**2 + self.v**2
        self.n_cp, self.n_ca = n_cp, n_ca
        self.cp_idx = inverse[:3 * n_cp].reshape(3, n_cp)
        self.ca_idx = inverse[3 * n_cp:].reshape(4, n_ca)
        self.cp_obs = np.asarray(cp["cphase"], float)
        self.cp_sig = np.maximum(np.asarray(cp["sigmacp"], float), SIGMA_FLOOR)
        self.ca_obs = np.asarray(ca["camp"], float)
        self.ca_sig = np.maximum(np.asarray(ca["sigmaca"], float), SIGMA_FLOOR)

    @property
    def n(self):
        """Total number of closures (phases + amplitudes); used for the degrees of freedom."""
        return self.n_cp + self.n_ca


def make_blocks(closures):
    """Turn the per-file closure lists into a list of ClosureBlock (one per file, same order).
    Everything that evaluates chi2 takes this list."""
    return [ClosureBlock(c["cphase"], c["logcamp"]) for c in closures]


def thin_closures(closures, max_points):
    """Keep at most max_points closures of each type per file, taken at regular intervals
    (None = all). Makes the differential evolution and the first polishing stage fast; the
    best candidates are polished afterwards on all the closures."""
    if max_points is None:
        return closures
    thinned = []
    for c in closures:
        out = {}
        for key, data in c.items():
            if len(data) <= max_points:
                out[key] = data
            else:
                out[key] = data[np.linspace(0, len(data) - 1, max_points, dtype=int)]
        thinned.append(out)
    return thinned


def split_by_time(closures, stride=5):
    """Split the closures into train and validation: one time stamp out of every `stride`
    goes to validation (stride = 5: 20% of the data, enough to measure chi2 while keeping 80%
    for the fit; whole time stamps are moved so that train and validation are independent).
    The fit never sees the validation closures; they give the validation chi2 (the final
    metric of the report) and the figure validation_closures.png."""
    train, validation = [], []
    for c in closures:
        tr, va = {}, {}
        for key, data in c.items():
            if len(data) == 0:
                tr[key], va[key] = data, data
                continue
            times = np.unique(data["time"])
            mask = np.isin(data["time"], times[::stride])
            tr[key], va[key] = data[~mask], data[mask]
        train.append(tr)
        validation.append(va)
    return train, validation


def wrap_phase(angle):
    """Wrap an angle to (-pi, pi]; used in the closure phase residuals."""
    return np.angle(np.exp(1j * angle))


class JointModel:
    """Joint model of all files: defines the parameters and their bounds, and computes
    visibilities, closure residuals, the analytic Jacobian and the reduced chi2.

    Parameter vector q:
      per file d: ncomp[d]-1 flux logits (the last component's logit is fixed to 0; the
                  flux fractions come from a softmax and add up to 1)
      per file d: core FWHM [uas]
      per file d other than the reference: core shift (sx, sy) [uas], i.e. the position of
                  the core of d in the frame of the reference core
      per shared jet k: x, y [uas, reference frame] and FWHM [uas]
    In file d (core at the origin) jet k sits at (X_k - sx_d, Y_k - sy_d).

    jet_map[d] lists the shared jets used by file d. In the final fit every file uses every
    jet; only the significance test builds models where one file leaves one jet out (see
    without_jet).
    In this script one JointModel is built for every number of components tried.
    Used throughout: per-file fits (band_model), joint search, polishing, significance test,
    report and figures.
    """

    def __init__(self, ncomp, jet_map, ref, n_jets, labels, cfg):
        self.ncomp = list(ncomp)
        self.jet_map = [list(m) for m in jet_map]
        self.ref = ref
        self.n_jets = n_jets
        self.D = len(ncomp)
        names, lower, upper = [], [], []

        def add(name, lo, hi):
            names.append(name)
            lower.append(lo)
            upper.append(hi)
            return len(names) - 1

        self.logit_idx = []
        for d in range(self.D):
            slots = ["core"] + [f"jet{k}" for k in self.jet_map[d]]
            self.logit_idx.append(np.array(
                [add(f"logit {slots[j]} [{labels[d]}]", -cfg["max_logit"], cfg["max_logit"])
                 for j in range(self.ncomp[d] - 1)], dtype=int))
        self.corefwhm_idx = [add(f"FWHM core [{labels[d]}]", cfg["min_fwhm"], cfg["max_core_fwhm"])
                             for d in range(self.D)]
        self.shift_idx = []
        for d in range(self.D):
            if d == ref or self.ncomp[d] < 2:
                self.shift_idx.append(None)
            else:
                ms = cfg["max_core_shift"]
                self.shift_idx.append((add(f"core shift x [{labels[d]}]", -ms, ms),
                                       add(f"core shift y [{labels[d]}]", -ms, ms)))
        self.jet_idx = []
        for k in range(n_jets):
            mj = cfg["max_jet_offset"]
            self.jet_idx.append((add(f"jet{k} x", -mj, mj), add(f"jet{k} y", -mj, mj),
                                 add(f"jet{k} FWHM", cfg["min_fwhm"], cfg["max_fwhm"])))
        self.names = names
        self.lower = np.array(lower, float)
        self.upper = np.array(upper, float)
        self.n_params = len(names)

    def fractions(self, q, d):
        """Flux fractions of file d (core first, then the jets of jet_map[d]), softmax of the
        logits; they add up to 1."""
        logits = np.append(q[self.logit_idx[d]], 0.0)
        logits -= logits.max()
        w = np.exp(logits)
        return w / w.sum()

    def shift(self, q, d):
        """Core shift (sx, sy) of file d in uas; (0, 0) for the reference file."""
        if self.shift_idx[d] is None:
            return np.zeros(2)
        return np.array([q[self.shift_idx[d][0]], q[self.shift_idx[d][1]]])

    def geometry(self, q, d):
        """x, y and FWHM of the components of file d in its own frame (core at the origin).
        Basis of _gaussians, components and the figures."""
        s = self.shift(q, d)
        xs, ys, fw = [0.0], [0.0], [q[self.corefwhm_idx[d]]]
        for k in self.jet_map[d]:
            ix, iy, iw = self.jet_idx[k]
            xs.append(q[ix] - s[0])
            ys.append(q[iy] - s[1])
            fw.append(q[iw])
        return np.array(xs), np.array(ys), np.array(fw)

    def components(self, q, d):
        """List [(fraction, x, y, FWHM, jet index or None for the core)] of file d, in its own
        frame. Used by the report and plot_components."""
        f = self.fractions(q, d)
        xs, ys, fw = self.geometry(q, d)
        ids = [None] + self.jet_map[d]
        return [(f[i], xs[i], ys[i], fw[i], ids[i]) for i in range(len(f))]

    def _gaussians(self, q, d, u, v, q2):
        """Visibility of each unit-flux Gaussian of file d at each (u, v): envelope
        exp(-2 pi^2 sigma^2 |uv|^2) times the phase of its position. One row per component;
        visibility and _dataset_terms combine the rows with the flux fractions."""
        xs, ys, fw = self.geometry(q, d)
        sig = fw * FWHM_TO_SIGMA_RAD
        env = np.exp(-2.0 * np.pi**2 * np.outer(sig**2, q2))
        phase = np.exp(TWO_PI_I_UAS * (np.outer(xs, u) + np.outer(ys, v)))
        return env * phase, sig

    def visibility(self, q, d, u, v):
        """Complex model visibility of file d (total flux 1) at arbitrary (u, v). Not used by
        the fit; used for the Jy flux scale, the self-calibration of the images, the dynamic
        range, the residual map and plot_uv."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        G, _ = self._gaussians(q, d, u, v, u**2 + v**2)
        return self.fractions(q, d) @ G

    def _dataset_terms(self, q, d, blk, need_jac):
        """Normalised residuals (obs - model)/sigma of the closure phases and log closure
        amplitudes of file d and, if need_jac, their analytic derivatives with respect to each
        parameter. Core of the fit: residuals, jacobian, chi2_red and filter_closures go
        through here.

        Model visibilities on the block's baselines are combined into closure phases
        arg(V1 V2 V3) and log closure amplitudes ln|V1| + ln|V2| - ln|V3| - ln|V4|; the phase
        residual is wrapped to (-pi, pi]. The Jacobian uses d arg(V) = Im(dV/V) and
        d ln|V| = Re(dV/V), summed over the baselines of each closure."""
        f = self.fractions(q, d)
        G, sig = self._gaussians(q, d, blk.u, blk.v, blk.q2)
        V = f @ G
        i1, i2, i3 = blk.cp_idx
        j1, j2, j3, j4 = blk.ca_idx
        r_cp = wrap_phase(blk.cp_obs - np.angle(V[i1] * V[i2] * V[i3])) / blk.cp_sig
        la = np.log(np.maximum(np.abs(V), VIS_FLOOR))
        r_ca = (blk.ca_obs - (la[j1] + la[j2] - la[j3] - la[j4])) / blk.ca_sig
        if not need_jac:
            return r_cp, r_ca

        dfw = -4.0 * np.pi**2 * sig * FWHM_TO_SIGMA_RAD
        rows, cols = [], []
        for j in range(len(f) - 1):
            rows.append(f[j] * (G[j] - V))
            cols.append(self.logit_idx[d][j])
        rows.append(f[0] * G[0] * dfw[0] * blk.q2)
        cols.append(self.corefwhm_idx[d])
        sx_row = np.zeros_like(V)
        sy_row = np.zeros_like(V)
        for slot, k in enumerate(self.jet_map[d]):
            c = slot + 1
            fg = f[c] * G[c]
            rx = fg * TWO_PI_I_UAS * blk.u
            ry = fg * TWO_PI_I_UAS * blk.v
            ix, iy, iw = self.jet_idx[k]
            rows += [rx, ry, fg * dfw[c] * blk.q2]
            cols += [ix, iy, iw]
            sx_row -= rx
            sy_row -= ry
        if self.shift_idx[d] is not None:
            rows += [sx_row, sy_row]
            cols += list(self.shift_idx[d])
        tiny = np.abs(V) < VIS_FLOOR
        W = np.array(rows) / np.where(tiny, 1.0, V)
        W[:, tiny] = 0.0  # log|V| is clamped there, so its derivative is zero
        J_cp = -np.imag(W[:, i1] + W[:, i2] + W[:, i3]) / blk.cp_sig
        J_ca = -np.real(W[:, j1] + W[:, j2] - W[:, j3] - W[:, j4]) / blk.ca_sig
        return r_cp, r_ca, J_cp, J_ca, cols

    def residuals(self, q, blocks):
        """All normalised residuals of all files in one vector. This is what least_squares
        minimises (run_least_squares) and the basis of every chi2."""
        out = []
        for d, blk in enumerate(blocks):
            if blk is None or blk.n == 0:
                continue
            out.extend(self._dataset_terms(q, d, blk, False))
        return np.concatenate(out)

    def jacobian(self, q, blocks):
        """Analytic Jacobian of residuals (rows = closures, columns = parameters). Used by
        least_squares to converge and by covariance for the 1-sigma errors."""
        n_rows = sum(blk.n for blk in blocks if blk is not None)
        J = np.zeros((n_rows, self.n_params))
        r0 = 0
        for d, blk in enumerate(blocks):
            if blk is None or blk.n == 0:
                continue
            _, _, J_cp, J_ca, cols = self._dataset_terms(q, d, blk, True)
            J[r0:r0 + blk.n_cp, cols] = J_cp.T
            r0 += blk.n_cp
            J[r0:r0 + blk.n_ca, cols] = J_ca.T
            r0 += blk.n_ca
        return J

    def chi2_red(self, q, blocks):
        """Reduced chi2 on closures used in the fit: sum(r^2) / (N - n_params).
        Used to compare and rank the candidates during the search."""
        r = self.residuals(q, blocks)
        return float(np.sum(r**2) / max(len(r) - self.n_params, 1))

    def chi2_red_validation(self, q, blocks, d=None):
        """Reduced chi2 on the validation closures, never seen by the fit: sum(r^2) / N (no
        parameters are subtracted because none were fitted to these data). With d, only for
        file d. This is the final metric reported in fit_summary.txt."""
        if d is not None:
            if blocks[d].n == 0:
                return np.nan
            r = np.concatenate(self._dataset_terms(q, d, blocks[d], False))
        else:
            r = self.residuals(q, blocks)
        return float(np.sum(r**2) / max(len(r), 1))

    def random_start(self, rng, cfg):
        """Random parameter vector within the bounds (logits in [-3, 3], jet FWHM between 30
        and 3000 uas). Used for the random starts of the joint search."""
        q = rng.uniform(self.lower, self.upper)
        for d in range(self.D):
            # logits in [-3, 3]: flux ratios between components from ~1/400 to ~400
            q[self.logit_idx[d]] = rng.uniform(-3.0, 3.0, size=len(self.logit_idx[d]))
            q[self.corefwhm_idx[d]] = rng.uniform(cfg["min_fwhm"], cfg["max_core_fwhm"])
        for k in range(self.n_jets):
            # jet FWHM between 30 and 3000 uas: the sizes of jet features seen in practice
            q[self.jet_idx[k][2]] = rng.uniform(max(cfg["min_fwhm"], 30.0),
                                                min(cfg["max_fwhm"], 3000.0))
        return q

    def perturb(self, q, rng, cfg):
        """Copy of q with small noise (logits, FWHM, core shifts, jet positions ~150 uas).
        Gives starts close to a good solution, to check it is not a local minimum."""
        # noise levels: logits +-0.3 (~30% in flux), FWHM +-15%, core shift 15% of its bound,
        # jet positions +-150 uas; small enough to stay near the solution, large enough to leave
        # a shallow local minimum
        p = q.copy()
        for d in range(self.D):
            p[self.logit_idx[d]] += rng.normal(0.0, 0.3, size=len(self.logit_idx[d]))
            p[self.corefwhm_idx[d]] *= rng.lognormal(0.0, 0.15)
            if self.shift_idx[d] is not None:
                p[list(self.shift_idx[d])] += rng.normal(0.0, 0.15 * cfg["max_core_shift"], size=2)
        for k in range(self.n_jets):
            ix, iy, iw = self.jet_idx[k]
            p[[ix, iy]] += rng.normal(0.0, 150.0, size=2)
            p[iw] *= rng.lognormal(0.0, 0.15)
        return np.clip(p, self.lower, self.upper)


class MeanSquareObjective:
    """Scalar objective (mean squared residual) for differential_evolution, which needs a
    number rather than a vector. Used in fit_band_task and in the joint search."""

    def __init__(self, model, blocks):
        self.model, self.blocks = model, blocks

    def __call__(self, q):
        return float(np.mean(self.model.residuals(q, self.blocks)**2))


def run_least_squares(model, start, blocks, max_nfev):
    """Polish a parameter vector with least squares (bounded trust region, analytic Jacobian).
    The soft_l1 loss down-weights closures with residuals > 2 sigma, so a few outliers cannot
    pull the fit. Every polish in the script goes through here."""
    # soft_l1 with f_scale = 2: residuals up to ~2 sigma count as normal chi2, larger ones grow
    # only linearly; tolerances 1e-9 so convergence is set by max_nfev or by the data
    return least_squares(
        model.residuals, start, jac=model.jacobian, args=(blocks,),
        bounds=(model.lower, model.upper), method="trf", x_scale="jac",
        loss="soft_l1", f_scale=2.0, ftol=1e-9, xtol=1e-9, gtol=1e-9,
        max_nfev=max_nfev,
    )


# State of each pool process (model, closures, max_nfev). The functions below live at module
# level so they can be sent to other processes.
_WORKER = {}


def _init_polish_worker(model, blocks, max_nfev):
    """Runs once in each process: stores the model and the closures, so they are not sent
    again with every candidate."""
    _WORKER.update(model=model, blocks=blocks, max_nfev=max_nfev)


def _polish_task(start):
    """Polish one start with the model stored in the process. Returns (q, status, nfev).
    Used in the two polishing stages of the joint search."""
    result = run_least_squares(_WORKER["model"], start, _WORKER["blocks"], _WORKER["max_nfev"])
    return result.x, int(result.status), int(result.nfev)


def _polish_model_task(item):
    """Like _polish_task, but each task carries its own model (item = (model, start)).
    Used in the final polishes and in the refits of the significance test."""
    model, start = item
    result = run_least_squares(model, start, _WORKER["blocks"], _WORKER["max_nfev"])
    return result.x, int(result.status), int(result.nfev)


def fit_band_task(task):
    """Independent fit of a single file (step 1): differential evolution on thinned closures,
    then least squares from the best solution and from perturbations of it.
    Returns (chi2_red, q). align_band_fits uses these results to build the joint start."""
    model, coarse, train, cfg, seed, extra_starts = task
    rng = np.random.default_rng(seed)
    x0 = extra_starts[0] if extra_starts else None
    de = differential_evolution(
        MeanSquareObjective(model, coarse), bounds=list(zip(model.lower, model.upper)),
        seed=seed, maxiter=cfg["band_de_maxiter"], popsize=cfg["band_de_popsize"],
        polish=False, updating="immediate", workers=1, x0=x0,
    )
    starts = [de.x] + list(extra_starts)
    starts += [model.perturb(de.x, rng, cfg) for _ in range(cfg["band_perturb"])]
    best = None
    for start in starts:
        result = run_least_squares(model, start, train, min(cfg["max_nfev"], 500))
        chi2 = model.chi2_red(result.x, train)
        if best is None or chi2 < best[0]:
            best = (chi2, result.x)
    return best


def parallel_map(func, items, workers, initializer=None, initargs=()):
    """Apply func to every item in parallel (one process per core) and return the results in
    the same order. With workers=1 it runs serially, which helps debugging."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [func(item) for item in items]
    with ProcessPoolExecutor(max_workers=min(workers, len(items)),
                             initializer=initializer, initargs=initargs) as pool:
        return list(pool.map(func, items))


def clean_path(text):
    """Clean a path typed or dragged into the terminal (quotes, '\\ ', '~')."""
    return os.path.expanduser(text.strip().strip("'\"").replace("\\ ", " "))


def ask_fov(prompt, default):
    """Ask for the search FOV in uas; an empty Enter returns the default."""
    while True:
        text = input(prompt).strip()
        if not text:
            return default
        try:
            value = float(text)
            if value > 0:
                return value
        except ValueError:
            pass
        print("  Enter a number > 0 (in uas), or press Enter for the default.")


def collect_specs(args):
    """Read the input: the UVFITS files with the search FOV of each one, from the command line
    as file[:fov_uas] or asked one by one. The number of components is not asked: it is what
    this script chooses. The specs go to load_dataset."""
    specs = []
    if args.files:
        for item in args.files:
            path, _, fov = item.partition(":")
            specs.append({"path": clean_path(path),
                          "fov": float(fov) if fov else args.max_jet_offset})
    else:
        print("Enter the UVFITS files one by one (empty Enter to finish).")
        while True:
            text = input(f"UVFITS file #{len(specs) + 1}: ").strip()
            if not text:
                if len(specs) >= 1:
                    break
                continue
            path = clean_path(text)
            if not os.path.isfile(path):
                print(f"  File not found: {path}")
                continue
            fov = ask_fov(f"  Search radius (FOV) around the core in uas "
                          f"[Enter = {args.max_jet_offset:.0f}]: ", args.max_jet_offset)
            specs.append({"path": path, "fov": fov})
    for spec in specs:
        if not os.path.isfile(spec["path"]):
            sys.exit(f"File not found: {spec['path']}")
    return specs


def load_dataset(spec, cfg):
    """Load one UVFITS with ehtim: optional coherent averaging, drop non-finite data, and form
    log closure amplitudes and closure phases (minimal independent set at each time,
    count="min"; only visibilities with SNR > snrcut; closure amplitudes debiased for the
    amplitude noise bias). Also returns the visibilities (used only for the Jy scale,
    self-calibrated images and plots), the frequency and the resolution 1/B_max ("beam"),
    which sets the scale of the revive check and of the residual maps."""
    # ehtim prints its own messages (welcome banner, header details); they are not needed here
    with contextlib.redirect_stdout(io.StringIO()):
        import ehtim as eh
        obs = eh.obsdata.load_uvfits(spec["path"])
    if cfg["avg"]:
        obs = obs.avg_coherent(cfg["avg"])
    good = (np.isfinite(obs.data["vis"]) & np.isfinite(obs.data["sigma"])
            & (obs.data["sigma"] > 0))
    if not np.all(good):
        obs.data = obs.data[good]
    # log closure amplitudes first (ehtim record-array mutation issue)
    logcamp = obs.c_amplitudes(vtype="vis", mode="all", count="min", ctype="logcamp",
                               debias=True, snrcut=cfg["snrcut"])
    cphase = obs.c_phases(vtype="vis", mode="all", count="min", ang_unit="rad",
                          snrcut=cfg["snrcut"])
    dat = obs.unpack(["u", "v", "vis", "sigma", "time", "t1", "t2"], mode="all", conj=False)
    return {
        "path": spec["path"], "name": os.path.basename(spec["path"]),
        "fov": float(spec["fov"]), "freq": float(obs.rf),
        # angular resolution 1/B_max [uas]: sets every scale-dependent threshold
        "beam": float(1.0 / np.max(np.hypot(dat["u"], dat["v"])) / RADPERUAS),
        "closures": {"cphase": cphase, "logcamp": logcamp},
        "u": np.asarray(dat["u"], float), "v": np.asarray(dat["v"], float),
        "time": np.asarray(dat["time"], float), "t1": np.asarray(dat["t1"]), "t2": np.asarray(dat["t2"]),
        "vis": np.asarray(dat["vis"]), "sigma": np.asarray(dat["sigma"]).real.astype(float),
    }


def short_names(names):
    """Shortest distinguishing part of the file names (e.g. C1, C2, X1 ...), used as file
    labels in fit_summary.txt."""
    parts = [n.split(".") for n in names]
    if len(names) > 1 and len({len(p) for p in parts}) == 1:
        diff = [i for i in range(len(parts[0])) if len({p[i] for p in parts}) > 1]
        out = [".".join(p[i] for i in diff) for p in parts]
        if diff and len(set(out)) == len(out):
            return out
    return [os.path.splitext(n)[0] for n in names]


def band_model(ncomp, label, cfg, fov):
    """JointModel for a single file (no core shift, jets free), with the jet search box
    limited to that file's FOV. Fitted by fit_band_task in step 1."""
    return JointModel([ncomp], [list(range(ncomp - 1))], 0, ncomp - 1, [label],
                      dict(cfg, max_jet_offset=fov))


def recentre_on_brightest(model, q):
    """Take a single-file fit and call its brightest component the core: returns the core and
    the other components with positions relative to it. Closures do not fix the absolute
    position, so this puts every per-file fit in the same convention before align_band_fits."""
    comps = model.components(q, 0)
    # criterion: the core is the brightest component of the map
    b = int(np.argmax([c[0] for c in comps]))
    cx, cy = comps[b][1], comps[b][2]
    rel = [(c[0], c[1] - cx, c[2] - cy, c[3]) for c in comps]
    return rel[b], [c for i, c in enumerate(rel) if i != b]


def align_band_fits(datasets, band_results, ref, cfg):
    """Match the jets of every per-file fit to those of one file (jref, the lowest frequency
    when all have the same number of components), with the Hungarian algorithm on position
    and FWHM differences. The median offset of the matched jets gives a first core shift per
    file. The result goes to seed_full_vector to build the starting point of the joint fit."""
    D = len(datasets)
    cores, jets = [], []
    for d in range(D):
        core, others = recentre_on_brightest(band_results[d]["model"], band_results[d]["q"])
        cores.append(core)
        jets.append(others)

    # the jets are defined by the file with most jets; with the same number in every file, the
    # lowest frequency, where the optically thin jet is brightest and most extended
    jref = max(range(D), key=lambda d: (len(jets[d]), -datasets[d]["freq"]))
    shared_xy = np.array([[c[1], c[2]] for c in jets[jref]]).reshape(-1, 2)
    shared_fw = np.array([c[3] for c in jets[jref]])
    n_jets = len(jets[jref])

    jet_map, offsets, matched_fw = [], np.zeros((D, 2)), [[] for _ in range(n_jets)]
    for d in range(D):
        if not jets[d]:
            jet_map.append([])
            continue
        p = np.array([[c[1], c[2]] for c in jets[d]])
        fw = np.array([c[3] for c in jets[d]])
        # matching cost in uas: distance between the jets plus half the FWHM difference (position
        # matters most; size only breaks ties between jets at similar distances)
        cost = (np.linalg.norm(p[:, None, :] - shared_xy[None, :, :], axis=2)
                + 0.5 * np.abs(fw[:, None] - shared_fw[None, :]))
        rows, cols = linear_sum_assignment(cost)
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
        jet_map.append([int(k) for k in cols])
        if d != jref:
            # median of the offsets of all matched jets: one badly matched jet cannot move it
            offsets[d] = np.median(shared_xy[cols] - p[rows], axis=0)
        for r, k in zip(rows, cols):
            matched_fw[k].append(fw[r])

    # offsets[d] = s_d - s_jref ; reference core fixed at origin => s_jref = -offsets[ref]
    shifts = np.clip(offsets - offsets[ref], -cfg["max_core_shift"], cfg["max_core_shift"])
    shared_ref = shared_xy - offsets[ref]
    return jet_map, n_jets, jref, cores, jets, shifts, shared_ref, matched_fw


def seed_full_vector(model, cores, jets, jet_map, shifts, shared_ref, matched_fw):
    """Starting vector q of the joint model built from the aligned per-file fits (fluxes,
    core FWHMs, core shifts, shared jet positions and the median of the matched FWHMs).
    Jets not matched in a file start with 0.2% of the core flux. It is the x0 of the joint
    differential evolution, one of the forced starts of the polishing, and the model used by
    filter_closures."""
    q = np.zeros(model.n_params)
    for d in range(model.D):
        fr = np.full(model.n_jets + 1, 2e-3 * cores[d][0])
        fr[0] = cores[d][0]
        for slot, k in enumerate(jet_map[d]):
            fr[1 + k] = max(jets[d][slot][0], 2e-3 * cores[d][0])
        q[model.logit_idx[d]] = np.log(fr[:-1] / fr[-1])
        q[model.corefwhm_idx[d]] = cores[d][3]
        if model.shift_idx[d] is not None:
            q[list(model.shift_idx[d])] = shifts[d]
    for k in range(model.n_jets):
        ix, iy, iw = model.jet_idx[k]
        q[ix], q[iy] = shared_ref[k]
        q[iw] = np.median(matched_fw[k]) if matched_fw[k] else 300.0  # 300 uas if never matched
    return np.clip(q, model.lower, model.upper)


def without_jet(full, q_full, keep, labels, cfg):
    """Build a smaller model where file d uses only the jets in keep[d], and copy the values of
    q_full into it. The significance test uses it to refit with one jet removed from one map;
    the refit only measures dchi2, the model with all jets is the one reported.
    Returns (model, q, keep)."""
    keep = [sorted(m) for m in keep]
    ncomp = [len(m) + 1 for m in keep]
    used = sorted({k for m in keep for k in m})
    renumber = {k: i for i, k in enumerate(used)}
    model = JointModel(ncomp, [[renumber[k] for k in m] for m in keep], full.ref, len(used), labels, cfg)
    q = np.zeros(model.n_params)
    for d in range(full.D):
        fr = full.fractions(q_full, d)
        f = np.maximum(np.array([fr[0]] + [fr[1 + full.jet_map[d].index(k)] for k in keep[d]]), 1e-12)
        q[model.logit_idx[d]] = np.log(f[:-1] / f[-1])
        q[model.corefwhm_idx[d]] = q_full[full.corefwhm_idx[d]]
        if model.shift_idx[d] is not None:
            q[list(model.shift_idx[d])] = full.shift(q_full, d)
    for k in used:
        q[list(model.jet_idx[renumber[k]])] = q_full[list(full.jet_idx[k])]
    return model, np.clip(q, model.lower, model.upper), keep


def set_fraction(model, q, d, slot, value):
    """Return q with component `slot` of file d set to flux fraction `value` (others rescaled).
    Used by revive_dead_jets to give a moved jet a starting flux."""
    f = model.fractions(q, d).copy()
    f[slot] = 0.0
    f *= (1.0 - value) / f.sum()
    f[slot] = value
    f = np.maximum(f, 1e-9)
    q = q.copy()
    q[model.logit_idx[d]] = np.clip(np.log(f[:-1] / f[-1]), model.lower[model.logit_idx[d]],
                                    model.upper[model.logit_idx[d]])
    return q


def residual_peaks(model, q, datasets, cfg, n_peaks=3):
    """Brightest peaks of the sum of the residual dirty maps of all files (each self-calibrated
    to the model and normalised by its rms), in the reference frame. Peaks within half a beam
    of a core are ignored. revive_dead_jets uses them as new places to try a jet."""
    live = [k for k in range(model.n_jets)
            if max(model.fractions(q, d)[1 + model.jet_map[d].index(k)]
                   for d in range(model.D) if k in model.jet_map[d]) >= cfg["dead_fraction"]]
    reach = max([np.hypot(q[model.jet_idx[k][0]], q[model.jet_idx[k][1]]) for k in live] + [1000.0])
    beam_max = max(ds["beam"] for ds in datasets)
    # map half-size: twice the distance of the farthest jet with flux plus 3 beams (at least
    # 1 mas), never beyond the search box; 301 x 301 pixels
    half = min(cfg["max_jet_offset"], 2.0 * reach + 3.0 * beam_max)
    grid = np.linspace(-half, half, 301)
    total = np.zeros((grid.size, grid.size))
    cores = []
    for d, ds in enumerate(datasets):
        s = model.shift(q, d)
        cores.append(s)
        scale = flux_scale(model, q, d, ds)
        model_vis = scale * model.visibility(q, d, ds["u"], ds["v"])
        res = self_calibrate(ds, model_vis) - model_vis
        ax = np.exp(-TWO_PI_I_UAS * np.outer(ds["u"], grid - s[0]))
        ay = np.exp(-TWO_PI_I_UAS * np.outer(ds["v"], grid - s[1]))
        img = np.real((ay * res[:, None]).T @ ax)  # rows = y, columns = x
        # each map is divided by its rms so that every file weighs the same in the sum
        total += img / max(np.std(img), 1e-12)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    for s, ds in zip(cores, datasets):  # residuals within half a beam of a core are not jets
        total[np.hypot(xx - s[0], yy - s[1]) < 0.5 * ds["beam"]] = -np.inf
    peaks = []
    for _ in range(n_peaks):
        iy, ix = np.unravel_index(np.argmax(total), total.shape)
        if not np.isfinite(total[iy, ix]):
            break
        peaks.append((grid[ix], grid[iy]))
        # the next peak must be at least one (largest) beam away: a different feature
        total[np.hypot(xx - grid[ix], yy - grid[iy]) < beam_max] = -np.inf
    return peaks


def revive_dead_jets(model, q, datasets, blocks, cfg, log):
    """A jet with almost no flux in every map (< dead_fraction) is useless where it is: move it
    to a residual-map peak, refit, and keep the change only if
      * chi2 drops by more than 0.5% (smaller changes are within what a refit alone gives), and
      * the moved jet stays a separate, fainter component: more than half a beam from every
        core and fainter than the core in every map (closer than half a beam it cannot be told
        apart from the core).
    Applied to the best fit before the final polishes."""
    chi2 = model.chi2_red(q, blocks)
    for k in range(model.n_jets):
        users = [d for d in range(model.D) if k in model.jet_map[d]]
        fr = [model.fractions(q, d)[1 + model.jet_map[d].index(k)] for d in users]
        if max(fr) >= cfg["dead_fraction"]:
            continue
        for x, y in residual_peaks(model, q, datasets, cfg):
            trial = q.copy()
            ix, iy, iw = model.jet_idx[k]
            # start: FWHM = the largest beam (a peak of the residual map is about one beam wide)
            # and 2% of the flux in every map; the least-squares refit adjusts both
            trial[ix], trial[iy], trial[iw] = x, y, min(max(ds["beam"] for ds in datasets), cfg["max_fwhm"])
            for d in users:
                trial = set_fraction(model, trial, d, 1 + model.jet_map[d].index(k), 0.02)
            trial = np.clip(trial, model.lower, model.upper)
            new = run_least_squares(model, trial, blocks, cfg["max_nfev"]).x
            new_chi2 = model.chi2_red(new, blocks)
            # a revived jet must stay a jet: > half a beam from every core and fainter than it
            ok = True
            for d in users:
                rel = new[[ix, iy]] - model.shift(new, d)
                f = model.fractions(new, d)
                ok &= np.hypot(*rel) > 0.5 * datasets[d]["beam"] and f[1 + model.jet_map[d].index(k)] < f[0]
            accepted = ok and new_chi2 < chi2 * 0.995
            log(f"jet with no flux in any map moved to the residual peak at ({x / 1000:.2f}, "
                f"{y / 1000:.2f}) mas: chi2_red {chi2:.3f} -> {new_chi2:.3f}, "
                + ("kept" if accepted else "not kept (no improvement)" if ok else
                   "not kept (it ended within half a beam of a core, or brighter than it)"))
            if accepted:
                q, chi2 = new, new_chi2
                break
    return q


def robust_keep_mask(residual, minimum_sigma=8.0):
    """Mask of residuals to keep: |r| below max(8 sigma, median + 6 MAD). Robust because the
    median and MAD are not driven by the outliers themselves. Used by filter_closures."""
    # 1.4826 * MAD is the standard deviation for Gaussian noise; 6 of them above the median is far
    # outside the noise, and the 8-sigma floor keeps every closure when the fit is already good
    absolute = np.abs(residual)
    if len(absolute) == 0:
        return np.ones(0, bool), minimum_sigma
    center = np.median(absolute)
    mad = 1.4826 * np.median(np.abs(absolute - center))
    limit = max(minimum_sigma, center + 6.0 * max(mad, 1e-6))
    return absolute <= limit, limit


def filter_closures(model, q, closures, labels, log):
    """Remove outlier closures (bad data) from the training set, judged by their residuals
    with respect to the seed model (robust_keep_mask). The filtered set is what the joint
    search and the final polishes fit; the validation closures are never filtered."""
    blocks = make_blocks(closures)
    filtered = []
    for d, (c, blk) in enumerate(zip(closures, blocks)):
        if blk.n == 0:
            filtered.append(c)
            continue
        r_cp, r_ca = model._dataset_terms(q, d, blk, False)
        keep_cp, lim_cp = robust_keep_mask(r_cp)
        keep_ca, lim_ca = robust_keep_mask(r_ca)
        filtered.append({"cphase": c["cphase"][keep_cp], "logcamp": c["logcamp"][keep_ca]})
        n_out = (~keep_cp).sum(), (~keep_ca).sum()
        if sum(n_out):
            log(f"      {labels[d]}: {n_out[0]} closure phases and {n_out[1]} log closure "
                f"amplitudes removed as outliers")
    return filtered


def covariance(model, q, blocks):
    """Covariance of the parameters, (J^T J)^-1 scaled by the reduced chi2. The square root of
    its diagonal gives the formal 1-sigma errors of the report, best_params.json and the error
    bars of core_shifts.png; derived quantities (r, PA) get theirs by error propagation."""
    J = model.jacobian(q, blocks)
    chi2 = model.chi2_red(q, blocks)
    # scaled by the reduced chi2 so the errors grow when the model does not reach the noise;
    # pseudo-inverse in case a parameter is not constrained (e.g. a jet with ~0 flux)
    return np.linalg.pinv(J.T @ J) * chi2


def flux_scale(model, q, d, ds):
    """Total flux in Jy of file d: least-squares scale of the model amplitudes (total flux 1)
    to the observed amplitudes. Closures do not constrain the total flux, so this only
    converts fractions into approximate Jy for the report, the images and plot_uv."""
    amp_obs = np.abs(ds["vis"])
    amp_mod = np.abs(model.visibility(q, d, ds["u"], ds["v"]))
    denom = np.sum(amp_mod**2)
    return float(np.sum(amp_obs * amp_mod) / denom) if denom > 0 else np.nan


def self_calibrate(ds, model_vis, n_iter=50):
    """Amplitude+phase station gains per integration so that vis ~ g1 conj(g2) model (antsol),
    and the visibilities corrected by them. Only used to build images (residual_peaks,
    dynamic_range); the fit itself uses closures only, which do not depend on station gains."""
    vis = ds["vis"].copy()
    w = 1.0 / np.maximum(ds["sigma"], 1e-12)**2
    for t in np.unique(ds["time"]):
        idx = np.flatnonzero(ds["time"] == t)
        sites, inv = np.unique(np.concatenate([ds["t1"][idx], ds["t2"][idx]]), return_inverse=True)
        a, b = inv[:len(idx)], inv[len(idx):]
        V, M, W = vis[idx], model_vis[idx], w[idx]
        g = np.ones(len(sites), complex)
        for _ in range(n_iter):
            num = np.zeros(len(sites), complex)
            den = np.zeros(len(sites))
            # V_ab = g_a conj(g_b) M_ab, used from both ends of the baseline
            np.add.at(num, a, W * V * g[b] * np.conj(M))
            np.add.at(den, a, W * np.abs(g[b] * M)**2)
            np.add.at(num, b, W * np.conj(V) * g[a] * M)
            np.add.at(den, b, W * np.abs(g[a] * M)**2)
            # half of the update per iteration (damping) for stable convergence; 50 iterations
            g = 0.5 * g + 0.5 * np.where(den > 0, num / np.maximum(den, 1e-30), g)
        vis[idx] = V / (g[a] * np.conj(g[b]))
    return vis


def dynamic_range(model, q, d, ds, scale, max_pix=301):
    """Peak of the naturally weighted dirty map / rms of the residual dirty map, after
    self-calibrating the visibilities to the fitted model. Reported per file as an extra
    check that nothing bright is left in the residuals."""
    half = ds["fov"]
    # pixel = beam/4 (well sampled), between 64 and max_pix pixels per side
    npix = int(min(max_pix, max(64, 2.0 * half / (ds["beam"] / 4.0))))
    grid = np.linspace(-half, half, npix)
    w = 1.0 / np.maximum(ds["sigma"], 1e-12)**2
    model_vis = scale * model.visibility(q, d, ds["u"], ds["v"])
    vis = self_calibrate(ds, model_vis)
    res = vis - model_vis
    ax = np.exp(-TWO_PI_I_UAS * np.outer(ds["u"], grid))
    ay = np.exp(-TWO_PI_I_UAS * np.outer(ds["v"], grid))
    dirty = lambda vis: np.real((ay * (w * vis)[:, None]).T @ ax) / w.sum()
    peak = float(dirty(vis).max())
    rms = float(np.std(dirty(res)))
    return peak, rms, peak / rms if rms > 0 else np.inf


def solution_distance(model, qa, qb):
    """RMS difference between two solutions in positions, FWHMs and flux fractions. Used in
    the report to show whether the final candidates converged to the same answer."""
    pos = [i for k in range(model.n_jets) for i in model.jet_idx[k][:2]]
    pos += [i for s in model.shift_idx if s is not None for i in s]
    fws = [model.jet_idx[k][2] for k in range(model.n_jets)] + list(model.corefwhm_idx)
    fr = np.concatenate([model.fractions(qa, d) - model.fractions(qb, d) for d in range(model.D)])
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x)**2))) if len(x) else 0.0
    return rms(qa[pos] - qb[pos]), rms(qa[fws] - qb[fws]), rms(fr)


def fit_once(args, cfg, datasets, ncomp_all, out_dir):
    """One complete fit with ncomp_all components in every map, written to out_dir: independent
    per-file fits, joint search, polishing, significance test, report (fit_summary.txt,
    best_params.json) and figures. Returns what main needs to compare the models: number of
    parameters, sum of squared training residuals and number of training closures (BIC),
    validation chi2, and the significance of every jet in every map (auto_summary.txt)."""
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    report = []

    def log(text="", show=True):
        if show:
            print(text, flush=True)
        report.append(text)

    def start_step(text):
        # prints "text ..." and leaves the line open; end_step closes it with the time taken
        print("  " + text + " ...", end="", flush=True)
        report.append(text + " ...")
        return time.time()

    def end_step(t0):
        print(f" done ({time.time() - t0:.0f} s)", flush=True)
        report[-1] += f" done ({time.time() - t0:.0f} s)"

    # The reference file (core at the origin) is the highest frequency unless --ref is given.
    labels = [f"{ds['name']} ({ds['freq'] / 1e9:.2f} GHz)" for ds in datasets]
    short = short_names([ds["name"] for ds in datasets])
    D = len(datasets)
    # highest frequency: smallest core shift (closest to the jet base) and best resolution
    ref = args.ref if args.ref is not None else max(range(D), key=lambda d: datasets[d]["freq"])

    # the header is printed once by main; here it only goes to fit_summary.txt
    log("Run log", False)
    log(f"{ncomp_all} components per map (1 core + {ncomp_all - 1} shared jets); reference map: "
        f"{labels[ref]}", False)
    log(f"Settings: preset {args.preset}, SNR cut {cfg['snrcut']:g}, "
        + (f"time averaging {cfg['avg']:g} s" if cfg["avg"] else "no time averaging"), False)
    log("", False)
    log(f"{'file':44s} {'CP':>6s} {'logCA':>6s} {'beam[uas]':>9s} {'FOV[uas]':>9s}", False)
    for ds, lab in zip(datasets, labels):
        c = ds["closures"]
        log(f"{lab:44s} {len(c['cphase']):6d} {len(c['logcamp']):6d} {ds['beam']:9.0f} "
            f"{ds['fov']:9.0f}", False)
    log("", False)

    # Train/validation split: the validation closures are never fitted and give the final chi2.
    full = [ds["closures"] for ds in datasets]
    used = thin_closures(full, cfg["max_points"])
    train, validation = split_by_time(used)

    # Step 1: every file is fitted on its own. Its jets are then matched across files
    # (align_band_fits) to build the starting point of the joint fit.
    t0 = start_step("[1/4] Fitting each file separately (starting point of the joint fit)")
    tasks = []
    for d, ds in enumerate(datasets):
        bm = band_model(ncomp_all, labels[d], cfg, ds["fov"])
        # differential evolution on <= 160 closures of each type (fast); polishing on all of them
        coarse = make_blocks(thin_closures([train[d]], 160))
        tasks.append((bm, coarse, make_blocks([train[d]]), cfg, args.seed + 101 * d, []))
    results = parallel_map(fit_band_task, tasks, args.workers)
    band_results = [{"model": t[0], "q": r[1], "chi2": r[0]} for t, r in zip(tasks, results)]
    end_step(t0)

    jet_map, n_jets, jref, cores, jets, shifts, shared_ref, matched_fw = align_band_fits(
        datasets, band_results, ref, cfg)
    # joint model: every map has the core and all the shared jets
    model = JointModel([n_jets + 1] * D, [list(range(n_jets))] * D, ref, n_jets, labels, cfg)
    q_seed = seed_full_vector(model, cores, jets, jet_map, shifts, shared_ref, matched_fw)
    joint_train = filter_closures(model, q_seed, train, short, log)
    joint_blocks = make_blocks(joint_train)
    val_blocks = make_blocks(validation)

    def make_solution(m, x, status, nfev):
        return {"q": x, "status": status, "nfev": nfev,
                "train_chi2": m.chi2_red(x, joint_blocks),
                "validation_chi2": m.chi2_red_validation(x, val_blocks)}

    # Step 2: global search of the joint model. Differential evolution starting from the seed,
    # plus random and perturbed starts; the best ones go to the polishing.
    t0 = start_step(f"[2/4] Global search of the joint model ({model.n_params} parameters)")
    rng = np.random.default_rng(args.seed)
    # thinned to <= 120 closures of each type per file: DE needs many evaluations, and only has
    # to find the right region; the polishing uses all the closures
    de_blocks = make_blocks(thin_closures(joint_train, 120))
    de = differential_evolution(
        MeanSquareObjective(model, de_blocks), bounds=list(zip(model.lower, model.upper)),
        seed=args.seed, maxiter=cfg["de_maxiter"], popsize=cfg["de_popsize"], polish=False,
        updating="immediate", workers=1, x0=q_seed,
    )
    forced = [q_seed.copy(), de.x.copy()]
    starts = [model.random_start(rng, cfg) for _ in range(cfg["n_random"])]
    starts += [model.perturb(q_seed, rng, cfg) for _ in range(10)]
    starts += [model.perturb(de.x, rng, cfg) for _ in range(10)]
    # the random starts are ranked on <= 400 closures of each type per file; the n_polish - 2
    # best ones are polished, together with the seed and the DE result (always kept)
    score_blocks = make_blocks(thin_closures(joint_train, 400))
    scores = np.array([model.chi2_red(q, score_blocks) for q in starts])
    chosen = forced + [starts[i] for i in np.argsort(scores)[:max(cfg["n_polish"] - 2, 0)]]
    end_step(t0)

    # Step 3: least-squares polishing. Stage 1: every candidate on a thinned training set;
    # stage 2: the best ones on all the training closures.
    t0 = start_step(f"[3/4] Refining the best {len(chosen)} candidates with least squares")
    stage1_blocks = make_blocks(thin_closures(joint_train, cfg["polish_points"]))
    stage1 = parallel_map(_polish_task, chosen, args.workers,
                          initializer=_init_polish_worker,
                          initargs=(model, stage1_blocks, cfg["max_nfev"]))
    stage1_chi2 = [model.chi2_red(x, stage1_blocks) for x, _, _ in stage1]
    best_idx = np.argsort(stage1_chi2)[:cfg["n_final"]]
    polished = parallel_map(_polish_task, [stage1[i][0] for i in best_idx], args.workers,
                            initializer=_init_polish_worker,
                            initargs=(model, joint_blocks, cfg["max_nfev"]))
    stage2 = sorted((x for x, _, _ in polished), key=lambda x: model.chi2_red(x, joint_blocks))
    revive_notes = []

    def log_revive(text):
        revive_notes.append(text)

    # a jet with ~0 flux in every map is moved to the brightest residual (revive_dead_jets)
    stage2[0] = revive_dead_jets(model, stage2[0], datasets, joint_blocks, cfg, log_revive)
    q_best = stage2[0]

    # Final polishes: the stage-2 solutions plus perturbations of the best one. Every map keeps
    # every shared jet (same number of Gaussians in all maps); its flux in each map stays free.
    tasks = [(model, x) for x in stage2]
    tasks += [(model, model.perturb(q_best, rng, cfg)) for _ in range(len(stage2))]
    finals = parallel_map(_polish_model_task, tasks, args.workers,
                          initializer=_init_polish_worker,
                          initargs=(None, joint_blocks, cfg["max_nfev"]))
    solutions = sorted((make_solution(model, *res) for res in finals), key=lambda s: s["train_chi2"])
    q = solutions[0]["q"]
    n_final_polishes = len(tasks)
    end_step(t0)
    for note in revive_notes:
        log("      " + note)

    # Significance of every jet in every map: refit without that jet in that map only and compare
    # chi2 (dchi2 in units of the reduced chi2 of the best fit). The jet stays in the model either
    # way; dchi2 < dchi2_min (default 9, ~3 sigma for one parameter) only marks it as not
    # significant in that map.
    chi2_best_red = model.chi2_red(q, joint_blocks)
    s_with = float(np.sum(model.residuals(q, joint_blocks)**2))
    t0 = start_step(f"[4/4] Significance test: refitting without each jet in each map "
                    f"({D * n_jets} refits)")
    pairs_dk, sig_tasks = [], []
    for d in range(D):
        for k in range(n_jets):
            keep = [list(range(n_jets)) for _ in range(D)]
            keep[d].remove(k)
            m_k, q_k, _ = without_jet(model, q, keep, labels, cfg)
            pairs_dk.append((d, k))
            sig_tasks.append((m_k, q_k))
    sig_res = parallel_map(_polish_model_task, sig_tasks, args.workers,
                           initializer=_init_polish_worker,
                           initargs=(None, joint_blocks, cfg["max_nfev"]))
    dchi2 = {}
    for (d, k), (m_k, _), res in zip(pairs_dk, sig_tasks, sig_res):
        s_without = float(np.sum(m_k.residuals(res[0], joint_blocks)**2))
        dchi2[(d, k)] = (s_without - s_with) / chi2_best_red
    end_step(t0)

    t0 = start_step("Errors, report and figures")
    best = solutions[0]
    cov = covariance(model, q, joint_blocks)
    err = np.sqrt(np.clip(np.diag(cov), 0, None))

    # jet labels J1..JN sorted by distance from the reference core
    order = np.argsort([np.hypot(q[model.jet_idx[k][0]], q[model.jet_idx[k][1]]) for k in range(n_jets)])
    jet_names = {int(k): (f"J{rank + 1}", rank) for rank, k in enumerate(order)}

    # Report: fit_summary.txt (main results first, everything else afterwards) and best_params.json.
    scales = [flux_scale(model, q, d, datasets[d]) for d in range(D)]
    drs = [dynamic_range(model, q, d, datasets[d], scales[d]) for d in range(D)]
    rad2deg = 180.0 / np.pi

    def lin_err(grad):
        return float(np.sqrt(max(grad @ cov @ grad, 0.0)))

    def jet_geometry(d, k):
        """x, y, r, PA of shared jet k seen from the core of file d, with formal errors."""
        ix, iy, _ = model.jet_idx[k]
        sh = model.shift(q, d)
        x, y = q[ix] - sh[0], q[iy] - sh[1]
        r = float(np.hypot(x, y))
        pa = float(np.degrees(np.arctan2(x, y)))
        g = {name: np.zeros(model.n_params) for name in ("x", "y", "r", "pa")}
        pairs = [(ix, iy, 1.0)]
        if model.shift_idx[d] is not None:
            pairs.append((*model.shift_idx[d], -1.0))
        for jx, jy, sign in pairs:
            g["x"][jx] = sign
            g["y"][jy] = sign
            if r > 0:
                g["r"][jx], g["r"][jy] = sign * x / r, sign * y / r
                g["pa"][jx], g["pa"][jy] = sign * rad2deg * y / r**2, -sign * rad2deg * x / r**2
        return (x, y, r, pa), {name: lin_err(v) for name, v in g.items()}

    names = [ds["name"] for ds in datasets]
    source = os.path.commonprefix(names).rstrip("._-") or "source"
    detected = {dk: v >= cfg["dchi2_min"] for dk, v in dchi2.items()}
    jets_sorted = list(order)

    summary = []
    put = summary.append
    put(f"{source}: blind multi-frequency closure fit, {time.strftime('%Y-%m-%d %H:%M')}")
    put(f"Components per map: {n_jets + 1} (1 core + {n_jets} shared jets), the same in every map")
    put("Files: " + ", ".join(f"{short[d]} ({datasets[d]['freq'] / 1e9:.2f} GHz, "
                              f"FOV {datasets[d]['fov'] / 1000:.1f} mas)" for d in range(D)))
    put(f"Reference: core of {short[ref]}. No prior model was used; the fit starts from the closures only.")
    put("Units: uas for positions and sizes, deg for angles (PA east of north), Jy for flux.")
    put("Errors are formal 1-sigma from the closure fit; real uncertainties are likely larger.")

    put("")
    put(f"Core shifts (position of each core relative to the {short[ref]} core)")
    put(f"  {'file':6s} {'GHz':>6s} {'dx':>8s} {'+-':>5s} {'dy':>8s} {'+-':>5s} {'|shift|':>8s} {'+-':>5s} "
        f"{'PA':>7s} {'+-':>5s}")
    for d in sorted(range(D), key=lambda d: datasets[d]["freq"]):
        sh = model.shift(q, d)
        if model.shift_idx[d] is None:
            note = "reference" if d == ref else "no jets, undefined"
            put(f"  {short[d]:6s} {datasets[d]['freq'] / 1e9:6.2f} {'0':>8s} {'':5s} {'0':>8s} {'':5s} "
                f"{'0':>8s} {'':5s} {'':>7s} {'':5s}  ({note})")
            continue
        sx, sy = model.shift_idx[d]
        r = float(np.hypot(*sh))
        g_r, g_pa = np.zeros(model.n_params), np.zeros(model.n_params)
        if r > 0:
            g_r[sx], g_r[sy] = sh[0] / r, sh[1] / r
            g_pa[sx], g_pa[sy] = rad2deg * sh[1] / r**2, -rad2deg * sh[0] / r**2
        put(f"  {short[d]:6s} {datasets[d]['freq'] / 1e9:6.2f} {sh[0]:8.1f} {err[sx]:5.1f} {sh[1]:8.1f} "
            f"{err[sy]:5.1f} {r:8.1f} {lin_err(g_r):5.1f} {np.degrees(np.arctan2(*sh)):7.1f} {lin_err(g_pa):5.1f}")

    put("")
    put("Components per band (core of each map at the origin)")
    put("  x = RA offset (east positive), y = Dec offset (north positive), r = distance to the core.")
    put("  frac = fraction of the total flux of the band (from closures); flux = frac x amplitude")
    put("  scale of the visibilities, so the Jy values are approximate. FWHM of circular Gaussians.")
    put("  Jets keep the same position and FWHM in every band; only their flux changes.")
    put("  Every band has the same Gaussians. dchi2 = increase of chi2 (in units of the reduced")
    put(f"  chi2) when that jet is removed from that band and the fit is redone; dchi2 < "
        f"{cfg['dchi2_min']:.0f} (~3 sigma)")
    put("  means the band does not need that Gaussian (marked n.s., not significant); it is still")
    put("  part of the model.")
    for d in sorted(range(D), key=lambda d: datasets[d]["freq"]):
        comps = model.components(q, d)
        f0, _, _, fw0, _ = comps[0]
        put("")
        put(f"  {short[d]}  {datasets[d]['freq'] / 1e9:.2f} GHz   total flux {scales[d]:.3f} Jy   "
            f"beam (1/Bmax) {datasets[d]['beam']:.0f} uas   dynamic range {drs[d][2]:.0f}")
        put(f"  {'comp':5s} {'flux':>7s} {'frac':>7s} {'x':>8s} {'+-':>4s} {'y':>8s} {'+-':>4s} "
            f"{'r':>8s} {'+-':>4s} {'PA':>7s} {'+-':>5s} {'FWHM':>7s} {'+-':>4s}"
            + f" {'dchi2':>8s}")
        put(f"  {'core':5s} {f0 * scales[d]:7.4f} {f0:7.4f} {0:8.0f} {'':4s} {0:8.0f} {'':4s} "
            f"{'':8s} {'':4s} {'':7s} {'':5s} {fw0:7.0f} {err[model.corefwhm_idx[d]]:4.0f}")
        missing = []
        for k in sorted(model.jet_map[d], key=lambda k: jet_names[k][1]):
            if not detected[(d, k)]:
                missing.append(jet_names[k][0])
            f = model.fractions(q, d)[1 + model.jet_map[d].index(k)]
            (x, y, r, pa), e = jet_geometry(d, k)
            iw = model.jet_idx[k][2]
            line = (f"  {jet_names[k][0]:5s} {f * scales[d]:7.4f} {f:7.4f} {x:8.0f} {e['x']:4.0f} {y:8.0f} "
                    f"{e['y']:4.0f} {r:8.0f} {e['r']:4.0f} {pa:7.1f} {e['pa']:5.1f} {q[iw]:7.0f} {err[iw]:4.0f}")
            v = dchi2[(d, k)]
            line += (f" {v:8.1f}" if abs(v) < 1e5 else f" {v:8.1e}") \
                + ("" if detected[(d, k)] else "  n.s.")
            put(line)
        if missing:
            put(f"  not significant in this band: {', '.join(missing)} "
                "(fitted, but the band is as well described without it)")

    put("")
    put("Core distance of each jet in every band, r +- error (uas)")
    put("  " + f"{'jet':5s}" + "".join(f"{short[d]:>16s}" for d in sorted(range(D), key=lambda d: datasets[d]["freq"])))
    for k in jets_sorted:
        cells = []
        for d in sorted(range(D), key=lambda d: datasets[d]["freq"]):
            if (d, k) in detected:
                (_, _, r, _), e = jet_geometry(d, k)
                cells.append(f"{r:.0f} +- {e['r']:.0f}" + ("" if detected[(d, k)] else "*"))
            else:
                cells.append("-")
        if any(c != "-" for c in cells):
            put("  " + f"{jet_names[k][0]:5s}" + "".join(f"{c:>16s}" for c in cells))
    put("  The differences between bands of the same jet are the core shifts projected on the jet.")
    put("  * not significant in that band: the position comes from the other bands.")

    put("")
    put("Fit quality and procedure")
    put("  chi2_red = reduced chi2 of closure phases + log closure amplitudes on the validation")
    put("  closures (every 5th time stamp), which are never used in the fit: sum(r^2)/N. It is ~1")
    put("  when the model describes the data down to the noise, and it does not decrease just")
    put("  because more Gaussians are added (unlike the chi2 of the fitted closures).")
    put(f"  {'file':8s} {'freq[GHz]':>9s} {'N valid':>8s} {'chi2_red':>9s}")
    for d in range(D):
        put(f"  {short[d]:8s} {datasets[d]['freq'] / 1e9:9.2f} {val_blocks[d].n:8d} "
            f"{model.chi2_red_validation(q, val_blocks, d):9.3f}")
    put(f"  {'all':8s} {'':9s} {sum(bl.n for bl in val_blocks):8d} {best['validation_chi2']:9.3f}")
    put("  Dynamic range = peak of the dirty map / rms of the residual dirty map (natural weighting,")
    put("  field = search FOV of each file), after self-calibrating the visibilities to the fitted model:")
    put(f"  {'file':8s} {'peak [Jy/beam]':>14s} {'rms [Jy/beam]':>14s} {'DR':>8s}")
    for d in range(D):
        put(f"  {short[d]:8s} {drs[d][0]:14.4f} {drs[d][1]:14.6f} {drs[d][2]:8.0f}")
    # solutions within 1% of the best chi2 are equally good fits: how many reach it shows
    # whether the minimum is found reliably
    n_same = sum(1 for sol in solutions if sol["train_chi2"] < best["train_chi2"] * 1.01)
    put("  Search:")
    put(f"    differential evolution: {de.nit} generations, {de.nfev} model evaluations")
    put(f"    least-squares polishes: {len(chosen)} candidates, best "
        f"{len(best_idx)} on all data")
    for note in revive_notes:
        put(f"    {note}")
    put(f"    same {n_jets + 1} Gaussians in every map, {n_final_polishes} final polishes; "
        f"significance: {len(dchi2)} refits without one jet in one map")
    put(f"    final solutions within 1% of the best chi2: {n_same}/{len(solutions)}")
    put(f"    model parameters: {model.n_params};  total time: {time.time() - t_start:.0f} s")

    # Everything else, kept at the end of fit_summary.txt for reference.
    original = []
    orig = original.append
    orig("")
    orig("")
    orig("Further output (for reference)")
    orig("")
    orig("Fit in the native format (core of each file at (0,0), uas)")
    orig("fraction = flux fraction from closures; flux~ = fraction x amplitude scale (approximate, Jy)")
    best_dump = {"files": [], "shared_jets": {}, "parameters": {}}
    for d in range(D):
        orig(f"\n{labels[d]}   amplitude scale (total flux~) = {scales[d]:.4f} Jy")
        orig(f"{'comp':6s} {'fraction':>9s} {'flux~[Jy]':>10s} {'x [uas]':>11s} {'y [uas]':>11s} {'FWHM [uas]':>11s}")
        comps = model.components(q, d)
        file_dump = {"file": datasets[d]["path"], "freq_hz": datasets[d]["freq"],
                     "flux_scale_jy": scales[d], "beam_uas": datasets[d]["beam"],
                     "fov_uas": datasets[d]["fov"], "dynamic_range": drs[d][2],
                     "dirty_peak_jy_beam": drs[d][0], "residual_rms_jy_beam": drs[d][1],
                     "components": []}
        for f, x, y, fw, k in sorted(comps, key=lambda c: -1 if c[4] is None else jet_names[c[4]][1]):
            name = "core" if k is None else jet_names[k][0]
            orig(f"{name:6s} {f:9.5f} {f * scales[d]:10.5f} {x:11.2f} {y:11.2f} {fw:11.2f}")
            file_dump["components"].append({"name": name, "fraction": f, "flux_jy": f * scales[d],
                                            "x_uas": x, "y_uas": y, "fwhm_uas": fw,
                                            "detected": k is None or bool(detected[(d, k)])})
        orig("Distances from the core:")
        for k in sorted(model.jet_map[d], key=lambda k: jet_names[k][1]):
            (x, y, dist, pa), e = jet_geometry(d, k)
            if detected[(d, k)]:
                weak = ""
            else:
                weak = "   (not significant in this band: dchi2 = %.1f)" % dchi2[(d, k)]
            orig(f"  dist core to jet {jet_names[k][1] + 1} ({jet_names[k][0]}): "
                 f"{dist:10.2f} +- {e['r']:6.2f} uas   (PA {pa:7.2f} +- {e['pa']:5.2f} deg){weak}")
            file_dump.setdefault("distances", {})[jet_names[k][0]] = {
                "dist_uas": dist, "err_uas": e["r"], "pa_deg": pa, "pa_err_deg": e["pa"]}
        best_dump["files"].append(file_dump)

    orig("\nShared jets in the reference frame (reference core at origin)")
    orig(f"{'jet':4s} {'x [uas]':>11s} {'+-':>7s} {'y [uas]':>11s} {'+-':>7s} {'FWHM':>9s} {'+-':>7s}  files")
    for k in order:
        ix, iy, iw = model.jet_idx[k]
        files_k = ", ".join(short[d] for d in range(D) if k in model.jet_map[d])
        orig(f"{jet_names[k][0]:4s} {q[ix]:11.2f} {err[ix]:7.2f} {q[iy]:11.2f} {err[iy]:7.2f} "
             f"{q[iw]:9.2f} {err[iw]:7.2f}  {files_k}")
        best_dump["shared_jets"][jet_names[k][0]] = {"x_uas": q[ix], "y_uas": q[iy], "fwhm_uas": q[iw],
                                                     "x_err": err[ix], "y_err": err[iy], "fwhm_err": err[iw]}

    orig("\nPairwise core shifts pos(A) - pos(B) [uas]")
    for a_ in range(D):
        for b_ in range(a_ + 1, D):
            if (model.shift_idx[a_] is None and a_ != ref) or (model.shift_idx[b_] is None and b_ != ref):
                continue
            dxy = model.shift(q, a_) - model.shift(q, b_)
            orig(f"  {short[a_]:>8s} - {short[b_]:<8s} dx = {dxy[0]:9.2f}  dy = {dxy[1]:9.2f}  "
                 f"|d| = {np.hypot(*dxy):9.2f}")

    orig("\nFinal candidates (ranked by the fit), validation chi2_red of each")
    orig(f"{'rank':>4s} {'chi2_red':>9s} {'status':>6s} {'nfev':>6s}")
    for i, sol in enumerate(solutions, 1):
        orig(f"{i:4d} {sol['validation_chi2']:9.3f} {sol['status']:6d} {sol['nfev']:6d}")
    orig("Stability vs best (RMS): positions [uas], FWHM [uas], fractions")
    for i, sol in enumerate(solutions[:10], 1):
        pr, fr_, ff = solution_distance(model, q, sol["q"])
        orig(f"  {i:3d}  pos {pr:10.2f}   fwhm {fr_:10.2f}   frac {ff:9.5f}")

    orig("\nAll parameters (best solution)")
    for name, value, e in zip(model.names, q, err):
        orig(f"  {name:60s} {value:12.4f} +- {e:9.4f}")
        best_dump["parameters"][name] = {"value": float(value), "err": float(e)}

    # Figures.
    os.makedirs(out_dir, exist_ok=True)  # again, in case it was moved/deleted while the fit ran
    others = [sol["q"] for sol in solutions[1:10]]
    plot_components(model, q, others, datasets, labels, jet_names, detected,
                    os.path.join(out_dir, "components.png"))
    plot_overlay(model, q, cov, datasets, labels, jet_names, detected, ref,
                 os.path.join(out_dir, "overlay_aligned.png"), zoom=args.zoom)
    plot_core_shifts(model, q, cov, datasets, labels, os.path.join(out_dir, "core_shifts.png"))
    plot_validation(model, q, validation, labels, os.path.join(out_dir, "validation_closures.png"))
    plot_uv(model, q, datasets, labels, scales, os.path.join(out_dir, "amplitude_vs_uvdist.png"),
            os.path.join(out_dir, "closure_phase_vs_uvdist.png"))

    end_step(t0)
    log(f"Total time: {time.time() - t_start:.0f} s", False)
    run_log = ["", ""] + report
    with open(os.path.join(out_dir, "fit_summary.txt"), "w") as handle:
        handle.write("\n".join(summary + original + run_log) + "\n")
    core_shift_dump = {}
    for d in range(D):
        sh = model.shift(q, d)
        e = (0.0, 0.0) if model.shift_idx[d] is None else (err[model.shift_idx[d][0]], err[model.shift_idx[d][1]])
        core_shift_dump[short[d]] = {"dx_uas": sh[0], "dy_uas": sh[1], "dx_err": e[0], "dy_err": e[1]}
    # Sum of squared residuals on all training closures (before the outlier filter), so that the
    # models with different numbers of components are compared on exactly the same data (BIC).
    train_blocks = make_blocks(train)
    s_train = float(np.sum(model.residuals(q, train_blocks)**2))
    n_train = int(sum(bl.n for bl in train_blocks))
    chi2_train = model.chi2_red(q, train_blocks)
    best_dump.update(reference_file=datasets[ref]["path"], reference_short=short[ref], source=source,
                     ncomp=n_jets + 1, n_params=model.n_params, core_shifts=core_shift_dump,
                     chi2_red_validation=best["validation_chi2"], chi2_sum_train=s_train,
                     n_train=n_train, chi2_red_train=chi2_train, preset=args.preset,
                     q=q.tolist(), names=model.names)
    with open(os.path.join(out_dir, "best_params.json"), "w") as handle:
        json.dump(best_dump, handle, indent=2, default=float)
    return {
        "ncomp": n_jets + 1, "n_params": model.n_params, "out_dir": out_dir,
        "s_train": s_train, "n_train": n_train, "chi2_train": chi2_train,
        "chi2_valid": best["validation_chi2"],
        "short": short, "freqs": [ds["freq"] for ds in datasets],
        "dchi2": {(short[d], jet_names[k][0]): v for (d, k), v in dchi2.items()},
        "jets": [jet_names[k][0] for k in order], "summary": summary,
    }


def write_auto_summary(root, runs, best_run, args, elapsed):
    """Write auto_summary.txt (comparison of all models, significance of each jet of the chosen
    model, and the fit summary of the chosen model) and auto_chi2.png, and print the comparison."""
    lines = []
    put = lines.append
    put(f"Automatic number of components, {time.strftime('%Y-%m-%d %H:%M')}")
    put("Files: " + ", ".join(f"{s} ({f / 1e9:.2f} GHz)" for s, f in zip(runs[0]["short"], runs[0]["freqs"])))
    put("Every map has the same Gaussians. A model with one more jet is kept if the chi2 of the")
    put("fitted closures drops by more than ln(N closures) per added parameter (BIC), with the drop")
    put("in units of the reduced chi2 of the larger model.")
    put("chi2_red = reduced chi2 of the validation closures (never fitted), all files together.")
    put("")
    put(f"  {'ncomp':>5s} {'params':>6s} {'chi2_red':>9s} {'drop':>10s} {'needed':>8s}  result")
    for r in runs:
        gain = "" if r["gain"] is None else f"{r['gain']:10.1f}"
        needed = "" if r["needed"] is None else f"{r['needed']:8.1f}"
        result = "first model" if r["gain"] is None else ("kept" if r["kept"] else "not kept")
        put(f"  {r['ncomp']:5d} {r['n_params']:6d} {r['chi2_valid']:9.3f} {gain:>10s} {needed:>8s}  {result}")
    put("  drop = chi2 drop with respect to the last kept model; needed = BIC penalty.")
    put("")
    stopped = "the largest number allowed (--max-comp)" if runs[-1]["ncomp"] >= args.max_comp \
        else f"{args.patience} models in a row that were not kept (--patience)"
    put(f"Chosen: {best_run['ncomp']} components in every map (search stopped at {stopped}).")
    put(f"Full results of the chosen model: {os.path.relpath(best_run['out_dir'], root)}/")
    put("")
    put("Significance of each jet of the chosen model in each map (dchi2 when the jet is removed")
    put(f"from that map and the fit is redone; below {args.dchi2_min:.0f} = not significant, n.s.):")
    colw = max(14, max(len(s) for s in best_run["short"]) + 2)
    put("  " + f"{'jet':5s}" + "".join(f"{s:>{colw}s}" for s in best_run["short"]))
    for jet in best_run["jets"]:
        cells = []
        for s in best_run["short"]:
            v = best_run["dchi2"].get((s, jet))
            if v is None:
                cells.append("-")
            else:
                cells.append((f"{v:.1f}" if abs(v) < 1e5 else f"{v:.1e}")
                             + ("" if v >= args.dchi2_min else " n.s."))
        put("  " + f"{jet:5s}" + "".join(f"{c:>{colw}s}" for c in cells))
    put("")
    put(f"Total time: {elapsed / 60:.1f} min for {len(runs)} models.")
    put("")
    put("")
    put("Fit summary of the chosen model")
    put("")
    lines += best_run["summary"]
    with open(os.path.join(root, "auto_summary.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:lines.index("Fit summary of the chosen model") - 2]), flush=True)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    n = [r["ncomp"] for r in runs]
    ax.plot(n, [r["chi2_valid"] for r in runs], "o-")
    ax.axvline(best_run["ncomp"], color="0.5", ls=":", label=f"chosen: {best_run['ncomp']}")
    ax.set_xlabel("number of components per map")
    ax.set_ylabel(r"validation $\chi^2_\mathrm{red}$")
    ax.set_xticks(n)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(root, "auto_chi2.png"), dpi=150)
    plt.close(fig)
    print(f"\nResults written to: {os.path.abspath(root)}", flush=True)


def parse_args():
    """Command-line options (see the module docstring for the usage)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="file.uvfits[:fov_uas]")
    p.add_argument("--preset", choices=PRESETS, default="serious")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--ref", type=int, default=None,
                   help="index (0-based, input order) of the reference file; default: highest frequency")
    p.add_argument("--max-core-shift", type=float, default=1000.0, help="uas")
    p.add_argument("--max-jet-offset", type=float, default=20000.0,
                   help="default search FOV (radius around the core, uas) for files without one")
    p.add_argument("--max-fwhm", type=float, default=8000.0, help="jet FWHM upper bound, uas")
    p.add_argument("--max-core-fwhm", type=float, default=1000.0, help="uas")
    p.add_argument("--min-fwhm", type=float, default=0.0, help="uas")
    p.add_argument("--snrcut", type=float, default=3.0)
    p.add_argument("--avg", type=float, default=0.0, help="coherent time averaging [s] before closures (0 = off)")
    p.add_argument("--seed", type=int, default=8101)
    p.add_argument("--zoom", type=float, default=750.0,
                   help="half width [uas] of the core zoom in overlay_aligned.png")
    p.add_argument("--min-comp", type=int, default=2, help="first number of components tried (>= 2)")
    p.add_argument("--max-comp", type=int, default=8, help="largest number of components tried")
    p.add_argument("--patience", type=int, default=2,
                   help="stop after this many models in a row that are not kept")
    p.add_argument("--dchi2-min", type=float, default=9.0,
                   help="delta chi2 (in units of the reduced chi2) needed for a jet to be "
                        "significant in a map (9 ~ 3 sigma for one parameter)")
    return p.parse_args()


def main():
    """Load the files once, fit N = min_comp, min_comp+1, ... components per map, keep or reject
    each model with the BIC criterion, and write the comparison (write_auto_summary)."""
    args = parse_args()
    if args.min_comp < 2:
        sys.exit("--min-comp must be >= 2 (core + at least one jet).")
    specs = collect_specs(args)
    # max_logit = 8 bounds the flux logits (a component can be down to ~e^-8 of another, i.e.
    # effectively zero); dead_fraction = 0.5% of the total flux is the level below which a jet
    # counts as "no flux" (revive_dead_jets)
    cfg = dict(PRESETS[args.preset])
    cfg.update(max_core_shift=args.max_core_shift, max_jet_offset=args.max_jet_offset,
               max_fwhm=args.max_fwhm, max_core_fwhm=args.max_core_fwhm, min_fwhm=args.min_fwhm,
               max_logit=8.0, snrcut=args.snrcut, avg=args.avg, dead_fraction=5e-3,
               dchi2_min=args.dchi2_min)
    # jets are shared, so the joint search box must hold the widest FOV requested
    cfg["max_jet_offset"] = max(spec["fov"] for spec in specs)
    t_start = time.time()
    root = args.out or time.strftime("auto_fit_%Y%m%d_%H%M%S")
    os.makedirs(root, exist_ok=True)

    # the closures are computed once and reused for every number of components
    datasets = [load_dataset(spec, cfg) for spec in specs]
    labels = [f"{ds['name']} ({ds['freq'] / 1e9:.2f} GHz)" for ds in datasets]
    ref = args.ref if args.ref is not None else max(range(len(datasets)), key=lambda d: datasets[d]["freq"])
    print(f"Components per map: from {args.min_comp} up to {args.max_comp}, the same in every map; "
          f"reference map: {labels[ref]}")
    print(f"Settings: preset {args.preset}, SNR cut {cfg['snrcut']:g}, "
          + (f"time averaging {cfg['avg']:g} s" if cfg["avg"] else "no time averaging"))
    print(f"\n{'file':44s} {'CP':>6s} {'logCA':>6s} {'beam[uas]':>9s} {'FOV[uas]':>9s}")
    for ds, lab in zip(datasets, labels):
        c = ds["closures"]
        print(f"{lab:44s} {len(c['cphase']):6d} {len(c['logcamp']):6d} {ds['beam']:9.0f} {ds['fov']:9.0f}")
        if len(c["cphase"]) + len(c["logcamp"]) == 0:
            sys.exit(f"No closures for {lab} (need >= 3 stations with SNR > snrcut).")

    runs, best_run, fails = [], None, 0
    for n in range(args.min_comp, args.max_comp + 1):
        print(f"\n{n} components per map", flush=True)
        res = fit_once(args, cfg, datasets, n, os.path.join(root, f"ncomp_{n}"))
        if best_run is None:
            res.update(gain=None, needed=None, kept=True)
            print(f"  validation chi2_red = {res['chi2_valid']:.3f}; first model", flush=True)
        else:
            # BIC: the chi2 drop with respect to the last kept model, in units of the reduced chi2
            # of the larger model, must exceed ln(N closures) per added parameter
            res["gain"] = (best_run["s_train"] - res["s_train"]) / res["chi2_train"]
            res["needed"] = (res["n_params"] - best_run["n_params"]) * np.log(res["n_train"])
            res["kept"] = res["gain"] > res["needed"]
            print(f"  validation chi2_red = {res['chi2_valid']:.3f}; chi2 drop {res['gain']:.1f}, "
                  f"needed > {res['needed']:.1f}: " + ("kept" if res["kept"] else "not kept"), flush=True)
        runs.append(res)
        if res["kept"]:
            best_run, fails = res, 0
        else:
            fails += 1
            if fails >= args.patience:
                break

    write_auto_summary(root, runs, best_run, args, time.time() - t_start)


if __name__ == "__main__":
    main()
