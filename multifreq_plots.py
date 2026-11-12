"""
Figures of the multi-frequency closure fit, shared by every fit script
(multifreq_fit_blind_full.py, multifreq_fit_auto_components.py, ...).

All functions take the fitted JointModel and its parameter vector q; they only draw,
they never change the fit.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

VIS_FLOOR = 1e-12
SIGMA_FLOOR = 1e-4

JET_COLORS = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown",
              "tab:pink", "tab:olive", "tab:cyan", "tab:gray"]


def grid(n, width=4.6, height=4.4):
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat[:n]


def not_fitted_jets(model, q, d):
    """Shared jets that file d does not fit: [(x, y, fwhm, k)] in the file-d frame (core at 0,0)."""
    s = model.shift(q, d)
    out = []
    for k in range(model.n_jets):
        if k not in model.jet_map[d]:
            ix, iy, iw = model.jet_idx[k]
            out.append((q[ix] - s[0], q[iy] - s[1], q[iw], k))
    return out


def plot_components(model, q_fit, others, datasets, labels, jet_names, detected, path):
    boxes = []
    for d in range(model.D):
        for f, x, y, fw, k in model.components(q_fit, d):
            if k is None or detected[(d, k)]:  # non-significant jets do not set the frame
                boxes.append((x - fw / 2, x + fw / 2, y - fw / 2, y + fw / 2))
        for x, y, fw, k in not_fitted_jets(model, q_fit, d):
            boxes.append((x - fw / 2, x + fw / 2, y - fw / 2, y + fw / 2))
    boxes = np.array(boxes)
    x0, x1 = boxes[:, 0].min(), boxes[:, 1].max()
    y0, y1 = boxes[:, 2].min(), boxes[:, 3].max()
    half = 0.55 * max(x1 - x0, y1 - y0, 400.0)
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    fig, axes = grid(model.D)
    for d, ax in enumerate(axes):
        for qo in others:
            for f, x, y, fw, k in model.components(qo, d):
                color = "tab:red" if k is None else JET_COLORS[jet_names[k][1] % len(JET_COLORS)]
                ax.add_patch(Circle((x, y), fw / 2, fill=False, color=color, alpha=0.12, lw=0.8))
        for f, x, y, fw, k in model.components(q_fit, d):
            sig = k is None or detected[(d, k)]
            if k is None:
                name, color = "core", "tab:red"
            else:
                name, idx = jet_names[k]
                color = JET_COLORS[idx % len(JET_COLORS)]
            # every map shows every fitted Gaussian; non-significant ones are dashed and faint
            ax.add_patch(Circle((x, y), fw / 2, fill=False, color=color, lw=1.8 if sig else 1.0,
                                ls="-" if sig else "--", alpha=1.0 if sig else 0.5))
            ax.scatter(x, y, s=40, facecolors="none", edgecolors=color, alpha=1.0 if sig else 0.5,
                       label=name if sig else f"{name} (n.s.)")
            ax.annotate(f"{name}\nf={f:.3f}", (x, y), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=color, alpha=1.0 if sig else 0.5)
        # jets fitted only in other bands: drawn where they would be, but not in this band's model
        for x, y, fw, k in not_fitted_jets(model, q_fit, d):
            name, idx = jet_names[k]
            color = JET_COLORS[idx % len(JET_COLORS)]
            ax.add_patch(Circle((x, y), fw / 2, fill=False, color=color, lw=1.0, ls=":", alpha=0.6))
            ax.scatter(x, y, s=30, marker="x", color=color, alpha=0.6, label=f"{name} (not fitted)")
            ax.annotate(f"{name}\nnot fitted", (x, y), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=color, alpha=0.6)
        ax.set_xlim(xc + half, xc - half)
        ax.set_ylim(yc - half, yc + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(labels[d], fontsize=10)
        ax.set_xlabel("relative RA [uas]")
        ax.set_ylabel("relative Dec [uas]")
        ax.grid(alpha=0.25)
        handles, names = ax.get_legend_handles_labels()
        rank = lambda n: (0, "") if n == "core" else (1, n.split()[0].zfill(4))
        pairs = sorted(zip(names, handles), key=lambda t: rank(t[0]))
        ax.legend([h for _, h in pairs], [n for n, _ in pairs], fontsize=7, loc="best")
    fig.suptitle("Fitted components (core of each file at origin; circle = FWHM)\n"
                 "dashed = not significant in that band; dotted x = fitted only in other bands "
                 "(shown, not in this band's model)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_core_shifts(model, q, cov, datasets, labels, path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    cmap = plt.get_cmap("viridis")
    freqs = np.array([ds["freq"] for ds in datasets])
    norm = plt.Normalize(freqs.min() / 1e9, max(freqs.max() / 1e9, freqs.min() / 1e9 + 1e-3))
    for d in range(model.D):
        s = model.shift(q, d)
        ex = ey = 0.0
        if model.shift_idx[d] is not None:
            sx, sy = model.shift_idx[d]
            ex, ey = np.sqrt(max(cov[sx, sx], 0)), np.sqrt(max(cov[sy, sy], 0))
        color = cmap(norm(freqs[d] / 1e9))
        ax.errorbar(s[0], s[1], xerr=ex, yerr=ey, fmt="o", color=color, capsize=3, label=labels[d])
    ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("core RA offset vs reference core [uas]")
    ax.set_ylabel("core Dec offset vs reference core [uas]")
    ax.set_title("Core positions in the reference frame (jets aligned)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title=r"error bars: 1$\sigma$", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay(model, q, cov, datasets, labels, jet_names, detected, ref, path, zoom=750.0):
    """All maps superposed in the reference frame: the shared (optically thin) jets coincide,
    and each core sits at its core shift. Left: whole source; right: the cores in a fixed
    window of +-zoom uas around the reference core. No error bars (cov is not used)."""
    by_freq = sorted(range(model.D), key=lambda d: datasets[d]["freq"])
    cmap = plt.get_cmap("plasma")
    core_color = {d: cmap(0.08 + 0.8 * i / max(len(by_freq) - 1, 1)) for i, d in enumerate(by_freq)}
    live = [k for k in range(model.n_jets) if any(detected.get((d, k), False) for d in range(model.D))]
    shifts = {d: model.shift(q, d) for d in range(model.D)}
    core_fw = {d: q[model.corefwhm_idx[d]] for d in range(model.D)}

    boxes = [(shifts[d][0] - core_fw[d] / 2, shifts[d][0] + core_fw[d] / 2,
              shifts[d][1] - core_fw[d] / 2, shifts[d][1] + core_fw[d] / 2) for d in range(model.D)]
    for k in live:
        ix, iy, iw = model.jet_idx[k]
        boxes.append((q[ix] - q[iw] / 2, q[ix] + q[iw] / 2, q[iy] - q[iw] / 2, q[iy] + q[iw] / 2))
    boxes = np.array(boxes)
    x0, x1, y0, y1 = boxes[:, 0].min(), boxes[:, 1].max(), boxes[:, 2].min(), boxes[:, 3].max()
    full_frame = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.55 * max(x1 - x0, y1 - y0, 400.0))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8))
    for ax, (xc, yc, half), title in [
            (axes[0], full_frame, "All maps superposed, optically thin jets aligned"),
            (axes[1], (0.0, 0.0, zoom), f"Zoom on the cores (+-{zoom:.0f} uas): core shift")]:
        for k in live:
            ix, iy, iw = model.jet_idx[k]
            name, idx = jet_names[k]
            color = JET_COLORS[idx % len(JET_COLORS)]
            ax.add_patch(Circle((q[ix], q[iy]), q[iw] / 2, facecolor=color, edgecolor="none", alpha=0.25))
            ax.add_patch(Circle((q[ix], q[iy]), q[iw] / 2, fill=False, edgecolor=color, lw=1.6))
            ax.annotate(name, (q[ix], q[iy]), ha="center", va="center", fontsize=10,
                        color=color, fontweight="bold", annotation_clip=True)
        path_xy = np.array([shifts[d] for d in by_freq])
        ax.plot(path_xy[:, 0], path_xy[:, 1], "-", color="0.5", lw=0.8, zorder=1)
        for d in by_freq:
            c = core_color[d]
            ax.add_patch(Circle(shifts[d], core_fw[d] / 2, fill=False, edgecolor=c, lw=2.0, ls="--"))
            ax.plot(shifts[d][0], shifts[d][1], "o", ms=5, color=c,
                    label=f"core {labels[d]}" + (" (reference)" if d == ref else ""))
        ax.set_xlim(xc + half, xc - half)
        ax.set_ylim(yc - half, yc + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("relative RA [uas]")
        ax.set_ylabel("relative Dec [uas]")
        ax.grid(alpha=0.25, ls="--")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Jets (filled, shared by all frequencies) and cores (dashed, one per file); "
                 f"circle = FWHM; origin = core of {labels[ref]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_validation(model, q, validation, labels, path):
    fig, axes = plt.subplots(2, model.D, figsize=(3.8 * model.D, 7.5), squeeze=False)
    for d in range(model.D):
        cp, ca = validation[d]["cphase"], validation[d]["logcamp"]
        ax = axes[0, d]
        if len(cp):
            v = [model.visibility(q, d, cp[f"u{i}"], cp[f"v{i}"]) for i in (1, 2, 3)]
            ax.scatter(np.angle(v[0] * v[1] * v[2]), cp["cphase"], s=8, alpha=0.5)
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], "k-", lw=1)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_title(labels[d], fontsize=9)
        ax.set_xlabel("model closure phase [rad]")
        ax.set_ylabel("observed closure phase [rad]")
        ax.grid(alpha=0.25)
        ax = axes[1, d]
        if len(ca):
            la = [np.log(np.maximum(np.abs(model.visibility(q, d, ca[f"u{i}"], ca[f"v{i}"])), VIS_FLOOR))
                  for i in (1, 2, 3, 4)]
            pred = la[0] + la[1] - la[2] - la[3]
            ax.scatter(pred, ca["camp"], s=8, alpha=0.5)
            both = np.concatenate([pred, ca["camp"]])
            lims = [np.nanpercentile(both, 1), np.nanpercentile(both, 99)]
            ax.plot(lims, lims, "k-", lw=1)
        ax.set_xlabel("model log closure amplitude")
        ax.set_ylabel("observed log closure amplitude")
        ax.grid(alpha=0.25)
    fig.suptitle("Held-out closures: observed vs model")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_uv(model, q, datasets, labels, scales, path_amp, path_cp):
    fig, axes = grid(model.D, 5.0, 4.0)
    for d, ax in enumerate(axes):
        ds = datasets[d]
        uvd = np.hypot(ds["u"], ds["v"])
        amp_mod = scales[d] * np.abs(model.visibility(q, d, ds["u"], ds["v"]))
        order = np.argsort(uvd)
        ax.errorbar(uvd, np.abs(ds["vis"]), yerr=ds["sigma"], fmt=".", color="tab:blue",
                    alpha=0.25, ms=3, label="data")
        ax.scatter(uvd[order], amp_mod[order], s=4, color="tab:red",
                   label=f"model (x{scales[d]:.3f} Jy)")
        ax.set_title(labels[d], fontsize=9)
        ax.set_xlabel(r"UV distance [$\lambda$]")
        ax.set_ylabel("Amplitude [Jy]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Amplitude vs UV distance")
    fig.tight_layout()
    fig.savefig(path_amp, dpi=130)
    plt.close(fig)

    fig, axes = grid(model.D, 5.0, 4.0)
    for d, ax in enumerate(axes):
        cp = datasets[d]["closures"]["cphase"]
        if len(cp):
            bl = np.max([np.hypot(cp[f"u{i}"], cp[f"v{i}"]) for i in (1, 2, 3)], axis=0)
            v = [model.visibility(q, d, cp[f"u{i}"], cp[f"v{i}"]) for i in (1, 2, 3)]
            order = np.argsort(bl)
            ax.errorbar(bl, np.degrees(cp["cphase"]),
                        yerr=np.degrees(np.maximum(np.asarray(cp["sigmacp"], float), SIGMA_FLOOR)),
                        fmt=".", color="tab:blue", alpha=0.2, ms=3, label="data")
            ax.scatter(bl[order], np.degrees(np.angle(v[0] * v[1] * v[2]))[order], s=4,
                       color="tab:red", label="model")
        ax.set_title(labels[d], fontsize=9)
        ax.set_xlabel(r"longest baseline of triangle [$\lambda$]")
        ax.set_ylabel("Closure phase [deg]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Closure phase vs UV distance")
    fig.tight_layout()
    fig.savefig(path_cp, dpi=130)
    plt.close(fig)
