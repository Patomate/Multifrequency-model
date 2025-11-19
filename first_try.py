import numpy as np
from scipy.optimize import least_squares
import ehtim as eh
import ehtim.const_def as ehc

# =========================================================
# 0. Helpers: µas <-> rad
# =========================================================

RADPERUAS = ehc.RADPERUAS
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def uas_to_rad(x_uas):
    return x_uas * RADPERUAS


def rad_to_uas(x_rad):
    return x_rad / RADPERUAS


# =========================================================
# 1. EHT array
# =========================================================

eht_array = eh.array.load_txt('EAVN_22GHz_array.txt')


# =========================================================
# 2. Frequencies and TRUE model parameters
# =========================================================

freqs_ghz = np.array([120.0, 230.0, 345.0])
nu0_ghz = 230.0
nu0 = nu0_ghz * 1e9

# True fluxes at reference frequency [Jy]
F_core_0_true = 0.8
F_k1_0_true = 0.4
F_k2_0_true = 0.2

# True spectral indices
alpha_core_true = 0.0
alpha_k1_true = -0.7
alpha_k2_true = -0.7

# Core-shift constant (x_core * nu ≈ const) in µas GHz
core_shift_const_uas_ghz = 4600.0

# True jet component physical positions (frequency-independent) in µas
X_KNOT1_TRUE_UAS, Y_KNOT1_TRUE_UAS = 60.0, 0.0
X_KNOT2_TRUE_UAS, Y_KNOT2_TRUE_UAS = 100.0, 0.0

x_knot1_true = uas_to_rad(X_KNOT1_TRUE_UAS)
y_knot1_true = uas_to_rad(Y_KNOT1_TRUE_UAS)
x_knot2_true = uas_to_rad(X_KNOT2_TRUE_UAS)
y_knot2_true = uas_to_rad(Y_KNOT2_TRUE_UAS)


def true_core_position(freq_ghz):
    """
    True physical core position at freq_ghz.
    Core shift law: x_core ∝ ν^-1, y_core = 0
    Returns (x_core, y_core) in radians.
    """
    x_uas = core_shift_const_uas_ghz / freq_ghz
    y_uas = 0.0
    return uas_to_rad(x_uas), uas_to_rad(y_uas)


def true_fluxes_at_freq(freq_ghz):
    """Return (core, knot1, knot2) fluxes at freq_ghz according to true spectral indices."""
    s = freq_ghz / nu0_ghz
    F_core = F_core_0_true * s**alpha_core_true
    F_k1 = F_k1_0_true * s**alpha_k1_true
    F_k2 = F_k2_0_true * s**alpha_k2_true
    return F_core, F_k1, F_k2


# =========================================================
# 3. Image geometry and Gaussian sizes
# =========================================================

npix = 128
fov_uas = 200.0       # total field of view in µas
fov = uas_to_rad(fov_uas)
psize = fov / npix

ra = 0.0   # hours
dec = 0.0  # degrees
mjd = 57757

# FWHM of Gaussians in µas (TRUE model)
fwhm_core_uas_true = 30.0
fwhm_k1_uas_true = 40.0
fwhm_k2_uas_true = 50.0

fwhm_core_true = uas_to_rad(fwhm_core_uas_true)
fwhm_k1_true = uas_to_rad(fwhm_k1_uas_true)
fwhm_k2_true = uas_to_rad(fwhm_k2_uas_true)

# (We keep sigma only for potential diagnostics; images use FWHM directly)
sigma_core = fwhm_core_true * FWHM_TO_SIGMA
sigma_k1 = fwhm_k1_true * FWHM_TO_SIGMA
sigma_k2 = fwhm_k2_true * FWHM_TO_SIGMA


# =========================================================
# 4. Build ehtim images from physical parameters
# =========================================================

def build_image_from_physical_params(freq_ghz, A_core_0, alpha_core,
                                     A_k1_0, alpha_k1,
                                     A_k2_0, alpha_k2,
                                     x_k1_phys, y_k1_phys,
                                     x_k2_phys, y_k2_phys,
                                     x_core_phys_dict, y_core_phys_dict,
                                     Dx_dict, Dy_dict):
    """
    Build an ehtim Image for a given frequency freq_ghz, using the
    reparameterization:

        - A_core(ν) = A_core,0 (ν/ν0)^α_core, etc.
        - physical positions r_j, r_c(ν)
        - map positions = r - Δν

    All inputs (positions, alignments) are in radians.
    """
    rf = freq_ghz * 1e9
    s = freq_ghz / nu0_ghz

    # Fluxes at this frequency
    A_core_nu = A_core_0 * s**alpha_core
    A_k1_nu = A_k1_0 * s**alpha_k1
    A_k2_nu = A_k2_0 * s**alpha_k2

    # Physical core position and alignment for this frequency
    x_core_phys = x_core_phys_dict[freq_ghz]
    y_core_phys = y_core_phys_dict[freq_ghz]
    Dx = Dx_dict[freq_ghz]
    Dy = Dy_dict[freq_ghz]

    # Map-frame positions: R = r_phys - Δν
    x_core_map = x_core_phys - Dx
    y_core_map = y_core_phys - Dy

    x_k1_map = x_k1_phys - Dx
    y_k1_map = y_k1_phys - Dy

    x_k2_map = x_k2_phys - Dx
    y_k2_map = y_k2_phys - Dy

    im = eh.image.Image(
        image=np.zeros((npix, npix)),
        psize=psize,
        ra=ra,
        dec=dec,
        rf=rf,
        source="toy_core_3comp",
        polrep="stokes",
        pol_prim="I"
    )

    # Add core and jet Gaussians (circular)
    im = im.add_gauss(A_core_nu, [fwhm_core_true, fwhm_core_true, 0.0, x_core_map, y_core_map])
    im = im.add_gauss(A_k1_nu, [fwhm_k1_true, fwhm_k1_true, 0.0, x_k1_map, y_k1_map])
    im = im.add_gauss(A_k2_nu, [fwhm_k2_true, fwhm_k2_true, 0.0, x_k2_map, y_k2_map])

    return im


def build_true_image(freq_ghz):
    """
    Build the TRUE sky image (using true parameters and no alignment).
    """
    F_core, F_k1, F_k2 = true_fluxes_at_freq(freq_ghz)
    x_core, y_core = true_core_position(freq_ghz)

    rf = freq_ghz * 1e9
    im = eh.image.Image(
        image=np.zeros((npix, npix)),
        psize=psize,
        ra=ra,
        dec=dec,
        rf=rf,
        source="toy_core_3comp",
        polrep="stokes",
        pol_prim="I"
    )

    im = im.add_gauss(F_core, [fwhm_core_true, fwhm_core_true, 0.0, x_core,       y_core])
    im = im.add_gauss(F_k1,   [fwhm_k1_true,  fwhm_k1_true,  0.0, x_knot1_true, y_knot1_true])
    im = im.add_gauss(F_k2,   [fwhm_k2_true,  fwhm_k2_true,  0.0, x_knot2_true, y_knot2_true])

    return im


# =========================================================
# 5. Simulate CLEAN EHT observations with ehtim (data)
# =========================================================

tint = 10.0    # s
tadv = 600.0   # s
tstart = 0.0   # h
tstop = 8.0    # h
bw = 4.0e9     # Hz

obs_template_by_freq = {}
obs_true_by_freq = {}
data_by_freq = {}

for fghz in freqs_ghz:
    rf = fghz * 1e9

    # 1) Empty Obsdata template defining baselines/times/sigmas
    obs_template = eht_array.obsdata(
        ra, dec, rf, bw,
        tint, tadv, tstart, tstop,
        mjd=mjd,
        polrep="stokes",
        tau=0.0,
        elevmin=10.0,
        elevmax=90.0,
        no_elevcut_space=False,
        timetype="UTC",
        fix_theta_GMST=False
    )
    obs_template_by_freq[fghz] = obs_template

    # 2) TRUE image and CLEAN visibilities, no noise
    im_true = build_true_image(fghz)
    obs_true = im_true.observe_same_nonoise(
        obs_template,
        sgrscat=False,
        ttype="direct",   # direct DTFT
        cache=False,
        fft_pad_factor=2,
        zero_empty_pol=True,
        verbose=True
    )

    obs_true_by_freq[fghz] = obs_true
    print(f"Simulated CLEAN obs at {fghz:.1f} GHz")

    # 3) Extract (u, v, vis, sigma) for fitting
    dat = obs_true.unpack(["u", "v", "vis", "sigma"], mode="all", conj=False)
    data_by_freq[fghz] = {
        "u": dat["u"],
        "v": dat["v"],
        "vis": dat["vis"],
        "sigma": dat["sigma"].real
    }


# =========================================================
# 6. Parameter vector layout and unpacking
# =========================================================
# p = [
#   0:  A_core_0
#   1:  alpha_core
#   2:  A_k1_0
#   3:  alpha_k1
#   4:  A_k2_0
#   5:  alpha_k2
#   6:  x_k1_phys
#   7:  y_k1_phys
#   8:  x_k2_phys
#   9:  y_k2_phys
#   10..: x_core_phys[f], y_core_phys[f] for each freq (order of freqs_ghz)
#   then: Dx[f], Dy[f] for f != nu0_ghz
# ]
#
# All positions and Dx,Dy are in radians.


def unpack_params(p, freqs_ghz, nu0_ghz):
    freqs_ghz = np.array(freqs_ghz, dtype=float)
    nu0_ghz = float(nu0_ghz)

    i = 0
    A_core_0 = p[i]; i += 1
    alpha_core = p[i]; i += 1

    A_k1_0 = p[i]; i += 1
    alpha_k1 = p[i]; i += 1

    A_k2_0 = p[i]; i += 1
    alpha_k2 = p[i]; i += 1

    x_k1_phys = p[i]; i += 1
    y_k1_phys = p[i]; i += 1
    x_k2_phys = p[i]; i += 1
    y_k2_phys = p[i]; i += 1

    x_core_phys = {}
    y_core_phys = {}
    for f in freqs_ghz:
        x_core_phys[f] = p[i]; i += 1
        y_core_phys[f] = p[i]; i += 1

    Dx = {}
    Dy = {}
    for f in freqs_ghz:
        if np.isclose(f, nu0_ghz):
            Dx[f] = 0.0
            Dy[f] = 0.0
        else:
            Dx[f] = p[i]; i += 1
            Dy[f] = p[i]; i += 1

    return dict(
        A_core_0=A_core_0, alpha_core=alpha_core,
        A_k1_0=A_k1_0, alpha_k1=alpha_k1,
        A_k2_0=A_k2_0, alpha_k2=alpha_k2,
        x_k1_phys=x_k1_phys, y_k1_phys=y_k1_phys,
        x_k2_phys=x_k2_phys, y_k2_phys=y_k2_phys,
        x_core_phys=x_core_phys, y_core_phys=y_core_phys,
        Dx=Dx, Dy=Dy
    )


# =========================================================
# 7. Global residual function using ehtim as the model
# =========================================================

def residuals_global(p, freqs_ghz, nu0_ghz,
                     data_by_freq, obs_template_by_freq):
    """
    For each frequency:
      - build an ehtim Image from the parameter vector p,
      - compute visibilities with observe_same_nonoise on the same uv-coverage,
      - compare model vs data visibilities (Re/Im) weighted by sigma.
    """
    params = unpack_params(p, freqs_ghz, nu0_ghz)

    A_core_0 = params["A_core_0"]
    alpha_core = params["alpha_core"]
    A_k1_0 = params["A_k1_0"]
    alpha_k1 = params["alpha_k1"]
    A_k2_0 = params["A_k2_0"]
    alpha_k2 = params["alpha_k2"]

    x_k1_phys = params["x_k1_phys"]
    y_k1_phys = params["y_k1_phys"]
    x_k2_phys = params["x_k2_phys"]
    y_k2_phys = params["y_k2_phys"]
    x_core_phys = params["x_core_phys"]
    y_core_phys = params["y_core_phys"]
    Dx = params["Dx"]
    Dy = params["Dy"]

    freqs_ghz = np.array(freqs_ghz, dtype=float)
    nu0_ghz = float(nu0_ghz)

    res_all = []

    for f in freqs_ghz:
        # Build model image at this frequency
        im_model = build_image_from_physical_params(
            f,
            A_core_0, alpha_core,
            A_k1_0, alpha_k1,
            A_k2_0, alpha_k2,
            x_k1_phys, y_k1_phys,
            x_k2_phys, y_k2_phys,
            x_core_phys, y_core_phys,
            Dx, Dy
        )

        # Use EXACT same uv-coverage as the data (obs_template_by_freq[f])
        obs_template = obs_template_by_freq[f]
        obs_model = im_model.observe_same_nonoise(
            obs_template,
            sgrscat=False,
            ttype="direct",
            cache=False,
            fft_pad_factor=2,
            zero_empty_pol=True,
            verbose=False
        )

        dat_model = obs_model.unpack(["vis"], mode="all", conj=False)
        vis_model = dat_model["vis"]

        # Data for this frequency
        d = data_by_freq[f]
        vis_data = d["vis"]
        sigma = d["sigma"]

        # Residuals: use Re and Im parts, normalized by sigma
        r_real = (vis_model.real - vis_data.real) / sigma
        r_imag = (vis_model.imag - vis_data.imag) / sigma

        res_all.append(r_real)
        res_all.append(r_imag)

    return np.concatenate(res_all)


# =========================================================
# 8. Initial guess and bounds (near TRUE)
# =========================================================

p0_list = []

# Fluxes and spectral indices (slightly perturbed)
p0_list.append(F_core_0_true * 0.9)         # A_core_0
p0_list.append(alpha_core_true + 0.1)       # alpha_core
p0_list.append(F_k1_0_true * 0.9)           # A_k1_0
p0_list.append(alpha_k1_true + 0.1)         # alpha_k1
p0_list.append(F_k2_0_true * 0.9)           # A_k2_0
p0_list.append(alpha_k2_true + 0.1)         # alpha_k2

# Knot positions (perturbed)
p0_list.append(x_knot1_true * 1.05)         # x_k1_phys
p0_list.append(y_knot1_true + uas_to_rad(5.0))
p0_list.append(x_knot2_true * 0.95)         # x_k2_phys
p0_list.append(y_knot2_true - uas_to_rad(5.0))

# Core positions per frequency (perturbed true positions)
for f in freqs_ghz:
    x_c_true, y_c_true = true_core_position(f)
    p0_list.append(x_c_true * 1.05)
    p0_list.append(y_c_true + uas_to_rad(5.0))

# Alignment parameters Δν: start at 0 for f != nu0
for f in freqs_ghz:
    if np.isclose(f, nu0_ghz):
        continue
    p0_list.append(0.0)   # Dx(f)
    p0_list.append(0.0)   # Dy(f)

p0 = np.array(p0_list, dtype=float)
print("\nNumber of fit parameters:", len(p0))

# Bounds
lower = []
upper = []

# A_core_0, A_k1_0, A_k2_0
for _ in range(3):
    lower.append(0.0)
    upper.append(10.0)

# alpha_core, alpha_k1, alpha_k2
for _ in range(3):
    lower.append(-3.0)
    upper.append(2.0)

# x_k1, y_k1, x_k2, y_k2
pos_min = uas_to_rad(-300.0)
pos_max = uas_to_rad(+300.0)
for _ in range(4):
    lower.append(pos_min)
    upper.append(pos_max)

# x_core_phys[f], y_core_phys[f]
for _ in freqs_ghz:
    lower.append(pos_min)
    upper.append(pos_max)
    lower.append(pos_min)
    upper.append(pos_max)

# Dx[f], Dy[f] for f != nu0
for f in freqs_ghz:
    if np.isclose(f, nu0_ghz):
        continue
    lower.append(pos_min)
    upper.append(pos_max)
    lower.append(pos_min)
    upper.append(pos_max)

lower = np.array(lower, dtype=float)
upper = np.array(upper, dtype=float)
assert lower.shape == p0.shape
assert upper.shape == p0.shape


# =========================================================
# 9. Run the fit (least squares)
# =========================================================

result = least_squares(
    residuals_global,
    x0=p0,
    bounds=(lower, upper),
    args=(freqs_ghz, nu0_ghz, data_by_freq, obs_template_by_freq),
    method='trf',
    x_scale='jac',
    verbose=2,
    max_nfev=200  # careful: this is expensive because we re-FT every eval
)

print("\nOptimization finished.")
print("Success:", result.success)
print("Message:", result.message)

p_best = result.x
res_best = result.fun
chi2_best = np.sum(res_best**2)
ndof_best = len(res_best) - len(p_best)
red_chi2_best = chi2_best / max(ndof_best, 1)
print("Rough reduced chi^2 (best-fit):", red_chi2_best)

best = unpack_params(p_best, freqs_ghz, nu0_ghz)

# Also compute χ² at TRUE parameters using the same forward operator
p_true_list = []

p_true_list.append(F_core_0_true)
p_true_list.append(alpha_core_true)
p_true_list.append(F_k1_0_true)
p_true_list.append(alpha_k1_true)
p_true_list.append(F_k2_0_true)
p_true_list.append(alpha_k2_true)
p_true_list.append(x_knot1_true)
p_true_list.append(y_knot1_true)
p_true_list.append(x_knot2_true)
p_true_list.append(y_knot2_true)

for f in freqs_ghz:
    x_c_true, y_c_true = true_core_position(f)
    p_true_list.append(x_c_true)
    p_true_list.append(y_c_true)

for f in freqs_ghz:
    if np.isclose(f, nu0_ghz):
        continue
    p_true_list.append(0.0)
    p_true_list.append(0.0)

p_true = np.array(p_true_list, dtype=float)
res_true = residuals_global(p_true, freqs_ghz, nu0_ghz, data_by_freq, obs_template_by_freq)
chi2_true = np.sum(res_true**2)
ndof_true = len(res_true) - len(p_true)
red_chi2_true = chi2_true / max(ndof_true, 1)

print("\n=== GLOBAL CHI^2 COMPARISON (same ehtim operator for data+model) ===")
print(f"  Reduced χ² at TRUE params     ≈ {red_chi2_true:.3e}")
print(f"  Reduced χ² at BEST-FIT params ≈ {red_chi2_best:.3e}")


# =========================================================
# 10. Pretty comparison: TRUE vs BEST-FIT parameters
# =========================================================

print("\n=== FLUXES AND SPECTRAL INDICES (true vs fit) ===")
print(f"Core  : F0 true = {F_core_0_true:.3f} Jy, fit = {best['A_core_0']:.3f} Jy, "
      f"Δ = {best['A_core_0'] - F_core_0_true:.3e}")
print(f"        alpha true = {alpha_core_true:.3f}, fit = {best['alpha_core']:.3f}, "
      f"Δ = {best['alpha_core'] - alpha_core_true:.3e}")

print(f"Knot1 : F0 true = {F_k1_0_true:.3f} Jy, fit = {best['A_k1_0']:.3f} Jy, "
      f"Δ = {best['A_k1_0'] - F_k1_0_true:.3e}")
print(f"        alpha true = {alpha_k1_true:.3f}, fit = {best['alpha_k1']:.3f}, "
      f"Δ = {best['alpha_k1'] - alpha_k1_true:.3e}")

print(f"Knot2 : F0 true = {F_k2_0_true:.3f} Jy, fit = {best['A_k2_0']:.3f} Jy, "
      f"Δ = {best['A_k2_0'] - F_k2_0_true:.3e}")
print(f"        alpha true = {alpha_k2_true:.3f}, fit = {best['alpha_k2']:.3f}, "
      f"Δ = {best['alpha_k2'] - alpha_k2_true:.3e}")

print("\n=== KNOT POSITIONS (true vs fit) in µas ===")
xk1_fit_uas = rad_to_uas(best["x_k1_phys"])
yk1_fit_uas = rad_to_uas(best["y_k1_phys"])
xk2_fit_uas = rad_to_uas(best["x_k2_phys"])
yk2_fit_uas = rad_to_uas(best["y_k2_phys"])

print(f"Knot1: true = ({X_KNOT1_TRUE_UAS:.1f}, {Y_KNOT1_TRUE_UAS:.1f}) µas, "
      f"fit = ({xk1_fit_uas:.3f}, {yk1_fit_uas:.3f}) µas, "
      f"Δ = ({xk1_fit_uas - X_KNOT1_TRUE_UAS:.3e}, {yk1_fit_uas - Y_KNOT1_TRUE_UAS:.3e})")
print(f"Knot2: true = ({X_KNOT2_TRUE_UAS:.1f}, {Y_KNOT2_TRUE_UAS:.1f}) µas, "
      f"fit = ({xk2_fit_uas:.3f}, {yk2_fit_uas:.3f}) µas, "
      f"Δ = ({xk2_fit_uas - X_KNOT2_TRUE_UAS:.3e}, {yk2_fit_uas - Y_KNOT2_TRUE_UAS:.3e})")

print("\n=== CORE POSITIONS AND ALIGNMENTS (true vs fit) in µas ===")
for f in freqs_ghz:
    x_true, y_true = true_core_position(f)
    x_true_uas = rad_to_uas(x_true)
    y_true_uas = rad_to_uas(y_true)

    x_fit_uas = rad_to_uas(best["x_core_phys"][f])
    y_fit_uas = rad_to_uas(best["y_core_phys"][f])

    Dx_fit_uas = rad_to_uas(best["Dx"][f])
    Dy_fit_uas = rad_to_uas(best["Dy"][f])

    print(f"\nν = {f:.1f} GHz:")
    print(f"  Core physical position true = ({x_true_uas:.3f}, {y_true_uas:.3f}) µas")
    print(f"                          fit  = ({x_fit_uas:.3f}, {y_fit_uas:.3f}) µas")
    print(f"                          Δpos = ({x_fit_uas - x_true_uas:.3e}, {y_fit_uas - y_true_uas:.3e}) µas")
    print(f"  Alignment Δν true = (0.0, 0.0) µas, fit = ({Dx_fit_uas:.3f}, {Dy_fit_uas:.3f}) µas")

print("\nDone.")
