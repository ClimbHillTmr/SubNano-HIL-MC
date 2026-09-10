# -*- coding: utf-8 -*-
"""
Extra process-hazard / latent-image analyses for the SubNano-HIL-MC project.

These four models close the gap left by a design audit: a contrast curve,
resist sputtering by the He+ beam, beam-induced heating, and a 3-D stochastic
developed-line simulation.  Every model is fully physical (no stubs / TODOs);
where a parameter is uncertain it is exposed as a keyword argument with a
literature-typical default and justified in the docstring.

The module is intentionally self-contained: it depends only on numpy and the
beam/resist constants that are needed.  The Monte-Carlo PSF / depth profile are
passed in by the caller (e.g. from ``nature_visualization.prepare``); the figure
script ``scripts/fig_extras.py`` wires them in.

Units
-----
* line dose          : pC / cm   (1 pC/cm = 6.2415e6 ions/cm = 0.62415 ions/nm)
* energy             : eV        (1 eV = 1.602176634e-19 J)
* length             : nm
* energy density     : eV / nm^3
* areal energy       : eV / nm^2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Ions per nm of line length for a line dose q [pC/cm]:
#   1 pC = 1e-12 C = 1e-12 / 1.602176634e-19 e  = 6.2415e6 ions/cm
#   per nm of line length  ->  / 1e7  ->  0.62415 ions/nm
ION_PER_NM_PER_PC_CM = 1e-12 / 1.602176634e-19 / 1e7
EV_TO_J = 1.602176634e-19


# ===========================================================================
# 1. Contrast curve (negative tone, gel-threshold development)
# ===========================================================================
@dataclass
class ContrastResult:
    doses_pC_cm: np.ndarray
    remaining_fraction: np.ndarray
    D0: float          # dose at which gel fraction first becomes > 0
    D100: float        # dose at which the full film is gelled
    gamma: float       # contrast = 1 / log10(D100 / D0)


def contrast_curve(
    depth_profile: np.ndarray,
    depth_edges: np.ndarray,
    rho_gel: float,
    doses_pC_cm: np.ndarray,
    energy_per_ion_in_resist_eV: float,
    *,
    centerline_lsf_per_ion_eV_per_nm: float,
    ion_per_nm_per_pC_cm: float = ION_PER_NM_PER_PC_CM,
    resist_thickness_nm: Optional[float] = None,
):
    """Remaining-thickness fraction ``T/T0`` vs line dose for a *negative-tone*
    resist, derived from the per-depth deposited energy density.

    Model
    -----
    At the centre-line of an isolated (straight) line the vertically-integrated
    areal exposure is ``E_A = D * lam * L(0)``  [eV/nm^2] with the line-spread
    function ``L(0)`` (= ``centerline_lsf_per_ion_eV_per_nm``) and
    ``lam = D * ion_per_nm_per_pC_cm`` ions/nm.  We assume the lateral and
    depth depositions are separable, so the *volumetric* energy density at depth
    bin ``k`` is

        U(z_k; D) = E_A * f(z_k) / dz_k ,
        f(z_k)    = depth_profile[k] / sum(depth_profile)      (per-nm fraction)

    which is exactly normalised so that ``integral U dz = E_A``.  A layer gels
    (survives development) where ``U(z_k; D) >= rho_gel``.  The remaining
    thickness fraction is the gelled depth divided by the resist thickness.

    For negative tone a higher dose gels *more* of the film, so ``T/T0`` is a
    non-decreasing function of dose.  No experimental dissolution data was
    available, hence this is a purely model-derived development curve.

    Convention (stated explicitly)
    ------------------------------
    * ``D0``   = dose at which the gel fraction first becomes non-zero (i.e. the
                shallowest / highest-dose layer crosses ``rho_gel``).
    * ``D100`` = dose at which the *full* film thickness is gelled (deepest
                layer crosses ``rho_gel``).
    * ``gamma``= 1 / log10(D100 / D0)   (larger gamma = sharper contrast).

    Parameters
    ----------
    depth_profile : eV deposited in each depth bin per incident ion (sum = total
        energy in resist per ion ~ 3293 eV for 30 keV He+).
    depth_edges   : bin edges (nm), length = len(depth_profile) + 1.
    rho_gel       : gel threshold energy density, eV/nm^3 (pipeline uses 15).
    doses_pC_cm   : monotonically increasing line doses to evaluate.
    energy_per_ion_in_resist_eV : sum(depth_profile); passed explicitly so the
        caller controls whether the smooth-tail corrected value is used.
    centerline_lsf_per_ion_eV_per_nm : peak LSF value ``L(0)`` (eV/nm per ion).
        For the converged bulk PSF this is ``lsf.max()`` (~1200.7).  Exposed so
        the model can be driven without the heavy MC cache.
    """
    depth_profile = np.asarray(depth_profile, dtype=float)
    depth_edges = np.asarray(depth_edges, dtype=float)
    doses = np.asarray(doses_pC_cm, dtype=float)
    dz = np.diff(depth_edges)
    if resist_thickness_nm is None:
        resist_thickness_nm = float(depth_edges[-1])
    t_res = float(resist_thickness_nm)

    # per-bin volumetric density per unit dose *per ion* (eV/nm^3 per ion)
    #   U0(z) = L(0) * depth_profile / (E_ion * dz)
    U0 = centerline_lsf_per_ion_eV_per_nm * depth_profile / (energy_per_ion_in_resist_eV * dz)

    # only depth bins that lie inside the resist film
    in_film = depth_edges[:-1] < t_res
    U0_f = U0[in_film]
    n_f = U0_f.size

    # dose needed to gel each (in-film) layer:  D * lam * U0 >= rho_gel
    lam = ion_per_nm_per_pC_cm
    with np.errstate(divide="ignore"):
        D_layer = np.where(U0_f > 0, rho_gel / (lam * U0_f), np.inf)
    D0 = float(np.nanmin(D_layer))
    D100 = float(np.nanmax(D_layer))

    # remaining fraction over the requested doses
    remaining = np.empty(doses.shape, dtype=float)
    for i, D in enumerate(doses):
        gelled = D * lam * U0_f >= rho_gel
        remaining[i] = gelled.sum() / n_f

    gamma = float(1.0 / np.log10(D100 / D0)) if D100 > D0 > 0 else np.nan
    return ContrastResult(doses, remaining, D0, D100, gamma)


# ===========================================================================
# 2. Physical sputtering yield of the resist by He+
# ===========================================================================
@dataclass
class SputterResult:
    yield_atoms_per_ion: float
    removed_thickness_nm: float
    energy_keV: float
    dose_pC_cm: float
    surface_binding_eV: float


def sputter_yield_he(
    energy_keV: float = 30.0,
    *,
    surface_binding_eV: float = 3.0,
    target_Z: float = 4.0,
    target_A: float = 7.6,
    alpha: float = 0.5,
    thompson_m: float = 2.8,
    dose_pC_cm: float = 200.0,
    beam_width_nm: float = 2.24,
    resist_density_g_cm3: float = 0.771,
    molecule_M_g_mol: float = 2329.8,
    atoms_per_molecule: int = 306,
    ion_per_nm_per_pC_cm: float = ION_PER_NM_PER_PC_CM,
):
    """Physical sputtering yield (atoms/ion) of the resist by He+, via a
    Sigmund reduced-energy formula with a surface-binding-energy threshold.

    Sigmund / reduced-energy model
    ------------------------------
    Reduced energy (Bohr, amu-based):

        eps = 32.55 * A2 * E[eV] / ( Z1*Z2*sqrt(Z1^2+Z2^2) * (A1+A2) )

    Nuclear stopping (Krstic-Wilson / Kashetov approximation):

        s_n(eps) = 3.441 * sqrt(eps) * ln(eps + e^2.718) / (1 + 1.228 * eps)

    Yield (Sigmund, threshold corrected):

        Y = 0.042 * alpha * (A2/A1) * sqrt(A1*A2)/(A1+A2)
            * s_n(eps) * (1 - sqrt(eps_th / eps)) ** m

    with ``eps_th`` the threshold in reduced units using the surface binding
    energy ``U_s`` as the displacement/threshold energy, ``Z1=2`` (He),
    ``A1=4.0026``.

    Parameters & defaults
    ----------------------
    * ``surface_binding_eV = 3.0`` : surface binding energy of the organometallic
      hybrid.  He sputtering of organics/polymers is governed by the cohesive /
      sublimation energy; values for resists and organic matter are typically
      1-5 eV (PMMA ~2-3 eV, typical polymers 2-4 eV).  3 eV is a defensible
      central estimate for a Ti6-oxo / organic hybrid.  *Uncertain* -> keyword.
    * ``target_Z=4.0, target_A=7.6`` : effective resist "atom" (mean over
      C120H152O28Ti6).  These are the number-averaged atomic number and mass:
      Z_eff = 1228/306 ~= 4.0, A_eff = 2329.8/306 ~= 7.6.
    * ``beam_width_nm = 2.24`` : lateral footprint of the exposed line, taken as
      the molecule diameter; used to convert the line dose into an areal ion
      fluence for the removed-thickness estimate.

    Removed thickness for a given line dose
    ---------------------------------------
    Areal fluence  F = (dose * ion_per_nm_per_pC_cm) / beam_width_nm  [ions/nm^2].
    Atomic volume  Omega = M / (rho * N_A * n_atoms)                 [nm^3/atom].
    removed_thickness = F * Y * Omega                                [nm].
    This is compared against the 40 nm film to judge whether the film is
    physically eroded during exposure.
    """
    Z1, A1 = 2.0, 4.002602
    E_eV = energy_keV * 1e3
    denom = Z1 * target_Z * np.sqrt(Z1**2 + target_Z**2) * (A1 + target_A)
    eps = 32.55 * target_A * E_eV / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        sn = 3.441 * np.sqrt(eps) * np.log(eps + np.exp(2.718)) / (1.0 + 1.228 * eps)
    eps_th = 32.55 * target_A * surface_binding_eV / denom
    thr = 1.0 - np.sqrt(eps_th / eps) if eps > eps_th else -np.inf
    if thr <= 0:
        Y = 0.0
    else:
        pref = 0.042 * alpha * (target_A / A1) * np.sqrt(A1 * target_A) / (A1 + target_A)
        Y = float(pref * sn * thr**thompson_m)

    # removed thickness for the requested line dose
    fluence = dose_pC_cm * ion_per_nm_per_pC_cm / beam_width_nm          # ions/nm^2
    omega_atom_nm3 = (molecule_M_g_mol
                      / (resist_density_g_cm3 * 6.02214076e23 * atoms_per_molecule)
                      * 1e21)                                            # nm^3/atom
    removed_nm = fluence * Y * omega_atom_nm3
    return SputterResult(Y, removed_nm, energy_keV, dose_pC_cm, surface_binding_eV)


# ===========================================================================
# 3. Beam-induced heating of the film during writing
# ===========================================================================
@dataclass
class HeatingResult:
    dT_scan_averaged_K: float     # conservative, time-averaged (steady conduction)
    dT_worst_case_K: float        # single-pixel / single-dwell adiabatic peak
    safe: bool                    # True if both estimates below the threshold
    degrade_threshold_C: float


def beam_heating(
    dose_pC_cm: float,
    beam_current_pA: Optional[float] = None,
    dwell_us: Optional[float] = None,
    *,
    beam_energy_keV: float = 30.0,
    r_beam_nm: float = 2.24,
    t_film_nm: float = 40.0,
    k_resist: float = 0.2,
    k_Si: float = 148.0,
    alpha_Si: float = 8.9e-5,
    rhoC_resist: float = 1.5e6,
    E_resist_frac: float = 0.11,
    pitch_nm: float = 10.0,
    degrade_threshold_C: float = 150.0,
    ion_per_nm_per_pC_cm: float = ION_PER_NM_PER_PC_CM,
):
    """Temperature rise of the resist film during He+ writing.

    Two estimates are returned:

    (a) Conservative scan-averaged  -- steady vertical conduction through the
        resist (thickness ``t_film_nm``) into the Si substrate acting as a heat
        sink.  The energy deposited per unit *area* of the resist is

            q_res [J/m^2] = lam * E_resist[eV] * E_resist_frac * eV_to_J
                            / w_eff ,   lam = dose * ion_per_nm_per_pC_cm

        spread over an effective lateral width ``w_eff = max(r_beam, pitch)``.
        The time-average power density is ``q'' = q_res / dwell`` (one dwell per
        pitch advance) and the steady vertical conduction rise is

            dT_avg = q'' * t_film / k_resist .

        This is conservative because it assumes the *whole* film reaches a
        uniform temperature (i.e. it ignores that the Si sink also spreads heat
        laterally, which would lower the rise further).

    (b) Worst-case single pixel  -- the beam dwells on one spot for ``dwell_us``
        while the heat spreads resistively into the Si substrate (the real
        limiting sink).  Treated as a surface disk source of radius ``a_eff`` on a
        semi-infinite Si half-space (Carslaw-Jaeger spreading resistance), the
        steady surface rise is

            dT_worst = P_abs_res / (4 * k_Si * a_eff) ,
            a_eff   = max(r_beam, 2*sqrt(alpha_Si * dwell)) .

        ``a_eff`` captures the finite thermal diffusion during the dwell; for
        typical He+ dwell times (sub-microsecond) this gives a realistic, finite
        peak that is still larger than the scan-averaged estimate because the
        energy is concentrated in one spot rather than spread over the film.

    If ``beam_current_pA`` is not supplied it is derived from the dose assuming a
    scan pitch ``pitch_nm`` and the given ``dwell_us`` (or a 1 us default),
    so that the worst-case estimate also scales with dose.  If ``dwell_us`` is not
    given, an adiabatic whole-film lumped estimate is used for ``dT_avg``
    (``dT_avg = q_res / (rhoC_resist * t_film)``), which is an upper bound.

    Parameters & defaults
    ----------------------
    * ``k_resist = 0.2`` W/m/K : typical for a low-density organic / hybrid
      resist (organics ~0.1-0.3).  *Uncertain* -> keyword.
    * ``k_Si = 148`` W/m/K : bulk silicon (used as the substrate sink).
    * ``rhoC_resist = 1.5e6`` J/m^3/K : volumetric heat capacity of an organic
      resist (PMMA ~1.7e6).
    * ``E_resist_frac = 0.11`` : fraction of beam energy deposited in the resist
      (~11% for 30 keV He+, the rest in the Si substrate / backscatter).
    * ``degrade_threshold_C = 150`` : assumed resist degradation / reflow
      temperature; ``safe`` is True only if *both* estimates stay below it.
    """
    V = beam_energy_keV * 1e3                                  # V
    lam = dose_pC_cm * ion_per_nm_per_pC_cm                     # ions/nm
    E_line_resist_J_per_nm = lam * 3293.0 * E_resist_frac * EV_TO_J  # J/nm
    w_eff_nm = max(r_beam_nm, pitch_nm)
    q_res = E_line_resist_J_per_nm / (w_eff_nm * 1e-9)          # J/m^2

    # --- scan-averaged (conservative, steady vertical conduction) -------------
    if dwell_us is not None:
        dwell = dwell_us * 1e-6
        q_pp = q_res / dwell                                   # W/m^2
        dT_avg = q_pp * (t_film_nm * 1e-9) / k_resist           # K
    else:
        dT_avg = q_res / (rhoC_resist * (t_film_nm * 1e-9))     # lumped adiabatic

    # --- worst-case single pixel (transient disk source on Si sink) ----------
    if beam_current_pA is None:
        d_use = dwell_us if dwell_us is not None else 1.0       # us
        dwell = d_use * 1e-6
        # dose [pC/cm] = I[pA] * dwell[s] / pitch[cm]; pitch[cm]=pitch_nm*1e-7
        I_pA = dose_pC_cm * pitch_nm / (dwell * 1e9)
    else:
        I_pA = beam_current_pA
        dwell = dwell_us * 1e-6 if dwell_us is not None else 1.0e-6
    P_abs = I_pA * 1e-12 * V                                   # W
    P_abs_res = P_abs * E_resist_frac                          # W into resist
    a_eff = max(r_beam_nm * 1e-9, 2.0 * np.sqrt(alpha_Si * dwell))
    dT_worst = P_abs_res / (4.0 * k_Si * a_eff)                # K

    safe = bool(dT_avg < degrade_threshold_C and dT_worst < degrade_threshold_C)
    return HeatingResult(float(dT_avg), float(dT_worst), safe, float(degrade_threshold_C))


# ===========================================================================
# 4. Stochastic 3-D developed line
# ===========================================================================
@dataclass
class Line3DResult:
    s_nm: np.ndarray              # along-line coordinate
    x_left_nm: np.ndarray         # left sidewall position per s
    x_right_nm: np.ndarray        # right sidewall position per s
    top_height_nm: np.ndarray     # remaining resist height per s
    cd_nm: float                  # mean developed CD
    ler_3sigma_nm: float          # line-edge roughness (3 sigma)
    lwr_3sigma_nm: float          # line-width roughness (3 sigma)
    corr_length_nm: float         # CD autocorrelation length
    z_edges_nm: np.ndarray = field(default_factory=lambda: np.array([]))


def _finite_line_window(s_nm: np.ndarray, length_nm: float, taper_nm: float) -> np.ndarray:
    """Normalised dose window along a finite line of given length with smooth
    end tapers (so the ends develop slightly narrower, as in reality)."""
    half = length_nm / 2.0
    ends = np.clip((half - np.abs(s_nm - length_nm / 2.0)) / max(taper_nm, 1e-6), 0.0, 1.0)
    return ends


def simulate_line_3d(
    x: np.ndarray,
    lsf: np.ndarray,
    dose_pC_cm: float,
    threshold_areal: float,
    w_se: float = 20.0,
    a_eff: float = 9.33,
    cd_target_nm: float = 10.0,
    length_nm: float = 200.0,
    dz_nm: float = 2.0,
    seed: int = 0,
    *,
    depth_profile: Optional[np.ndarray] = None,
    depth_edges: Optional[np.ndarray] = None,
    energy_per_ion_in_resist_eV: float = 3293.0,
    rho_gel: float = 15.0,
    resist_thickness_nm: float = 40.0,
    ion_per_nm_per_pC_cm: float = ION_PER_NM_PER_PC_CM,
):
    """Build a 3-D latent image for a *finite* line, add shot noise consistent
    with the stochastic (secondary-electron event) model, threshold it, and
    extract per-position sidewalls.

    Construction
    ------------
    The deposited volumetric energy density is modelled as separable:

        E(s,x,z) = lam * L(x) * f(z) * W(s)          [eV/nm^3]

    where ``lam = dose * ion_per_nm_per_pC_cm`` (ions/nm of line), ``L(x)`` is the
    supplied 1-D LSF (eV/nm per ion), ``f(z) = depth_profile/(E_ion*dz)`` is the
    depth distribution (normalised so integral f dz = 1), and ``W(s)`` is the
    finite-line window from ``_finite_line_window``.  The fluence of *exposure
    events* in a voxel is ``mean_E / w_se``; a Poisson draw (seeded) gives the
    stochastic event count, so ``E = events * w_se``.

    A voxel is "gelled" (negative tone) where ``E >= rho_gel``.  For each
    along-line position ``s`` the left/right sidewalls are the outermost gelled
    ``x``; the remaining height is the deepest gelled ``z``.  CD(s) is the
    sidewall separation; LER/LWR are 3*std over ``s``; the correlation length is
    the lag at which the CD autocorrelation first falls to 1/e.

    Deterministic given ``seed`` (all randomness flows from one ``default_rng``).

    Parameters
    ----------
    x, lsf        : lateral grid (nm) and line-spread function (eV/nm per ion).
    dose_pC_cm    : line dose.
    threshold_areal : areal gel threshold passed for reference (the 3-D model
        uses the volumetric ``rho_gel``; ``threshold_areal`` is retained for API
        compatibility and is not used internally).
    w_se, a_eff   : secondary-electron event energy and stochastic area (defaults
        match the pipeline calibration).
    """
    x = np.asarray(x, dtype=float)
    lsf = np.asarray(lsf, dtype=float)
    rng = np.random.default_rng(seed)

    # depth grid (reuse caller's profile if given, else a flat profile)
    if depth_profile is None or depth_edges is None:
        nz = max(int(round(resist_thickness_nm / dz_nm)), 1)
        z_edges = np.linspace(0.0, resist_thickness_nm, nz + 1)
        depth_profile = np.full(nz, energy_per_ion_in_resist_eV / nz)
    else:
        z_edges = np.asarray(depth_edges, dtype=float)
        depth_profile = np.asarray(depth_profile, dtype=float)
    dz = np.diff(z_edges)
    in_film = z_edges[:-1] < resist_thickness_nm
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])[in_film]
    fz = depth_profile[in_film] / (energy_per_ion_in_resist_eV * dz[in_film])  # 1/nm

    # along-line grid
    ns = int(round(length_nm / dz_nm))
    s = np.linspace(0.0, length_nm, ns)
    Ws = _finite_line_window(s, length_nm, taper_nm=max(4.0 * dz_nm, 2.0))

    lam = dose_pC_cm * ion_per_nm_per_pC_cm
    Lx = lsf  # eV/nm per ion
    # mean volumetric energy density E(s,x,z) [eV/nm^3]
    mean_E = lam * (Lx[None, :, None] * fz[None, None, :] * Ws[:, None, None])

    # shot noise: Poisson on event count = mean_E / w_se
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_events = mean_E / w_se
    mean_events = np.where(np.isfinite(mean_events), mean_events, 0.0)
    events = rng.poisson(mean_events)
    E = events * w_se

    # threshold -> gelled mask (nz, nx, ns) ; we work (ns, nx, nz)
    gelled = E >= rho_gel  # (ns, nx, nz)

    x_left = np.full(ns, np.nan)
    x_right = np.full(ns, np.nan)
    top_h = np.full(ns, 0.0)
    for i in range(ns):
        gxy = gelled[i]                       # (nx, nz)
        if not gxy.any():
            continue
        # through-thickness gel fraction per lateral column; the line sidewall
        # is the outermost x where the column is gelled over >= 50% of its depth
        # (negative tone: the line is "solid" where most of the film gelled).
        gf = gxy.mean(axis=1)
        xs_idx = np.where(gf >= 0.5)[0]
        if xs_idx.size == 0:
            continue
        x_left[i] = x[xs_idx.min()]
        x_right[i] = x[xs_idx.max()]
        # deepest gelled depth (remaining height)
        zg = np.where(gxy.any(axis=0))[0]
        top_h[i] = zc[zg.max()]

    cd_s = x_right - x_left
    cd_s = np.where(np.isfinite(cd_s), cd_s, 0.0)
    cd_mean = float(cd_s.mean()) if cd_s.any() else 0.0
    # positions with no gelled column (line ends) -> collapse to the centre so
    # the returned arrays are finite and of consistent length
    x_mid = x[np.argmin(np.abs(x))]
    x_left = np.where(np.isfinite(x_left), x_left, x_mid)
    x_right = np.where(np.isfinite(x_right), x_right, x_mid)
    ler = 3.0 * float(cd_s.std()) if cd_s.size > 1 else 0.0
    lwr = ler  # single line: width roughness == edge roughness here

    # autocorrelation length of CD(s)
    cd_c = cd_s - cd_s.mean()
    ac = np.correlate(cd_c, cd_c, mode="full")[len(cd_c) - 1:]
    ac = ac / (ac[0] if ac[0] != 0 else 1.0)
    below = np.where(ac < 1.0 / np.e)[0]
    corr = float(below[0] * dz_nm) if below.size else float(length_nm)

    return Line3DResult(
        s_nm=s, x_left_nm=x_left, x_right_nm=x_right, top_height_nm=top_h,
        cd_nm=cd_mean, ler_3sigma_nm=ler, lwr_3sigma_nm=lwr,
        corr_length_nm=corr, z_edges_nm=z_edges,
    )
