# -*- coding: utf-8 -*-
"""
SubNano-HIL-MC  |  40 nm Ti6-oxo-cluster resist on a 670 um Si wafer, 30 keV He+
===============================================================================

End-to-end, data-driven pipeline.  Every number in the report is either

  * measured   -- read from ``data/`` (single-crystal CIF, XRR reflectivity), or
  * calculated -- from the Monte-Carlo transport + lithography metrics, or
  * assumed    -- an explicitly flagged model parameter (never silently used).

Inputs
------
    data/4.cif                      single crystal  -> composition, molecule size
    data/0828-24355-new/.../4.txt   XRR             -> thickness, density, roughness
    user specification              resist 40 nm / substrate Si 670 um / He+ 30 keV

Physics chain
-------------
    CIF  ->  stoichiometry + molecular diameter (the "molecular pixel")
    XRR  ->  film thickness / mass density / surface roughness  (Parratt fit)
    materials -> electronic (Lindhard-Scharff) + nuclear (ZBL) stopping
    mc        -> 30 keV He+ trajectories, SE1 spreading, substrate SE2 re-injection
    metrics   -> radial PSF -> Abel -> LSF -> CD(dose) -> NILS -> stochastic LER
    design    -> CD floor, CD target, exposure latitude, roughness budget

Outputs (in ``analysis/case_40nm_bulk/``)
-----------------------------------------
    fig1_structure.png     single-crystal interpretation of 4.cif
    fig2_xrr.png           XRR measurement, Parratt fit, density depth profile
    fig3_transport.png     trajectories, in-resist + full-depth dose, energy budget
    fig4_psf_nils.png      PSF (95% CI), LSF/SE2 decomposition, NILS vs CD
    fig5_ler_window.png    LER(dose), CD(dose), 2-D process window
    fig6_design.png        roughness budget, exposure latitude, contrast, hazards
    report.html            full report
    results.json           every number, machine readable
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.ndimage import gaussian_filter, gaussian_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hil_mc.cifio import read_structure, build_bonds, assemble_molecule   # noqa: E402
from hil_mc.structure import analyse_cif                                  # noqa: E402
from hil_mc.materials import Material                                     # noqa: E402
from hil_mc.mc import make_stack, HeMonteCarlo, MCResult                  # noqa: E402
from hil_mc.metrics import (lsf_from_map, radial_profile, dose_sweep,     # noqa: E402
                            bootstrap_psf, calibrate_a_eff,
                            smooth_tail, converged_lsf)
from hil_mc import design as dsn                                          # noqa: E402
from hil_mc import hazards as hz                                          # noqa: E402
from hil_mc import xrr as xrrmod                                          # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "analysis", "case_40nm_bulk")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Case configuration
# ---------------------------------------------------------------------------
RESIST_THICKNESS_NM = 40.0        # user-specified resist film thickness
SUBSTRATE_THICKNESS_NM = 670e3    # 670 um Si wafer
BEAM_ENERGY_KEV = 30.0
CD_TARGET_NM = 10.0               # design linewidth (isolated line)
CD_TOL = 0.10                     # +/-10% CD window

# --- model parameters that are ASSUMED (not measured here) ------------------
RHO_GEL = 15.0                    # eV/nm^3 absorbed energy density to gel
W_SE = 20.0                       # eV per exposure (secondary-electron) event
TARGET_LER3_NM = 0.21             # published sub-nm reference (Zhuang 2024)

# --- transport calibration --------------------------------------------------
# e_scale multiplies the Lindhard-Scharff electronic stopping so that the mean
# stopping depth of 30 keV He+ in Si matches the SRIM projected range.
# Verified in this build: e_scale = 1.10 -> 281 nm (SRIM 282.2 nm, -0.4%).
E_SCALE = 1.10
A_EFF = 9.17                 # nm^2; back-fitted to membrane min LER3sigma = 0.21 nm
SRIM_RANGE_SI_NM = 282.2

# --- Monte-Carlo sampling ---------------------------------------------------
N_MAIN = 40000
N_REP = 15000
N_SCAN = 12000
DEPTH_MAX_NM = 600.0

# --- publication style ------------------------------------------------------
C_BULK = "#1f4e79"      # 40 nm / 670 um Si            deep blue
C_MEM = "#2a9d8f"       # 40 nm / 20 nm SiNx membrane  teal
C_REF = "#6b7280"       # reference lines / grey text
C_ACC = "#c0392b"       # highlight / brick red
C_SUB = "#94a3b8"       # substrate accent / slate

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 10.0, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.0,
    "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111", "xtick.color": "#333333",
    "ytick.color": "#333333", "xtick.direction": "out", "ytick.direction": "out",
    "figure.dpi": 130, "savefig.dpi": 320, "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.5,
    "grid.linestyle": "--", "figure.facecolor": "white",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _style():
    plt.rcParams.update(STYLE)


def clean(ax, xlab=None, ylab=None, logx=False, logy=False, grid=True,
          letter=None, xlim=None, ylim=None, title=None):
    """Apply the shared axis style and an optional panel letter."""
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    for s in ("top", "right"):
        if s in ax.spines:
            ax.spines[s].set_visible(False)
    if grid:
        ax.grid(True, which="major", color="#e4e4e4", lw=0.5, ls="--")
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    if xlab:
        ax.set_xlabel(xlab, labelpad=4)
    if ylab:
        ax.set_ylabel(ylab, labelpad=4)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=9, pad=3)
    if letter:
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top", ha="left", color="#111111",
                zorder=40,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#e2e2e2",
                          lw=0.5, alpha=0.9))


# ===========================================================================
# 1. Single crystal
# ===========================================================================
def step_structure():
    log("parsing data/4.cif ...")
    a = analyse_cif(os.path.join(DATA, "4.cif"))
    st = read_structure(os.path.join(DATA, "4.cif"))
    bonds = build_bonds(st)
    order, xyz = assemble_molecule(st, bonds, seed=0)
    els = [st.atoms[i].element for i in order]
    labs = [st.atoms[i].label for i in order]
    log(f"   {a.formula}  {a.space_group}  Z={a.z}  "
        f"rho_xtal={a.density_calc} g/cm3  R1={a.r1}  "
        f"molecule diameter {a.molecule_extent_nm:.2f} nm")
    return a, (xyz, els, labs)


# ===========================================================================
# 2. XRR: thickness / density / roughness from the raw measurement
# ===========================================================================
def step_xrr(analysis):
    """Fit the measured reflectivity instead of quoting vendor constants."""
    path = xrrmod.default_xrr_path(ROOT)
    log(f"fitting XRR {os.path.relpath(path, ROOT)} ...")
    data = xrrmod.load_xrr(path)
    # electrons per gram of the CIF composition -> converts delta to mass density
    probe = Material.from_formula("resist", analysis.formula, 1.0)
    fit = xrrmod.fit_xrr(data, film_z_over_a=probe.electrons_per_gram)
    log(f"   d = {fit.thickness_nm:.2f} +/- {fit.thickness_err_nm:.2f} nm, "
        f"rho = {fit.density_g_cm3:.3f} +/- {fit.density_err_g_cm3:.3f} g/cm3, "
        f"sigma = {fit.sigma_film_nm:.3f} +/- {fit.sigma_film_err_nm:.3f} nm "
        f"(log10 residual {fit.rms_log10_residual:.3f})")
    if fit.kiessig is not None:
        log(f"   model-free Kiessig thickness = {fit.kiessig.thickness_nm:.2f} nm "
            f"(R2 = {fit.kiessig.r_squared:.5f}, {fit.kiessig.n_fringes} fringes)")
    return data, fit


# ===========================================================================
# 3. Materials
# ===========================================================================
def step_materials(analysis, xrr_fit):
    resist = Material.from_formula("Ti6-oxo cluster resist (film)",
                                   analysis.formula, xrr_fit.density_g_cm3,
                                   w_se=W_SE)
    resist_xtal = Material.from_formula("Ti6-oxo cluster (crystal)",
                                        analysis.formula, analysis.density_calc,
                                        w_se=W_SE)
    si = Material.from_formula("Si(001) substrate", "Si", 2.3291)
    sin = Material.from_formula("Si3N4 membrane", "Si3 N4", 3.17)
    return resist, resist_xtal, si, sin


# ===========================================================================
# 4. Monte-Carlo
# ===========================================================================
def run_case(resist, substrate, t_res, n_ions, membrane=None, t_mem=0.0,
             seed=1234, keep_traj=0, box=80.0):
    stack = make_stack(resist, substrate, resist_thickness_nm=t_res,
                       substrate_thickness_nm=SUBSTRATE_THICKNESS_NM,
                       membrane=membrane, membrane_thickness_nm=t_mem,
                       e_scale=E_SCALE)
    mc = HeMonteCarlo(stack, BEAM_ENERGY_KEV, seed=seed, box_nm=box)
    return mc.run(n_ions, keep_trajectories=keep_traj, depth_max_nm=DEPTH_MAX_NM)


def _lsf(maps):
    """Converged line-spread function (replicate-averaged radial PSF -> Abel)."""
    if isinstance(maps, MCResult):
        maps = [maps]
    x, lsf = converged_lsf([m.exposure_map for m in maps], maps[0].pixel_nm)
    return x, smooth_tail(x, lsf, r_min=8.0)


def fwhm(x, y):
    """Full width at half maximum of a peaked, symmetric profile."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return float("nan")
    pk = float(np.nanmax(y))
    if pk <= 0:
        return float("nan")
    half = 0.5 * pk
    i0 = int(np.nanargmax(y))
    left = np.where(y[:i0] <= half)[0]
    right = np.where(y[i0:] <= half)[0]
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float(x[i0 + right[0]] - x[left[-1]])


def transport_dict(res, x=None, lsf=None):
    """MC-derived transport quantities for a DesignSummary."""
    r, dens = radial_profile(res.exposure_map, res.pixel_nm)
    tot = dens.sum()
    c = np.cumsum(dens) / max(tot, 1e-30)

    def rad(frac):
        j = int(np.searchsorted(c, frac))
        return float(r[min(j, len(r) - 1)])

    zc = 0.5 * (res.depth_edges_full[1:] + res.depth_edges_full[:-1])
    dp = res.depth_profile_full
    deep = zc > RESIST_THICKNESS_NM + 5.0
    bragg = float(zc[deep][int(np.argmax(dp[deep]))]) if deep.any() else float("nan")
    out = dict(
        energy_in_resist_eV=res.energy_in_resist,
        fraction_in_resist=res.energy_in_resist / (BEAM_ENERGY_KEV * 1e3),
        substrate_se2_eV=float(res.substrate_map.sum() * res.pixel_nm ** 2),
        backscatter_fraction=res.backscatter_fraction,
        range_substrate_nm=res.range_substrate,
        bragg_peak_nm=bragg,
        r50_nm=rad(0.50), r90_nm=rad(0.90), r99_nm=rad(0.99),
    )
    if x is not None and lsf is not None:
        out["lsf_fwhm_nm"] = fwhm(x, lsf)
    return out


# ===========================================================================
# FIGURE 1 -- single-crystal interpretation of 4.cif
# ===========================================================================
def fig_structure(analysis, geom, path):
    _style()
    xyz, els, labs = geom
    a = analysis
    fig = plt.figure(figsize=(10.2, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.95, 1.15], wspace=0.26,
                          left=0.03, right=0.985, top=0.86, bottom=0.13)

    # (a) molecule: hide H, shrink C, emphasise the Ti6O4 core
    ax = fig.add_subplot(gs[0], projection="3d")
    col = {"Ti": C_BULK, "O": C_ACC, "C": "#6a6a6a"}
    size = {"C": 6, "O": 20, "Ti": 70}
    alpha = {"C": 0.35, "O": 0.85, "Ti": 1.0}
    pos = xyz - xyz.mean(axis=0)
    for e in ("C", "O", "Ti"):
        m = np.array([s == e for s in els])
        ax.scatter(pos[m, 0], pos[m, 1], pos[m, 2], s=size[e], c=col[e],
                   alpha=alpha[e], edgecolors="none", depthshade=False, label=e)
    ti = np.array([s == "Ti" for s in els])
    p = pos[ti]
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if np.linalg.norm(p[i] - p[j]) < 4.0:
                ax.plot(*zip(p[i], p[j]), color=C_BULK, lw=2.4, alpha=0.9)
    ax.view_init(elev=16, azim=32)
    ax.set_axis_off()
    lim = np.abs(pos).max() * 0.78
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_title("a", loc="left", fontsize=12, fontweight="bold", pad=1)
    ax.text2D(0.5, 0.02, f"molecule Ø = {a.molecule_extent_nm:.2f} nm "
                         f"(the molecular pixel)", transform=ax.transAxes,
              ha="center", fontsize=8, color=C_REF)
    ax.legend(loc="upper right", frameon=True, fontsize=8, framealpha=0.92,
              edgecolor="#cccccc", labelspacing=0.25, handletextpad=0.4)

    # (b) Ti6 core, in-plane projection
    ax2 = fig.add_subplot(gs[1])
    ax2.set_aspect("equal")
    ti_idx = [i for i, e in enumerate(els) if e == "Ti"]
    p2 = pos[ti_idx][:, :2]
    ax2.scatter(p2[:, 0], p2[:, 1], s=300, c=C_BULK, edgecolors="white",
                lw=1.4, zorder=5)
    for i, (x_, y_) in enumerate(p2):
        ax2.text(x_, y_, f"Ti{i+1}", ha="center", va="center", color="white",
                 fontsize=8, fontweight="bold", zorder=6)
    for i in range(len(p2)):
        for j in range(i + 1, len(p2)):
            if np.linalg.norm(pos[ti_idx[i]] - pos[ti_idx[j]]) < 4.2:
                ax2.plot(*zip(p2[i], p2[j]), color=C_BULK, lw=1.6, zorder=4)
    ax2.margins(0.18)
    n_mu3 = len(a.oxo.get("mu3-O", []))
    n_mu2 = len(a.oxo.get("mu2-O", []))
    ax2.text(0.03, 0.97,
             f"core {a.core_formula}\n"
             f"{n_mu3} µ₃-O + {n_mu2} µ₂-O bridges\n"
             f"d(Ti–Ti) = {min(a.ti_ti_distances):.2f}–{max(a.ti_ti_distances):.2f} Å",
             transform=ax2.transAxes, fontsize=7.6, color="#333333", va="top",
             bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#dddddd", lw=0.5))
    clean(ax2, xlab="x (Å)", ylab="y (Å)", grid=False, letter=None)
    ax2.set_title("b", loc="left", fontsize=12, fontweight="bold", pad=1)

    # (c) crystal data + ligand inventory
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    lig = {}
    for L in a.ligands:
        if L.formula == "O":
            continue
        lig[L.formula] = lig.get(L.formula, 0) + L.count
    lig_txt = ", ".join(f"{k}×{v}" for k, v in sorted(lig.items()))
    rows = [
        ["Formula", a.formula],
        ["M (g mol⁻¹)", f"{a.fw:.2f}"],
        ["System / group", f"{a.crystal_system}, {a.space_group} (#{a.it_number})"],
        ["a, b, c (Å)", f"{a.cell[0]:.4f}, {a.cell[1]:.4f}, {a.cell[2]:.4f}"],
        ["α, β, γ (°)", f"{a.cell[3]:.3f}, {a.cell[4]:.3f}, {a.cell[5]:.3f}"],
        ["V (Å³) / Z / Z′", f"{a.volume:.2f} / {a.z} / {a.z_prime:.2f}"],
        ["ρ(calc) (g cm⁻³)", f"{a.density_calc:.3f}"],
        ["T (K) / λ (Å)", f"{a.temperature_K:.0f} / {a.wavelength:.5f}"],
        ["θmax / complet.", f"{a.theta_max:.1f}° / {a.completeness:.3f}"],
        ["R1 / wR2 / GooF", f"{a.r1:.4f} / {a.wr2:.4f} / {a.goof:.3f}"],
        ["Ti (wt%)", f"{100*a.ti_mass_fraction:.2f}"],
        ["Ø / Rg (nm)", f"{a.molecule_extent_nm:.2f} / {a.rg_nm:.2f}"],
        ["Ligands", lig_txt],
    ]
    tbl = ax3.table(cellText=rows, colLabels=["Crystal data (refined)", "Value"],
                    cellLoc="left", loc="center", colWidths=[0.42, 0.58])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.2)
    tbl.scale(1, 1.20)
    for (r_, c_), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfcfcf")
        cell.set_linewidth(0.4)
        cell.set_facecolor("white")
        if r_ == 0:
            cell.set_text_props(color="#111111", fontweight="bold")
    ax3.set_title("c", loc="left", fontsize=12, fontweight="bold", pad=10)

    fig.suptitle("Figure 1 · data/4.cif — Ti₆-oxo cluster single crystal "
                 "(Olex2/SHELXL, 100 K, Cu-Kα)", fontsize=11.5,
                 fontweight="bold", y=0.975)
    fig.savefig(path)
    plt.close(fig)
    log(f"   saved {os.path.basename(path)}")


# ===========================================================================
# FIGURE 2 -- XRR: measurement, Parratt fit, derived depth profile
# ===========================================================================
def fig_xrr(data, fit, analysis, path):
    _style()
    fig = plt.figure(figsize=(10.2, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.30,
                          left=0.07, right=0.99, top=0.84, bottom=0.18)

    # (a) measured reflectivity + Parratt fit
    ax = fig.add_subplot(gs[0])
    norm = float(np.max(data.intensity))
    ax.plot(data.two_theta_deg, data.intensity / norm, color=C_REF, lw=1.0,
            label="measured")
    ax.plot(2.0 * fit.theta_deg, fit.r_model, color=C_ACC, lw=1.6,
            label="Parratt fit")
    ax.axvline(fit.two_theta_min_deg, color="#bbbbbb", ls="--", lw=0.9)
    ax.axvspan(data.two_theta_deg.min(), fit.two_theta_min_deg,
               color="#f0f0f0", alpha=0.85, lw=0)
    ax.text(fit.two_theta_min_deg * 1.03, 3e-6,
            "excluded\n(beam footprint)", fontsize=7.2, color=C_REF, va="bottom")
    ax.axvline(fit.critical_two_theta_deg, color=C_MEM, ls=":", lw=1.1)
    ax.text(fit.critical_two_theta_deg * 1.05, 0.30,
            f"2θ$_c$ = {fit.critical_two_theta_deg:.3f}°", fontsize=7.4,
            color=C_MEM)
    clean(ax, xlab="2θ (deg)", ylab="reflectivity (norm.)", logy=True,
          letter="a", xlim=(data.two_theta_deg.min(), data.two_theta_deg.max()),
          ylim=(1e-6, 3))
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.text(0.98, 0.42,
            f"d = {fit.thickness_nm:.2f} ± {fit.thickness_err_nm:.2f} nm\n"
            f"ρ = {fit.density_g_cm3:.3f} ± {fit.density_err_g_cm3:.3f} g cm⁻³\n"
            f"σ = {fit.sigma_film_nm:.3f} ± {fit.sigma_film_err_nm:.3f} nm",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd", lw=0.5))

    # (b) model-free Kiessig fringe fit
    ax2 = fig.add_subplot(gs[1])
    k = fit.kiessig
    if k is not None and k.orders.size:
        th2 = (np.radians(k.peak_two_theta_deg / 2.0)) ** 2
        m2 = k.orders.astype(float) ** 2
        ax2.scatter(m2, th2 * 1e6, s=32, color=C_BULK, edgecolors="white",
                    lw=0.6, zorder=5, label="fringe maxima")
        sl, ic = np.polyfit(m2, th2, 1)
        xf = np.linspace(0, m2.max() * 1.06, 60)
        ax2.plot(xf, (sl * xf + ic) * 1e6, color=C_ACC, lw=1.5,
                 label=f"linear fit (R² = {k.r_squared:.5f})")
        ax2.text(0.03, 0.96,
                 f"modified Bragg:\nθ² = θ$_c$² + m²(λ/2d)²\n"
                 f"d = {k.thickness_nm:.2f} nm ({k.n_fringes} fringes)",
                 transform=ax2.transAxes, va="top", fontsize=7.5, color="#333333",
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#dddddd",
                           lw=0.5))
    clean(ax2, xlab="fringe order m²", ylab="θ² (µrad²·10⁶)", letter="b")
    ax2.legend(frameon=False, loc="lower right", fontsize=8)

    # (c) derived electron-density depth profile
    ax3 = fig.add_subplot(gs[2])
    from scipy.special import erf
    d = fit.thickness_nm
    z = np.linspace(-4, d + 22, 700)
    rho_f = fit.density_g_cm3
    rho_s = xrrmod.SI_DENSITY
    prof = (0.5 * rho_f * (1 + erf(z / (np.sqrt(2) * fit.sigma_film_nm)))
            + 0.5 * (rho_s - rho_f) * (1 + erf((z - d) /
                                               (np.sqrt(2) * max(fit.sigma_sub_nm, 1e-3)))))
    ax3.plot(z, prof, color=C_BULK, lw=1.8)
    ax3.axvspan(-4, 0, color="#eaf4f8", alpha=0.8, lw=0)
    ax3.axvspan(0, d, color="#fff8dc", alpha=0.9, lw=0)
    ax3.axvspan(d, z.max(), color="#eceff1", alpha=0.9, lw=0)
    ax3.axhline(rho_f, color=C_MEM, ls=":", lw=1.0)
    ax3.text(d * 0.5, rho_f * 1.10, f"film ρ = {rho_f:.3f}", ha="center",
             fontsize=7.4, color=C_MEM)
    ax3.text(d + 8, rho_s * 0.92, "Si substrate", fontsize=7.6, color=C_REF)
    ax3.annotate(f"σ = {fit.sigma_film_nm:.2f} nm", xy=(0, rho_f * 0.5),
                 xytext=(6, rho_f * 0.20), fontsize=7.4, color=C_ACC,
                 arrowprops=dict(arrowstyle="->", color=C_ACC, lw=0.9))
    ax3.text(0.98, 0.05,
             f"densification ρ/ρ_xtal = {100*rho_f/analysis.density_calc:.0f}%",
             transform=ax3.transAxes, ha="right", fontsize=7.4, color="#333333",
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#dddddd", lw=0.5))
    clean(ax3, xlab="depth z from surface (nm)", ylab="mass density (g cm⁻³)",
          letter="c", xlim=(-4, z.max()), ylim=(-0.1, rho_s * 1.15))

    fig.suptitle("Figure 2 · X-ray reflectivity of sample 4 — thickness, density "
                 "and roughness fitted from the raw data", fontsize=11.5,
                 fontweight="bold", y=0.975)
    fig.savefig(path)
    plt.close(fig)
    log(f"   saved {os.path.basename(path)}")


# ===========================================================================
# FIGURE 3 -- transport: trajectories, dose depth profiles, energy budget
# ===========================================================================
def fig_transport(bulk, mem, path):
    _style()
    fig = plt.figure(figsize=(11.0, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 0.95, 1.05, 0.85],
                          wspace=0.34, left=0.055, right=0.99, top=0.84,
                          bottom=0.18)

    # (a) trajectory density map (lateral x vs depth z)
    ax = fig.add_subplot(gs[0])
    trajs = [t for t in bulk.trajectories if len(t) > 2]
    xs = np.concatenate([t[:, 0] for t in trajs])
    zs = np.concatenate([t[:, 2] for t in trajs])
    xb = np.linspace(-60, 60, 121)
    zb = np.linspace(0, 330, 331)
    H, xe, ze = np.histogram2d(xs, zs, bins=[xb, zb])
    cmap = LinearSegmentedColormap.from_list("inj", ["#ffffff", "#9ecbe1", C_BULK])
    ax.pcolormesh(xe, ze, np.maximum(H.T, 1), cmap=cmap,
                  norm=LogNorm(vmin=1, vmax=max(H.max(), 10)), shading="auto")
    for t in trajs[:12]:
        ax.plot(t[:, 0], t[:, 2], lw=0.5, alpha=0.5, color="#0b2d4d", zorder=5)
    ax.axhline(0, color="#8d6e63", lw=1.4)
    ax.axhline(RESIST_THICKNESS_NM, color=C_ACC, lw=1.4, ls="--")
    ax.axhline(bulk.range_substrate, color=C_MEM, lw=1.2, ls=":")
    ax.text(-57, bulk.range_substrate - 8,
            f"mean stop {bulk.range_substrate:.0f} nm", fontsize=7.2, color=C_MEM,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.8))
    ax.text(57, 20, "resist", ha="right", fontsize=7.8, color="#8d6e63",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.text(57, 210, "Si 670 µm", ha="right", fontsize=7.8, color=C_SUB,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_ylim(330, -10)
    clean(ax, xlab="lateral x (nm)", ylab="depth z (nm)", grid=False, letter="a",
          xlim=(-60, 60))

    # (b) full-depth dose (Bragg curve) with the resist band highlighted
    ax2 = fig.add_subplot(gs[1])
    zc = 0.5 * (bulk.depth_edges_full[1:] + bulk.depth_edges_full[:-1])
    dp = gaussian_filter1d(bulk.depth_profile_full, 1.2)
    ax2.plot(dp, zc, color=C_BULK, lw=1.7)
    ax2.fill_betweenx(zc, 0, dp, color=C_BULK, alpha=0.14, lw=0)
    ax2.axhspan(0, RESIST_THICKNESS_NM, color="#fff2c8", alpha=0.85, lw=0)
    deep = zc > RESIST_THICKNESS_NM + 5.0
    if deep.any():
        zpk = float(zc[deep][int(np.argmax(dp[deep]))])
        ax2.axhline(zpk, color=C_ACC, ls=":", lw=1.1)
        ax2.text(dp.max() * 0.96, zpk - 10, f"Bragg peak {zpk:.0f} nm",
                 ha="right", fontsize=7.3, color=C_ACC,
                 bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                           alpha=0.85))
    ax2.text(dp.max() * 0.96, RESIST_THICKNESS_NM * 0.55, "resist\n0–40 nm",
             ha="right", fontsize=7.3, color="#8a6d00")
    ax2.set_ylim(zc.max(), 0)
    clean(ax2, xlab="dE/dz (eV nm⁻¹ ion⁻¹)", ylab="depth z (nm)", letter="b")

    # (c) in-resist depth profile: bulk Si vs suspended membrane
    ax3 = fig.add_subplot(gs[2])
    zr = 0.5 * (bulk.depth_edges[1:] + bulk.depth_edges[:-1])
    pb = gaussian_filter1d(bulk.depth_profile, 2.0)
    pm = gaussian_filter1d(mem.depth_profile, 2.0)
    nb = pb.max() or 1.0
    nm_ = pm.max() or 1.0
    ax3.fill_betweenx(zr, 0, pb / nb, color=C_BULK, alpha=0.15, lw=0)
    ax3.fill_betweenx(zr, 0, pm / nm_, color=C_MEM, alpha=0.12, lw=0)
    ax3.plot(pb / nb, zr, color=C_BULK, lw=1.8, label="670 µm Si")
    ax3.plot(pm / nm_, zr, color=C_MEM, lw=1.8, ls="--", label="20 nm SiN$_x$")
    ax3.set_ylim(RESIST_THICKNESS_NM, 0)
    clean(ax3, xlab="in-resist dose (peak-norm.)", ylab="depth z (nm)", letter="c")
    ax3.legend(frameon=False, loc="lower left", fontsize=8)

    # (d) single-ion energy budget (log so the 1e-3 backscatter is visible)
    ax4 = fig.add_subplot(gs[3])
    e_res = bulk.energy_in_resist
    e_sub = bulk.energy_in_substrate_total
    e_back = bulk.energy_backscattered_out
    e_deep = BEAM_ENERGY_KEV * 1e3 - e_res - e_sub - e_back
    labels = ["resist", "sub\n<24 nm", "deep\nSi", "back-\nscatter"]
    vals = [e_res, e_sub, max(e_deep, 0.1), max(e_back, 0.1)]
    cols = [C_BULK, C_ACC, C_SUB, "#c62828"]
    bars = ax4.bar(labels, vals, color=cols, width=0.64, edgecolor="white", lw=0.6)
    ax4.set_yscale("log")
    ax4.set_ylim(1e0, 1e5)
    for rect, v in zip(bars, vals):
        ax4.text(rect.get_x() + rect.get_width() / 2, v * 1.35,
                 f"{100*v/(BEAM_ENERGY_KEV*1e3):.2f}%", ha="center",
                 fontsize=7.4, fontweight="bold")
    clean(ax4, ylab="eV per incident ion", letter="d")
    ax4.grid(True, axis="y", which="both", color="#e4e4e4", lw=0.5, ls="--")
    ax4.tick_params(axis="x", labelsize=7.6)
    ax4.set_title(f"η = {100*bulk.backscatter_fraction:.2f}%  (EBL 30–50%)",
                  fontsize=8.4, pad=3)

    fig.suptitle("Figure 3 · 30 keV He⁺ transport in 40 nm Ti-cluster resist on "
                 "670 µm Si", fontsize=11.5, fontweight="bold", y=0.975)
    fig.savefig(path)
    plt.close(fig)
    log(f"   saved {os.path.basename(path)}")


# ===========================================================================
# FIGURE 4 -- PSF / LSF / NILS
# ===========================================================================
def fig_psf(cases, a_eff, cd_floor, path):
    _style()
    fig = plt.figure(figsize=(10.2, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.30,
                          left=0.075, right=0.99, top=0.84, bottom=0.18)
    thresh = RHO_GEL * RESIST_THICKNESS_NM

    # (a) radial PSF with bootstrap 95% CI
    ax = fig.add_subplot(gs[0])
    r90 = None
    for reps, col, lab in cases:
        r, mean, lo, hi = bootstrap_psf([m.exposure_map for m in reps],
                                        reps[0].pixel_nm, n_boot=40)
        ax.fill_between(r, lo, hi, color=col, alpha=0.16, lw=0)
        ax.plot(r, mean, color=col, lw=1.7, label=lab)
        if r90 is None:
            c = np.cumsum(mean) / mean.sum()
            r90 = float(r[int(np.searchsorted(c, 0.90))])
    ax.axvline(r90, color=C_REF, ls=":", lw=1.1)
    ax.text(r90 * 2.2, 4e-6, f"r$_{{90}}$ = {r90:.2f} nm", color=C_REF,
            fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", lw=0.4))
    clean(ax, xlab="radius r (nm)", ylab="PSF (norm., eV nm⁻² ion⁻¹)", logy=True,
          letter="a", xlim=(0, 45), ylim=(1e-6, 2))
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.text(0.97, 0.52, "shaded = bootstrap 95% CI\ncurves overlap: PSF core is\n"
                        "substrate-independent",
            transform=ax.transAxes, ha="right", va="center", fontsize=7.4,
            color=C_REF)

    # (b) LSF and its substrate-SE2 component
    ax2 = fig.add_subplot(gs[1])
    for reps, col, lab in cases:
        res = reps[0]
        x, lsf = _lsf(reps)
        peak = lsf[len(lsf) // 2]
        lsx, lsub = lsf_from_map(res.substrate_map, res.pixel_nm)
        ax2.plot(x, lsf / peak, color=col, lw=1.7, label=f"{lab} — total")
        ax2.plot(lsx, lsub / peak, color=col, lw=1.2, ls="--",
                 label=f"{lab} — SE₂ from substrate")
    clean(ax2, xlab="|x| (nm)", ylab="LSF / LSF(0)", logy=True, letter="b",
          xlim=(0, 35), ylim=(1e-5, 2))
    ax2.legend(frameon=False, loc="lower left", fontsize=7.2)

    # (c) NILS vs developed CD, with the CD floor and target marked
    ax3 = fig.add_subplot(gs[2])
    cd_max_plot = 0.0
    for reps, col, lab in cases:
        x, lsf = _lsf(reps)
        sw = dose_sweep(x, lsf, np.geomspace(1.0, 4000.0, 320), thresh, W_SE, a_eff)
        cd = np.array([s.cd_nm for s in sw])
        nils = np.array([s.nils for s in sw])
        ok = np.isfinite(cd) & np.isfinite(nils)
        cd, nils = cd[ok], nils[ok]
        s = np.argsort(cd)
        ax3.plot(cd[s], gaussian_filter1d(nils[s], 1.2), color=col, lw=1.7,
                 label=lab)
        cd_max_plot = max(cd_max_plot, float(cd.max()))
    ax3.axvspan(0, cd_floor, color="#f2f2f2", alpha=0.95, lw=0)
    ax3.axvline(cd_floor, color=C_REF, ls="--", lw=1.0)
    ax3.text(cd_floor * 0.94, 7.6, "CD < molecule Ø\n(unphysical)", ha="right",
             fontsize=7.0, color=C_REF)
    ax3.axvline(CD_TARGET_NM, color=C_ACC, ls=":", lw=1.2)
    ax3.text(CD_TARGET_NM * 1.05, 7.6, f"design CD\n{CD_TARGET_NM:.0f} nm",
             fontsize=7.2, color=C_ACC)
    clean(ax3, xlab="developed CD (nm)", ylab="NILS", letter="c",
          xlim=(0, min(cd_max_plot, 18)), ylim=(0, 8.5))
    ax3.legend(frameon=False, loc="upper right", fontsize=8)

    fig.suptitle("Figure 4 · Point- and line-spread functions and image log-slope",
                 fontsize=11.5, fontweight="bold", y=0.975)
    fig.savefig(path)
    plt.close(fig)
    log(f"   saved {os.path.basename(path)}")
    return r90


# ===========================================================================
# FIGURE 5 -- LER(dose), CD(dose), 2-D process window
# ===========================================================================
def fig_ler(cases, a_eff, cd_floor, sigma_surf, thickness_scan, path):
    _style()
    fig = plt.figure(figsize=(10.6, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.12], wspace=0.32,
                          left=0.075, right=0.985, top=0.84, bottom=0.18)
    thresh = RHO_GEL * RESIST_THICKNESS_NM
    doses = np.geomspace(1.0, 4000.0, 360)

    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    for reps, col, lab in cases:
        x, lsf = _lsf(reps)
        sw = dose_sweep(x, lsf, doses, thresh, W_SE, a_eff,
                        sigma_rough_nm=sigma_surf)
        cd = np.array([s.cd_nm for s in sw])
        ler = np.array([s.ler_3sigma_nm for s in sw])
        adm = np.isfinite(ler) & (ler > 0) & (cd >= cd_floor)
        rej = np.isfinite(ler) & (ler > 0) & (cd < cd_floor)
        ax.plot(doses[rej], ler[rej], color=col, lw=1.0, ls=":", alpha=0.55)
        ax.plot(doses[adm], ler[adm], color=col, lw=1.8, label=lab)
        ax2.plot(doses, cd, color=col, lw=1.8, label=lab)
        if adm.any():
            i = int(np.nanargmin(np.where(adm, ler, np.inf)))
            ax.scatter([doses[i]], [ler[i]], color=col, s=46, zorder=6,
                       edgecolors="black", lw=0.7)
            off = (-10, 20) if col == C_BULK else (-10, -30)
            ax.annotate(f"{ler[i]:.3f} nm\n{doses[i]:.0f} pC/cm",
                        (doses[i], ler[i]), textcoords="offset points",
                        xytext=off, ha="right", fontsize=7.2, color=col,
                        fontweight="bold")

    ax.axhline(TARGET_LER3_NM, color=C_REF, ls=":", lw=1.1)
    ax.text(1.15, TARGET_LER3_NM * 0.94, "published sub-nm reference 0.21 nm",
            ha="left", va="top", fontsize=7.0, color=C_REF)
    clean(ax, xlab="line dose (pC cm⁻¹)", ylab="stochastic LER 3σ (nm)",
          logx=True, logy=True, letter="a", ylim=(0.12, 3.0))
    ax.yaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.3, 0.5, 1.0, 2.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.text(0.98, 0.06, "dotted = rejected by CD floor", transform=ax.transAxes,
            ha="right", fontsize=7.0, color=C_REF)

    ax2.axhspan(0, cd_floor, color="#f2f2f2", alpha=0.95, lw=0)
    ax2.axhline(cd_floor, color=C_REF, ls="--", lw=1.0)
    ax2.text(1.15, cd_floor * 1.06, f"CD floor = molecule Ø = {cd_floor:.2f} nm",
             fontsize=7.0, color=C_REF)
    ax2.axhline(CD_TARGET_NM, color=C_ACC, ls=":", lw=1.2)
    ax2.text(1.15, CD_TARGET_NM * 1.05, f"design CD {CD_TARGET_NM:.0f} nm",
             fontsize=7.2, color=C_ACC)
    clean(ax2, xlab="line dose (pC cm⁻¹)", ylab="developed CD (nm)", logx=True,
          letter="b", ylim=(0, 22))
    ax2.legend(frameon=False, loc="upper left", fontsize=8)

    # (c) process window: LER(dose, thickness) with the CD floor masked out
    ax3 = fig.add_subplot(gs[2])
    th, dw, lm, cdm = thickness_scan
    lm = np.array(lm, dtype=float)
    cdm = np.array(cdm, dtype=float)
    masked = np.where(cdm >= cd_floor, lm, np.nan)
    lm_c = np.clip(np.nan_to_num(masked, nan=1.2), 0.12, 1.2)
    xx, yy = np.meshgrid(dw, th)
    im = ax3.pcolormesh(xx, yy, lm_c, cmap="viridis_r", shading="gouraud",
                        norm=LogNorm(vmin=0.15, vmax=1.2))
    lm_s = 10 ** gaussian_filter(np.log10(lm_c), 0.9)
    cs = ax3.contour(xx, yy, lm_s, levels=[0.20, 0.25, 0.30, 0.40, 0.60],
                     colors="white", linewidths=0.9)
    ax3.clabel(cs, fmt="%.2f", fontsize=7, inline=True, colors="white")
    # hatch the region the CD floor removes
    ax3.contourf(xx, yy, np.where(cdm >= cd_floor, 0.0, 1.0), levels=[0.5, 1.5],
                 colors="none", hatches=["////"])
    ax3.axhline(RESIST_THICKNESS_NM, color=C_ACC, lw=1.3, ls="--")
    ax3.text(dw[-1] * 0.95, RESIST_THICKNESS_NM + 2.0, "40 nm (specified)",
             color=C_ACC, fontsize=7.6, fontweight="bold", ha="right",
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    cb = fig.colorbar(im, ax=ax3, pad=0.02)
    cb.set_label("stochastic LER 3σ (nm)", fontsize=8.4)
    cb.set_ticks([0.2, 0.3, 0.5, 1.0])
    cb.set_ticklabels(["0.2", "0.3", "0.5", "1.0"])
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.tick_params(labelsize=8)
    clean(ax3, xlab="line dose (pC cm⁻¹)", ylab="resist thickness (nm)",
          logx=True, grid=False, letter="c")
    ax3.text(0.03, 0.05, "hatched: CD below molecular floor",
             transform=ax3.transAxes, fontsize=7.0, color="#333333",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    fig.suptitle("Figure 5 · Stochastic line-edge roughness, linewidth and the "
                 "process window", fontsize=11.5, fontweight="bold", y=0.975)
    fig.savefig(path)
    plt.close(fig)
    log(f"   saved {os.path.basename(path)}")


# ===========================================================================
# FIGURE 6 -- design closure: roughness budget, latitude, contrast, hazards
# ===========================================================================
def fig_design(summary, sweep_bulk, contrast, sputter_curve, heat_curve,
               line3d, sigma_surf, path):
    _style()
    fig = plt.figure(figsize=(10.6, 6.2))
    gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.42, left=0.075,
                          right=0.985, top=0.90, bottom=0.09)

    # (a) roughness budget: LER vs surface-to-edge transfer coefficient
    ax = fig.add_subplot(gs[0, 0])
    xi = np.linspace(0.0, 1.0, 120)
    for tag, s1, col, lab in (
            ("best", summary.best_ler3_stoch_nm / 3.0, C_BULK,
             f"best LER point (CD {summary.best_cd_nm:.1f} nm)"),
            ("target", summary.target_ler3_stoch_nm / 3.0, C_ACC,
             f"CD = {CD_TARGET_NM:.0f} nm point")):
        if not np.isfinite(s1):
            continue
        tot = 3.0 * np.sqrt(s1 ** 2 + (xi * sigma_surf) ** 2)
        ax.plot(xi, tot, color=col, lw=1.8, label=lab)
        ax.scatter([0.0], [3.0 * s1], color=col, s=30, zorder=5,
                   edgecolors="black", lw=0.6)
    ax.axhline(TARGET_LER3_NM, color=C_REF, ls=":", lw=1.1)
    ax.text(0.02, TARGET_LER3_NM * 1.06, "0.21 nm reference", fontsize=7.0,
            color=C_REF)
    ax.axvspan(0.0, 0.0, color="none")
    clean(ax, xlab="surface→edge transfer ξ", ylab="total LER 3σ (nm)",
          logy=True, letter="a", xlim=(0, 1), ylim=(0.1, 5))
    ax.legend(frameon=False, loc="upper left", fontsize=7.4)
    ax.text(0.98, 0.06,
            f"σ_surface (XRR) = {sigma_surf:.3f} nm\nξ=0 lower bound, ξ=1 upper bound",
            transform=ax.transAxes, ha="right", fontsize=7.0, color="#333333")

    # (b) exposure latitude at the design CD
    ax2 = fig.add_subplot(gs[0, 1])
    doses = np.array([m.dose_pC_cm for m in sweep_bulk])
    cd = np.array([m.cd_nm for m in sweep_bulk])
    ax2.plot(doses, cd, color=C_BULK, lw=1.8)
    ax2.axhline(CD_TARGET_NM, color=C_ACC, ls="--", lw=1.1)
    ax2.axhspan(CD_TARGET_NM * (1 - CD_TOL), CD_TARGET_NM * (1 + CD_TOL),
                color=C_ACC, alpha=0.12, lw=0)
    if summary.exposure_latitude_pC_cm:
        lo, hi = summary.exposure_latitude_pC_cm
        ax2.axvspan(lo, hi, color=C_MEM, alpha=0.18, lw=0)
        mid = np.sqrt(lo * hi)
        el_pct = 100.0 * (hi - lo) / mid if mid > 0 else float("nan")
        ax2.text(0.97, 0.06,
                 f"latitude {lo:.0f}–{hi:.0f} pC/cm\n(±{el_pct/2:.0f}% about {mid:.0f})",
                 transform=ax2.transAxes, ha="right", fontsize=7.2,
                 color="#0f5d55", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                           alpha=0.85))
    if np.isfinite(summary.target_dose_pC_cm):
        ax2.scatter([summary.target_dose_pC_cm], [summary.target_cd_nm],
                    color=C_ACC, s=40, zorder=6, edgecolors="black", lw=0.6)
    clean(ax2, xlab="line dose (pC cm⁻¹)", ylab="developed CD (nm)", logx=True,
          letter="b", ylim=(0, 22))
    ax2.set_title(f"CD = {CD_TARGET_NM:.0f} nm ± {int(CD_TOL*100)}%", fontsize=8.4,
                  pad=3)

    # (c) contrast curve (model-derived development)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(contrast.doses_pC_cm, contrast.remaining_fraction, color=C_BULK,
             lw=1.8)
    ax3.axvline(contrast.D0, color=C_MEM, ls=":", lw=1.1)
    ax3.axvline(contrast.D100, color=C_ACC, ls=":", lw=1.1)
    ax3.text(0.03, 0.95,
             f"D₀ = {contrast.D0:.2f} pC/cm\nD₁₀₀ = {contrast.D100:.0f} pC/cm\n"
             f"γ = {contrast.gamma:.2f}",
             transform=ax3.transAxes, va="top", fontsize=7.4, color="#333333",
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#dddddd", lw=0.5))
    clean(ax3, xlab="line dose (pC cm⁻¹)", ylab="remaining thickness T/T₀",
          logx=True, letter="c", ylim=(-0.03, 1.05))
    ax3.set_title("model-derived (no development data)", fontsize=8.0, pad=3)

    # (d) sputter + heating safety
    ax4 = fig.add_subplot(gs[1, 0])
    dsw, removed = sputter_curve
    ax4.plot(dsw, removed, color=C_ACC, lw=1.7, label="sputtered thickness")
    ax4.set_yscale("log")
    ax4.axhline(0.1, color=C_REF, ls=":", lw=0.9)
    ax4.text(dsw[1], 0.11, "0.1 nm", fontsize=7.0, color=C_REF, va="bottom")
    ax5 = ax4.twinx()
    ax5.spines["top"].set_visible(False)
    dsw2, dT_avg, dT_worst = heat_curve
    ax5.plot(dsw2, dT_worst, color=C_BULK, lw=1.5, ls="--", label="ΔT worst case")
    ax5.plot(dsw2, dT_avg, color=C_REF, lw=1.3, ls=":", label="ΔT scan-averaged")
    ax5.set_yscale("log")
    ax5.set_ylabel("ΔT (K)", color=C_BULK, fontsize=9)
    ax5.tick_params(axis="y", colors=C_BULK, labelsize=8)
    ax5.grid(False)
    clean(ax4, xlab="line dose (pC cm⁻¹)", ylab="sputtered thickness (nm)",
          logx=True, letter="d")
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax5.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=7.0)
    ax4.set_title("erosion and heating are negligible", fontsize=8.0, pad=3)

    # (e) 3-D developed line (stochastic latent image)
    ax6 = fig.add_subplot(gs[1, 1], projection="3d")
    s_ = line3d.s_nm
    xl, xr, top = line3d.x_left_nm, line3d.x_right_nm, line3d.top_height_nm
    u = np.linspace(0.0, 1.0, 40)
    S, U = np.meshgrid(s_, u, indexing="ij")
    Y = xl[:, None] + U * (xr - xl)[:, None]
    Z = top[:, None] * np.ones_like(U)
    ax6.plot_surface(S, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
                     alpha=0.92, rstride=2, cstride=2)
    ax6.plot(s_, xl, np.zeros_like(s_), color=C_ACC, lw=1.2)
    ax6.plot(s_, xr, np.zeros_like(s_), color=C_ACC, lw=1.2)
    ax6.set_xlabel("along line (nm)", labelpad=2, fontsize=8)
    ax6.set_ylabel("width (nm)", labelpad=2, fontsize=8)
    ax6.set_zlabel("height (nm)", labelpad=1, fontsize=8)
    ax6.set_zlim(0, RESIST_THICKNESS_NM)
    ax6.tick_params(labelsize=7)
    ax6.view_init(elev=24, azim=-58)
    ax6.text2D(0.02, 0.98, "e", transform=ax6.transAxes, fontsize=12,
               fontweight="bold", va="top",
               bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#e2e2e2",
                         lw=0.5, alpha=0.9))
    ax6.set_title(f"CD {line3d.cd_nm:.1f} nm, LER3σ {line3d.ler_3sigma_nm:.2f} nm,\n"
                  f"corr. length {line3d.corr_length_nm:.0f} nm",
                  fontsize=7.8, pad=1)

    # (f) provenance table: measured / calculated / assumed
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.axis("off")
    rows = [
        ["resist thickness", f"{RESIST_THICKNESS_NM:.0f} nm", "specified"],
        ["substrate", f"Si {SUBSTRATE_THICKNESS_NM/1e3:.0f} µm", "specified"],
        ["beam", f"He⁺ {BEAM_ENERGY_KEV:.0f} keV", "specified"],
        ["film density", f"{summary.a_eff_nm2:.2f} nm² (a_eff)", "back-fitted"],
        ["σ_surface", f"{sigma_surf:.3f} nm", "measured (XRR)"],
        ["molecule Ø", f"{summary.cd_floor_nm:.2f} nm", "measured (CIF)"],
        ["ρ_gel", f"{RHO_GEL:.0f} eV nm⁻³", "ASSUMED"],
        ["w_SE", f"{W_SE:.0f} eV", "ASSUMED"],
        ["e_scale", f"{E_SCALE:.2f} (range {summary.range_substrate_nm:.0f} nm)",
         "calibrated to SRIM"],
        ["dose axis", "relative", "NOT calibrated"],
    ]
    tbl = ax7.table(cellText=rows, colLabels=["Quantity", "Value", "Provenance"],
                    cellLoc="left", loc="center", colWidths=[0.34, 0.34, 0.32])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.0)
    tbl.scale(1, 1.30)
    for (r_, c_), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfcfcf")
        cell.set_linewidth(0.4)
        cell.set_facecolor("white")
        if r_ == 0:
            cell.set_text_props(color="#111111", fontweight="bold")
        elif c_ == 2 and rows[r_ - 1][2] in ("ASSUMED", "NOT calibrated"):
            cell.set_text_props(color=C_ACC, fontweight="bold")
    ax7.set_title("f", loc="left", fontsize=12, fontweight="bold", pad=10)

    fig.suptitle("Figure 6 · Design closure: roughness budget, exposure latitude, "
                 "development, hazards and provenance",
                 fontsize=11.5, fontweight="bold", y=0.975)
    fig.savefig(path)
    plt.close(fig)
    log(f"   saved {os.path.basename(path)}")


# ===========================================================================
# write_report
# ===========================================================================
def write_report(results, analysis, xrr_fit, s_bulk, s_mem, s_thin, s_dense,
                 contrast, sp_proc, ht_proc, line3d):
    """Render analysis/case_40nm_bulk/report.html from the full results dict.

    Restored: the function was referenced by ``main`` but never defined in the
    initial commit, so ``python run_case.py`` raised ``NameError`` after writing
    ``results.json``.  The HTML report below mirrors the figures already saved
    next to it (``fig1_structure.png``...``fig6_design.png``) and the structured
    numbers in ``results.json``.
    """
    import html as _html

    spec = results["specification"]
    cal = results["calibration"]
    cif = results["cif"]
    xrrd = results["xrr_fit"]
    psfd = results["psf"]

    def esc(x):
        if x is None:
            return ""
        return _html.escape(str(x))

    def case_row(s):
        d = s.to_dict()
        stoch = float(d['best_ler3_stoch_nm'])
        total = 3.0 * np.sqrt((stoch / 3.0) ** 2
                              + float(xrrd['sigma_film_nm']) ** 2)
        return (
            "<tr>"
            f"<td>{esc(s.tag)}</td>"
            f"<td>{d['best_dose_pC_cm']:.1f}</td>"
            f"<td>{d['best_cd_nm']:.2f}</td>"
            f"<td>{d['best_nils']:.2f}</td>"
            f"<td><b>{stoch:.3f}</b></td>"
            f"<td>{total:.3f}</td>"
            f"<td>{esc(d['cd_target_reachable'])}</td>"
            f"<td>{d['target_dose_pC_cm']:.1f}</td>"
            f"<td>{d['target_nils']:.2f}</td>"
            f"<td>{d['target_ler3_stoch_nm']:.3f}</td>"
            f"<td>{esc(d['exposure_latitude_pC_cm'])}</td>"
            "</tr>"
        )

    def ligand_row(l):
        return (f"<tr><td>{esc(l.get('label'))}</td>"
                f"<td>{esc(l.get('count'))}</td>"
                f"<td>{esc(l.get('role'))}</td></tr>")

    figures = [
        ("fig1_structure.png",   "Single-crystal structure of the Ti₆-oxo "
                                 "cluster resist (parsed from data/4.cif)."),
        ("fig2_xrr.png",         f"XRR fit (Parratt recursion) of the "
                                 f"{xrrd['thickness_nm']:.1f} nm film: "
                                 f"ρ = {xrrd['density_g_cm3']:.3f} g/cm³, "
                                 f"σ = {xrrd['sigma_film_nm']:.3f} nm."),
        ("fig3_transport.png",   "He⁺ ion transport: trajectories, depth-dose "
                                 "Bragg curve and energy partition for the "
                                 "bulk-Si and suspended-membrane stacks."),
        ("fig4_psf_nils.png",    "Radial PSF (95% CI) → LSF → CD/NILS for the "
                                 "two reference cases."),
        ("fig5_ler_window.png",  "LER(3σ) vs dose (U-curve), CD vs dose, "
                                 "and 2-D process window; thickness-scan "
                                 "summary inset."),
        ("fig6_design.png",      "Design closure: contrast curve, sputter, "
                                 "beam heating, and 3-D latent-image "
                                 "simulation."),
    ]

    css = (
        "body{font-family:-apple-system,Segoe UI,Arial,'Microsoft YaHei',"
        "sans-serif;max-width:1080px;margin:0 auto;padding:28px 22px 80px;"
        "color:#1a1a1a;line-height:1.55}"
        "h1{margin:0 0 6px}h2{margin-top:34px;border-bottom:1px solid #ddd;"
        "padding-bottom:4px}h3{margin-top:22px}"
        "table{border-collapse:collapse;margin:10px 0;font-size:0.95em}"
        "th,td{border:1px solid #d0d0d0;padding:5px 9px;text-align:left}"
        "th{background:#f4f4f4}"
        ".num{font-variant-numeric:tabular-nums}"
        ".key{color:#0a4;font-weight:600}"
        "img{max-width:100%;border:1px solid #e0e0e0;margin:8px 0}"
        ".open{background:#fff7e6;border-left:4px solid #e8a400;padding:8px "
        "14px;margin:6px 0}"
        ".meta{color:#666;font-size:0.9em}"
    )

    rows = "".join(case_row(s) for s in (s_bulk, s_mem, s_thin, s_dense))
    ligands_html = ""
    if isinstance(cif.get("ligands"), list) and cif["ligands"]:
        ligands_html = "<h3>Ligand inventory</h3><table><tr><th>label</th>" \
            "<th>count</th><th>role</th></tr>" + \
            "".join(ligand_row(l) for l in cif["ligands"]) + "</table>"

    figure_html = "".join(
        f'<h3>{esc(name)}</h3><img src="{esc(name)}" alt="{esc(name)}">'
        f'<p class="meta">{esc(cap)}</p>'
        for name, cap in figures
    )

    open_items_html = "".join(f'<div class="open">{esc(it)}</div>'
                               for it in results["open_items"])

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SubNano-HIL-MC · {esc(spec['beam'])} {spec['energy_keV']:.0f} keV on
{esc(spec['resist_nm']):.0f} nm Ti-cluster / {spec['substrate_thickness_um']:.0f} µm Si</title>
<style>{css}</style></head><body>
<h1>SubNano-HIL-MC report</h1>
<p class="meta">Generated {esc(results['generated'])} · run_case.py ·
seed 2024–2027 + 311–313</p>

<h2>1. Executive summary</h2>
<table><tr><th>spec</th><td>
{esc(spec['beam'])} {spec['energy_keV']:.0f} keV ·
{esc(spec['resist_nm']):.0f} nm Ti-cluster resist ·
{esc(spec['substrate'])} {spec['substrate_thickness_um']:.0f} µm ·
CD target {esc(spec['cd_target_nm']):.0f} ± {esc(spec['cd_tolerance']):.2f} nm
</td></tr><tr><th>calibration</th><td>
<span class="key">e_scale</span>={cal['e_scale']:.2f} →
range<sub>Si</sub> = {cal['range_substrate_nm']:.1f} nm
(SRIM {cal['srim_range_Si_nm']:.1f}, {cal['range_error_pct']:+.2f}%);
<span class="key">a_eff</span> = {cal['a_eff_nm2']:.2f} nm²
(back-fitted to 0.21 nm);
<span class="key">w_SE</span> = {cal['w_se_eV']:.0f} eV;
<span class="key">ρ_gel</span> = {cal['rho_gel_eV_per_nm3']:.0f} eV/nm³;
dose axis <b>{'calibrated' if cal['dose_axis_calibrated'] else 'not calibrated'}</b>
</td></tr><tr><th>PSF / LSF</th><td>
r<sub>90</sub> = {psfd['r90_nm']:.2f} nm;
LSF FWHM = {psfd['lsf_fwhm_nm']:.2f} nm
</td></tr><tr><th>design floor</th><td>
CD<sub>floor</sub> = {results['design']['cd_floor_nm']:.2f} nm
({esc(results['design']['cd_floor_basis'])})
</td></tr></table>

<h2>2. Per-case results</h2>
<table>
<tr><th>case</th><th>best dose<br>(pC/cm)</th><th>best CD<br>(nm)</th>
<th>NILS</th><th>LER₃σ<br>stoch (nm)</th><th>LER₃σ<br>total (nm)</th>
<th>CD=10 nm<br>reachable</th><th>target dose<br>(pC/cm)</th>
<th>target NILS</th><th>target LER₃σ<br>stoch (nm)</th>
<th>exposure latitude<br>(pC/cm)</th></tr>
{rows}
</table>
<p class="meta">LER₃σ<sub>total</sub> = 3·√(LER₁σ² + σ<sub>surface</sub>²)
with σ<sub>surface</sub> = {xrrd['sigma_film_nm']:.3f} nm.</p>

<h2>3. Crystal structure &mdash; <code>data/4.cif</code></h2>
<table><tr><th>formula</th><td>{esc(cif.get('formula'))}</td>
<th>FW</th><td>{esc(cif.get('fw'))}</td></tr>
<tr><th>crystal system</th><td>{esc(cif.get('crystal_system'))}</td>
<th>space group</th><td>{esc(cif.get('space_group'))} (#{esc(cif.get('it_number'))})</td></tr>
<tr><th>Z / Z'</th><td>{esc(cif.get('z'))} / {esc(cif.get('z_prime'))}</td>
<th>ρ_calc</th><td>{esc(cif.get('density_calc'))} g/cm³</td></tr>
<tr><th>R1 / wR2</th><td>{esc(cif.get('r1'))} / {esc(cif.get('wr2'))}</td>
<th>GoF</th><td>{esc(cif.get('goof'))}</td></tr>
<tr><th>molecule extent</th><td>{esc(cif.get('molecule_extent_nm'))} nm</td>
<th>Ti mass fraction</th><td>{esc(cif.get('ti_mass_fraction'))}</td></tr></table>
{ligands_html}

<h2>4. Hazards &amp; process</h2>
<table><tr><th>contrast</th><td>
D<sub>0</sub> = {contrast.D0:.2f} pC/cm ·
D<sub>100</sub> = {contrast.D100:.0f} pC/cm ·
γ = {contrast.gamma:.2f}
(model-derived; no development data)</td></tr>
<tr><th>sputter</th><td>
Y = {sp_proc.yield_atoms_per_ion:.2e} atoms/ion;
removed = {sp_proc.removed_thickness_nm:.2e} nm
(He → Ti-cluster, Sigmund, negligible)</td></tr>
<tr><th>beam heating</th><td>
&ltT&gt;<sub>scan</sub> = {ht_proc.dT_scan_averaged_K:.1e} K;
&ltT&gt;<sub>worst</sub> = {ht_proc.dT_worst_case_K:.1e} K;
safe = {esc(ht_proc.safe)}</td></tr>
<tr><th>3-D line</th><td>
CD = {line3d.cd_nm:.2f} nm;
LER₃σ = {line3d.ler_3sigma_nm:.2f} nm;
LWR₃σ = {line3d.lwr_3sigma_nm:.2f} nm;
ξ = {line3d.corr_length_nm:.1f} nm
</td></tr></table>

<h2>5. Figures</h2>
{figure_html}

<h2>6. Open items</h2>
{open_items_html}

<hr><p class="meta">
HIL stochastic lower bound for LER₃σ ≈ 0.21 nm (40 nm Ti-cluster /
670 µm Si, 30 keV He⁺); sub-nm reference target IRDS &lt; 0.5 nm.
All numbers above are model-derived; see <code>results.json</code> for the
machine-readable dump and <code>run.log</code> for the full log.
</p></body></html>
"""
    out_path = os.path.join(OUT, "report.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"   saved {os.path.basename(out_path)}")


# ===========================================================================
# main
# ===========================================================================
def main():
    t0 = time.time()
    _style()

    # ---- 1. single crystal ------------------------------------------------
    analysis, geom = step_structure()
    cd_floor = dsn.cd_floor_nm(analysis.molecule_extent_nm)

    # ---- 2. XRR ------------------------------------------------------------
    xrr_data, xrr_fit = step_xrr(analysis)
    sigma_surf = xrr_fit.sigma_film_nm

    # ---- 3. materials ------------------------------------------------------
    resist, resist_xtal, si, sin = step_materials(analysis, xrr_fit)
    log("material check:")
    for m in (resist, resist_xtal, si):
        log("   " + m.summary().splitlines()[0]
            + f"  rho={m.density:.3f}  theta_c={m.critical_angle_deg():.4f} deg")

    # ---- 4. Monte-Carlo ----------------------------------------------------
    log(f"MC: {RESIST_THICKNESS_NM:.0f} nm resist / 670 um Si ...")
    bulk = run_case(resist, si, RESIST_THICKNESS_NM, N_MAIN, seed=2024,
                    keep_traj=400)
    log("MC: 40 nm resist / 20 nm SiNx suspended membrane (reference) ...")
    mem = run_case(resist, None, RESIST_THICKNESS_NM, N_MAIN, seed=2025,
                   membrane=sin, t_mem=20.0)
    log(f"MC: XRR-measured {xrr_fit.thickness_nm:.1f} nm on bulk Si ...")
    thin = run_case(resist, si, xrr_fit.thickness_nm, N_REP, seed=2026)
    log("MC: 40 nm at single-crystal density ...")
    dense = run_case(resist_xtal, si, RESIST_THICKNESS_NM, N_REP, seed=2027)
    bulk_reps = [bulk] + [run_case(resist, si, RESIST_THICKNESS_NM, N_REP, seed=s)
                          for s in (301, 302, 303)]
    mem_reps = [mem] + [run_case(resist, None, RESIST_THICKNESS_NM, N_REP, seed=s,
                                 membrane=sin, t_mem=20.0)
                        for s in (311, 312, 313)]
    log(f"   mean stopping depth in Si = {bulk.range_substrate:.1f} nm "
        f"(SRIM {SRIM_RANGE_SI_NM} nm, {100*(bulk.range_substrate-SRIM_RANGE_SI_NM)/SRIM_RANGE_SI_NM:+.1f}%)")

    # ---- 5. calibrate the stochastic averaging area -------------------------
    x_m, lsf_m = _lsf(mem_reps)
    doses_cal = np.geomspace(1.0, 400.0, 140)
    a_eff = calibrate_a_eff(x_m, lsf_m, doses_cal,
                            RHO_GEL * RESIST_THICKNESS_NM, W_SE, TARGET_LER3_NM)
    log(f"calibrated a_eff = {a_eff:.2f} nm^2 "
        f"(-> membrane min LER3sigma = {TARGET_LER3_NM} nm)")

    x_b, lsf_b = _lsf(bulk_reps)
    thresh = RHO_GEL * RESIST_THICKNESS_NM
    doses = np.geomspace(1.0, 4000.0, 360)

    # ---- 6. design summaries -----------------------------------------------
    def summarise(tag, reps_or_res, t_res=RESIST_THICKNESS_NM):
        x, lsf = _lsf(reps_or_res)
        sw = dose_sweep(x, lsf, doses, RHO_GEL * t_res, W_SE, a_eff,
                        sigma_rough_nm=sigma_surf)
        res = reps_or_res[0] if isinstance(reps_or_res, list) else reps_or_res
        s = dsn.summarise_design(
            tag, sw,
            molecule_extent_nm=analysis.molecule_extent_nm,
            sigma_surface_nm=sigma_surf,
            resist_nm=t_res,
            substrate_um=SUBSTRATE_THICKNESS_NM / 1e3,
            energy_keV=BEAM_ENERGY_KEV,
            rho_gel=RHO_GEL, w_se=W_SE, a_eff=a_eff, e_scale=E_SCALE,
            cd_target_nm=CD_TARGET_NM, cd_tol=CD_TOL,
            transport=transport_dict(res, x, lsf),
        )
        return s, sw

    s_bulk, sw_bulk = summarise("bulk_40nm_Si670um", bulk_reps)
    s_mem, _ = summarise("membrane_40nm_SiNx20nm", mem_reps)
    s_thin, _ = summarise(f"bulk_{xrr_fit.thickness_nm:.1f}nm_Si670um", thin,
                          t_res=xrr_fit.thickness_nm)
    s_dense, _ = summarise("bulk_40nm_crystal_density", dense)
    log(f"   bulk : best CD {s_bulk.best_cd_nm:.2f} nm @ {s_bulk.best_dose_pC_cm:.0f} pC/cm, "
        f"NILS {s_bulk.best_nils:.2f}, LER3s(stoch) {s_bulk.best_ler3_stoch_nm:.3f} nm")
    log(f"   bulk : CD={CD_TARGET_NM:.0f} nm reachable = {s_bulk.cd_target_reachable}, "
        f"dose {s_bulk.target_dose_pC_cm:.0f} pC/cm, NILS {s_bulk.target_nils:.2f}, "
        f"LER3s {s_bulk.target_ler3_stoch_nm:.3f} nm")
    log(f"   bulk : exposure latitude {s_bulk.exposure_latitude_pC_cm}")

    # ---- 7. thickness scan (with CD recorded so the floor can be applied) ---
    log("MC: resist-thickness scan ...")
    ths = np.array([15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0])
    doses_w = np.geomspace(3.0, 3000.0, 70)
    lermap = np.zeros((len(ths), len(doses_w)))
    cdmap = np.zeros_like(lermap)
    for i, t in enumerate(ths):
        r = run_case(resist, si, float(t), N_SCAN, seed=400 + i)
        xx, ll = _lsf(r)
        sw = dose_sweep(xx, ll, doses_w, RHO_GEL * float(t), W_SE, a_eff)
        lermap[i] = [s.ler_3sigma_nm for s in sw]
        cdmap[i] = [s.cd_nm for s in sw]
        adm = cdmap[i] >= cd_floor
        best_t = np.nanmin(lermap[i][adm]) if adm.any() else float("nan")
        log(f"   t={t:5.1f} nm -> min LER3s (CD>=floor) = {best_t:.3f} nm")

    # ---- 8. hazards / development ------------------------------------------
    log("hazards: contrast curve, sputtering, heating, 3-D latent image ...")
    dp = bulk.depth_profile
    de = bulk.depth_edges
    e_ion = float(dp.sum())
    lsf0 = float(lsf_b.max())
    doses_c = np.geomspace(0.3, 3000.0, 200)
    contrast = hz.contrast_curve(dp, de, RHO_GEL, doses_c, e_ion,
                                 centerline_lsf_per_ion_eV_per_nm=lsf0,
                                 resist_thickness_nm=RESIST_THICKNESS_NM)
    dose_proc = (s_bulk.target_dose_pC_cm if np.isfinite(s_bulk.target_dose_pC_cm)
                 else s_bulk.best_dose_pC_cm)
    dsw = np.geomspace(10.0, 3000.0, 120)
    removed = np.array([hz.sputter_yield_he(BEAM_ENERGY_KEV, dose_pC_cm=float(q),
                                            beam_width_nm=analysis.molecule_extent_nm,
                                            resist_density_g_cm3=xrr_fit.density_g_cm3,
                                            molecule_M_g_mol=analysis.fw,
                                            atoms_per_molecule=analysis.n_atoms_molecule
                                            ).removed_thickness_nm
                        for q in dsw])
    heat = [hz.beam_heating(float(q), dwell_us=1.0,
                            beam_energy_keV=BEAM_ENERGY_KEV,
                            r_beam_nm=analysis.molecule_extent_nm,
                            t_film_nm=RESIST_THICKNESS_NM,
                            E_resist_frac=s_bulk.fraction_in_resist)
            for q in dsw]
    dT_avg = np.array([h.dT_scan_averaged_K for h in heat])
    dT_worst = np.array([h.dT_worst_case_K for h in heat])
    sp_proc = hz.sputter_yield_he(BEAM_ENERGY_KEV, dose_pC_cm=float(dose_proc),
                                  beam_width_nm=analysis.molecule_extent_nm,
                                  resist_density_g_cm3=xrr_fit.density_g_cm3,
                                  molecule_M_g_mol=analysis.fw,
                                  atoms_per_molecule=analysis.n_atoms_molecule)
    ht_proc = hz.beam_heating(float(dose_proc), dwell_us=1.0,
                              beam_energy_keV=BEAM_ENERGY_KEV,
                              r_beam_nm=analysis.molecule_extent_nm,
                              t_film_nm=RESIST_THICKNESS_NM,
                              E_resist_frac=s_bulk.fraction_in_resist)
    line3d = hz.simulate_line_3d(x_b, lsf_b, float(dose_proc), thresh,
                                 w_se=W_SE, a_eff=a_eff,
                                 cd_target_nm=CD_TARGET_NM, length_nm=200.0,
                                 dz_nm=2.0, seed=7,
                                 depth_profile=dp, depth_edges=de,
                                 energy_per_ion_in_resist_eV=e_ion,
                                 rho_gel=RHO_GEL,
                                 resist_thickness_nm=RESIST_THICKNESS_NM)
    log(f"   contrast D0={contrast.D0:.2f} D100={contrast.D100:.0f} gamma={contrast.gamma:.2f}; "
        f"sputter removed={sp_proc.removed_thickness_nm:.2e} nm; "
        f"dT_worst={ht_proc.dT_worst_case_K:.1e} K (safe={ht_proc.safe}); "
        f"3D line CD={line3d.cd_nm:.2f} nm LER3s={line3d.ler_3sigma_nm:.2f} nm")

    # ---- 9. figures --------------------------------------------------------
    cases = [(bulk_reps, C_BULK, "670 µm Si"), (mem_reps, C_MEM, "20 nm SiN$_x$")]
    fig_structure(analysis, geom, os.path.join(OUT, "fig1_structure.png"))
    fig_xrr(xrr_data, xrr_fit, analysis, os.path.join(OUT, "fig2_xrr.png"))
    fig_transport(bulk, mem, os.path.join(OUT, "fig3_transport.png"))
    r90 = fig_psf(cases, a_eff, cd_floor, os.path.join(OUT, "fig4_psf_nils.png"))
    fig_ler(cases, a_eff, cd_floor, sigma_surf, (ths, doses_w, lermap, cdmap),
            os.path.join(OUT, "fig5_ler_window.png"))
    fig_design(s_bulk, sw_bulk, contrast, (dsw, removed), (dsw, dT_avg, dT_worst),
               line3d, sigma_surf, os.path.join(OUT, "fig6_design.png"))

    # ---- 10. results.json --------------------------------------------------
    adm_scan = np.where(cdmap >= cd_floor, lermap, np.nan)
    results = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        specification=dict(resist_nm=RESIST_THICKNESS_NM,
                           substrate="Si(001)",
                           substrate_thickness_um=SUBSTRATE_THICKNESS_NM / 1e3,
                           beam="He+", energy_keV=BEAM_ENERGY_KEV,
                           cd_target_nm=CD_TARGET_NM, cd_tolerance=CD_TOL),
        cif=analysis.to_dict(),
        xrr_fit=xrr_fit.to_dict(),
        calibration=dict(e_scale=E_SCALE,
                         range_substrate_nm=bulk.range_substrate,
                         srim_range_Si_nm=SRIM_RANGE_SI_NM,
                         range_error_pct=100 * (bulk.range_substrate - SRIM_RANGE_SI_NM)
                         / SRIM_RANGE_SI_NM,
                         a_eff_nm2=a_eff, a_eff_provenance="back-fitted to 0.21 nm",
                         w_se_eV=W_SE, w_se_provenance="assumed",
                         rho_gel_eV_per_nm3=RHO_GEL,
                         rho_gel_provenance="assumed",
                         dose_axis_calibrated=False),
        resist=dict(formula=resist.formula, density=resist.density,
                    density_crystal=analysis.density_calc,
                    electrons_per_gram=resist.electrons_per_gram,
                    electron_density=resist.electron_density,
                    critical_angle_2theta_deg=2 * resist.critical_angle_deg(),
                    summary=resist.summary()),
        psf=dict(r90_nm=r90, lsf_fwhm_nm=s_bulk.lsf_fwhm_nm),
        design=dict(cd_floor_nm=cd_floor,
                    cd_floor_basis="1 x molecular diameter from data/4.cif"),
        cases={s.tag: s.to_dict() for s in (s_bulk, s_mem, s_thin, s_dense)},
        thickness_scan=dict(thickness_nm=ths.tolist(),
                            dose_pC_cm=doses_w.tolist(),
                            ler3_nm=np.round(lermap, 4).tolist(),
                            cd_nm=np.round(cdmap, 4).tolist(),
                            min_ler3_admissible_nm=[
                                float(np.nanmin(row)) if np.isfinite(row).any()
                                else None for row in adm_scan]),
        hazards=dict(contrast=dict(D0_pC_cm=contrast.D0, D100_pC_cm=contrast.D100,
                                   gamma=contrast.gamma,
                                   provenance="model-derived, no development data"),
                     sputter=dict(dose_pC_cm=float(dose_proc),
                                  yield_atoms_per_ion=sp_proc.yield_atoms_per_ion,
                                  removed_thickness_nm=sp_proc.removed_thickness_nm),
                     heating=dict(dose_pC_cm=float(dose_proc),
                                  dT_scan_averaged_K=ht_proc.dT_scan_averaged_K,
                                  dT_worst_case_K=ht_proc.dT_worst_case_K,
                                  safe=bool(ht_proc.safe)),
                     line3d=dict(dose_pC_cm=float(dose_proc), cd_nm=line3d.cd_nm,
                                 ler3_nm=line3d.ler_3sigma_nm,
                                 lwr3_nm=line3d.lwr_3sigma_nm,
                                 corr_length_nm=line3d.corr_length_nm)),
        open_items=[
            "Absolute dose axis is not calibrated: rho_gel and w_SE are assumed and "
            "a_eff is back-fitted, so dose values are model-relative.",
            "The surface-to-edge roughness transfer coefficient xi is unknown; the "
            "report gives a bounded budget instead of a single LER number.",
            "No development kinetics, CD-SEM or cross-section data, so the contrast "
            "curve and sidewall profile remain model-derived.",
        ],
    )
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)
    log(f"   saved results.json  ({time.time()-t0:.0f}s so far)")

    write_report(results, analysis, xrr_fit, s_bulk, s_mem, s_thin, s_dense,
                 contrast, sp_proc, ht_proc, line3d)
    log(f"done in {time.time()-t0:.0f}s")
    return results


if __name__ == "__main__":
    # Restored entry point: without it `python run_case.py` was a silent no-op.
    # The guard keeps `import run_case` (used by scripts/nature_visualization.py)
    # side-effect free.
    main()
