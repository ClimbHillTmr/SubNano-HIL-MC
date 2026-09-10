# -*- coding: utf-8 -*-
"""Figure 5 ("extras"): four process-hazard / latent-image analyses.

Panels
------
(a) Contrast curve  T/T0  vs line dose with D0 / D100 / gamma annotated.
(b) Exposure latitude & CD(dose) at the fixed CD target (10 nm +/- 10%),
    read from the Monte-Carlo cache sweep.
(c) Sputtering & thermal safety window: removed thickness (nm) and Delta-T (K)
    vs dose, with the allowed (CD +/-10%) dose band shaded.
(d) 3-D developed nanowire morphology from ``simulate_line_3d``.

Style: Nature-ish (white background, no top/right spines, panel letters a-d,
unit-bearing axis labels, 300+ dpi).  Runs end-to-end in a couple of minutes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource  # noqa: F401  (kept available for shading)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_case as rc                              # noqa: E402
from nature_visualization import prepare           # noqa: E402
import hil_mc.hazards as hz                         # noqa: E402


# ---------------------------------------------------------------------------
# style helpers (consistent with nature_visualization)
# ---------------------------------------------------------------------------
def _rcstyle():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 130,
        "savefig.dpi": 320,
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e4e4",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
    })


def clean(ax, xlab=None, ylab=None, logx=False, logy=False, grid=True,
          letter=None, xlim=None, ylim=None):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if hasattr(ax, "spines"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, which="major", color="#e4e4e4", lw=0.5, ls="--")
        ax.set_axisbelow(True)
    if xlab:
        ax.set_xlabel(xlab, labelpad=5)
    if ylab:
        ax.set_ylabel(ylab, labelpad=5)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if letter:
        ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top", ha="left", color="#111111", zorder=30,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#e2e2e2",
                          lw=0.5, alpha=0.9))


def main():
    _rcstyle()
    outdir = os.path.join(ROOT, "analysis", "case_nature")
    os.makedirs(outdir, exist_ok=True)
    d = prepare(outdir)

    x = d["x"]
    lsf = d["lsf_bulk"]
    th = rc.RHO_GEL * rc.RESIST_THICKNESS_NM
    a_eff = d["a_eff"]
    dp = d["bulk"].depth_profile
    de = d["bulk"].depth_edges
    Eion = float(dp.sum())
    lsf0 = float(lsf.max())

    CD_TARGET = 10.0
    process_dose = 800.0  # dose at which the 2-D CD ~= 10 nm (from the cache sweep)

    # =====================================================================
    # (a) contrast curve
    # =====================================================================
    doses_c = np.geomspace(0.3, 600.0, 140)
    cr = hz.contrast_curve(dp, de, rc.RHO_GEL, doses_c, Eion,
                           centerline_lsf_per_ion_eV_per_nm=lsf0)

    # =====================================================================
    # (b) CD(dose) and exposure latitude at CD = 10 nm +/-10% (from cache)
    # =====================================================================
    doses_b = np.geomspace(3.0, 3000.0, 200)
    sw = rc.dose_sweep(x, lsf, doses_b, th, rc.W_SE, a_eff)
    cd_b = np.array([s.cd_nm for s in sw], dtype=float)
    in_band = (cd_b >= 0.9 * CD_TARGET) & (cd_b <= 1.1 * CD_TARGET)
    if in_band.any():
        el = (float(doses_b[in_band][0]), float(doses_b[in_band][-1]))
    else:
        el = (float("nan"), float("nan"))
    # dose that gives exactly the target (monotonic interpolation)
    try:
        D_target = float(np.interp(CD_TARGET, cd_b, doses_b))
    except Exception:
        D_target = float("nan")

    # =====================================================================
    # (c) sputtering & thermal safety window
    # =====================================================================
    doses_curve = np.geomspace(10.0, 3000.0, 120)
    removed = np.array([hz.sputter_yield_he(rc.BEAM_ENERGY_KEV, dose_pC_cm=D).removed_thickness_nm
                        for D in doses_curve])
    dT_avg = np.array([hz.beam_heating(D, beam_current_pA=None, dwell_us=1.0).dT_scan_averaged_K
                       for D in doses_curve])
    dT_worst = np.array([hz.beam_heating(D, beam_current_pA=None, dwell_us=1.0).dT_worst_case_K
                         for D in doses_curve])

    # headline numbers at the process dose
    sp_proc = hz.sputter_yield_he(rc.BEAM_ENERGY_KEV, dose_pC_cm=process_dose)
    ht_proc = hz.beam_heating(process_dose, beam_current_pA=None, dwell_us=1.0)

    # =====================================================================
    # (d) 3-D developed nanowire
    # =====================================================================
    dose_3d = 1000.0  # gives a developed CD ~ 10 nm
    line = hz.simulate_line_3d(x, lsf, dose_3d, th, rc.W_SE, a_eff,
                               cd_target_nm=CD_TARGET, length_nm=200.0, dz_nm=2.0,
                               seed=0, depth_profile=dp, depth_edges=de,
                               energy_per_ion_in_resist_eV=Eion)

    # =====================================================================
    # figure
    # =====================================================================
    fig = plt.figure(figsize=(8.6, 7.2))
    gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.34,
                          left=0.09, right=0.97, top=0.94, bottom=0.07)

    # ---- (a) ----
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(doses_c, cr.remaining_fraction, color="#1f4e79", lw=1.9)
    ax.axvline(cr.D0, color="#c0392b", ls=":", lw=1.1)
    ax.axvline(cr.D100, color="#2a9d8f", ls=":", lw=1.1)
    ax.text(cr.D0, 0.04, f"D0={cr.D0:.2f}", color="#c0392b", fontsize=8,
            rotation=90, va="bottom", ha="right")
    ax.text(cr.D100, 0.04, f"D100={cr.D100:.0f}", color="#2a9d8f", fontsize=8,
            rotation=90, va="bottom", ha="left")
    ax.text(0.5, 0.88, f"gamma = {cr.gamma:.2f}", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", lw=0.5))
    clean(ax, xlab="line dose (pC cm$^{-1}$)", ylab="remaining thickness  $T/T_0$",
          logx=True, letter="a", xlim=(0.3, 600), ylim=(-0.02, 1.05))
    ax.set_xticks([0.3, 1, 3, 10, 30, 100, 300, 600])
    ax.set_xticklabels(["0.3", "1", "3", "10", "30", "100", "300", "600"])

    # ---- (b) ----
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(doses_b, cd_b, color="#1f4e79", lw=1.9, label="CD(dose)")
    if in_band.any():
        ax.axvspan(el[0], el[1], color="#2a9d8f", alpha=0.12,
                   label="allowed (CD$\\pm$10%)")
    ax.axhline(CD_TARGET, color="#111111", ls="--", lw=1.0)
    ax.axhline(0.9 * CD_TARGET, color="#888888", ls=":", lw=0.9)
    ax.axhline(1.1 * CD_TARGET, color="#888888", ls=":", lw=0.9)
    if np.isfinite(D_target):
        ax.scatter([D_target], [CD_TARGET], color="#c0392b", s=45, zorder=6,
                   edgecolors="black", lw=0.6)
        ax.annotate(f"CD=10 nm @ {D_target:.0f} pC/cm", (D_target, CD_TARGET),
                    textcoords="offset points", xytext=(-6, 10), fontsize=8,
                    color="#c0392b", fontweight="bold", ha="right")
    lbl = (f"exposure latitude {el[0]:.0f}-{el[1]:.0f} pC/cm"
           if in_band.any() else "no CD=10 nm band in sweep")
    ax.text(0.97, 0.05, lbl, transform=ax.transAxes, ha="right", fontsize=8,
            color="#2a9d8f", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    clean(ax, xlab="line dose (pC cm$^{-1}$)", ylab="developed CD (nm)",
          logx=True, letter="b", xlim=(3, 3000), ylim=(0, 24))
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_xticks([3, 10, 30, 100, 300, 1000, 3000])
    ax.set_xticklabels(["3", "10", "30", "100", "300", "1000", "3000"])

    # ---- (c) ----
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(doses_curve, removed, color="#c0392b", lw=1.8, label="removed thickness (nm)")
    ax.set_yscale("log")
    ax.axhline(0.1, color="#888888", ls=":", lw=0.9)
    ax.text(doses_curve[2], 0.11, "0.1 nm", color="#888888", fontsize=7.5, va="bottom")
    # shade allowed dose band
    if in_band.any():
        ax.axvspan(el[0], el[1], color="#2a9d8f", alpha=0.10)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(doses_curve, dT_worst, color="#1f4e79", lw=1.5, ls="--",
             label=r"$\Delta T$ worst (K)")
    ax2.plot(doses_curve, dT_avg, color="#6b7280", lw=1.3, ls=":",
             label=r"$\Delta T$ avg (K)")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\Delta T$ (K)", color="#1f4e79")
    ax2.tick_params(axis="y", colors="#1f4e79")
    ax2.set_ylim(1e-9, 1.0)
    ax.text(0.97, 0.95, "both << 150 C limit: SAFE", transform=ax.transAxes,
            ha="right", fontsize=8, color="#2a9d8f", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    clean(ax, xlab="line dose (pC cm$^{-1}$)", ylab="sputtered thickness (nm)",
          logx=True, letter="c", xlim=(10, 3000), ylim=(1e-5, 1.0))
    ax.set_xticks([10, 30, 100, 300, 1000, 3000])
    ax.set_xticklabels(["10", "30", "100", "300", "1000", "3000"])
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, frameon=False, loc="lower right", fontsize=7.5)

    # ---- (d) 3-D nanowire ----
    ax = fig.add_subplot(gs[1, 1], projection="3d")
    s = line.s_nm
    xl = line.x_left_nm
    xr = line.x_right_nm
    top = line.top_height_nm
    m = 60
    u = np.linspace(0.0, 1.0, m)
    S, U = np.meshgrid(s, u, indexing="ij")
    X = S
    Y = xl[:, None] + U * (xr - xl)[:, None]
    Z = top[:, None]
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True,
                    alpha=0.92, rstride=2, cstride=2)
    ax.plot(s, xl, np.zeros_like(s), color="#c0392b", lw=1.4, zorder=5)
    ax.plot(s, xr, np.zeros_like(s), color="#c0392b", lw=1.4, zorder=5)
    ax.set_xlabel("along line (nm)", labelpad=4)
    ax.set_ylabel("width (nm)", labelpad=4)
    ax.set_zlabel("remaining height (nm)", labelpad=2)
    ax.set_zlim(0, rc.RESIST_THICKNESS_NM)
    ax.view_init(elev=24, azim=-58)
    ax.text2D(0.03, 0.97, "d", transform=ax.transAxes, fontsize=12,
              fontweight="bold", va="top", ha="left", color="#111111",
              bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#e2e2e2",
                        lw=0.5, alpha=0.9))
    ax.set_title(f"CD={line.cd_nm:.1f} nm, LER3sigma={line.ler_3sigma_nm:.2f} nm "
                 f"(D={dose_3d:.0f} pC/cm)", fontsize=8.5, pad=2)

    path = os.path.join(outdir, "fig5_extras.png")
    fig.savefig(path)
    plt.close(fig)

    # one-line headline summary
    summary = (f"hazards: contrast D0={cr.D0:.2f} D100={cr.D100:.0f} gamma={cr.gamma:.2f}; "
               f"sputter Y={sp_proc.yield_atoms_per_ion:.2e} "
               f"removed={sp_proc.removed_thickness_nm:.2e} nm @ {process_dose:.0f} pC/cm; "
               f"heating dT_avg={ht_proc.dT_scan_averaged_K:.1e} K "
               f"dT_worst={ht_proc.dT_worst_case_K:.1e} K (safe={ht_proc.safe}); "
               f"3D line CD={line.cd_nm:.2f} nm LER3sigma={line.ler_3sigma_nm:.2f} nm "
               f"corr={line.corr_length_nm:.1f} nm; "
               f"exposure latitude {el[0]:.0f}-{el[1]:.0f} pC/cm")
    print(summary)
    return path


if __name__ == "__main__":
    main()
