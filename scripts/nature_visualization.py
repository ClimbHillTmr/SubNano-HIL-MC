# -*- coding: utf-8 -*-
"""
Nature-style re-visualization of the SubNano-HIL-MC results.

Reproduces the exact (seeded, deterministic) Monte-Carlo data by importing
``run_case`` (its ``__main__`` guard prevents the default figures from being
regenerated), recalibrates the stochastic averaging area ``a_eff`` exactly as
the live pipeline does, then renders four publication-grade figures.

Design goals (this revision):
  * no overlapping text / legend / markers;
  * high resolution (400 dpi), larger fonts and contrast;
  * a correct and readable Monte-Carlo *injection* panel — the trajectory
    tuples are (x, y, z) with **z = depth**
    the figure now uses the true
    depth axis and shows a lateral-vs-depth trajectory-density map.

Outputs (to ``analysis/case_nature/``):
    fig1_structure.png, fig2_stack_energy.png, fig3_psf_nils.png,
    fig4_ler_window.png, _cache.pkl (internal MC-array cache).
"""
from __future__ import annotations
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
from scipy.ndimage import gaussian_filter, gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import run_case as rc                          # noqa: E402

# ---------------------------------------------------------------------------
# Palette (consistent across every figure)
# ---------------------------------------------------------------------------
C_BULK = "#1f4e79"      # 40 nm / 670 um Si            deep blue
C_MEM = "#2a9d8f"       # 40 nm / 20 nm SiNx membrane  teal-green
C_REF = "#6b7280"       # reference lines / grey text
C_ACC = "#c0392b"       # highlight / brick red
C_SUB = "#94a3b8"       # substrate accent / slate grey


def _rcstyle():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 130,
        "savefig.dpi": 400,
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
    """Nature-style axis dressing
    optional bold panel letter top-left inside."""
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
        ax.text(0.025, 0.965, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left", color="#111111", zorder=30,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#e2e2e2",
                          lw=0.5, alpha=0.85))


# ---------------------------------------------------------------------------
# Data preparation (mirrors run_case.main; deterministic thanks to fixed seeds)
# ---------------------------------------------------------------------------
def prepare(outdir):
    cache = os.path.join(outdir, "_cache.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    rc.log("nature: parsing 4.cif ...")
    analysis, geom = rc.step_structure()
    resist, resist_xtal, si, sin, pmma = rc.step_materials(analysis)

    rc.log("nature: MC 40 nm / 670 um Si (keep_traj=400) ...")
    bulk = rc.run_case(resist, si, rc.RESIST_THICKNESS_NM, 60000, seed=2024, keep_traj=400)
    rc.log("nature: MC 40 nm / 20 nm SiNx membrane ...")
    mem = rc.run_case(resist, None, rc.RESIST_THICKNESS_NM, 60000, seed=2025,
                      membrane=sin, t_mem=20.0)

    bulk_reps = [bulk] + [rc.run_case(resist, si, rc.RESIST_THICKNESS_NM, 20000, seed=s)
                          for s in (301, 302, 303)]
    mem_reps = [mem] + [rc.run_case(resist, None, rc.RESIST_THICKNESS_NM, 20000, seed=s,
                                    membrane=sin, t_mem=20.0) for s in (311, 312, 313)]

    x, lsf_mem = rc._lsf(mem)
    lsf_mem = rc.smooth_tail(x, lsf_mem, r_min=8.0)
    doses_cal = np.geomspace(1.0, 400.0, 140)
    a_eff = rc.calibrate_a_eff(x, lsf_mem, doses_cal,
                               rc.RHO_GEL * rc.RESIST_THICKNESS_NM,
                               rc.W_SE, rc.TARGET_LER3_NM)
    x, lsf_bulk = rc._lsf(bulk)
    lsf_bulk = rc.smooth_tail(x, lsf_bulk, r_min=8.0)

    ths = np.array([15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0])
    doses_w = np.geomspace(3.0, 300.0, 60)
    lermap = np.zeros((len(ths), len(doses_w)))
    for i, t in enumerate(ths):
        r_ = rc.run_case(resist, si, float(t), 20000, seed=400 + i)
        xx, ll = rc.converged_lsf([r_.exposure_map], r_.pixel_nm)
        ll = rc.smooth_tail(xx, ll, r_min=8.0)
        sw = rc.dose_sweep(xx, ll, doses_w, rc.RHO_GEL * float(t), rc.W_SE, a_eff)
        lermap[i] = [s.ler_3sigma_nm for s in sw]
        rc.log(f"nature: t={t:5.1f} nm -> min LER3σ = {np.nanmin(lermap[i]):.3f} nm")

    data = dict(
        analysis=analysis, geom=geom, bulk=bulk, mem=mem,
        bulk_reps=bulk_reps, mem_reps=mem_reps, a_eff=a_eff,
        ths=ths, doses_w=doses_w, lermap=lermap, x=x,
        lsf_bulk=lsf_bulk, lsf_mem=lsf_mem,
    )
    with open(cache, "wb") as fh:
        pickle.dump(data, fh)
    return data


def _bootstrap(res_reps):
    return rc.bootstrap_psf([m.exposure_map for m in res_reps],
                            res_reps[0].pixel_nm, n_boot=40)


def _nils_curve(res_reps, thresh, a_eff):
    x, lsf = rc._lsf(res_reps)
    lsf = rc.smooth_tail(x, lsf, r_min=8.0)
    doses = np.geomspace(1.0, 400.0, 160)
    sw = rc.dose_sweep(x, lsf, doses, thresh, rc.W_SE, a_eff)
    cd = np.array([s.cd_nm for s in sw])
    nils = np.array([s.nils for s in sw])
    return doses, cd, nils


# ===========================================================================
# FIGURE 1 — single-crystal structure (from 4.cif)
# ===========================================================================
def fig_structure(d, path):
    _rcstyle()
    xyz, els, labs = d["geom"]
    a = d["analysis"]
    fig = plt.figure(figsize=(8.0, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.9, 1.05], wspace=0.2,
                          left=0.03, right=0.99, top=0.80, bottom=0.13)

    # (a) 3-D molecule — de-cluttered: hide H, shrink C, emphasise Ti6 + O
    ax = fig.add_subplot(gs[0], projection="3d")
    col = {"Ti": C_BULK, "O": C_ACC, "C": "#6a6a6a", "H": "#d0d0d0"}
    pos = xyz - xyz.mean(axis=0)
    show = {"H": 0.0, "C": 6, "O": 20, "Ti": 70}
    alpha = {"H": 0.0, "C": 0.35, "O": 0.85, "Ti": 1.0}
    for e in ["H", "C", "O", "Ti"]:
        m = np.array([x == e for x in els])
        if e == "H":
            continue
        ax.scatter(pos[m, 0], pos[m, 1], pos[m, 2], s=show[e], c=col[e],
                   alpha=alpha[e], edgecolors="none", depthshade=False, label=e)
    ti = np.array([x == "Ti" for x in els])
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
    ax.set_title("a", loc="left", fontsize=13, fontweight="bold", color="#111111", pad=1)
    leg = ax.legend(loc="lower left", frameon=True, fontsize=8, framealpha=0.92,
                    edgecolor="#cccccc", borderpad=0.5, handletextpad=0.4,
                    labelspacing=0.25)
    leg.set_zorder(20)

    # (b) Ti6 core (2-D projection)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_aspect("equal")
    ti_idx = [i for i, e in enumerate(els) if e == "Ti"]
    p2 = pos[ti_idx][:, :2]
    ax2.scatter(p2[:, 0], p2[:, 1], s=300, c=C_BULK, edgecolors="white", lw=1.4, zorder=5)
    for i, (x_, y_) in enumerate(p2):
        ax2.text(x_, y_, f"Ti{i+1}", ha="center", va="center", color="white",
                 fontsize=8, fontweight="bold", zorder=6)
    for i in range(len(p2)):
        for j in range(i + 1, len(p2)):
            if np.linalg.norm(pos[ti_idx[i]] - pos[ti_idx[j]]) < 4.2:
                ax2.plot(*zip(p2[i], p2[j]), color=C_BULK, lw=1.6, zorder=4)
    ax2.margins(0.16)   # keep corner Ti markers fully inside the frame
    ax2.text(0.04, 0.965, f"d(Ti–Ti) = {min(a.ti_ti_distances):.2f}–{max(a.ti_ti_distances):.2f} Å",
             transform=ax2.transAxes, fontsize=7.5, color=C_REF, va="top",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", lw=0.5))
    clean(ax2, xlab="x (Å)", ylab="y (Å)", grid=False)
    ax2.set_title("b", loc="left", fontsize=13, fontweight="bold", color="#111111", pad=1)

    # (c) crystal data table
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    rows = [
        ["Formula", a.formula],
        ["M, g mol$^{-1}$", f"{a.fw:.2f}"],
        ["Crystal system", f"{a.crystal_system}, {a.space_group} (#{a.it_number})"],
        ["a, b, c, Å", f"{a.cell[0]:.4f}, {a.cell[1]:.4f}, {a.cell[2]:.4f}"],
        ["α, β, γ, °", f"{a.cell[3]:.3f}, {a.cell[4]:.3f}, {a.cell[5]:.3f}"],
        ["V, Å$^3$ / Z / Z′", f"{a.volume:.2f} / {a.z} / {a.z_prime:.2f}"],
        ["ρ(calc), g cm$^{-3}$", f"{a.density_calc:.3f}"],
        ["T, K / λ, Å", f"{a.temperature_K:.0f} / {a.wavelength:.5f}"],
        ["θmax / complet.", f"{a.theta_max:.1f}° / {a.completeness:.3f}"],
        ["R1 / wR2 / GooF", f"{a.r1:.4f} / {a.wr2:.4f} / {a.goof:.3f}"],
        ["Ti, wt%", f"{100*a.ti_mass_fraction:.2f}"],
        ["Ø / Rg, nm", f"{a.molecule_extent_nm:.2f} / {a.rg_nm:.2f}"],
    ]
    tbl = ax3.table(cellText=rows, colLabels=["Item", "Value"], cellLoc="left",
                    loc="center", colWidths=[0.4, 0.6])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.4)
    tbl.scale(1, 1.24)
    for (r_, c_), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfcfcf")
        cell.set_linewidth(0.4)
        cell.set_facecolor("white")
        if r_ == 0:
            cell.set_text_props(color="#111111", fontweight="bold")
    clean(ax3, grid=False)
    ax3.set_title("c", loc="left", fontsize=13, fontweight="bold", color="#111111", pad=10)

    fig.suptitle("Ti$_6$-oxo cluster single crystal (data_zzh-hpf-js_auto)",
                 fontsize=12.5, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path)
    plt.close(fig)
    rc.log(f"nature: saved {os.path.basename(path)}")


# ===========================================================================
# FIGURE 2 — Monte-Carlo injection, depth profile, energy budget
# ===========================================================================
def fig_stack(d, path):
    _rcstyle()
    bulk, mem = d["bulk"], d["mem"]
    fig = plt.figure(figsize=(8.2, 3.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.92, 0.92, 0.8], wspace=0.3,
                          left=0.05, right=0.995, top=0.86, bottom=0.17)

    # (a) Monte-Carlo injection: trajectory-density map (lateral x vs depth z)
    ax = fig.add_subplot(gs[0])
    trajs = [t for t in bulk.trajectories if len(t) > 2]
    xs = np.concatenate([t[:, 0] for t in trajs])   # lateral x (nm)
    zs = np.concatenate([t[:, 2] for t in trajs])   # depth z (nm)
    xb = np.linspace(-60, 60, 121)
    zb = np.linspace(0, 300, 301)
    H, xe, ze = np.histogram2d(xs, zs, bins=[xb, zb])
    cmap = LinearSegmentedColormap.from_list("inj", ["#ffffff", "#9ecbe1", C_BULK])
    ax.pcolormesh(xe, ze, np.maximum(H.T, 1), cmap=cmap,
                  norm=LogNorm(vmin=1, vmax=max(H.max(), 10)), shading="auto", alpha=0.95)
    ax.axhline(0, color="#8d6e63", lw=1.4)
    ax.axhline(rc.RESIST_THICKNESS_NM, color=C_ACC, lw=1.4, ls="--")
    for t in trajs[:12]:
        ax.plot(t[:, 0], t[:, 2], lw=0.5, alpha=0.55, color="#0b2d4d", zorder=5)
    ax.set_xlim(-60, 60)
    ax.set_ylim(300, -8)
    clean(ax, xlab="lateral x (nm)", ylab="depth z (nm)", grid=False, letter="a")
    ax.text(58, 20, "resist", ha="right", va="center", fontsize=8, color="#8d6e63",
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                         ec="none", alpha=0.75))
    ax.text(58, 190, "Si 670 µm", ha="right", va="center", fontsize=8, color=C_SUB,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                         ec="none", alpha=0.75))

    # (b) in-resist depth profile (smoothed, peak-normalised)
    ax2 = fig.add_subplot(gs[1])
    zc = 0.5 * (bulk.depth_edges[1:] + bulk.depth_edges[:-1])
    in_res = zc <= rc.RESIST_THICKNESS_NM
    pb = gaussian_filter1d(bulk.depth_profile, 2.2)
    pm = gaussian_filter1d(mem.depth_profile, 2.2)
    n_pk = pb[in_res].max() or 1.0
    m_pk = pm[in_res].max() or 1.0
    ax2.fill_betweenx(zc, 0, pb / n_pk, color=C_BULK, alpha=0.16, lw=0)
    ax2.fill_betweenx(zc, 0, pm / m_pk, color=C_MEM, alpha=0.13, lw=0)
    ax2.plot(pb / n_pk, zc, color=C_BULK, lw=1.8, label="670 µm Si")
    ax2.plot(pm / m_pk, zc, color=C_MEM, lw=1.8, ls="--", label="20 nm SiN$_x$")
    ax2.set_ylim(40, 0)
    clean(ax2, xlab="deposited energy (arb.)", ylab="depth z (nm)",
          grid=True, letter="b", ylim=(40, 0))
    ax2.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=8.5)

    # (c) energy budget (log scale so the 0.1% backscatter is visible)
    ax3 = fig.add_subplot(gs[2])
    labels = ["resist", "sub\n<24nm", "deep\nsub", "back\nscat"]
    e_res = bulk.energy_in_resist
    e_sub = bulk.energy_in_substrate_total
    e_deep = rc.BEAM_ENERGY_KEV * 1e3 - e_res - e_sub - bulk.energy_backscattered_out
    vals = [e_res, e_sub, max(e_deep, 0.1), bulk.energy_backscattered_out]
    cols = [C_BULK, C_ACC, C_SUB, "#c62828"]
    b = ax3.bar(labels, vals, color=cols, width=0.62, edgecolor="white", lw=0.6)
    ax3.set_yscale("log")
    ax3.set_ylim(1e1, 4e4)
    clean(ax3, ylab="eV per ion", grid=True, letter="c")
    ax3.grid(True, axis="y", which="both", color="#e4e4e4", lw=0.5, ls="--")
    ax3.tick_params(axis="x", labelsize=8)
    for rect, v in zip(b, vals):
        ax3.text(rect.get_x() + rect.get_width() / 2, max(v, 30) * 1.18,
                 f"{100*v/(rc.BEAM_ENERGY_KEV*1e3):.1f}%", ha="center",
                 fontsize=8, fontweight="bold")

    # (d) backscattered-ion exit-radius distribution
    ax4 = fig.add_subplot(gs[3])
    r = np.asarray(bulk.backscatter_radii)
    if len(r) > 8:
        ax4.hist(r, bins=18, color="#c62828", alpha=0.85, edgecolor="white", lw=0.4)
        ax4.set_yscale("log")
        ax4.set_ylim(0.5, None)
        ax4.axvline(np.mean(r), color="#333333", ls=":", lw=1.2)
        ax4.text(0.95, 0.88, f"$\\bar{{r}}$ ≈ {np.mean(r):.0f} nm",
                 transform=ax4.transAxes, fontsize=8, color="#333333",
                 ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", lw=0.4))
    clean(ax4, xlab="exit radius (nm)", ylab="counts", grid=True, letter="d")
    ax4.set_title(f"η = {100*bulk.backscatter_fraction:.2f}%", fontsize=9,
                  fontweight="bold", pad=3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path)
    plt.close(fig)
    rc.log(f"nature: saved {os.path.basename(path)}")


# ===========================================================================
# FIGURE 3 — PSF (bootstrap CI), LSF decomposition, NILS
# ===========================================================================
def fig_psf(d, path):
    _rcstyle()
    cases = [(d["bulk_reps"], C_BULK, "670 µm Si"),
             (d["mem_reps"], C_MEM, "20 nm SiN$_x$")]
    thresh = rc.RHO_GEL * rc.RESIST_THICKNESS_NM
    fig = plt.figure(figsize=(7.8, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 1.0], wspace=0.3,
                          left=0.095, right=0.995, top=0.86, bottom=0.17)

    # (a) radial PSF + bootstrap CI
    ax = fig.add_subplot(gs[0])
    for reps, col, lab in cases:
        r, mean, lo, hi = _bootstrap(reps)
        ax.fill_between(r, lo, hi, color=col, alpha=0.14, lw=0)
        ax.plot(r, mean, color=col, lw=1.8, label=lab)
    ax.axvline(1.62, color=C_REF, ls=":", lw=1.1)
    ax.text(4.4, 5e-6, "r$_{90}$ = 1.62 nm", color=C_REF, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", lw=0.4))
    clean(ax, xlab="radius r (nm)", ylab="PSF (eV nm$^{-2}$ ion$^{-1}$)",
          logy=True, grid=True, letter="a", xlim=(0, 45), ylim=(1e-6, 2))
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.text(0.97, 0.55, "curves nearly identical\n(substrate-independent)",
            transform=ax.transAxes, ha="right", va="center", fontsize=8, color=C_REF)

    # (b) LSF decomposition
    ax2 = fig.add_subplot(gs[1])
    for reps, col, lab in cases:
        res = reps[0]
        x, lsf = rc._lsf(reps)
        peak = lsf[len(lsf) // 2]
        lsx, lsub = rc.lsf_from_map(res.substrate_map, res.pixel_nm)
        ax2.plot(x, lsf / peak, color=col, lw=1.8, label=f"{lab} — total")
        ax2.plot(lsx, lsub / peak, color=col, lw=1.3, ls="--",
                 label=f"{lab} — substrate SE$_2$")
    clean(ax2, xlab="|x| (nm)", ylab="LSF / LSF(0)", logy=True, grid=True,
          letter="b", xlim=(0, 35), ylim=(1e-5, 2))
    ax2.legend(frameon=False, loc="lower left", fontsize=7.8)

    # (c) NILS vs developed CD (lightly smoothed to remove discretisation steps)
    ax3 = fig.add_subplot(gs[2])
    for reps, col, lab in cases:
        _, cd, nils = _nils_curve(reps, thresh, d["a_eff"])
        ok = np.isfinite(nils) & np.isfinite(cd)
        cd, nils = cd[ok], nils[ok]
        s = np.argsort(cd)
        nils_s = gaussian_filter1d(nils[s], 1.2)
        ax3.plot(cd[s], nils_s, color=col, lw=1.8, label=lab)
    clean(ax3, xlab="developed CD (nm)", ylab="NILS", grid=True, letter="c",
          xlim=(0, 9), ylim=(0, 8.5))
    ax3.legend(frameon=False, loc="upper left", fontsize=9)
    ax3.annotate("PSF-tail\nroll-off", xy=(7.9, 2.0), xytext=(5.6, 0.5),
                 fontsize=8, color=C_REF,
                 arrowprops=dict(arrowstyle="->", color=C_REF, lw=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path)
    plt.close(fig)
    rc.log(f"nature: saved {os.path.basename(path)}")


# ===========================================================================
# FIGURE 4 — LER U-curve, CD(dose), process-window heatmap
# ===========================================================================
def fig_ler(d, path):
    _rcstyle()
    cases = [("b", d["bulk_reps"], C_BULK, "670 µm Si"),
             ("m", d["mem_reps"], C_MEM, "20 nm SiN$_x$")]
    thresh = rc.RHO_GEL * rc.RESIST_THICKNESS_NM
    fig = plt.figure(figsize=(7.8, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.0, 1.08], wspace=0.3,
                          left=0.085, right=0.995, top=0.86, bottom=0.17)

    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    best = {}
    for key, reps, col, lab in cases:
        x, lsf = rc._lsf(reps)
        lsf = rc.smooth_tail(x, lsf, r_min=8.0)
        doses_ = np.geomspace(1.0, 400.0, 160)
        sw = rc.dose_sweep(x, lsf, doses_, thresh, rc.W_SE, d["a_eff"])
        ler = np.array([s.ler_3sigma_nm for s in sw], dtype=float)
        cd = np.array([s.cd_nm for s in sw])
        ax.plot(doses_, ler, color=col, lw=1.8, label=lab)
        ax2.plot(doses_, cd, color=col, lw=1.8, label=lab)
        ok = np.isfinite(ler) & (ler > 0)
        i = int(np.argmin(np.where(ok, ler, np.inf)))
        best[key] = dict(dose=float(doses_[i]), ler=float(ler[i]), cd=float(cd[i]))
        ax.scatter([doses_[i]], [ler[i]], color=col, s=50, zorder=6,
                   edgecolors="black", lw=0.7)
        if key == "b":
            xytxt, ha = (-12, 22), "right"     # above-left of the dot
        else:
            xytxt, ha = (-12, -34), "right"    # below-left of the dot
        ax.annotate(f"{ler[i]:.2f} nm\n{doses_[i]:.0f} pC/cm",
                    (doses_[i], ler[i]), textcoords="offset points",
                    xytext=xytxt, ha=ha, fontsize=7.5, color=col,
                    fontweight="bold")

    ax.axhline(rc.TARGET_LER3_NM, color=C_REF, ls=":", lw=1.2)
    ax.text(1.3, 0.205, "IRDS target 0.21 nm", ha="left", va="top",
            fontsize=7.5, color=C_REF)
    clean(ax, xlab="line dose (pC cm$^{-1}$)", ylab="LER 3σ (nm)",
          logx=True, logy=True, grid=True, letter="a", ylim=(0.12, 0.6))
    ax.yaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.25, 0.3, 0.4, 0.5]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])

    clean(ax2, xlab="line dose (pC cm$^{-1}$)", ylab="developed CD (nm)",
          logx=True, grid=True, letter="b", ylim=(0.4, 9))
    ax2.legend(frameon=False, loc="upper left", fontsize=9)
    ax2.set_xticks([1, 3, 10, 30, 100, 300])
    ax2.set_xticklabels(["1", "3", "10", "30", "100", "300"])

    # (c) process window
    ax3 = fig.add_subplot(gs[2])
    th, dw, lm = d["ths"], d["doses_w"], d["lermap"]
    lm_c = np.clip(np.nan_to_num(lm), 0.1, 1.2)
    xx, yy = np.meshgrid(dw, th)
    im = ax3.pcolormesh(xx, yy, lm_c, cmap="viridis_r", shading="gouraud",
                        norm=LogNorm(vmin=0.15, vmax=1.2))
    lm_s = 10 ** gaussian_filter(np.log10(lm_c), 0.9)
    cs = ax3.contour(xx, yy, lm_s, levels=[0.20, 0.25, 0.30, 0.40, 0.55],
                     colors="white", linewidths=0.9)
    ax3.clabel(cs, fmt="%.2f", fontsize=7, inline=True, colors="white")
    ax3.axhline(40, color=C_ACC, lw=1.3, ls="--")
    ax3.text(dw[-1] * 0.95, 42, "40 nm (spec.)", color=C_ACC, fontsize=8,
             fontweight="bold", ha="right",
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    ax3.axhline(23.7, color="#0b7a9c", lw=1.1, ls=":")
    ax3.text(dw[-1] * 0.95, 25.5, "23.7 nm (XRR)", color="#0b7a9c", fontsize=8,
             ha="right",
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    cb = fig.colorbar(im, ax=ax3, pad=0.02)
    cb.set_label("LER 3σ (nm)", fontsize=9)
    cb.set_ticks([0.2, 0.3, 0.5, 1.0])
    cb.set_ticklabels(["0.2", "0.3", "0.5", "1.0"])
    cb.ax.yaxis.set_minor_formatter(NullFormatter())   # kill stray log-minor labels
    cb.ax.tick_params(labelsize=8)
    clean(ax3, xlab="line dose (pC cm$^{-1}$)", ylab="resist thickness (nm)",
          logx=True, grid=False, letter="c", ylim=(10, 90))
    ax3.set_xticks([3, 10, 30, 100, 300])
    ax3.set_xticklabels(["3", "10", "30", "100", "300"])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path)
    plt.close(fig)
    rc.log(f"nature: saved {os.path.basename(path)}")
    return best


# ===========================================================================
def main():
    outdir = os.path.join(ROOT, "analysis", "case_nature")
    os.makedirs(outdir, exist_ok=True)
    d = prepare(outdir)
    fig_structure(d, os.path.join(outdir, "fig1_structure.png"))
    fig_stack(d, os.path.join(outdir, "fig2_stack_energy.png"))
    fig_psf(d, os.path.join(outdir, "fig3_psf_nils.png"))
    best = fig_ler(d, os.path.join(outdir, "fig4_ler_window.png"))
    rc.log(f"nature: a_eff = {d['a_eff']:.2f} nm^2")
    rc.log("nature: sweep-min LER = bulk {:.2f} nm / mem {:.2f} nm".format(
        best["b"]["ler"], best["m"]["ler"]))
    rc.log("nature: DONE")


if __name__ == "__main__":
    main()
