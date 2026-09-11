# -*- coding: utf-8 -*-
"""Helium-ion (HIL) vs electron-beam (EBL) lithography on one identical stack.

Purpose
-------
Score, on the *same* material stack and with the *same* resist chemistry, how
the two beams differ in (i) exposure delocalisation and (ii) long-range
proximity background, and translate that difference into the quantity the user
cares about: the line-edge roughness (LER).

Conditions
----------
    HIL :  30 keV He+   and 100 keV He+   <-- first-principles Monte-Carlo (this repo)
    EBL :  30 kV  e-    and 100 kV  e-    <-- literature double-Gaussian PSF

Stack: Ti6-oxo cluster resist 40 nm / Si(001) 670 um (unchanged from the main case).

Why the comparison is fair
--------------------------
* Both beams are pushed through **one shared edge model**: the developed edge is
  where the line-spread function crosses the resist threshold, the image
  log-slope is |d ln E / dx| at that crossing, and

      LER(3 sigma) = 3 * CD / (NILS * sqrt(n_A * a_eff)) .

  ``n_A = rho_gel * t_resist / w_se`` and ``a_eff`` are resist/statistics
  properties, so they are held identical for the two beams.  Because the edge
  position depends only on the *ratio* threshold/dose, sweeping that ratio is
  equivalent to sweeping dose and the resulting (CD, NILS) locus is independent
  of the absolute dose calibration -- which is what makes a helium Monte-Carlo
  (absolute eV/nm per ion) comparable with an electron fit (arbitrary units).
* LER reduces to ``3 * sigma_edge / sqrt(n_A * a_eff)`` with
  ``sigma_edge = 1 / ILS``; the CD enters only through where the edge sits.

Honest caveats are written into ``results.json`` -> ``provenance`` and repeated
in ``report.html``.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import run_case as rc                                                     # noqa: E402
from hil_mc import ebeam as eb                                            # noqa: E402
from hil_mc.materials import Material                                     # noqa: E402
from hil_mc.mc import make_stack, HeMonteCarlo                            # noqa: E402
from hil_mc.metrics import converged_lsf, radial_profile                   # noqa: E402

OUT = os.path.join(ROOT, "analysis", "hil_vs_ebl")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
VOLTAGES = (30.0, 100.0)
N_IONS = 20_000          # per Monte-Carlo replicate
N_REP = 3                # replicates -> converged (replicate-averaged) LSF
DEPTH_MAX_NM = 2500.0    # 100 keV He+ range in Si is ~720 nm in this model

# Resist / statistics constants -- IDENTICAL for both beams (controlled test).
RESIST_NM = rc.RESIST_THICKNESS_NM
RHO_GEL = rc.RHO_GEL
W_SE = rc.W_SE
N_A = RHO_GEL * RESIST_NM / W_SE            # exposure events / nm^2 at the edge
A_EFF = 9.17                                # nm^2, project calibration
# (results.json.calibration.a_eff_nm2; back-fitted so the HIL membrane baseline
#  gives LER(3sigma) = 0.21 nm, the Zhuang 2024 published value).

# EBL secondary parameterisation, used as a sensitivity band.
EBL_ALPHA_SPOT_NM = 3.0   # representative high-resolution Gaussian probe

# common lateral grid for the edge model
X = np.arange(-100.0, 100.0 + 1e-9, 0.05)

# CD probe points for the comparison table [nm]
CD_PROBES = (5.0, 10.0, 20.0, 50.0, 100.0)
PITCHES = np.array([25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0])

# palette -- warm = ion, cool = electron (matches the repo's figure style)
C = {
    "HIL 30 keV": "#C0392B",
    "HIL 100 keV": "#E67E22",
    "EBL 30 kV": "#1f4e79",
    "EBL 100 kV": "#2a9d8f",
}

_LOG_LINES: list = []


def log(msg: str) -> None:
    line = msg
    print(line, flush=True)
    _LOG_LINES.append(line)


# ---------------------------------------------------------------------------
def build_materials():
    """Same Ti6-oxo film (XRR density) + Si(001) as the main case."""
    resist = Material.from_formula("Ti6-oxo cluster resist (film)",
                                   "C120 H152 O28 Ti6", 0.77076, w_se=W_SE)
    si = Material.from_formula("Si(001) substrate", "Si", 2.3291)
    return resist, si


def run_hil(voltage_kev: float, resist: Material, si: Material,
            n_ions: int, seed: int):
    stack = make_stack(resist, si, resist_thickness_nm=RESIST_NM,
                       substrate_thickness_nm=rc.SUBSTRATE_THICKNESS_NM,
                       e_scale=rc.E_SCALE)
    mc = HeMonteCarlo(stack, voltage_kev, seed=seed, box_nm=80.0)
    return mc.run(n_ions, keep_trajectories=0, depth_max_nm=DEPTH_MAX_NM)


def envelope_from_radial(r, dens):
    """r50 / r90 / r99 from a radial areal-density profile (area weighted)."""
    w = np.asarray(dens, float) * 2.0 * np.pi * np.asarray(r, float)
    c = np.cumsum(w) / max(w.sum(), 1e-30)
    out = {}
    for f in (0.5, 0.9, 0.99):
        j = int(np.searchsorted(c, f))
        out[f"r{int(100*f)}_nm"] = float(np.asarray(r)[min(j, len(r) - 1)])
    return out


def lsf_fwhm(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pk = float(np.nanmax(y))
    if not np.isfinite(pk) or pk <= 0:
        return float("nan")
    half = 0.5 * pk
    i0 = int(np.nanargmax(y))
    left = np.where(y[:i0] <= half)[0]
    right = np.where(y[i0:] <= half)[0]
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float(x[i0 + right[0]] - x[left[-1]])


def interp_ler_at_cd(cd, ler, target) -> float:
    """LER at a target CD, interpolated in log-log and only on the monotonic part."""
    cd = np.asarray(cd, float)
    ler = np.asarray(ler, float)
    m = np.isfinite(cd) & np.isfinite(ler) & (cd > 0) & (ler > 0)
    if m.sum() < 3:
        return float("nan")
    xs, ys = cd[m], ler[m]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    xs, idx = np.unique(xs, return_index=True)
    ys = ys[idx]
    if xs.size < 3 or target < xs[0] or target > xs[-1]:
        return float("nan")
    return float(np.exp(np.interp(math.log(target), np.log(xs), np.log(ys))))


def loglog_slope(cd, ler, target) -> float:
    """Local log-log slope d ln(LER) / d ln(CD) at a target CD."""
    cd = np.asarray(cd, float)
    ler = np.asarray(ler, float)
    m = np.isfinite(cd) & np.isfinite(ler) & (cd > 0) & (ler > 0)
    if m.sum() < 5:
        return float("nan")
    xs, ys = np.log(cd[m]), np.log(ler[m])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    lt = math.log(target)
    if lt < xs[0] or lt > xs[-1]:
        return float("nan")
    i = int(np.clip(np.searchsorted(xs, lt), 2, len(xs) - 3))
    sl = xs[i - 2:i + 3]
    sy = ys[i - 2:i + 3]
    return float(np.polyfit(sl, sy, 1)[0])


# ---------------------------------------------------------------------------
def analyse():
    resist, si = build_materials()
    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stack": {"resist_nm": RESIST_NM, "substrate_um": rc.SUBSTRATE_THICKNESS_NM / 1e3,
                  "resist_formula": "C120 H152 O28 Ti6",
                  "resist_density_g_cm3": 0.77076,
                  "beam_energies": list(VOLTAGES)},
        "shared_resist_model": {
            "rho_gel_eV_per_nm3": RHO_GEL, "w_se_eV": W_SE,
            "n_edge_events_per_nm2": N_A, "a_eff_nm2": A_EFF,
            "note": ("Identical for both beams: the comparison isolates the PSF shape. "
                     "a_eff is the project calibration (0.21 nm membrane anchor)."),
        },
        "hil": {},
        "ebl_literature": {},
        "ebl_resist_scaled": {},
        "comparison": {},
        "provenance": {},
    }

    curves = {}

    # ---------------- HIL: first-principles Monte-Carlo -------------------
    for v in VOLTAGES:
        t0 = time.time()
        log(f"[HIL] {v:.0f} keV He+  MC x{N_REP} replicates x{N_IONS} ions ...")
        reps = [run_hil(v, resist, si, N_IONS, seed=9100 + 17 * i)
                for i in range(N_REP)]
        pix = reps[0].pixel_nm
        x_h, lsf_h = converged_lsf([r.exposure_map for r in reps], pix)
        # common grid (zero outside the 80 nm radial support of the Abel transform)
        lsf_c = np.interp(X, x_h, lsf_h, left=0.0, right=0.0)

        rr, pp = radial_profile(reps[0].exposure_map, pix)
        pavg = np.zeros_like(rr)
        for r in reps:
            r_i, p_i = radial_profile(r.exposure_map, pix)
            pavg += np.interp(rr, r_i, p_i)
        pavg /= len(reps)

        e_dir = float(np.mean([r.direct_map.sum() * r.pixel_nm ** 2 for r in reps]))
        e_se2 = float(np.mean([r.substrate_map.sum() * r.pixel_nm ** 2 for r in reps]))
        res = {
            "energy_in_resist_eV": float(np.mean([r.energy_in_resist for r in reps])),
            "fraction_in_resist": float(np.mean([r.energy_in_resist for r in reps]))
            / (v * 1e3),
            "substrate_se2_eV": e_se2,
            "proximity_background_ratio": e_se2 / max(e_dir, 1e-30),
            "ion_backscatter_fraction": float(np.mean([r.backscatter_fraction
                                                       for r in reps])),
            "range_substrate_nm": float(np.mean([r.range_substrate for r in reps])),
            "lsf_fwhm_nm": lsf_fwhm(X, lsf_c),
            "n_ions_total": N_IONS * N_REP,
            "seconds": round(time.time() - t0, 1),
        }
        res.update(envelope_from_radial(rr, pavg))
        # normalise the radial areal density so that  int p(r) 2 pi r dr = 1,
        # i.e. the same normalisation as the analytical EBL PSF.
        dr = float(rr[1] - rr[0])
        integ = float(np.sum(pavg * 2.0 * np.pi * rr) * dr)
        res["_psf_r_nm"] = rr.tolist()
        res["_psf_dens"] = (pavg / max(integ, 1e-30)).tolist()
        cur = eb.sweep_edge(X, lsf_c, n_points=340, floor_rel=2e-4)
        cd, ni, ler = eb.ler_curve(cur, N_A, A_EFF)
        res["best"] = eb.best_ler(cur, N_A, A_EFF, cd_min_nm=2.24)
        res["_edge_sweep"] = cur
        curves[f"HIL {v:.0f} keV"] = (cd, ni, ler, lsf_c, pavg, rr)
        results["hil"][f"{v:.0f} keV"] = res
        log(f"        range={res['range_substrate_nm']:.0f} nm  FWHM={res['lsf_fwhm_nm']:.2f} nm"
            f"  E_res={res['energy_in_resist_eV']:.0f} eV ({100*res['fraction_in_resist']:.2f}%)"
            f"  eta_ion={100*res['ion_backscatter_fraction']:.3f}%"
            f"  prox_bg={100*res['proximity_background_ratio']:.1f}%"
            f"  best LER3s={res['best'].get('ler3_nm', float('nan')):.3f} nm"
            f" @CD={res['best'].get('cd_nm', float('nan')):.1f} nm  ({res['seconds']}s)")

    # ---------------- EBL: literature double-Gaussian ---------------------
    for tag, maker in (("ebl_literature", lambda v: eb.literature_psf(v)),
                       ("ebl_resist_scaled",
                        lambda v: eb.resist_scaled_psf(v, RESIST_NM, EBL_ALPHA_SPOT_NM))):
        disp = "EBL" if tag == "ebl_literature" else "EBL*"
        for v in VOLTAGES:
            psf = maker(v)
            lsf_c = psf.lsf(X)
            env = psf.envelope()
            res = {
                "alpha_nm": psf.alpha_nm, "beta_nm": psf.beta_nm, "eta": psf.eta,
                "backscatter_energy_fraction": psf.backscatter_energy_fraction,
                "pedestal_over_peak": psf.pedestal_over_peak,
                "r50_nm": env["r50_nm"], "r90_nm": env["r90_nm"], "r99_nm": env["r99_nm"],
                "lsf_fwhm_nm": lsf_fwhm(X, lsf_c),
                "mtf": {f"{int(p)} nm": float(psf.mtf(p)) for p in PITCHES},
            }
            cur = eb.sweep_edge(X, lsf_c, n_points=340, floor_rel=2e-4)
            cd, ni, ler = eb.ler_curve(cur, N_A, A_EFF)
            res["best"] = eb.best_ler(cur, N_A, A_EFF, cd_min_nm=2.24)
            res["_edge_sweep"] = cur
            curves[f"{disp} {v:.0f} kV"] = (cd, ni, ler, lsf_c, None, None)
            results[tag][f"{v:.0f} kV"] = res
            log(f"[{disp:9s}] {v:.0f} kV"
                f"  alpha={psf.alpha_nm:.2f} nm  beta={psf.beta_nm/1000:.2f} um"
                f"  FWHM={res['lsf_fwhm_nm']:.2f} nm"
                f"  best LER3s={res['best'].get('ler3_nm', float('nan')):.3f} nm"
                f" @CD={res['best'].get('cd_nm', float('nan')):.1f} nm")

    # ---------------- MTF of the HIL PSF (same footing) -------------------
    for v in VOLTAGES:
        key = f"{v:.0f} keV"
        rr = np.asarray(results["hil"][key]["_psf_r_nm"])
        pp = np.asarray(results["hil"][key]["_psf_dens"])
        results["hil"][key]["mtf"] = {
            f"{int(p)} nm": eb.mtf_from_radial(rr, pp, p) for p in PITCHES}
        results["hil"][key]["mtf_note"] = (
            "Numerical Hankel transform of the Monte-Carlo radial PSF; the radial "
            "support is 80 nm, so pitches >> 80 nm are only indicative.")

    # ---------------- comparison table ------------------------------------
    # curves is already keyed by display name (HIL 30 keV / EBL 100 kV / EBL* ...)
    table = {}
    for name, (cd, ni, ler, _lsf_c, _p, _r) in curves.items():
        row = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            sig = cd / ni
        m = np.isfinite(sig) & (cd > 0) & np.isfinite(ler)
        row["best_ler3_nm"] = float(np.nanmin(ler[m])) if m.any() else float("nan")
        if m.any():
            j = int(np.nanargmin(ler[m]))
            row["best_ler3_cd_nm"] = float(cd[m][j])
            row["sigma_edge_at_best_nm"] = float(sig[m][j])
        for p in CD_PROBES:
            row[f"ler3_at_CD{int(p)}nm"] = interp_ler_at_cd(cd, ler, p)
            row[f"loglog_slope_at_CD{int(p)}nm"] = loglog_slope(cd, ler, p)
        table[name] = row
    results["comparison"] = table

    # ---------------- provenance / caveats --------------------------------
    results["provenance"] = {
        "HIL": ("30/100 keV He+ Monte-Carlo on Ti6-oxo 40 nm / Si, e_scale="
                f"{rc.E_SCALE}; PSF/LSF from replicate-averaged radial profile + Abel "
                "transform. Absolute dose axis uncalibrated (rho_gel, w_se assumed)."),
        "EBL": ("Double-Gaussian with literature (alpha, beta, eta). Reference: "
                + eb.CITATION),
        "EBL_literature_caveat": ("Tabulated parameters are for a 0.5 um resist on Si. "
                                  "alpha grows steeply with resist thickness (Eq. 10), so "
                                  "for a 40 nm film the pure scattering term is much "
                                  "smaller; the 'ebl_resist_scaled' branch applies Eq. 10 "
                                  f"plus a {EBL_ALPHA_SPOT_NM:.0f} nm probe spot as the "
                                  "sensitivity band. Real tools always add a probe term, "
                                  "which the literature alpha implicitly carries."),
        "model_limit": ("The project's LER expression has no development-kinetics term, "
                        "so it reports the stochastic (shot-noise) lower bound, not an "
                        "absolute prediction; and it assumes the same a_eff and n_A for "
                        "both beams to isolate the PSF-shape contribution."),
        "metrics_fix": ("converged_lsf() now mirrors the positive-half radial LSF onto "
                        "the full x axis (metrics.py). Before the fix CD was a half "
                        "width and NILS was 2x too small; LER was unaffected because it "
                        "scales as NILS/CD."),
    }

    # ---------------- figure ----------------------------------------------
    make_figure(results, curves)

    # strip non-serialisable sweep objects
    for grp in ("hil", "ebl_literature", "ebl_resist_scaled"):
        for k, v in results[grp].items():
            v.pop("_edge_sweep", None)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)
    write_report(results, table)
    with open(os.path.join(OUT, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LOG_LINES) + "\n")
    log(f"saved -> {os.path.relpath(OUT, ROOT)}")
    return results


# ---------------------------------------------------------------------------
def make_figure(results, curves):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.6,
        "axes.titlesize": 9.5,
    })
    fig, ax = plt.subplots(2, 3, figsize=(11.4, 6.9))

    def style(a, grid_axis="both"):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, which="major", ls=":", lw=0.5, color="#cccccc", alpha=0.9)
        a.set_axisbelow(True)
        a.tick_params(direction="out", length=3)
        return a

    # (a) radial PSF -------------------------------------------------------
    a = style(ax[0, 0])
    r_eb = np.logspace(-1, 5, 600)
    for v in VOLTAGES:
        p = eb.literature_psf(v)
        a.plot(r_eb, p.psf_r(r_eb) * 2 * np.pi * r_eb, color=C[f"EBL {v:.0f} kV"], lw=1.3,
               label=f"EBL {v:.0f} kV")
    for v in VOLTAGES:
        key = f"{v:.0f} keV"
        rr = np.asarray(results["hil"][key]["_psf_r_nm"])
        pp = np.asarray(results["hil"][key]["_psf_dens"])
        w = pp * 2 * np.pi * rr
        a.plot(rr, w, color=C[f"HIL {v:.0f} keV"], lw=1.6,
               label=f"HIL {v:.0f} keV")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(0.3, 3e4); a.set_ylim(1e-7, 3e-1)
    a.set_xlabel("radius  r  (nm)")
    a.set_ylabel(r"radial energy density  (2$\pi$r·PSF, per ion)")
    a.set_title("(a)  exposure delocalisation", loc="left")
    a.legend(frameon=False, loc="upper right")
    a.annotate("EBL backscatter halo\n$\\beta$ ≈ 4–31 µm", xy=(3e3, 2e-5),
               xytext=(120, 3e-6), fontsize=7.2, color="#444444",
               arrowprops=dict(arrowstyle="->", lw=0.6, color="#888888"))

    # (b) LSF near the edge ------------------------------------------------
    a = style(ax[0, 1])
    for name in ("HIL 30 keV", "HIL 100 keV", "EBL 30 kV", "EBL 100 kV"):
        cd, ni, ler, lsf_c, _p, _r = curves[name]
        a.plot(X, lsf_c / np.nanmax(lsf_c), color=C[name], lw=1.5, label=name)
    a.set_yscale("log")
    a.set_xlim(-25, 25); a.set_ylim(1e-5, 1.6)
    a.set_xlabel("lateral position  x  (nm)")
    a.set_ylabel("normalised line-spread function")
    a.set_title("(b)  edge profile (isolated line)", loc="left")
    a.legend(frameon=False, loc="upper right")

    # (c) NILS vs CD -------------------------------------------------------
    a = style(ax[0, 2])
    for name in ("HIL 30 keV", "HIL 100 keV", "EBL 30 kV", "EBL 100 kV"):
        cd, ni, ler, lsf_c, _p, _r = curves[name]
        m = np.isfinite(ni) & (cd > 0)
        a.plot(cd[m], ni[m], color=C[name], lw=1.5, label=name)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(1, 200); a.set_ylim(1e-2, 30)
    a.set_xlabel("developed linewidth  CD  (nm)")
    a.set_ylabel("NILS  (dimensionless)")
    a.set_title("(c)  edge sharpness", loc="left")
    a.legend(frameon=False, loc="upper left", fontsize=7.2)

    # (d) LER vs CD --------------------------------------------------------
    a = style(ax[1, 0])
    for name in ("HIL 30 keV", "HIL 100 keV", "EBL 30 kV", "EBL 100 kV"):
        cd, ni, ler, lsf_c, _p, _r = curves[name]
        m = np.isfinite(ler) & (cd > 0)
        a.plot(cd[m], ler[m], color=C[name], lw=1.6, label=name)
    for v in VOLTAGES:
        cd, ni, ler, _l, _p, _r = curves[f"EBL* {v:.0f} kV"]
        m = np.isfinite(ler) & (cd > 0)
        a.plot(cd[m], ler[m], color="#7f8c8d", ls="--", lw=1.0,
               label=f"EBL* {v:.0f} kV (40 nm-scaled)")
    a.axhline(0.21, color="#8a6d00", ls=":", lw=1.0)
    a.text(1.25, 0.235, "IRDS sub-nm target 0.21 nm", fontsize=6.8, color="#8a6d00")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(1, 200); a.set_ylim(1e-2, 1e3)
    a.set_xlabel("developed linewidth  CD  (nm)")
    a.set_ylabel(r"stochastic LER 3$\sigma$  (nm)")
    a.set_title("(d)  line-edge roughness", loc="left")
    a.legend(frameon=False, loc="upper right", fontsize=6.9)

    # (e) MTF vs pitch -----------------------------------------------------
    a = style(ax[1, 1])
    for v in VOLTAGES:
        p = eb.literature_psf(v)
        a.plot(PITCHES, [float(p.mtf(x)) for x in PITCHES], color=C[f"EBL {v:.0f} kV"],
               lw=1.4, marker="o", ms=2.6, label=f"EBL {v:.0f} kV")
    for v in VOLTAGES:
        key = f"{v:.0f} keV"
        mtf = results["hil"][key]["mtf"]
        a.plot(PITCHES, [mtf[f"{int(x)} nm"] for x in PITCHES],
               color=C[f"HIL {v:.0f} keV"], lw=1.7, marker="s", ms=2.8,
               label=f"HIL {v:.0f} keV")
    a.axhline(1.0 / (1.0 + 0.74), color="#1f4e79", ls=":", lw=0.9)
    a.text(27, 0.585, r"EBL $\eta$-limited contrast  1/(1+$\eta$)", fontsize=6.6,
           color="#1f4e79")
    a.set_xscale("log"); a.set_ylim(0.0, 1.08)
    a.set_xlabel("grating pitch  p  (nm)")
    a.set_ylabel("modulation transfer  M(p)")
    a.set_title("(e)  proximity contrast (grating)", loc="left")
    a.legend(frameon=False, loc="lower left", fontsize=7.0)

    # (f) proximity background --------------------------------------------
    a = style(ax[1, 2])
    labels, vals, cols = [], [], []
    for v in VOLTAGES:
        key = f"{v:.0f} keV"
        labels.append(f"HIL\n{v:.0f} keV")
        vals.append(100.0 * results["hil"][key]["proximity_background_ratio"])
        cols.append(C[f"HIL {v:.0f} keV"])
    for v in VOLTAGES:
        labels.append(f"EBL\n{v:.0f} kV")
        vals.append(100.0 * results["ebl_literature"][f"{v:.0f} kV"]["eta"]
                    / (1.0 + results["ebl_literature"][f"{v:.0f} kV"]["eta"]))
        cols.append(C[f"EBL {v:.0f} kV"])
    b = a.bar(range(len(vals)), vals, color=cols, width=0.62)
    for rect, val in zip(b, vals):
        a.text(rect.get_x() + rect.get_width() / 2, val * 1.35 if val > 1 else 0.02,
               f"{val:.2f}%" if val < 10 else f"{val:.0f}%", ha="center", fontsize=7.0)
    a.set_yscale("log"); a.set_ylim(1e-3, 3e2)
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels, fontsize=7.4)
    a.set_ylabel("long-range background  (% of local dose)")
    a.set_title("(f)  proximity background (single line)", loc="left")
    a.text(0.02, 0.94, "HIL: substrate SE2 + ion backscatter\nEBL: backscattered electrons "
                       r"$\eta/(1+\eta)$", transform=a.transAxes, fontsize=6.8,
           va="top", color="#444444")

    fig.tight_layout(pad=0.9)
    path = os.path.join(OUT, "fig_hil_vs_ebl.png")
    fig.savefig(path, dpi=400, facecolor="white")
    plt.close(fig)
    log(f"figure -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
def write_report(results, table):
    def fmt(x, nd=3):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "—"
        return "—" if not np.isfinite(v) else f"{v:.{nd}f}"

    names = ["HIL 30 keV", "HIL 100 keV", "EBL 30 kV", "EBL 100 kV"]

    rows_psf = []
    for v in VOLTAGES:
        h = results["hil"][f"{v:.0f} keV"]
        e = results["ebl_literature"][f"{v:.0f} kV"]
        rows_psf.append(
            f"<tr><td>HIL {v:.0f} keV</td><td>{h['lsf_fwhm_nm']:.2f}</td>"
            f"<td>{h['r50_nm']:.2f}</td><td>{h['r90_nm']:.2f}</td>"
            f"<td>{h['r99_nm']:.2f}</td><td>—</td>"
            f"<td>{100*h['proximity_background_ratio']:.2f}%</td></tr>")
        rows_psf.append(
            f"<tr><td>EBL {v:.0f} kV</td><td>{e['lsf_fwhm_nm']:.1f}</td>"
            f"<td>{e['r50_nm']:.1f}</td><td>{e['r90_nm']/1000:.1f} µm</td>"
            f"<td>{e['r99_nm']/1000:.1f} µm</td><td>{e['eta']:.2f}</td>"
            f"<td>{100*e['backscatter_energy_fraction']:.1f}%</td></tr>")

    th = "".join(f"<th>CD = {int(p)} nm</th>" for p in CD_PROBES)
    rows_ler = []
    for n in names:
        r = table[n]
        tds = "".join(f"<td>{fmt(r[f'ler3_at_CD{int(p)}nm'], 2)}</td>" for p in CD_PROBES)
        rows_ler.append(f"<tr><td>{n}</td>{tds}</tr>")

    rows_best = []
    for n in names:
        r = table[n]
        rows_best.append(
            f"<tr><td>{n}</td><td>{fmt(r['best_ler3_nm'], 3)}</td>"
            f"<td>{fmt(r.get('best_ler3_cd_nm'), 1)}</td>"
            f"<td>{fmt(r.get('sigma_edge_at_best_nm'), 2)}</td></tr>")

    hil30 = table["HIL 30 keV"]; ebl30 = table["EBL 30 kV"]; ebl100 = table["EBL 100 kV"]
    ratio30 = (ebl30.get("best_ler3_nm", float("nan"))
               / hil30.get("best_ler3_nm", float("nan")))
    ratio100 = (ebl100.get("best_ler3_nm", float("nan"))
                / hil30.get("best_ler3_nm", float("nan")))

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>HIL vs EBL — 电子扩散 / 邻近效应 / LER</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1000px;
      margin:32px auto;padding:0 20px;color:#1a1a1a;line-height:1.62;font-size:15px}}
 h1{{font-size:24px;margin-bottom:4px}} h2{{font-size:18px;margin-top:30px;
      border-bottom:1px solid #e6e6e6;padding-bottom:5px}}
 .sub{{color:#666;font-size:13px;margin-bottom:22px}}
 table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}}
 th,td{{border:1px solid #e3e3e3;padding:6px 9px;text-align:center}}
 th{{background:#f6f8fa;font-weight:600}} td:first-child{{text-align:left}}
 .key{{background:#fff8e6;border-left:4px solid #d9a300;padding:11px 15px;margin:16px 0}}
 .warn{{background:#fdeeee;border-left:4px solid #c0392b;padding:11px 15px;margin:16px 0}}
 .ok{{background:#eef7f2;border-left:4px solid #2a9d8f;padding:11px 15px;margin:16px 0}}
 code{{background:#f3f3f3;padding:1px 5px;border-radius:3px;font-size:13px}}
 img{{width:100%;margin:14px 0;border:1px solid #eee}}
 .foot{{color:#777;font-size:12.5px;margin-top:30px;border-top:1px solid #eee;padding-top:12px}}
</style></head><body>

<h1>氦离子光刻 vs 电子束光刻：电子扩散、邻近效应与线边缘粗糙度</h1>
<div class="sub">同一叠层：Ti₆-氧簇胶 40 nm / Si(001) 670 µm ｜ He⁺ 30 / 100 keV（蒙特卡洛）
｜ e⁻ 30 / 100 kV（文献双高斯 PSF）<br>生成于 {results['generated']} ·
<code>scripts/compare_hil_ebl.py</code></div>

<h2>1. 结论摘要</h2>
<div class="key"><b>核心发现</b>：在<u>孤立线</u>口径下，HIL 的边缘锐度由二次电子离域决定
（LSF 半高宽约 {results['hil']['30 keV']['lsf_fwhm_nm']:.1f} nm），而 EBL 由前散射范围 α 决定。
文献参数下 30 kV EBL 的 α ≈ {results['ebl_literature']['30 kV']['alpha_nm']:.0f} nm、
100 kV ≈ {results['ebl_literature']['100 kV']['alpha_nm']:.1f} nm —— 即
<b>升压只把 EBL 的模糊从几十 nm 压到约 10 nm</b>，而 HIL 天然处于 ~1 nm 量级。</div>
<div class="ok"><b>最关键的物理差异不是"边缘有多陡"，而是"能量落到哪里"</b>：
EBL 有 <b>{100*results['ebl_literature']['30 kV']['backscatter_energy_fraction']:.0f}%</b>
的沉积能量来自背散射电子、铺展在 β ≈ {results['ebl_literature']['30 kV']['beta_nm']/1000:.0f}–{results['ebl_literature']['100 kV']['beta_nm']/1000:.0f} µm 尺度上
（η ≈ 0.74）；HIL 的对应长程本底只有
<b>{100*results['hil']['30 keV']['proximity_background_ratio']:.2f}%</b>
（衬底 SE2 回注），离子背散射仅 {100*results['hil']['30 keV']['ion_backscatter_fraction']:.3f}%。
这正是 HIL 在<b>密排图形（光栅）</b>中邻近效应可忽略的根源。</div>
<div class="warn"><b>必须诚实说明的边界</b>：在<u>大线宽</u>（CD ≳ 50 nm）下，双高斯 EBL 的
边缘斜率随 CD 增大而变陡，其 LER 会逼近甚至优于 HIL 的幂律尾部（见 §4 表与图 d）。
HIL 的决定性优势集中在 <b>小线宽（CD ≲ 20 nm）</b>，也正是亚纳米 LER 与高分辨光栅所关注的区间；
密排图形中 EBL 还要再叠加 η≈0.74 的图案密度依赖本底，HIL 没有这一项。</div>

<h2>2. 方法：为什么两条路径可以同台比较</h2>
<p>两支束流走<b>同一套显影边缘模型</b>：显影边界是线扩散函数跨越抗蚀剂阈值的位置，
像对数斜率 NILS = CD·|dlnE/dx|，随机 LER 用项目既有公式</p>
<p style="text-align:center"><code>LER(3σ) = 3·CD / (NILS·√(n_A·a_eff)) = 3·σ_edge / √(n_A·a_eff)</code></p>
<p>其中 n_A = ρ_gel·t/ω_SE = {N_A:.0f} events/nm²、a_eff = {A_EFF:.2f} nm²
<b>两支束流取同一组值</b>（抗蚀剂化学相同），因此差异<b>完全归因于 PSF 形状</b>。
又因显影边界只取决于「阈值/剂量」之比，扫描该比值等价于扫描剂量，
所得 (CD, NILS) 轨迹与绝对剂量标定无关——这才使 He⁺ 蒙特卡洛的绝对单位（eV/nm/ion）
与电子的任意单位可以直接对照。注意 LER 可化简为 3σ_edge/√(n_A·a_eff)，
<b>CD 只通过"边缘落在哪里"进入</b>。</p>

<h2>3. 曝光离域与邻近本底</h2>
<img src="fig_hil_vs_ebl.png" alt="HIL vs EBL comparison">
<table>
<tr><th>束流</th><th>LSF FWHM (nm)</th><th>PSF r50 (nm)</th><th>PSF r90</th><th>PSF r99</th>
<th>η (背散射比)</th><th>长程本底占比</th></tr>
{''.join(rows_psf)}
</table>
<p><b>读法</b>：EBL 的 r90/r99 落在 µm 量级——那不是"分辨率"，而是
<b>背散射晕</b>（42.5% 的能量铺在 β 尺度上）。对孤立线而言该晕近乎均匀、不改变局部梯度，
但它意味着近一半的入射能量并不贡献图形对比度，并造成图案密度依赖的邻近效应。
HIL 无此通道。</p>

<h2>4. 结果：LER 对比</h2>
<table>
<tr><th>束流</th><th>最佳 LER 3σ (nm)</th><th>对应 CD (nm)</th><th>边缘模糊 σ_edge (nm)</th></tr>
{''.join(rows_best)}
</table>
<p>固定线宽下的随机 LER 3σ（单位 nm）：</p>
<table><tr><th>束流</th>{th}</tr>
{''.join(rows_ler)}
</table>
<div class="key"><b>量级判断</b>：以 HIL 30 keV 为基准，文献参数下 30 kV EBL 的最佳 LER 约为其
<b>{ratio30:.1f} 倍</b>，100 kV EBL 约为 <b>{ratio100:.1f} 倍</b>。
升压对 EBL 有实质收益（前散射 α 由 {results['ebl_literature']['30 kV']['alpha_nm']:.0f} nm
降到 {results['ebl_literature']['100 kV']['alpha_nm']:.1f} nm），但仍显著落后；
HIL 30→100 keV 的 LER 几乎不变，说明其极限由二次电子离域而非束流能量设定——
这本身是工艺鲁棒性优势。</div>

<h2>5. 邻近效应（光栅口径）</h2>
<p>调制传递函数 M(p) 直接给出图形对比度。EBL 因 η 较大，细光栅的对比度被压制在
1/(1+η) = {1/(1+0.74):.2f} 附近；HIL 则趋近 1。这与参考文献（Langmuir 2025, 41, 29152，
Si 光栅的 HIL 制备）中"氦离子光刻邻近效应可忽略"的观察一致。</p>
<table><tr><th>M(p)</th>{''.join(f'<th>{int(p)} nm</th>' for p in PITCHES)}</tr>
<tr><td>HIL 30 keV</td>{''.join(f"<td>{results['hil']['30 keV']['mtf'][f'{int(p)} nm']:.3f}</td>" for p in PITCHES)}</tr>
<tr><td>EBL 30 kV</td>{''.join(f"<td>{results['ebl_literature']['30 kV']['mtf'][f'{int(p)} nm']:.3f}</td>" for p in PITCHES)}</tr>
<tr><td>EBL 100 kV</td>{''.join(f"<td>{results['ebl_literature']['100 kV']['mtf'][f'{int(p)} nm']:.3f}</td>" for p in PITCHES)}</tr>
</table>

<h2>6. 口径与局限</h2>
<ul>
<li><b>EBL 参数为文献假设值</b>：{results['provenance']['EBL_literature_caveat']}</li>
<li><b>HIL 的绝对剂量轴未标定</b>（ρ_gel、ω_SE 为假设，a_eff 为反标定），故 LER 是随机下限而非绝对预测。</li>
<li>两支束流共用 a_eff 与 n_A，属受控比较，用于<b>隔离 PSF 形状贡献</b>。</li>
<li>模型无显影动力学项，未包含抗蚀剂溶解随机性与线宽粗糙度（LWR）的独立贡献。</li>
<li><b>本次顺带修复</b>：<code>metrics.converged_lsf()</code> 原先只返回正半轴，导致 CD 被当作半宽、
NILS 低估 2×（LER 不受影响）。已改为镜像为完整对称网格。</li>
</ul>

<div class="foot">数值见 <code>results.json</code>；复现 <code>python scripts/compare_hil_ebl.py</code>。
参考文献：M. Chen et al., <i>Nanofabrication of Silicon Gratings Enabled by Vapor-Phase,
Highly Uniform Self-Assembled Monolayer Resists</i>, Langmuir 2025, 41(43), 29152；
EBL 双高斯参数见 {eb.CITATION}</div>
</body></html>"""
    path = os.path.join(OUT, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"report -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    analyse()
