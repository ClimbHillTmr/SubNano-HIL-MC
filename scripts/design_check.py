"""Design verification: check the current HIL-MC implementation against the
design specification (40 nm resist / 670 um Si / 30 keV He+) and against the
numbers quoted in the earlier "final experiment" suite.

Usage:  python scripts/design_check.py
It reuses the cached Monte-Carlo arrays (analysis/case_nature/_cache.pkl)
so it runs in seconds.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import run_case as rc                      # noqa: E402
from nature_visualization import prepare   # noqa: E402

OUT = os.path.join(ROOT, "analysis", "case_nature")
TARGET_CD_NM = 10.0          # design line width (isolated line)
CD_TOL = 0.10                # +/-10 % CD tolerance for the process window
SIGMA_XRR_NM = 0.7262        # XRR-measured surface roughness (sample 4)


def metrics_at_cd(doses, cd, nils, ler, cd_target, tol=CD_TOL):
    """Best (min-LER) point constrained to |CD - cd_target| <= tol*cd_target."""
    ok = np.isfinite(cd) & np.isfinite(ler) & (ler > 0) & \
         (np.abs(cd - cd_target) <= tol * cd_target)
    if not ok.any():
        return None
    i = int(np.argmin(np.where(ok, ler, np.inf)))
    return dict(dose=float(doses[i]), cd=float(cd[i]), nils=float(nils[i]),
                ler=float(ler[i]))


def window_at_cd(doses, cd, ler, cd_target, tol=CD_TOL, slack=1.10):
    """Dose range where CD is on target and LER <= slack * (best LER)."""
    ok = np.isfinite(cd) & np.isfinite(ler) & (ler > 0) & \
         (np.abs(cd - cd_target) <= tol * cd_target)
    if not ok.sum():
        return None
    dd, ll = doses[ok], ler[ok]
    keep = ll <= slack * ll.min()
    if not keep.any():
        return None
    return dict(lo=float(dd[keep].min()), hi=float(dd[keep].max()),
                n=int(keep.sum()), ler_min=float(ll.min()))


def fwhm(x, y):
    """FWHM of a peaked 1-D curve; mirrors the curve about its peak first so
    that half-maximum crossings exist on both sides (the LSF is stored on a
    half grid x >= 0 with its maximum at x = 0)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    i0 = int(np.argmax(y))
    xs = np.concatenate([2 * x[i0] - x[:i0][::-1], x[i0:]])
    ys = np.concatenate([y[:i0][::-1], y[i0:]])
    return _fwhm_core(xs, ys)


def _fwhm_core(x, y):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    pk = y.max()
    if not np.isfinite(pk) or pk <= 0:
        return float("nan")
    half = 0.5 * pk
    i0 = int(np.argmax(y))
    below = np.flatnonzero(y <= half)
    if below.size == 0:
        return float("nan")
    if i0 == 0:                     # peak sits on the grid edge (half-profile)
        return 2.0 * float(x[below[0]] - x[i0])
    right = below[below > i0]
    left = below[below < i0]
    if right.size == 0 or left.size == 0:
        return float("nan")
    return float(x[right[0]] - x[left[-1]])


def main():
    d = prepare(OUT)
    bulk_reps = d["bulk_reps"]
    a_eff = d["a_eff"]
    bulk = d["bulk"]

    # ---------- LSF / dose sweep (40 nm / 670 um Si) ------------------------
    x, lsf = rc.converged_lsf([m.exposure_map for m in bulk_reps],
                              bulk_reps[0].pixel_nm)
    lsf = rc.smooth_tail(x, lsf, r_min=8.0)
    thresh = rc.RHO_GEL * rc.RESIST_THICKNESS_NM
    doses = np.geomspace(0.3, 5000.0, 700)
    sw = rc.dose_sweep(x, lsf, doses, thresh, rc.W_SE, a_eff)
    cd = np.array([s.cd_nm for s in sw])
    nils = np.array([s.nils for s in sw])
    ler = np.array([s.ler_3sigma_nm for s in sw], dtype=float)

    # ---------- (1) ion end-of-range / stopping depth in Si -----------------
    # NOTE: MCResult.depth_profile spans 0 .. t_resist only (linspace(0, t, 81)),
    # i.e. the model has NO substrate depth-dose curve.  The stopping
    # distribution in Si is therefore recovered from the trajectory end points.
    z_end = np.array([t[:, 2].max() for t in bulk.trajectories if len(t) > 2])
    z_end = z_end[np.isfinite(z_end)]
    stopped = z_end[z_end < 0.98 * rc.SUBSTRATE_THICKNESS_NM]   # not transmitted
    hist, edges = np.histogram(z_end, bins=60)
    zc = 0.5 * (edges[1:] + edges[:-1])
    z_peak = float(zc[np.argmax(hist)])
    z_mean = float(z_end.mean())
    frac_penetrate = float((z_end > rc.RESIST_THICKNESS_NM).mean())
    frac_stop_in_si = float(stopped.size / max(z_end.size, 1))
    # authoritative model output: mean stopping depth inside the substrate
    range_sub = float(getattr(bulk, "range_substrate", float("nan")))
    srim = 282.2

    # ---------- (2) PSF / LSF width ----------------------------------------
    fwhm_lsf = fwhm(x, lsf)

    # ---------- (3) unconstrained optimum (what the code reports today) -----
    ok = np.isfinite(ler) & (ler > 0)
    i_min = int(np.argmin(np.where(ok, ler, np.inf)))
    free = dict(dose=float(doses[i_min]), cd=float(cd[i_min]),
                nils=float(nils[i_min]), ler=float(ler[i_min]))

    # ---------- (4) CD-constrained optimum (design: 10 nm line) -------------
    constrained = metrics_at_cd(doses, cd, nils, ler, TARGET_CD_NM)
    win = window_at_cd(doses, cd, ler, TARGET_CD_NM)
    cd_max = float(np.nanmax(cd))

    # ---------- (5) LER with XRR roughness folded in ------------------------
    def with_rough(l):
        return float(np.hypot(l, SIGMA_XRR_NM))
    ler_tot_free = with_rough(free["ler"])
    ler_tot_cd = with_rough(constrained["ler"]) if constrained else None

    # ---------- (6) physical sanity: molecular-size floor -------------------
    mol = d["analysis"].molecule_extent_nm

    # CD / NILS / LER along the low-dose branch (where the reported "optimum"
    # and the thickness-scan minimum actually sit)
    probe_doses = [1.0, 3.0, 5.0, 10.0, 40.0, 100.0, 250.0, 665.0, 2000.0]
    probe = []
    for pd in probe_doses:
        j = int(np.argmin(np.abs(doses - pd)))
        probe.append(dict(dose=float(doses[j]), cd=float(cd[j]),
                          nils=float(nils[j]), ler=float(ler[j])))
    degenerate = [p for p in probe if p["cd"] < mol]

    res = {
        "design": dict(resist_nm=rc.RESIST_THICKNESS_NM,
                       substrate_um=rc.SUBSTRATE_THICKNESS_NM / 1e3,
                       energy_keV=rc.BEAM_ENERGY_KEV, target_cd_nm=TARGET_CD_NM,
                       tolerance=CD_TOL),
        "si_stopping": dict(
            range_substrate_nm=range_sub,
            srim_reference_nm=srim,
            range_error_pct=100 * (range_sub - srim) / srim,
            end_of_range_peak_nm=z_peak,
            end_of_range_mean_traj_nm=z_mean,
            n_trajectories=int(z_end.size),
            fraction_penetrating_resist=frac_penetrate,
            fraction_stopping_in_si=frac_stop_in_si,
            depth_profile_in_substrate=False,
            note="MCResult.depth_profile covers 0..t_resist only; the Si "
                 "depth-dose curve is NOT modelled (only the mean stopping "
                 "depth and the raw trajectory end points are available)."),
        "lsf_fwhm_nm": fwhm_lsf,
        "r50_nm": float(bulk.r50_nm) if hasattr(bulk, "r50_nm") else None,
        "cd_max_nm": cd_max,
        "free_optimum": free,
        "cd_constrained_optimum": constrained,
        "process_window_pC_cm": win,
        "roughness": dict(sigma_xrr_nm=SIGMA_XRR_NM,
                          ler_stochastic_free=free["ler"],
                          ler_total_free=ler_tot_free,
                          ler_stochastic_at_cd=constrained["ler"] if constrained else None,
                          ler_total_at_cd=ler_tot_cd),
        "molecule_extent_nm": mol,
        "cd_below_molecule_size": bool(constrained and constrained["cd"] < mol),
        "dose_probe": probe,
        "degenerate_points": degenerate,
        "unconstrained_optimum_is_physical": bool(free["cd"] >= mol),
    }

    path = os.path.join(OUT, "design_check.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    # ------------------------------ report ---------------------------------
    print("=" * 74)
    print("DESIGN VERIFICATION  40 nm Ti6-oxo / 670 um Si / 30 keV He+")
    print("=" * 74)
    print(f"(1) mean stopping depth in Si   : {range_sub:8.1f} nm  "
          f"[SRIM {srim} nm -> err {100*(range_sub-srim)/srim:+.1f} %]")
    print(f"    trajectory end-of-range     : peak {z_peak:.1f} nm / "
          f"mean {z_mean:.1f} nm (n={z_end.size} kept trajectories)")
    print(f"    ions penetrating the resist : {100*frac_penetrate:8.1f} %")
    print(f"    Si depth-dose curve         : NOT MODELLED "
          f"(depth_profile spans 0..{rc.RESIST_THICKNESS_NM:.0f} nm only) "
          f"[earlier claim: Bragg peak ~160 nm]")
    print(f"(2) LSF FWHM                    : {fwhm_lsf:8.2f} nm   "
          f"[earlier claim ~1.1 nm]")
    print(f"(3) max CD reachable in sweep   : {cd_max:8.2f} nm   "
          f"[design target {TARGET_CD_NM:.0f} nm]")
    print("-" * 74)
    print("unconstrained optimum (as reported today):")
    print(f"    dose {free['dose']:.1f} pC/cm | CD {free['cd']:.2f} nm | "
          f"NILS {free['nils']:.2f} | LER3s {free['ler']:.3f} nm")
    if constrained:
        print(f"CD-constrained (CD = {TARGET_CD_NM:.0f} nm +/-{int(CD_TOL*100)}%):")
        print(f"    dose {constrained['dose']:.1f} pC/cm | CD {constrained['cd']:.2f} nm | "
              f"NILS {constrained['nils']:.2f} | LER3s {constrained['ler']:.3f} nm")
    else:
        print(f"CD-constrained: ** unreachable ** (max CD {cd_max:.2f} nm < "
              f"{TARGET_CD_NM*(1-CD_TOL):.1f} nm)")
    if win:
        print(f"    process window: {win['lo']:.1f} - {win['hi']:.1f} pC/cm "
              f"(LER <= 1.10 x min = {win['ler_min']:.3f} nm)")
    print("-" * 74)
    print("roughness coupling (XRR sigma = %.3f nm):" % SIGMA_XRR_NM)
    print(f"    stochastic only  : {free['ler']:.3f} nm (current model output)")
    print(f"    + sigma in quad. : {ler_tot_free:.3f} nm "
          f"[earlier claim 0.87 nm]")
    print("-" * 74)
    print("-" * 74)
    print("dose -> CD / NILS / LER trace (CD floor = molecule size %.2f nm):" % mol)
    print("    dose[pC/cm]     CD[nm]    NILS    LER3s[nm]   verdict")
    for p in probe:
        v = "CD < molecule size" if p["cd"] < mol else "ok"
        print(f"    {p['dose']:10.1f} {p['cd']:9.2f} {p['nils']:7.2f} "
              f"{p['ler']:11.3f}   {v}")
    print(f"molecule size (4.cif): {mol:.2f} nm  -> unconstrained optimum "
          f"{'PHYSICAL' if res['unconstrained_optimum_is_physical'] else 'UNPHYSICAL'}")
    print(f"saved: {path}")
    return res


if __name__ == "__main__":
    main()
