"""Lithography metrics derived from the Monte-Carlo energy-deposition maps.

Chain:  MC exposure map  ->  line-spread function (LSF)
                         ->  threshold development: CD(dose)
                         ->  image log-slope: NILS
                         ->  stochastic (shot-noise) LER
                         ->  process window

Conventions
-----------
* ``lsf``      : energy deposited per unit length of one ion, per unit lateral
                 distance  [eV/nm per ion]  -- integral over y and z.
* line dose q  : pC/cm  ->  lambda = 0.6242 * q  ions per nm of line length.
* areal energy : E_A(x) = lambda * lsf(x)                     [eV/nm^2]
* threshold    : E_th = rho_th * t_resist                     [eV/nm^2]
* NILS         = CD * |d ln E_A / dx|  at the edge
* LER (1 sigma) = 1 / ( ILS * sqrt(n_A * a_eff) )    [nm]
                 = CD / ( NILS * sqrt(n_A * a_eff) )
  with n_A = areal density of exposure *events* at the edge (= E_th / w_se)
  and a_eff the stochastic averaging (correlation) area.

All functions here are deterministic (no random numbers) so results are
reproducible across runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

ION_PER_NM_PER_PC_CM = 1e-12 / 1.602176634e-19 / 1e7     # 0.6242 ions/nm per pC/cm


def radial_profile(map2d: np.ndarray, pixel_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged areal energy density [eV/nm^2 per ion]."""
    n = map2d.shape[0]
    ax = (np.arange(n) - n / 2 + 0.5) * pixel_nm
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    r = np.hypot(xx, yy)
    # deterministic radial bin edges (avoid float-sensitive np.arange stop)
    nr = n // 2
    edges = np.linspace(0.0, ax.max(), nr + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    idx = np.clip(idx, 0, nr - 1)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    # sum of areal densities over annulus pixels -> multiply by pixel^2 to get
    # the energy in the annulus, then divide by annulus area -> eV/nm^2 per ion
    s = np.bincount(idx, weights=map2d.ravel() * (pixel_nm ** 2), minlength=nr)[:nr]
    dens = s / area
    return 0.5 * (edges[1:] + edges[:-1]), dens


def lsf_from_map(map2d: np.ndarray, pixel_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Line-spread function: integrate the 2D areal map over y."""
    prof = map2d.sum(axis=1) * pixel_nm          # eV/nm per ion
    n = map2d.shape[0]
    x = (np.arange(n) - n / 2 + 0.5) * pixel_nm
    return x, prof


def smooth_tail(x: np.ndarray, y: np.ndarray, r_min: float = 6.0,
                n_keep: int = 3) -> np.ndarray:
    """Replace the noisy far tail by a power-law fit (He PSF ~ r^-4)."""
    out = y.copy()
    m = np.abs(x) >= r_min
    if m.sum() > 20:
        k = np.polyfit(np.log(np.abs(x[m])), np.log(np.maximum(y[m], 1e-12)), 1)
        fit = np.exp(np.polyval(k, np.log(np.maximum(np.abs(x), 1e-3))))
        out[m] = np.maximum(out[m], 0.0)
        # blend: use the fit only where the MC data falls below it
        sel = m & (out < fit * 3)
        out[sel] = fit[sel]
    return out


def lsf_from_radial(r: np.ndarray, pr: np.ndarray,
                    x: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Line-spread function from an azimuthally-averaged circularly-symmetric PSF.

    For a 2D PSF ``p(r)`` (``pr`` is the areal density at radius ``r``), the
    line-spread along the x-axis is the Abel-type integral

        L(x) = 2 * ∫_x^∞  p(r) * r / sqrt(r^2 - x^2)  dr .

    The radial PSF is far better sampled at large ``r`` (∝ r) than the column
    integral, so this gives a much more stable edge slope (NILS) than taking
    ``lsf_from_map`` of a single noisy MC map.

    The singularity at ``r = x`` is removed with the substitution
    ``r = x / cos(theta)`` (theta in [0, pi/2)), giving a smooth integrand
    ``p(x/cos) * x / cos^2``.
    """
    r = np.asarray(r, dtype=float)
    pr = np.asarray(pr, dtype=float)
    if x is None:
        x = r
    x = np.asarray(x, dtype=float)
    from scipy.interpolate import interp1d
    from scipy.integrate import trapezoid as trapz
    f = interp1d(r, pr, bounds_error=False, fill_value=0.0)
    rmax = r[-1]
    base = 2.0 * trapz(pr, r)                          # L(0)
    xabs = np.abs(x)
    L = np.where(xabs <= r[0], base, 0.0)
    sel = xabs > r[0]
    if not sel.any():
        return x, L
    xs = xabs[sel]
    n_th = 2000
    th = np.linspace(0.0, np.pi / 2 - 1e-9, n_th)
    ct = np.cos(th)[:, None]
    rr = xs[None, :] / ct                             # (n_th, n_sel)
    # substitution r = x/cos(theta):  integrand = p(r) * x / cos^2(theta)
    g = np.where(rr <= rmax, f(rr) * xs[None, :] / (ct ** 2), 0.0)
    L[sel] = 2.0 * trapz(g, th, axis=0)
    return x, L


def converged_lsf(maps, pixel_nm: float, r_min: float = 8.0, symmetric: bool = True):
    """Average the radial PSF over several MC exposure maps, then Abel-transform
    to a well-converged line-spread function (avoids far-tail MC noise).

    Returns (x, lsf) where ``x`` is the lateral coordinate in nm and ``lsf`` is
    in eV/nm per ion.

    ``symmetric`` (default) mirrors the positive-half result onto the full x
    axis.  The radial grid produced by :func:`radial_profile` covers ``r >= 0``
    only, but the line-spread function is even in ``x`` -- ``L(-x) = L(x)`` --
    and :func:`evaluate_line` requires a grid symmetric about 0 because it takes
    ``CD = x[i1] - x[i0]``.  Passing the un-mirrored half-axis silently reports a
    **half width** as the CD and halves the NILS (the LER is unaffected because
    it depends on ``NILS / CD``).  Set ``symmetric=False`` only to recover the
    historical half-axis behaviour.
    """
    rs, ps = [], []
    for m in maps:
        rr, pp = radial_profile(m, pixel_nm)
        rs.append(rr)
        ps.append(pp)
    r0 = rs[0]
    pavg = np.zeros_like(r0)
    for rr, pp in zip(rs, ps):
        pavg += np.interp(r0, rr, pp)
    pavg /= len(maps)
    # power-law smoothing of the far radial tail (only where the MC is below it)
    pavg = smooth_tail(r0, pavg, r_min=r_min)
    x, lsf = lsf_from_radial(r0, pavg, r0)
    if symmetric:
        x = np.concatenate([-x[::-1], x])
        lsf = np.concatenate([lsf[::-1], lsf])
    return x, lsf


@dataclass
class LineMetrics:
    dose_pC_cm: float
    lambda_ions_per_nm: float
    cd_nm: float
    nils: float
    ils: float
    ler_1sigma_nm: float
    ler_3sigma_nm: float
    events_per_nm2_edge: float
    ler_total_3sigma_nm: float = 0.0
    """Total line-edge roughness at 3 sigma [nm] = stochastic + roughness.

    ``ler_total_3sigma_nm = 3 * sqrt(ler_1sigma_nm**2 + sigma_rough_nm**2)``.

    * ``ler_3sigma_nm`` (unchanged) is the *stochastic* (shot-noise) only term.
    * ``ler_total_3sigma_nm`` adds the high-frequency line-edge roughness
      ``sigma_rough_nm`` in quadrature. ``sigma_rough_nm`` is the *effective*
      edge contribution; feeding the raw XRR surface RMS is an upper bound
      because it also contains low-frequency content that does not blur the
      line edge. With ``sigma_rough_nm = 0`` the two fields are identical.
    """


def evaluate_line(x: np.ndarray, lsf: np.ndarray, dose_pC_cm: float,
                  threshold_areal: float, w_se: float = 20.0,
                  a_eff: float = 7.5, *, sigma_rough_nm: float = 0.0,
                  cd_min_nm: Optional[float] = None) -> LineMetrics:
    """Threshold development + NILS + stochastic LER for a single-pixel line.

    Parameters
    ----------
    x : np.ndarray
        Lateral coordinate grid [nm] (monotonic, symmetric about 0 is typical).
    lsf : np.ndarray
        Line-spread function [eV/nm per ion], same shape as ``x``.
    dose_pC_cm : float
        Line dose [pC/cm].
    threshold_areal : float
        Development threshold areal energy density [eV/nm^2].
    w_se : float
        Effective secondary-electron spreading width [nm] (stochastic model).
    a_eff : float
        Stochastic averaging (correlation) area [nm^2].
    sigma_rough_nm : float, keyword-only
        Effective high-frequency line-edge roughness contribution [nm] to add
        in quadrature to the stochastic LER (default 0 -> unchanged behaviour).
    cd_min_nm : float or None, keyword-only
        Reserved CD floor [nm]. ``evaluate_line`` always *computes* the metrics
        regardless of this value; the actual rejection of points below the floor
        is performed by :func:`best_for_cd` / :func:`exposure_latitude`.

    Returns
    -------
    LineMetrics
        Deterministic metrics for the given dose. ``ler_total_3sigma_nm``
        carries the stochastic + roughness total roughness.
    """
    # cd_min_nm is intentionally accepted for API symmetry with the sweep-level
    # helpers; rejection is done downstream so the sample is always computed.
    _ = cd_min_nm
    lam = dose_pC_cm * ION_PER_NM_PER_PC_CM
    E = lam * lsf                                        # eV/nm^2
    # --- edge position: last x where E >= E_th  (negative-tone: exposed = kept)
    above = E >= threshold_areal
    if not above.any():
        nan = float("nan")
        return LineMetrics(dose_pC_cm, lam, 0.0, 0.0, 0.0, nan, nan, 0.0,
                           nan if sigma_rough_nm == 0.0 else nan)
    i0, i1 = np.flatnonzero(above)[[0, -1]]
    cd = (x[i1] - x[i0])
    # sub-pixel edge refinement by log-linear interpolation on both edges
    def refine(i_in, i_out):
        if i_out < 0 or i_out >= len(E):
            return x[i_in]
        a, b = E[i_in], E[i_out]
        if b <= 0 or a <= 0:
            return x[i_in]
        la, lb = np.log(a), np.log(b)
        if la == lb:
            return x[i_in]
        t = (np.log(threshold_areal) - la) / (lb - la)
        return x[i_in] + t * (x[i_out] - x[i_in])
    xr = refine(i1, min(i1 + 1, len(E) - 1))
    xl = refine(i0, max(i0 - 1, 0))
    cd = max(xr - xl, 1e-3)
    # --- image log slope at the right edge
    def slope(xp):
        i = int(np.clip(np.searchsorted(x, xp), 1, len(x) - 2))
        dlnE = (np.log(max(E[i + 1], 1e-12)) - np.log(max(E[i - 1], 1e-12))) / (x[i + 1] - x[i - 1])
        return abs(dlnE)
    ils = 0.5 * (slope(xr) + slope(xl))
    nils = cd * ils
    # --- stochastic LER (1 sigma)
    n_edge = threshold_areal / w_se                      # events/nm^2 at the edge
    ler1 = cd / (nils * np.sqrt(max(n_edge * a_eff, 1e-9))) if nils > 0 else float("nan")
    # --- total LER (stochastic + effective roughness) in quadrature
    ler_total1 = math.sqrt(float(ler1) ** 2 + float(sigma_rough_nm) ** 2)
    ler_total3 = 3.0 * ler_total1
    return LineMetrics(dose_pC_cm, lam, cd, nils, ils, ler1, 3 * ler1, n_edge,
                       ler_total3)


def dose_sweep(x: np.ndarray, lsf: np.ndarray, doses: np.ndarray,
               threshold_areal: float, w_se: float = 20.0,
               a_eff: float = 7.5, *, sigma_rough_nm: float = 0.0,
               cd_min_nm: Optional[float] = None) -> list:
    """Evaluate :func:`evaluate_line` for every dose in ``doses``.

    Parameters mirror :func:`evaluate_line`. The returned list contains one
    :class:`LineMetrics` per dose, in the same order as ``doses``. This function
    is deterministic and returns *every* sample (no CD filtering) so callers can
    post-process with :func:`best_for_cd` / :func:`exposure_latitude`.

    Returns
    -------
    list[LineMetrics]
    """
    return [evaluate_line(x, lsf, float(d), threshold_areal, w_se, a_eff,
                          sigma_rough_nm=sigma_rough_nm, cd_min_nm=cd_min_nm)
            for d in doses]


def _as_sweep(sweep_or_arrays):
    """Normalise the ``sweep`` argument accepted by the helpers.

    Accepts either a ``list[LineMetrics]`` (as produced by :func:`dose_sweep`)
    or a ``(sweep, doses)`` tuple, for API convenience. The doses array is not
    required (each :class:`LineMetrics` already carries its ``dose_pC_cm``).
    """
    if (isinstance(sweep_or_arrays, (tuple, list)) and len(sweep_or_arrays) == 2
            and isinstance(sweep_or_arrays[0], list)
            and sweep_or_arrays[0]
            and isinstance(sweep_or_arrays[0][0], LineMetrics)):
        return list(sweep_or_arrays[0])
    if isinstance(sweep_or_arrays, list) and sweep_or_arrays \
            and isinstance(sweep_or_arrays[0], LineMetrics):
        return list(sweep_or_arrays)
    raise TypeError(
        "expected a list[LineMetrics] or (list[LineMetrics], doses) tuple, "
        f"got {type(sweep_or_arrays)!r}"
    )


def best_for_cd(sweep, cd_target_nm: float, tol: float = 0.10,
                cd_min_nm: Optional[float] = None):
    """Lowest-stochastic-LER point whose CD is on target.

    Deterministic selection over an already-computed sweep (no recomputation).

    Parameters
    ----------
    sweep : list[LineMetrics] or (list[LineMetrics], doses)
        Samples from :func:`dose_sweep`.
    cd_target_nm : float
        Target critical dimension [nm].
    tol : float
        Relative CD tolerance; a point qualifies when
        ``abs(cd - cd_target_nm) <= tol * cd_target_nm``.
    cd_min_nm : float or None
        If given, additionally require ``cd >= cd_min_nm`` [nm].

    Returns
    -------
    LineMetrics or None
        The qualifying sample with the smallest ``ler_3sigma_nm`` (stochastic
        only). Returns ``None`` when no sample satisfies the CD window.
    """
    sweep = _as_sweep(sweep)
    best = None
    for m in sweep:
        if not np.isfinite(m.cd_nm) or m.cd_nm <= 0:
            continue
        if cd_min_nm is not None and m.cd_nm < cd_min_nm:
            continue
        if abs(m.cd_nm - cd_target_nm) <= tol * cd_target_nm:
            if best is None or m.ler_3sigma_nm < best.ler_3sigma_nm:
                best = m
    return best


def exposure_latitude(sweep, cd_target_nm: float, tol: float = 0.10,
                      slack: float = 1.10, cd_min_nm: Optional[float] = None):
    """Exposure latitude = dose window where CD is on target and LER is tight.

    Deterministic reduction over an already-computed sweep (no recomputation).

    Parameters
    ----------
    sweep : list[LineMetrics] or (list[LineMetrics], doses)
        Samples from :func:`dose_sweep`.
    cd_target_nm : float
        Target critical dimension [nm].
    tol : float
        Relative CD tolerance for "on target": ``abs(cd - cd_target) <= tol*cd``.
    slack : float
        LER acceptance multiplier. A dose qualifies when its stochastic
        ``ler_3sigma_nm <= slack * ler_best`` where ``ler_best`` is the minimum
        LER among the on-target points.
    cd_min_nm : float or None
        If given, additionally require ``cd >= cd_min_nm`` [nm].

    Returns
    -------
    tuple[float, float, float] or None
        ``(dose_lo_pC_cm, dose_hi_pC_cm, ler_min_3sigma_nm)`` for the contiguous
        qualifying dose range (``dose_lo <= dose_hi``, both inside the swept
        dose range). Returns ``None`` when no point qualifies.
    """
    sweep = _as_sweep(sweep)
    best = best_for_cd(sweep, cd_target_nm, tol=tol, cd_min_nm=cd_min_nm)
    if best is None:
        return None
    ler_max = slack * best.ler_3sigma_nm
    qual = [m for m in sweep
            if np.isfinite(m.cd_nm) and m.cd_nm > 0
            and (cd_min_nm is None or m.cd_nm >= cd_min_nm)
            and abs(m.cd_nm - cd_target_nm) <= tol * cd_target_nm
            and m.ler_3sigma_nm <= ler_max]
    if not qual:
        return None
    doses = sorted(m.dose_pC_cm for m in qual)
    ler_min = min(m.ler_3sigma_nm for m in qual)
    return (doses[0], doses[-1], ler_min)


def bootstrap_psf(maps: list, pixel_nm: float, n_boot: int = 30,
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap 95% CI of the radial PSF from several independent MC runs."""
    rng = np.random.default_rng(seed)
    curves = []
    rs = []
    for m in maps:
        r, d = radial_profile(m, pixel_nm)
        rs.append(r)
        curves.append(d)
    # align all curves to the shortest radial grid (avoids off-by-one bins)
    k = min(len(d) for d in curves)
    r_ref = rs[0][:k]
    curves = np.array([d[:k] for d in curves])
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(curves), len(curves))
        means.append(curves[idx].mean(axis=0))
    means = np.array(means)
    mean = means.mean(axis=0)
    std = means.std(axis=0)
    norm = mean.max() + 1e-12
    hi_ci = (mean + 1.96 * std) / norm
    lo_ci = np.maximum(mean - 1.96 * std, 1e-6) / norm
    return r_ref, mean / norm, lo_ci, hi_ci


def calibrate_a_eff(x, lsf, doses, threshold_areal, w_se, target_ler3_nm=0.21):
    """Pick the stochastic averaging area so that min LER(3sigma) = target."""
    def min_ler(a):
        vals = [evaluate_line(x, lsf, float(d), threshold_areal, w_se, a).ler_3sigma_nm
                for d in doses]
        vals = np.array(vals, dtype=float)
        return np.nanmin(vals)
    lo, hi = 1e-2, 1e4
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if min_ler(mid) > target_ler3_nm:      # larger a_eff -> smaller LER
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)
