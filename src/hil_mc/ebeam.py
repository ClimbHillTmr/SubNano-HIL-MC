# -*- coding: utf-8 -*-
"""Analytical, literature-parameterised electron-beam (EBL) point-spread function.

Why this module exists
----------------------
The rest of ``hil_mc`` models a **helium-ion** beam from first principles
(Lindhard-Scharff / ZBL stopping + Monte-Carlo transport).  To quantify what
the *electron* beam does differently -- and in particular how strongly it
delocalises the exposure and how much long-range "proximity" background it
creates -- we do **not** re-derive electron transport.  Instead we use the
established industry model, the **double-Gaussian PSF**, whose three
parameters (alpha, beta, eta) are published for a wide range of beam energies:

    f(r) = 1 / ((1 + eta) * pi) * [ (1/a^2) exp(-r^2/a^2)
                                  + (eta/b^2) exp(-r^2/b^2) ]      [1/nm^2]

    normalised so that  2*pi * int_0^inf f(r) r dr = 1 .

* ``alpha`` : forward-scattering range  (beam + in-resist scattering)
* ``beta``  : back-scattering range     (substrate, um scale)
* ``eta``   : back-scattered / forward-scattered energy ratio

The corresponding **line-spread function** (what a straight line actually
exposes) is analytic:

    L(x) = int_{-inf}^{inf} f(sqrt(x^2+y^2)) dy
         = 1 / (sqrt(pi) (1 + eta)) * [ exp(-x^2/a^2)/a
                                      + eta exp(-x^2/b^2)/b ]     [1/nm]

Reference parameters
--------------------
Table I of *Proximity Effect in E-beam Lithography* (Georgia Tech Nanolithography
group, ``nanolithography.gatech.edu/proximity.pdf``, values reproduced from
ref. [13] therein), for 0.5 um resist on Si:

    beam energy   alpha        beta        eta
    ---------------------------------------------
      20 keV      0.12 um      2.0  um     0.74
      50 keV      0.024 um     9.5  um     0.74
     100 keV      0.0073 um   31.2  um     0.74

These are **literature-assumed** inputs in the sense of the project convention
(measured / calculated / assumed): they are not fitted to our sample.  They are
also quoted for a 0.5 um resist, whereas our resist is 40 nm -- ``alpha`` grows
steeply with resist thickness, and the same reference gives the empirical
scaling (its eq. 10)

    d_f [nm] = 0.9 * (R_t [nm] / V_b [kV]) ** 1.5

which reproduces the tabulated 20 keV / 500 nm value (0.9*(500/20)^1.5 = 113 nm
vs 120 nm tabulated).  Applying it to a 40 nm film gives a *much* smaller
forward blur, so this module exposes both parameterisations and lets the caller
show the sensitivity band instead of silently picking one.

Secondary-electron delocalisation is deliberately **not** added on top of
``alpha``: in the double-Gaussian model ``alpha`` is an effective width that
already absorbs the resolution of the tool and of the exposure process, and
adding a second blur would double-count it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import j0

CITATION = ("Proximity Effect in E-beam Lithography, Georgia Tech Nanolithography "
            "group (nanolithography.gatech.edu/proximity.pdf), Table I after ref. [13]; "
            "alpha-thickness scaling from its Eq. (10).")

# (alpha_nm, beta_nm, eta) for 0.5 um resist on Si -- literature, see docstring.
LITERATURE_TABLE: Dict[float, Tuple[float, float, float]] = {
    20.0: (120.0, 2_000.0, 0.74),
    50.0: (24.0, 9_500.0, 0.74),
    100.0: (7.3, 31_200.0, 0.74),
}


# ---------------------------------------------------------------------------
def alpha_scatter_nm(resist_thickness_nm: float, voltage_kv: float) -> float:
    """Forward-scattering beam broadening in the resist [nm], Eq. (10).

    ``d_f = 0.9 * (R_t / V_b) ** 1.5`` with ``R_t`` in nm and ``V_b`` in kV.
    """
    if resist_thickness_nm <= 0 or voltage_kv <= 0:
        raise ValueError("resist_thickness_nm and voltage_kv must be positive")
    return 0.9 * (float(resist_thickness_nm) / float(voltage_kv)) ** 1.5


def interp_literature_params(voltage_kv: float) -> Tuple[float, float, float]:
    """Log-log interpolate (alpha, beta, eta) of :data:`LITERATURE_TABLE`.

    ``alpha`` and ``beta`` are interpolated on log-log axes (both scale
    approximately as powers of the beam energy -- ``beta ~ E^1.7``); ``eta`` is
    weakly energy dependent and taken as the (constant) tabulated value.
    """
    e = float(voltage_kv)
    grid = np.array(sorted(LITERATURE_TABLE), dtype=float)
    if e <= grid[0]:
        return LITERATURE_TABLE[float(grid[0])]
    if e >= grid[-1]:
        return LITERATURE_TABLE[float(grid[-1])]
    a = np.array([LITERATURE_TABLE[float(x)][0] for x in grid])
    b = np.array([LITERATURE_TABLE[float(x)][1] for x in grid])
    eta = np.array([LITERATURE_TABLE[float(x)][2] for x in grid])
    la = float(np.interp(math.log(e), np.log(grid), np.log(a)))
    lb = float(np.interp(math.log(e), np.log(grid), np.log(b)))
    le = float(np.interp(math.log(e), np.log(grid), np.log(eta)))
    return math.exp(la), math.exp(lb), math.exp(le)


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EbPSF:
    """A double-Gaussian EBL point-spread function."""

    voltage_kv: float
    alpha_nm: float
    beta_nm: float
    eta: float
    label: str = "EBL"
    source: str = CITATION
    mode: str = "literature"

    # -- 2-D radial energy density [1/nm^2], unit integral ------------------
    def psf_r(self, r):
        r = np.asarray(r, dtype=float)
        a2 = self.alpha_nm ** 2
        b2 = self.beta_nm ** 2
        core = np.exp(-r ** 2 / a2) / a2
        halo = self.eta * np.exp(-r ** 2 / b2) / b2
        return (core + halo) / (math.pi * (1.0 + self.eta))

    # -- analytic line-spread function [1/nm], unit integral -----------------
    def lsf(self, x):
        x = np.asarray(x, dtype=float)
        a, b = self.alpha_nm, self.beta_nm
        core = np.exp(-x ** 2 / a ** 2) / a
        halo = self.eta * np.exp(-x ** 2 / b ** 2) / b
        return (core + halo) / (math.sqrt(math.pi) * (1.0 + self.eta))

    # -- enclosed-energy fraction within radius r ---------------------------
    def enclosed_fraction(self, r):
        r = np.asarray(r, dtype=float)
        fa = 1.0 - np.exp(-r ** 2 / self.alpha_nm ** 2)
        fb = 1.0 - np.exp(-r ** 2 / self.beta_nm ** 2)
        return (fa + self.eta * fb) / (1.0 + self.eta)

    # -- radius enclosing a given energy fraction (r50 / r90 / r99) ---------
    def radius_enclosing(self, fraction: float) -> float:
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be in (0, 1)")
        hi = self.alpha_nm
        while self.enclosed_fraction(hi) < fraction:
            hi *= 2.0
            if hi > 1e9:                       # dominated by the beta term
                break
        lo = 0.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.enclosed_fraction(mid) < fraction:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def envelope(self) -> Dict[str, float]:
        """r50 / r90 / r99 of the PSF [nm]."""
        return {f"r{int(100*f)}_nm": self.radius_enclosing(f)
                for f in (0.5, 0.9, 0.99)}

    # -- modulation transfer function at grating pitch p [nm] ---------------
    def mtf(self, pitch_nm):
        p = np.asarray(pitch_nm, dtype=float)
        return (np.exp(-(math.pi * self.alpha_nm / p) ** 2)
                + self.eta * np.exp(-(math.pi * self.beta_nm / p) ** 2)) / (1.0 + self.eta)

    # -- energy bookkeeping -------------------------------------------------
    @property
    def backscatter_energy_fraction(self) -> float:
        """Fraction of the deposited energy that arrives via backscattering."""
        return self.eta / (1.0 + self.eta)

    @property
    def pedestal_over_peak(self) -> float:
        """Long-range background relative to the forward peak (2-D PSF)."""
        return (self.eta * self.alpha_nm ** 2) / (self.beta_nm ** 2)

    def summary(self) -> Dict[str, float]:
        out = {
            "voltage_kv": self.voltage_kv,
            "alpha_nm": self.alpha_nm,
            "beta_nm": self.beta_nm,
            "eta": self.eta,
            "backscatter_energy_fraction": self.backscatter_energy_fraction,
            "pedestal_over_peak": self.pedestal_over_peak,
            "mode": self.mode,
        }
        out.update(self.envelope())
        return out


# ---------------------------------------------------------------------------
def literature_psf(voltage_kv: float) -> EbPSF:
    """Conventional EBL PSF: published (alpha, beta, eta) for a 0.5 um resist."""
    a, b, e = interp_literature_params(voltage_kv)
    return EbPSF(voltage_kv=voltage_kv, alpha_nm=a, beta_nm=b, eta=e,
                 label=f"EBL {voltage_kv:.0f} kV (literature, 0.5 um resist)",
                 mode="literature")


def resist_scaled_psf(voltage_kv: float, resist_thickness_nm: float,
                      alpha_spot_nm: float = 0.0,
                      beta_nm: Optional[float] = None,
                      eta: Optional[float] = None) -> EbPSF:
    """EBL PSF whose forward term is scaled to the actual resist thickness.

    ``alpha`` is built from the in-resist forward scattering (Eq. 10) combined
    in quadrature with an optional probe-spot width ``alpha_spot_nm``; ``beta``
    and ``eta`` default to the literature values at this voltage.
    """
    a_sc = alpha_scatter_nm(resist_thickness_nm, voltage_kv)
    a = math.hypot(a_sc, float(alpha_spot_nm))
    _, b_lit, e_lit = interp_literature_params(voltage_kv)
    return EbPSF(voltage_kv=voltage_kv, alpha_nm=a,
                 beta_nm=b_lit if beta_nm is None else float(beta_nm),
                 eta=e_lit if eta is None else float(eta),
                 label=(f"EBL {voltage_kv:.0f} kV "
                        f"(alpha scaled to {resist_thickness_nm:.0f} nm resist)"),
                 mode="resist_scaled")


# ---------------------------------------------------------------------------
def mtf_from_radial(r, p_r, pitch_nm) -> float:
    """Modulation transfer function of an arbitrary circularly symmetric PSF.

    ``M(f) = int_0^inf p(r) J0(2 pi f r) 2 pi r dr / int_0^inf p(r) 2 pi r dr``
    with ``f = 1/pitch``.  Used to score the *measured* helium-ion radial PSF on
    exactly the same footing as the analytical EBL one.
    """
    r = np.asarray(r, dtype=float)
    p_r = np.asarray(p_r, dtype=float)
    from scipy.integrate import trapezoid
    weight = p_r * 2.0 * np.pi * r
    num = trapezoid(weight * j0(2.0 * np.pi * r / float(pitch_nm)), r)
    den = trapezoid(weight, r)
    return float(num / den) if den else float("nan")


@dataclass
class EdgeSample:
    """One point of a developed-edge sweep (scale invariant in dose)."""

    threshold_rel: float          # threshold as a fraction of the LSF peak
    cd_nm: float
    nils: float
    ils_per_nm: float

    @property
    def ler1_nm(self) -> float:
        return self.cd_nm / self.nils if self.nils > 0 else float("nan")


def sweep_edge(x, lsf, n_points: int = 320,
               floor_rel: float = 1e-4) -> List[EdgeSample]:
    """Threshold sweep giving the *shape-limited* (CD, NILS) trade-off.

    The developed-edge position only depends on the ratio of the development
    threshold to the line dose, so sweeping that ratio is equivalent to sweeping
    dose, and the resulting ``(CD, NILS)`` locus is **independent of the
    absolute dose calibration**.  That is what makes a helium-ion (absolute
    Monte-Carlo units) vs electron (arbitrary units) comparison meaningful: both
    beams are scored on the same locus, with the same resist threshold
    chemistry handled downstream.

    Returns one :class:`EdgeSample` per threshold, from high threshold (narrow
    line) to low threshold (wide, over-exposed line).
    """
    x = np.asarray(x, dtype=float)
    lsf = np.asarray(lsf, dtype=float)
    peak = float(np.max(lsf))
    if not np.isfinite(peak) or peak <= 0:
        return []
    ths = np.geomspace(peak, max(peak * floor_rel, 1e-300), n_points)
    out: List[EdgeSample] = []
    for t in ths:
        m = _edge_from_threshold(x, lsf, float(t))
        if m is not None:
            out.append(m)
    return out


def _edge_from_threshold(x, lsf, th: float) -> Optional[EdgeSample]:
    """First/last crossing of ``lsf`` through ``th`` plus the edge log-slope.

    Mirrors the conventions of :func:`hil_mc.metrics.evaluate_line` (negative
    tone: exposed = kept; log-linear sub-pixel edge refinement; |d ln E / dx|).
    """
    above = lsf >= th
    if not above.any():
        return None
    idx = np.flatnonzero(above)
    i0, i1 = int(idx[0]), int(idx[-1])

    def refine(i_in: int, i_out: int) -> float:
        if i_out < 0 or i_out >= len(lsf):
            return float(x[i_in])
        a, b = float(lsf[i_in]), float(lsf[i_out])
        if a <= 0 or b <= 0 or a == b:
            return float(x[i_in])
        la, lb = math.log(a), math.log(b)
        if la == lb:
            return float(x[i_in])
        t = (math.log(th) - la) / (lb - la)
        return float(x[i_in] + t * (x[i_out] - x[i_in]))

    xr = refine(i1, min(i1 + 1, len(lsf) - 1))
    xl = refine(i0, max(i0 - 1, 0))
    cd = max(abs(xr - xl), 1e-3)

    def slope(xp: float) -> float:
        i = int(np.clip(np.searchsorted(x, xp), 1, len(x) - 2))
        dx = x[i + 1] - x[i - 1]
        if dx <= 0:
            return 0.0
        return abs((math.log(max(lsf[i + 1], 1e-300))
                    - math.log(max(lsf[i - 1], 1e-300))) / dx)

    ils = 0.5 * (slope(xr) + slope(xl))
    return EdgeSample(threshold_rel=th, cd_nm=cd, nils=cd * ils, ils_per_nm=ils)


def ler3_3sigma(cd_nm: float, nils: float, n_edge_events_per_nm2: float,
                a_eff_nm2: float) -> float:
    """Stochastic LER (3 sigma) from the project's shot-noise expression.

    ``sigma_1sigma = CD / (NILS * sqrt(n_A * a_eff))`` with ``n_A`` the areal
    density of exposure events at the edge.  Identical formula to
    :func:`hil_mc.metrics.evaluate_line`, factored out here so the helium and
    electron branches share one implementation.
    """
    if nils <= 0 or cd_nm <= 0:
        return float("nan")
    return 3.0 * cd_nm / (nils * math.sqrt(max(n_edge_events_per_nm2 * a_eff_nm2, 1e-12)))


def ler_curve(samples: Sequence[EdgeSample], n_edge_events_per_nm2: float,
              a_eff_nm2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(cd, nils, ler3)`` arrays from an edge sweep."""
    cd = np.array([s.cd_nm for s in samples], dtype=float)
    ni = np.array([s.nils for s in samples], dtype=float)
    ler = np.array([ler3_3sigma(c, n, n_edge_events_per_nm2, a_eff_nm2)
                    for c, n in zip(cd, ni)], dtype=float)
    return cd, ni, ler


def ler_at_cd(samples: Sequence[EdgeSample], cd_target_nm: float,
              n_edge_events_per_nm2: float, a_eff_nm2: float) -> float:
    """LER (3 sigma) of the sample whose developed CD is closest to ``cd_target``."""
    finite = [(s.cd_nm, s.nils) for s in samples
              if np.isfinite(s.cd_nm) and s.cd_nm > 0 and np.isfinite(s.nils)]
    if not finite:
        return float("nan")
    cd, ni = min(finite, key=lambda t: abs(t[0] - cd_target_nm))
    return ler3_3sigma(cd, ni, n_edge_events_per_nm2, a_eff_nm2)


def best_ler(samples: Sequence[EdgeSample], n_edge_events_per_nm2: float,
             a_eff_nm2: float, cd_min_nm: float = 0.0) -> Dict[str, float]:
    """Lowest LER (3 sigma) over the sweep, subject to a CD floor."""
    best = None
    for s in samples:
        if not (np.isfinite(s.cd_nm) and s.cd_nm > cd_min_nm and np.isfinite(s.nils)):
            continue
        v = ler3_3sigma(s.cd_nm, s.nils, n_edge_events_per_nm2, a_eff_nm2)
        if not np.isfinite(v):
            continue
        if best is None or v < best["ler3_nm"]:
            best = {"ler3_nm": v, "cd_nm": s.cd_nm, "nils": s.nils,
                    "threshold_rel": s.threshold_rel}
    return best if best is not None else {}
