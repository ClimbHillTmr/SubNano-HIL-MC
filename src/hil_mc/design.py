"""Design-level metric selection and uncertainty budgeting for HIL lines.

This module closes three modelling gaps that a design audit found in the raw
``metrics`` layer (see ``analysis/design_verification.md``):

1. **CD floor.**  ``metrics.dose_sweep`` returns *every* dose sample, including
   a degenerate low-dose corner where the developed line is thinner than a
   single resist molecule.  Those points are numerically valid but physically
   meaningless, and picking the global LER minimum lands in that corner.
   :func:`optimum_with_floor` rejects them with an explicit, documented floor.

2. **Roughness coupling.**  The stochastic (shot-noise) LER is a *lower bound*.
   The measured XRR surface RMS is a *different* quantity (long-range areal
   roughness).  :func:`roughness_budget` reports the LER for a range of
   transfer coefficients instead of silently picking one, so the reader sees
   the lower bound, the hard upper bound, and the plausible band.

3. **Dose-axis provenance.**  The absolute dose axis rests on assumed
   ``rho_gel`` / ``w_se`` and a back-fitted ``a_eff``.  :func:`DesignSummary`
   carries those inputs alongside every result so no number is quoted without
   its calibration provenance.

Conventions (fixed here, and used by every caller)
--------------------------------------------------
* ``sigma`` quantities are 1-sigma; ``ler3`` quantities are 3-sigma.
* The total edge roughness combines in quadrature **at 1 sigma**:

      LER_total(3s) = 3 * sqrt( sigma_stoch**2 + (xi * sigma_surface)**2 )

  with ``xi`` in [0, 1] the (unknown, experiment-dependent) fraction of the
  measured areal roughness that actually appears on the line edge.  ``xi = 0``
  is the stochastic lower bound; ``xi = 1`` is the hard upper bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import LineMetrics, best_for_cd, exposure_latitude

# Default transfer coefficients reported in the roughness budget.
DEFAULT_XI = (0.0, 0.1, 0.3, 0.5, 1.0)

# A developed line narrower than this many molecular diameters is rejected as
# unphysical.  1.0 is deliberately permissive (one molecule wide); the resist
# molecule diameter comes from the CIF, so the floor is data-driven.
CD_FLOOR_MOLECULES = 1.0


def cd_floor_nm(molecule_extent_nm: float,
                molecules: float = CD_FLOOR_MOLECULES) -> float:
    """Physical lower bound on a developed CD, in nm.

    A negative-tone line cannot be narrower than the cross-linked cluster that
    forms it, so the CIF-derived molecule diameter sets the floor.
    """
    return float(molecules) * float(molecule_extent_nm)


def _finite(sweep: Sequence[LineMetrics]) -> List[LineMetrics]:
    return [m for m in sweep
            if np.isfinite(m.cd_nm) and m.cd_nm > 0
            and np.isfinite(m.ler_3sigma_nm) and m.ler_3sigma_nm > 0
            and np.isfinite(m.nils) and m.nils > 0]


def optimum_with_floor(sweep: Sequence[LineMetrics],
                       cd_min_nm: float) -> Optional[LineMetrics]:
    """Lowest-stochastic-LER sample whose developed CD clears the floor.

    Returns ``None`` when no sample in the sweep reaches ``cd_min_nm``.
    """
    cand = [m for m in _finite(sweep) if m.cd_nm >= cd_min_nm]
    if not cand:
        return None
    return min(cand, key=lambda m: m.ler_3sigma_nm)


def peak_nils(sweep: Sequence[LineMetrics],
              cd_min_nm: float = 0.0) -> Optional[LineMetrics]:
    """Sample with the largest NILS among physically admissible points."""
    cand = [m for m in _finite(sweep) if m.cd_nm >= cd_min_nm]
    if not cand:
        return None
    return max(cand, key=lambda m: m.nils)


def degenerate_points(sweep: Sequence[LineMetrics],
                      cd_min_nm: float) -> List[LineMetrics]:
    """Samples rejected by the CD floor (kept for reporting/plotting)."""
    return [m for m in _finite(sweep) if m.cd_nm < cd_min_nm]


def roughness_budget(sigma_stoch_nm: float,
                     sigma_surface_nm: float,
                     xi: Sequence[float] = DEFAULT_XI) -> Dict[str, float]:
    """LER(3 sigma) for a range of surface-to-edge roughness transfers.

    Parameters
    ----------
    sigma_stoch_nm : 1-sigma stochastic (shot-noise) edge roughness [nm].
    sigma_surface_nm : measured 1-sigma areal (XRR) roughness [nm].
    xi : transfer coefficients to report (0 = lower bound, 1 = upper bound).

    Returns
    -------
    dict mapping ``"xi=0.30"`` -> LER 3-sigma [nm].
    """
    out: Dict[str, float] = {}
    for f in xi:
        s = math.sqrt(float(sigma_stoch_nm) ** 2 + (float(f) * float(sigma_surface_nm)) ** 2)
        out[f"xi={float(f):.2f}"] = 3.0 * s
    return out


@dataclass
class DesignSummary:
    """Every design number for one case, with its calibration provenance."""

    tag: str
    # --- geometry / beam
    resist_nm: float
    substrate_um: float
    energy_keV: float
    # --- calibration provenance (assumed or back-fitted, never measured here)
    rho_gel_eV_per_nm3: float
    w_se_eV: float
    a_eff_nm2: float
    e_scale: float
    dose_axis_calibrated: bool = False
    # --- transport
    energy_in_resist_eV: float = 0.0
    fraction_in_resist: float = 0.0
    substrate_se2_eV: float = 0.0
    backscatter_fraction: float = 0.0
    range_substrate_nm: float = 0.0
    bragg_peak_nm: float = 0.0
    r50_nm: float = 0.0
    r90_nm: float = 0.0
    r99_nm: float = 0.0
    lsf_fwhm_nm: float = 0.0
    # --- lithography (floor-constrained)
    cd_floor_nm: float = 0.0
    best_dose_pC_cm: float = float("nan")
    best_cd_nm: float = float("nan")
    best_nils: float = float("nan")
    best_ler3_stoch_nm: float = float("nan")
    # --- lithography at the design CD target
    cd_target_nm: float = float("nan")
    cd_target_reachable: bool = False
    target_dose_pC_cm: float = float("nan")
    target_cd_nm: float = float("nan")
    target_nils: float = float("nan")
    target_ler3_stoch_nm: float = float("nan")
    exposure_latitude_pC_cm: Optional[List[float]] = None
    # --- roughness budget
    sigma_surface_nm: float = 0.0
    ler3_budget_best: Dict[str, float] = field(default_factory=dict)
    ler3_budget_target: Dict[str, float] = field(default_factory=dict)
    # --- diagnostics
    n_degenerate_samples: int = 0
    cd_max_nm: float = float("nan")

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if self.exposure_latitude_pC_cm is not None:
            d["exposure_latitude_pC_cm"] = list(self.exposure_latitude_pC_cm)
        return d


def summarise_design(tag: str,
                     sweep: Sequence[LineMetrics],
                     *,
                     molecule_extent_nm: float,
                     sigma_surface_nm: float,
                     resist_nm: float,
                     substrate_um: float,
                     energy_keV: float,
                     rho_gel: float,
                     w_se: float,
                     a_eff: float,
                     e_scale: float,
                     cd_target_nm: float = 10.0,
                     cd_tol: float = 0.10,
                     transport: Optional[dict] = None,
                     xi: Sequence[float] = DEFAULT_XI) -> DesignSummary:
    """Build a :class:`DesignSummary` from a dose sweep.

    ``transport`` may carry the MC-derived quantities (energy budget, ranges,
    PSF radii, LSF FWHM); they are copied verbatim so the summary is the single
    place a report needs to read.
    """
    floor = cd_floor_nm(molecule_extent_nm)
    fin = _finite(sweep)
    best = optimum_with_floor(sweep, floor)
    tgt = best_for_cd(list(sweep), cd_target_nm, tol=cd_tol, cd_min_nm=floor)
    el = exposure_latitude(list(sweep), cd_target_nm, tol=cd_tol,
                           slack=1.10, cd_min_nm=floor)

    s = DesignSummary(
        tag=tag,
        resist_nm=float(resist_nm),
        substrate_um=float(substrate_um),
        energy_keV=float(energy_keV),
        rho_gel_eV_per_nm3=float(rho_gel),
        w_se_eV=float(w_se),
        a_eff_nm2=float(a_eff),
        e_scale=float(e_scale),
        dose_axis_calibrated=False,
        cd_floor_nm=floor,
        cd_target_nm=float(cd_target_nm),
        sigma_surface_nm=float(sigma_surface_nm),
        n_degenerate_samples=len(degenerate_points(sweep, floor)),
        cd_max_nm=float(max((m.cd_nm for m in fin), default=float("nan"))),
    )
    if transport:
        for k, v in transport.items():
            if hasattr(s, k):
                setattr(s, k, float(v))

    if best is not None:
        s.best_dose_pC_cm = float(best.dose_pC_cm)
        s.best_cd_nm = float(best.cd_nm)
        s.best_nils = float(best.nils)
        s.best_ler3_stoch_nm = float(best.ler_3sigma_nm)
        s.ler3_budget_best = roughness_budget(best.ler_1sigma_nm,
                                              sigma_surface_nm, xi)
    if tgt is not None:
        s.cd_target_reachable = True
        s.target_dose_pC_cm = float(tgt.dose_pC_cm)
        s.target_cd_nm = float(tgt.cd_nm)
        s.target_nils = float(tgt.nils)
        s.target_ler3_stoch_nm = float(tgt.ler_3sigma_nm)
        s.ler3_budget_target = roughness_budget(tgt.ler_1sigma_nm,
                                                sigma_surface_nm, xi)
    if el is not None:
        s.exposure_latitude_pC_cm = [float(el[0]), float(el[1])]
    return s
