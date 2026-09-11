"""End-to-end invariants of the live HIL pipeline (run_case.py + hil_mc).

These tests are deliberately small enough to run in CI (a few thousand ions),
but they exercise the same code path as the production run:
CIF -> material -> Monte-Carlo transport -> PSF/LSF -> NILS/LER.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_case as rc                                    # noqa: E402
from hil_mc.structure import analyse_cif                 # noqa: E402
from hil_mc.mc import make_stack, HeMonteCarlo           # noqa: E402

CIF = os.path.join(ROOT, "data", "4.cif")


# ---------------------------------------------------------------------------
# design constants
# ---------------------------------------------------------------------------
def test_design_constants():
    assert rc.RESIST_THICKNESS_NM == 40.0
    assert rc.SUBSTRATE_THICKNESS_NM == 670_000.0
    assert rc.BEAM_ENERGY_KEV == 30.0
    assert abs(rc.A_EFF - 9.17) < 0.5       # back-fitted calibration constant
    # XRR fit is the source of truth for film density & surface roughness;
    # verify the live fit (loose bounds cover fit uncertainty).
    analysis = analyse_cif(CIF)
    _xrr_data, xrr_fit = rc.step_xrr(analysis)
    assert 0.5 < xrr_fit.density_g_cm3 < 0.9
    assert 0.3 < xrr_fit.sigma_film_nm < 1.5


# ---------------------------------------------------------------------------
# CIF interpretation
# ---------------------------------------------------------------------------
def test_cif_interpretation_is_stable():
    a = analyse_cif(CIF)
    assert a.formula == "C120 H152 O28 Ti6"
    assert a.space_group == "P -1"
    assert abs(a.density_calc - 1.354) < 0.01
    assert abs(a.r1 - 0.0448) < 1e-6
    assert a.z == 1
    assert len(a.ti_ti_distances) == 15          # 6 Ti -> 15 pairs
    assert min(a.ti_ti_distances) > 2.5          # no absurd contacts
    assert a.ti_mass_fraction > 0.10             # inorganic core fraction


# ---------------------------------------------------------------------------
# Monte-Carlo transport
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_run():
    analysis = analyse_cif(CIF)
    _xrr_data, xrr_fit = rc.step_xrr(analysis)
    resist, _xtal, si, _sin = rc.step_materials(analysis, xrr_fit)
    stack = make_stack(resist, si, rc.RESIST_THICKNESS_NM,
                       rc.SUBSTRATE_THICKNESS_NM, e_scale=rc.E_SCALE)
    mc = HeMonteCarlo(stack, rc.BEAM_ENERGY_KEV, seed=99, box_nm=80.0)
    return mc.run(3000, keep_trajectories=100)


def test_transport_budgets_are_physical(small_run):
    r = small_run
    e_beam = rc.BEAM_ENERGY_KEV * 1e3
    assert 0.0 < r.energy_in_resist < e_beam
    # 30 keV He+ deposits roughly 5-20 % of its energy in a 40 nm film
    assert 0.03 < r.energy_in_resist / e_beam < 0.25
    assert 0.0 <= r.backscatter_fraction < 0.02        # He is ~1000x lighter than e-
    assert r.energy_backscattered_out >= 0.0
    # energy conservation: nothing may exceed the beam energy
    total = r.energy_in_resist + r.energy_in_substrate_total
    assert total <= e_beam * 1.001


def test_maps_are_finite_and_normalised(small_run):
    r = small_run
    for m in (r.direct_map, r.nuclear_map, r.exposure_map, r.substrate_map):
        assert np.all(np.isfinite(m))
        assert m.min() >= 0.0
    # exposure = direct + SE1 spreading + SE2, so it must dominate direct
    assert r.exposure_map.sum() >= r.direct_map.sum()
    assert r.exposure_map.shape == r.direct_map.shape
    # areal density integrates to the energy actually deposited in the resist
    integ = r.exposure_map.sum() * r.pixel_nm ** 2
    assert abs(integ - r.energy_in_resist) / r.energy_in_resist < 0.25


def test_psf_is_sub_nanometre(small_run):
    r = small_run
    rad = rc.radial_profile(r.exposure_map, r.pixel_nm)
    rr, pr = rad[0], rad[1]
    cdf = np.cumsum(pr * rr) / np.sum(pr * rr)
    r50 = rr[np.searchsorted(cdf, 0.50)]
    r90 = rr[np.searchsorted(cdf, 0.90)]
    assert r50 < 1.5            # half the energy inside ~0.6 nm
    assert r90 < 4.0            # 90 % inside a few nm -> He+ is highly collimated


def test_trajectories_record_depth_correctly(small_run):
    """"Regression guard: column 2 is depth; earlier code plotted column 0."""
    r = small_run
    tr = [t for t in r.trajectories if len(t) > 5]
    assert len(tr) >= 50
    z = np.concatenate([t[:, 2] for t in tr])
    x = np.concatenate([t[:, 0] for t in tr])
    assert z.max() > 3 * rc.RESIST_THICKNESS_NM     # ions do penetrate deeply
    # Lateral spread is physically unbounded now that ions are no longer killed
    # at the 80 nm box (defect-2 fix); it must still stay within a few projected
    # ranges of the stopping depth.  A runaway into thousands of nm would signal
    # a transport bug rather than legitimate straggle.
    max_depth = rc.RESIST_THICKNESS_NM + r.range_substrate
    assert np.abs(x).max() < 3.0 * max_depth


# ---------------------------------------------------------------------------
# litho metrics
# ---------------------------------------------------------------------------
def test_lsf_and_nils_are_consistent():
    analysis = analyse_cif(CIF)
    _xrr_data, xrr_fit = rc.step_xrr(analysis)
    resist, _x, si, _s = rc.step_materials(analysis, xrr_fit)
    res = rc.run_case(resist, si, rc.RESIST_THICKNESS_NM, 4000, seed=7)
    x, lsf = rc.converged_lsf([res.exposure_map], res.pixel_nm)
    lsf = rc.smooth_tail(x, lsf, r_min=8.0)
    assert np.all(np.isfinite(lsf))
    assert lsf.max() > 0
    # LSF must decay monotonically away from the centre in the far tail
    tail = lsf[len(lsf) // 2:]
    assert tail[0] >= tail[-1]

    thresh = rc.RHO_GEL * rc.RESIST_THICKNESS_NM
    doses = np.geomspace(1.0, 400.0, 60)
    sw = rc.dose_sweep(x, lsf, doses, thresh, rc.W_SE, rc.A_EFF)
    cds = np.array([s.cd_nm for s in sw])
    lers = np.array([s.ler_3sigma_nm for s in sw], dtype=float)
    assert np.all(cds >= 0.0)
    assert np.all(np.isfinite(cds))
    assert np.nanmin(lers) > 0.0
    assert np.nanmax(cds) > 3.0        # the sweep must actually print lines
