# -*- coding: utf-8 -*-
"""Tests for the extra process-hazard / latent-image models in
``src/hil_mc/hazards.py``.

All tests run on light synthetic inputs (no Monte-Carlo cache) so they are
fast (< 30 s) and deterministic.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

import hil_mc.hazards as hz


# ---------------------------------------------------------------------------
# shared synthetic fixtures
# ---------------------------------------------------------------------------
def _synthetic_lsf(nx=241, peak=1200.0, sigma=2.0, span=30.0):
    x = np.linspace(-span, span, nx)
    lsf = peak * np.exp(-(x / sigma) ** 2)
    return x, lsf


def _synthetic_depth(nz=20, t=40.0, e_ion=3293.0):
    z_edges = np.linspace(0.0, t, nz + 1)
    depth_profile = np.full(nz, e_ion / nz)  # flat depth (uniform dose)
    return depth_profile, z_edges


def _synthetic_depth_nonuniform(nz=20, t=40.0, e_ion=3293.0):
    # realistic: more energy deposited near the surface than at the film bottom
    zc = np.linspace(0.0, t, nz, endpoint=False) + t / nz / 2.0
    prof = 1.0 - 0.6 * zc / t
    prof = prof / prof.sum() * e_ion
    z_edges = np.linspace(0.0, t, nz + 1)
    return prof, z_edges


# ---------------------------------------------------------------------------
# 1. contrast curve
# ---------------------------------------------------------------------------
def test_contrast_monotonic_bounded():
    x, _ = _synthetic_lsf()
    dp, de = _synthetic_depth_nonuniform()
    doses = np.geomspace(0.3, 500.0, 80)
    cr = hz.contrast_curve(dp, de, 15.0, doses, dp.sum(),
                           centerline_lsf_per_ion_eV_per_nm=1200.0)
    rem = cr.remaining_fraction
    # monotonic non-decreasing with dose
    assert np.all(np.diff(rem) >= -1e-12)
    # bounded in [0, 1]
    assert rem.min() >= 0.0 and rem.max() <= 1.0
    assert 0.0 <= cr.D0 < cr.D100
    assert cr.gamma > 0.0


def test_contrast_deterministic():
    dp, de = _synthetic_depth_nonuniform()
    doses = np.geomspace(0.3, 500.0, 80)
    a = hz.contrast_curve(dp, de, 15.0, doses, dp.sum(),
                          centerline_lsf_per_ion_eV_per_nm=1200.0)
    b = hz.contrast_curve(dp, de, 15.0, doses, dp.sum(),
                          centerline_lsf_per_ion_eV_per_nm=1200.0)
    assert np.array_equal(a.remaining_fraction, b.remaining_fraction)
    assert a.D0 == b.D0 and a.D100 == b.D100


# ---------------------------------------------------------------------------
# 2. sputtering
# ---------------------------------------------------------------------------
def test_sputter_positive_small():
    sp = hz.sputter_yield_he(30.0, dose_pC_cm=200.0)
    assert sp.yield_atoms_per_ion > 0.0
    assert sp.yield_atoms_per_ion < 1.0
    # removed thickness is a tiny fraction of 40 nm
    assert 0.0 < sp.removed_thickness_nm < 0.1


def test_sputter_increases_with_energy():
    # From threshold up to the nuclear-stopping peak the yield rises; for He on
    # this light target the peak sits below ~0.1 keV, so we test that range.
    energies = np.array([0.01, 0.02, 0.05, 0.08])
    ys = [hz.sputter_yield_he(float(E), dose_pC_cm=200.0).yield_atoms_per_ion
          for E in energies]
    assert np.all(np.diff(ys) > 0.0)


# ---------------------------------------------------------------------------
# 3. beam heating
# ---------------------------------------------------------------------------
def test_heating_positive_finite_and_worst_gt_avg():
    doses = np.array([50.0, 100.0, 200.0, 300.0])
    prev_avg = prev_worst = -np.inf
    for D in doses:
        h = hz.beam_heating(D, beam_current_pA=None, dwell_us=1.0)
        assert np.isfinite(h.dT_scan_averaged_K) and h.dT_scan_averaged_K > 0.0
        assert np.isfinite(h.dT_worst_case_K) and h.dT_worst_case_K > 0.0
        # worst-case (concentrated) > scan-averaged
        assert h.dT_worst_case_K > h.dT_scan_averaged_K
        # increasing with dose
        assert h.dT_scan_averaged_K > prev_avg
        assert h.dT_worst_case_K > prev_worst
        prev_avg, prev_worst = h.dT_scan_averaged_K, h.dT_worst_case_K


# ---------------------------------------------------------------------------
# 4. 3-D stochastic line
# ---------------------------------------------------------------------------
def test_line3d_shape_and_cd():
    x, lsf = _synthetic_lsf()
    dp, de = _synthetic_depth()
    r = hz.simulate_line_3d(x, lsf, 1000.0, 15.0 * 40.0, 20.0, 9.33,
                           cd_target_nm=10.0, seed=0,
                           depth_profile=dp, depth_edges=de,
                           energy_per_ion_in_resist_eV=3293.0)
    # finite, consistent-length arrays
    assert np.all(np.isfinite(r.x_left_nm))
    assert np.all(np.isfinite(r.x_right_nm))
    assert len(r.x_left_nm) == len(r.x_right_nm) == len(r.s_nm)
    # recovered CD within ~25% of the 10 nm target
    assert abs(r.cd_nm - 10.0) / 10.0 < 0.25
    # LER strictly positive and finite
    assert np.isfinite(r.ler_3sigma_nm) and r.ler_3sigma_nm > 0.0
    # correlation length finite and positive
    assert np.isfinite(r.corr_length_nm) and r.corr_length_nm > 0.0


def test_line3d_seeded_noise():
    x, lsf = _synthetic_lsf()
    dp, de = _synthetic_depth()
    kw = dict(depth_profile=dp, depth_edges=de,
              energy_per_ion_in_resist_eV=3293.0)
    r0a = hz.simulate_line_3d(x, lsf, 1000.0, 15.0 * 40.0, 20.0, 9.33, seed=0, **kw)
    r0b = hz.simulate_line_3d(x, lsf, 1000.0, 15.0 * 40.0, 20.0, 9.33, seed=0, **kw)
    r1 = hz.simulate_line_3d(x, lsf, 1000.0, 15.0 * 40.0, 20.0, 9.33, seed=1, **kw)
    # identical for the same seed
    assert r0a.cd_nm == r0b.cd_nm
    assert np.array_equal(r0a.x_left_nm, r0b.x_left_nm)
    # different for a different seed (noise is actually seeded)
    assert r0a.cd_nm != r1.cd_nm
    assert not np.array_equal(r0a.x_left_nm, r1.x_left_nm)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
