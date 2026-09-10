"""Deterministic tests for hil_mc.metrics (no Monte-Carlo; analytic LSF).

The line-spread function is a Gaussian
    lsf(x) = A * exp(-x**2 / (2 s**2))
on a symmetric x grid, so ``evaluate_line`` receives a well-formed input and
every metric is reproducible.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the in-tree package importable when running `pytest` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hil_mc.metrics import (  # noqa: E402
    ION_PER_NM_PER_PC_CM,
    best_for_cd,
    dose_sweep,
    evaluate_line,
    exposure_latitude,
)

# --- shared analytic exposure ------------------------------------------------
S_NM = 3.0
A_EV_NM = 4.0
THRESH = 20.0  # eV/nm^2


def make_lsf(n=601, half=30.0):
    x = np.linspace(-half, half, n)
    lsf = A_EV_NM * np.exp(-(x**2) / (2 * S_NM**2))
    return x, lsf


def cd_analytic(dose_pC_cm):
    """Threshold crossing CD for the Gaussian LSF (sub-pixel refinement is tiny)."""
    lam = dose_pC_cm * ION_PER_NM_PER_PC_CM
    ratio = lam * A_EV_NM / THRESH
    if ratio <= 1.0:
        return 0.0
    return 2.0 * S_NM * math.sqrt(2.0 * math.log(ratio))


def sweep_over_decade(sigma_rough_nm=0.0):
    x, lsf = make_lsf()
    doses = np.logspace(1, 2, 12)  # 10 .. 100 pC/cm (a decade)
    return x, lsf, doses, dose_sweep(
        x, lsf, doses, THRESH, sigma_rough_nm=sigma_rough_nm)


def test_ler_decreases_with_dose():
    """Stochastic LER must fall monotonically with dose (shot noise ~ 1/sqrt(n))."""
    _, _, doses, sweep = sweep_over_decade()
    ler = np.array([m.ler_3sigma_nm for m in sweep], dtype=float)
    # all finite and well above threshold at these doses
    assert np.all(np.isfinite(ler))
    # strictly decreasing across the decade (before any tail roll-off)
    assert np.all(np.diff(ler) < 0), "LER must decrease with dose"
    # LER ~ 1/sqrt(n) with n proportional to dose, so low-dose/high-dose LER
    # ratio should track sqrt(dose_high/dose_low); allow a factor-of-2 slack.
    dose_ratio = doses[-1] / doses[0]
    ler_ratio = ler[0] / ler[-1]
    assert 0.5 * math.sqrt(dose_ratio) < ler_ratio < 2.0 * math.sqrt(dose_ratio)


def test_ler_total_quadrature():
    """ler_total_3sigma = 3*sqrt(ler_1sigma^2 + sigma^2); sigma=0 keeps stochastic."""
    x, lsf = make_lsf()
    dose = 50.0
    for sig in (0.0, 0.2, 0.7262, 1.5):
        m = evaluate_line(x, lsf, dose, THRESH, sigma_rough_nm=sig)
        expected = 3.0 * math.sqrt(m.ler_1sigma_nm**2 + sig**2)
        assert m.ler_total_3sigma_nm == pytest.approx(expected, rel=1e-9)
    # backward compatibility: sigma=0 -> identical to stochastic 3sigma term
    m0 = evaluate_line(x, lsf, dose, THRESH)  # default sigma=0
    assert m0.ler_total_3sigma_nm == pytest.approx(m0.ler_3sigma_nm, rel=1e-12)
    assert m0.ler_total_3sigma_nm == pytest.approx(
        evaluate_line(x, lsf, dose, THRESH, sigma_rough_nm=0.0).ler_total_3sigma_nm)


def test_best_for_cd_within_tolerance_and_min_ler():
    """best_for_cd returns an on-target point with the minimum stochastic LER."""
    _, _, _, sweep = sweep_over_decade()
    cd_target = 10.0
    tol = 0.10
    best = best_for_cd(sweep, cd_target_nm=cd_target, tol=tol)
    assert best is not None
    # on target
    assert abs(best.cd_nm - cd_target) <= tol * cd_target
    # minimum LER among all qualifying points
    qualifying = [m for m in sweep
                  if np.isfinite(m.cd_nm) and m.cd_nm > 0
                  and abs(m.cd_nm - cd_target) <= tol * cd_target]
    assert best.ler_3sigma_nm == min(m.ler_3sigma_nm for m in qualifying)


def test_best_for_cd_unreachable_returns_none():
    """best_for_cd returns None when the CD target is outside the swept range."""
    _, _, _, sweep = sweep_over_decade()
    assert best_for_cd(sweep, cd_target_nm=999.0) is None
    # and when cd_min_nm excludes every on-target point
    assert best_for_cd(sweep, cd_target_nm=10.0, tol=0.10, cd_min_nm=12.0) is None


def test_exposure_latitude_window():
    """Exposure latitude: lo<=hi, inside dose range, every dose on target."""
    _, _, doses, sweep = sweep_over_decade()
    cd_target = 10.0
    tol = 0.10
    out = exposure_latitude(sweep, cd_target_nm=cd_target, tol=tol, slack=1.10)
    assert out is not None
    dose_lo, dose_hi, ler_min = out
    assert dose_lo <= dose_hi
    assert dose_lo >= doses.min()
    assert dose_hi <= doses.max()
    # spot-check: every dose in [lo, hi] is on target and LER within slack
    best = best_for_cd(sweep, cd_target_nm=cd_target, tol=tol)
    for m in sweep:
        if dose_lo <= m.dose_pC_cm <= dose_hi:
            assert abs(m.cd_nm - cd_target) <= tol * cd_target
            assert m.ler_3sigma_nm <= (1.10 * best.ler_3sigma_nm) + 1e-9
    # ler_min equals the best on-target LER
    assert ler_min == pytest.approx(best.ler_3sigma_nm, rel=1e-9)


def test_exposure_latitude_none_when_empty():
    """Returns None when no point can satisfy the CD window."""
    _, _, _, sweep = sweep_over_decade()
    assert exposure_latitude(sweep, cd_target_nm=999.0) is None


def test_backward_compat_positional_call():
    """Positional 6-arg call still valid and matches analytic expectation."""
    x, lsf = make_lsf()
    dose = 50.0
    # positional call exactly as run_case.py / scripts do today
    m_pos = evaluate_line(x, lsf, dose, THRESH, 20.0, 7.5)
    # explicit keyword equivalent
    m_kw = evaluate_line(x, lsf, dose_pC_cm=dose, threshold_areal=THRESH,
                         w_se=20.0, a_eff=7.5)
    assert m_pos.cd_nm == pytest.approx(m_kw.cd_nm, rel=1e-12)
    assert m_pos.ler_3sigma_nm == pytest.approx(m_kw.ler_3sigma_nm, rel=1e-12)
    # stochastic 3sigma == 3 * 1sigma (internal consistency)
    assert m_pos.ler_3sigma_nm == pytest.approx(3.0 * m_pos.ler_1sigma_nm, rel=1e-9)
    # CD matches the analytic Gaussian threshold crossing (within sub-pixel error)
    assert m_pos.cd_nm == pytest.approx(cd_analytic(dose), abs=0.15)
    assert m_pos.cd_nm > 0


def test_dose_sweep_keyword_sigma_propagates():
    """dose_sweep forwards sigma_rough_nm to each LineMetrics."""
    x, lsf, doses, sweep = sweep_over_decade(sigma_rough_nm=0.7262)
    for m in sweep:
        assert m.ler_total_3sigma_nm != pytest.approx(m.ler_3sigma_nm, rel=1e-6)
    # default sigma -> total equals stochastic
    _, _, _, sweep0 = sweep_over_decade()
    for m in sweep0:
        assert m.ler_total_3sigma_nm == pytest.approx(m.ler_3sigma_nm, rel=1e-12)
