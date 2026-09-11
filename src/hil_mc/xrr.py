"""X-ray reflectivity (XRR) forward model and fit for the resist film.

The pipeline previously carried the XRR thickness / density / roughness as
hard-coded constants copied from a vendor report.  This module derives them
from the raw measurement in ``data/0828-24355-new/.../4.txt`` so every number
used downstream is traceable to data.

Physics
-------
* **Optical constants.**  For X-rays the refractive index is
  ``n = 1 - delta - i*beta`` with

      delta = r_e * lambda^2 * rho_e / (2*pi),

  ``r_e = 2.8179403262e-6 nm`` the Thomson radius, ``lambda`` the wavelength and
  ``rho_e`` the electron number density.  ``beta`` follows from the mass
  attenuation coefficient; for a low-Z organic film at Cu-Ka it is ~1e-3 of
  ``delta`` and is carried as a fitted scale on a literature-typical ratio.
* **Parratt recursion** (Parratt, Phys. Rev. 95, 359 (1954)) gives the exact
  specular reflectivity of a stratified stack.
* **Nevot-Croce factor** ``exp(-2*kz_j*kz_{j+1}*sigma^2)`` damps each interface
  with its RMS roughness ``sigma``.

Fit conventions
---------------
* Fitting is done on ``log10`` reflectivity, which weights the fringe decades
  evenly instead of letting the total-reflection plateau dominate.
* Data below ``theta_min`` (default: 2-theta = 0.70 deg) is excluded because the
  beam footprint exceeds the sample there and the measured plateau is a
  geometry artefact, not reflectivity.
* Reported uncertainties are 1-sigma from the Jacobian at the optimum; they are
  *fit* uncertainties and do not include systematic (footprint, resolution)
  error.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

R_E_NM = 2.8179403262e-6          # Thomson scattering length, nm
CU_KA_NM = 0.154184               # Cu-Kalpha1 wavelength, nm
N_A = 6.02214076e23

# electrons per gram (Z/A) and mass attenuation used for the two materials.
# Si: Z/A = 14/28.0855; the Ti6-oxo film Z/A comes from the CIF formula.
SI_DENSITY = 2.3290               # g/cm^3
SI_Z_OVER_A = 14.0 / 28.0855
# beta/delta for Cu-Ka: Si ~ 0.017, low-Z organics ~ 0.004 (literature-typical).
SI_BETA_OVER_DELTA = 0.017
FILM_BETA_OVER_DELTA = 0.004


def electron_density_per_nm3(density_g_cm3: float, z_over_a: float) -> float:
    """Electron number density in electrons / nm^3."""
    # (g/cm^3) * (mol e-/g) * (e-/mol) = e-/cm^3, then /1e21 -> e-/nm^3
    return float(density_g_cm3) * float(z_over_a) * N_A / 1e21


def delta_from_density(density_g_cm3: float, z_over_a: float,
                       wavelength_nm: float = CU_KA_NM) -> float:
    """Real part of the refractive-index decrement ``delta``."""
    rho_e = electron_density_per_nm3(density_g_cm3, z_over_a)
    return R_E_NM * wavelength_nm ** 2 * rho_e / (2.0 * math.pi)


def density_from_delta(delta: float, z_over_a: float,
                       wavelength_nm: float = CU_KA_NM) -> float:
    """Invert :func:`delta_from_density` (mass density in g/cm^3)."""
    rho_e = delta * 2.0 * math.pi / (R_E_NM * wavelength_nm ** 2)   # e-/nm^3
    return rho_e * 1e21 / (float(z_over_a) * N_A)


def critical_angle_deg(delta: float) -> float:
    """Total-reflection critical angle (deg) for a given ``delta``."""
    return math.degrees(math.sqrt(max(2.0 * delta, 0.0)))


def parratt_reflectivity(theta_deg: np.ndarray,
                         thickness_nm: float,
                         delta_film: float,
                         delta_sub: float,
                         sigma_film_nm: float,
                         sigma_sub_nm: float,
                         *,
                         beta_film: Optional[float] = None,
                         beta_sub: Optional[float] = None,
                         wavelength_nm: float = CU_KA_NM) -> np.ndarray:
    """Specular reflectivity of one film on a semi-infinite substrate.

    Parameters
    ----------
    theta_deg : incidence angles (grazing, deg) -- i.e. 2-theta / 2.
    thickness_nm : film thickness.
    delta_film, delta_sub : refractive-index decrements.
    sigma_film_nm : RMS roughness of the vacuum/film interface.
    sigma_sub_nm : RMS roughness of the film/substrate interface.

    Returns
    -------
    Reflectivity in [0, 1], same shape as ``theta_deg``.
    """
    if beta_film is None:
        beta_film = FILM_BETA_OVER_DELTA * delta_film
    if beta_sub is None:
        beta_sub = SI_BETA_OVER_DELTA * delta_sub

    th = np.radians(np.asarray(theta_deg, dtype=float))
    k0 = 2.0 * math.pi / wavelength_nm
    c2 = np.cos(th) ** 2

    # kz in vacuum / film / substrate
    kz0 = k0 * np.sqrt(np.asarray(1.0 - c2, dtype=complex))
    kz1 = k0 * np.sqrt((1.0 - 2.0 * delta_film - 2j * beta_film) - c2)
    kz2 = k0 * np.sqrt((1.0 - 2.0 * delta_sub - 2j * beta_sub) - c2)

    def fresnel(kza, kzb, sigma):
        r = (kza - kzb) / (kza + kzb)
        return r * np.exp(-2.0 * kza * kzb * (sigma ** 2))

    r01 = fresnel(kz0, kz1, sigma_film_nm)
    r12 = fresnel(kz1, kz2, sigma_sub_nm)
    phase = np.exp(2j * kz1 * thickness_nm)
    r = (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)
    return np.abs(r) ** 2


@dataclass
class XRRData:
    two_theta_deg: np.ndarray
    intensity: np.ndarray
    path: str

    @property
    def theta_deg(self) -> np.ndarray:
        return self.two_theta_deg / 2.0


def load_xrr(path: str) -> XRRData:
    """Read the Bruker/COMMANDER two-column export (2-theta, counts)."""
    tt, inten = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue                     # header row
            tt.append(a)
            inten.append(b)
    if not tt:
        raise ValueError(f"no numeric XRR rows found in {path}")
    return XRRData(np.asarray(tt), np.asarray(inten), path)


@dataclass
class KiessigFit:
    thickness_nm: float
    critical_angle_deg: float
    r_squared: float
    n_fringes: int
    orders: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    peak_two_theta_deg: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def kiessig_thickness(data: XRRData,
                      *,
                      wavelength_nm: float = CU_KA_NM,
                      prominence: float = 0.05,
                      two_theta_min: float = 0.9) -> KiessigFit:
    """Model-free thickness from Kiessig fringes (modified Bragg law).

    ``theta_m^2 = theta_c^2 + (m*lambda / (2 d))^2`` is linear in ``m^2``, so a
    straight-line fit of ``theta^2`` against ``m^2`` gives both the thickness
    and the critical angle without assuming a density.  The fringe order offset
    is scanned and the offset with the best linearity is kept.
    """
    from scipy.signal import find_peaks

    m_keep = data.two_theta_deg >= two_theta_min
    tt = data.two_theta_deg[m_keep]
    inten = data.intensity[m_keep]
    log_i = np.log10(np.maximum(inten, 1.0))
    peaks, _ = find_peaks(log_i, prominence=prominence)
    if peaks.size < 3:
        raise ValueError("too few Kiessig fringes to fit a thickness")

    theta = np.radians(tt[peaks] / 2.0)
    best: Optional[KiessigFit] = None
    for offset in range(1, 8):
        m = np.arange(peaks.size) + offset
        slope, intercept = np.polyfit(m ** 2, theta ** 2, 1)
        if slope <= 0:
            continue
        d = wavelength_nm / (2.0 * math.sqrt(slope))
        r2 = float(np.corrcoef(m ** 2, theta ** 2)[0, 1] ** 2)
        cand = KiessigFit(
            thickness_nm=float(d),
            critical_angle_deg=float(math.degrees(math.sqrt(max(intercept, 0.0)))),
            r_squared=r2,
            n_fringes=int(peaks.size),
            orders=m,
            peak_two_theta_deg=tt[peaks],
        )
        if best is None or cand.r_squared > best.r_squared:
            best = cand
    if best is None:
        raise ValueError("Kiessig fit failed (no positive slope)")
    return best


@dataclass
class XRRFit:
    thickness_nm: float
    density_g_cm3: float
    sigma_film_nm: float
    sigma_sub_nm: float
    sigma_film_err_nm: float
    thickness_err_nm: float
    density_err_g_cm3: float
    delta_film: float
    critical_angle_deg: float
    critical_two_theta_deg: float
    electron_density_per_nm3: float
    rms_log10_residual: float
    n_points: int
    two_theta_min_deg: float
    kiessig: Optional[KiessigFit] = None
    theta_deg: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    r_measured: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    r_model: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if not isinstance(v, np.ndarray) and k != "kiessig"}
        if self.kiessig is not None:
            d["kiessig"] = dict(thickness_nm=self.kiessig.thickness_nm,
                                critical_angle_deg=self.kiessig.critical_angle_deg,
                                r_squared=self.kiessig.r_squared,
                                n_fringes=self.kiessig.n_fringes)
        return d


def fit_xrr(data: XRRData,
            *,
            film_z_over_a: float,
            wavelength_nm: float = CU_KA_NM,
            two_theta_min: float = 0.70,
            thickness_guess_nm: Optional[float] = None,
            density_guess_g_cm3: float = 0.9,
            sigma_guess_nm: float = 0.7) -> XRRFit:
    """Fit thickness / density / roughness with the Parratt model.

    ``film_z_over_a`` (electrons per gram) comes from the CIF formula so the
    fitted ``delta`` converts to a mass density without extra assumptions.
    """
    from scipy.optimize import least_squares

    kie: Optional[KiessigFit]
    try:
        kie = kiessig_thickness(data, wavelength_nm=wavelength_nm)
    except Exception:
        kie = None
    if thickness_guess_nm is None:
        thickness_guess_nm = kie.thickness_nm if kie else 24.0

    keep = data.two_theta_deg >= two_theta_min
    th = data.theta_deg[keep]
    inten = data.intensity[keep]
    # normalise to the plateau just above the excluded region so the model,
    # which is a true reflectivity in [0, 1], is comparable to counts.
    norm = float(np.max(data.intensity))
    r_meas = np.maximum(inten / norm, 1e-12)
    y = np.log10(r_meas)

    delta_sub = delta_from_density(SI_DENSITY, SI_Z_OVER_A, wavelength_nm)

    def unpack(p):
        thick, dens, sig_f, sig_s, log_scale, log_bkg = p
        return thick, dens, sig_f, sig_s, log_scale, log_bkg

    def model_log(p):
        thick, dens, sig_f, sig_s, log_scale, log_bkg = unpack(p)
        delta_f = delta_from_density(dens, film_z_over_a, wavelength_nm)
        r = parratt_reflectivity(th, thick, delta_f, delta_sub,
                                 abs(sig_f), abs(sig_s),
                                 wavelength_nm=wavelength_nm)
        return np.log10(np.maximum(r * 10.0 ** log_scale + 10.0 ** log_bkg, 1e-14))

    def resid(p):
        return model_log(p) - y

    p0 = np.array([thickness_guess_nm, density_guess_g_cm3, sigma_guess_nm,
                   sigma_guess_nm * 0.6, 0.0, -7.0])
    lo = np.array([thickness_guess_nm * 0.5, 0.3, 0.05, 0.05, -2.0, -12.0])
    hi = np.array([thickness_guess_nm * 2.0, 2.5, 3.0, 3.0, 2.0, -3.0])
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    sol = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac",
                        max_nfev=20000)
    thick, dens, sig_f, sig_s, log_scale, log_bkg = unpack(sol.x)
    sig_f, sig_s = abs(sig_f), abs(sig_s)

    # 1-sigma parameter errors from the Jacobian at the optimum
    res = sol.fun
    dof = max(res.size - sol.x.size, 1)
    s2 = float(res @ res) / dof
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        err = np.full(sol.x.size, np.nan)

    delta_f = delta_from_density(dens, film_z_over_a, wavelength_nm)
    return XRRFit(
        thickness_nm=float(thick),
        density_g_cm3=float(dens),
        sigma_film_nm=float(sig_f),
        sigma_sub_nm=float(sig_s),
        thickness_err_nm=float(err[0]),
        density_err_g_cm3=float(err[1]),
        sigma_film_err_nm=float(err[2]),
        delta_film=float(delta_f),
        critical_angle_deg=float(critical_angle_deg(delta_f)),
        critical_two_theta_deg=float(2.0 * critical_angle_deg(delta_f)),
        electron_density_per_nm3=float(electron_density_per_nm3(dens, film_z_over_a)),
        rms_log10_residual=float(np.sqrt(np.mean(res ** 2))),
        n_points=int(th.size),
        two_theta_min_deg=float(two_theta_min),
        kiessig=kie,
        theta_deg=th,
        r_measured=r_meas,
        r_model=10.0 ** model_log(sol.x),
    )


def default_xrr_path(root: str) -> str:
    """Locate the sample-4 XRR export inside the repository ``data`` tree."""
    return os.path.join(root, "data", "0828-24355-new", "0828-24355-new",
                        "DATA", "4.txt")
