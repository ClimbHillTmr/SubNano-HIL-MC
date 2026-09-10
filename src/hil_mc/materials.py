"""Material model derived from the single-crystal structure (or any composition).

The composition of the resist is taken from the CIF (C120 H152 O28 Ti6, a Ti6-oxo
cluster), the film density from XRR (Leptos fit of sample 4).  From these we
derive everything the Monte-Carlo transport needs:

* electron density / electrons per gram  (-> XRR critical angle, Bethe)
* Lindhard-Scharff electronic stopping  (Bragg additivity, S_e ~ v ~ sqrt(E))
* ZBL universal nuclear stopping         (Bragg additivity over elements)
* screened-Rutherford deflection coefficients matched to the ZBL nuclear
  stopping, so that lateral straggle is consistent with SRIM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
NA = 6.02214076e23          # 1/mol
E2 = 14.3996                # e^2 in eV*Angstrom
A0 = 0.529177               # Bohr radius (Angstrom)
AMU_KEV = 24.795            # energy (keV) of a 1 amu particle moving at v0

# elemental data:  Z, A (g/mol), mean ionisation potential J (eV, ICRU-37/49)
ELEMENTS: Dict[str, Tuple[float, float, float]] = {
    "H":  (1.0,   1.008,   19.2),
    "B":  (5.0,  10.811,   76.0),
    "C":  (6.0,  12.011,   78.0),
    "N":  (7.0,  14.007,   82.0),
    "O":  (8.0,  15.999,   95.0),
    "F":  (9.0,  18.998,  115.0),
    "Si": (14.0, 28.0855, 173.0),
    "Ti": (22.0, 47.867,  233.0),
    "Zr": (40.0, 91.224,  393.0),
    "Hf": (72.0, 178.49,  650.0),
    "Zn": (30.0, 65.38,   330.0),
    "Sn": (50.0, 118.71,  488.0),
}


# --------------------------------------------------------------------------
def parse_formula(formula: str) -> Dict[str, int]:
    """'C120 H152 O28 Ti6' -> {'C':120, 'H':152, 'O':28, 'Ti':6}"""
    out: Dict[str, int] = {}
    for tok in formula.replace("·", " ").split():
        digits = "".join(ch for ch in tok if ch.isdigit())
        el = "".join(ch for ch in tok if ch.isalpha())
        if not el:
            continue
        out[el] = out.get(el, 0) + (int(digits) if digits else 1)
    return out


@dataclass
class Material:
    name: str
    composition: Dict[str, int]         # atoms per formula unit
    density: float                      # g/cm^3
    formula: str = ""
    molar_mass: float = 0.0
    mass_fraction: Dict[str, float] = field(default_factory=dict)
    atom_fraction: Dict[str, float] = field(default_factory=dict)
    electron_fraction: Dict[str, float] = field(default_factory=dict)
    n_atoms: Dict[str, float] = field(default_factory=dict)   # 1/Angstrom^3
    electrons_per_gram: float = 0.0     # mol e-/g
    electron_density: float = 0.0       # e-/Angstrom^3
    mean_J: float = 0.0                 # eV (Bragg additivity)
    # secondary-electron / exposure parameters
    w_se: float = 20.0                  # eV per created SE (exposure event)
    se_sigma: float = 0.65              # nm, lateral spread of SE energy
    se_lambda_z: float = 5.0            # nm, SE attenuation length (depth)
    se_escape: float = 4.0              # nm, SE escape depth to an interface

    def __post_init__(self):
        self._finalise()

    # ------------------------------------------------------------------
    @classmethod
    def from_formula(cls, name: str, formula: str, density: float, **kw) -> "Material":
        return cls(name=name, composition=parse_formula(formula), density=density,
                   formula=formula, **kw)

    def _finalise(self):
        comp = self.composition
        self.molar_mass = sum(ELEMENTS[e][1] * n for e, n in comp.items())
        tot_atoms = sum(comp.values())
        self.mass_fraction = {e: ELEMENTS[e][1] * n / self.molar_mass for e, n in comp.items()}
        self.atom_fraction = {e: n / tot_atoms for e, n in comp.items()}
        e_counts = {e: ELEMENTS[e][0] * n for e, n in comp.items()}
        tot_e = sum(e_counts.values())
        self.electron_fraction = {e: v / tot_e for e, v in e_counts.items()}
        self.electrons_per_gram = tot_e / self.molar_mass          # mol e-/g
        # number densities: 1/Angstrom^3
        for e, n in comp.items():
            self.n_atoms[e] = self.density * NA / self.molar_mass * n * 1e-24
        self.electron_density = sum(ELEMENTS[e][0] * v for e, v in self.n_atoms.items())
        # Bragg additivity for the mean ionisation potential
        num = den = 0.0
        for e, n in comp.items():
            z, a, j = ELEMENTS[e]
            num += n * z * math.log(j)
            den += n * z
        self.mean_J = math.exp(num / den) if den else 1.0
        if not self.formula:
            self.formula = "".join(f"{e}{n}" for e, n in sorted(comp.items()))

    # ------------------------------------------------------------------
    def xrr_delta(self, wavelength_nm: float = 0.15406) -> float:
        """Refractive index decrement delta = r_e lambda^2 n_e / (2 pi)."""
        r_e = 2.817940e-5   # Angstrom
        lam = wavelength_nm * 10.0
        return r_e * lam ** 2 * self.electron_density / (2 * math.pi)

    def critical_angle_deg(self, wavelength_nm: float = 0.15406) -> float:
        return math.degrees(math.sqrt(2 * self.xrr_delta(wavelength_nm)))

    def summary(self) -> str:
        lines = [
            f"Material            : {self.name}",
            f"Formula             : {self.formula}   M = {self.molar_mass:.2f} g/mol",
            f"Density             : {self.density:.4f} g/cm^3",
            f"Electrons / gram    : {self.electrons_per_gram:.4f} mol e-/g",
            f"Electron density    : {self.electron_density:.4e} e-/cm^3"
            f"  (={self.electron_density:.4f} e-/A^3)",
            f"Mean ionisation J   : {self.mean_J:.1f} eV",
            f"Critical angle (CuKa): theta_c = {self.critical_angle_deg():.4f} deg"
            f"  (2theta_c = {2*self.critical_angle_deg():.4f} deg)",
            "Mass fractions      : "
            + ", ".join(f"{e} {100*v:.2f}%" for e, v in
                        sorted(self.mass_fraction.items(), key=lambda kv: -kv[1])),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
#  Transport for a He+ projectile (Z1 = 2, M1 = 4.0026 amu)
# --------------------------------------------------------------------------
Z1 = 2.0
M1 = 4.0026
LS_PREFACTOR = 8 * math.pi * E2 * A0      # 191.6 eV*A^2
LS_XI = Z1 ** (1.0 / 6.0)                 # 1.1225


def v_over_v0(energy_kev: np.ndarray) -> np.ndarray:
    return np.sqrt(energy_kev / (AMU_KEV * M1))


def universal_screening_length(z2: float) -> float:
    """ZBL universal screening length a_U (Angstrom)."""
    return 0.8853 * A0 / (Z1 ** 0.23 + z2 ** 0.23)


def reduced_energy(energy_kev: np.ndarray, z2: float, m2: float) -> np.ndarray:
    a_u = universal_screening_length(z2)
    # e = E * a * M2 / (Z1 Z2 e^2 (M1+M2));  E in eV
    return (energy_kev * 1e3) * a_u * m2 / (Z1 * z2 * E2 * (M1 + m2))


def zbl_nuclear_stopping_reduced(eps: np.ndarray) -> np.ndarray:
    """Universal (reduced) nuclear stopping s_n(eps), ZBL 1985."""
    eps = np.asarray(eps, dtype=float)
    low = eps < 30.0
    hi = ~low
    out = np.empty_like(eps)
    e = np.clip(eps[low], 1e-12, None)
    out[low] = 0.5 * np.log(1.0 + 1.1383 * e) / (
        e + 0.01321 * e ** 0.21226 + 0.19593 * np.sqrt(e))
    eh = np.clip(eps[hi], 1e-12, None)
    out[hi] = np.log(eh) / (2.0 * eh)
    return out


class HeTransport:
    """Pre-computed He+ transport coefficients for one material.

    All cross-sections use the screened-Rutherford form

        dsigma/dOmega = (Z1 Z2 e^2 / 4E)^2 / (sin^2(theta/2) + alpha)^2

    with alpha(eps) obtained by matching the transport integral to the ZBL
    universal nuclear stopping, i.e. the deflection statistics are consistent
    with SRIM's nuclear stopping.
    """

    GRID_E = np.logspace(0.0, 3.0, 241)     # keV, 1 keV .. 1 MeV

    def __init__(self, mat: Material, theta_cut_deg: float = 5.0,
                 e_scale: float = 1.0):
        self.mat = mat
        self.e_scale = e_scale
        self.sc = math.sin(math.radians(theta_cut_deg) / 2.0) ** 2
        self.elements = list(mat.n_atoms.keys())
        self.z = np.array([ELEMENTS[e][0] for e in self.elements])
        self.m = np.array([ELEMENTS[e][1] for e in self.elements])
        self.n = np.array([mat.n_atoms[e] for e in self.elements])   # 1/A^3

        # ---- electronic stopping coefficient: S_e = K_e * sqrt(E_keV) [eV/nm]
        ke = np.zeros_like(self.z)
        for i, z2 in enumerate(self.z):
            ke[i] = (LS_PREFACTOR * LS_XI * (Z1 * z2)
                     / (Z1 ** (2 / 3) + z2 ** (2 / 3)) ** 1.5
                     / math.sqrt(AMU_KEV * M1))
        self.K_e = float(np.sum(self.n * ke) * 10.0 * e_scale)   # eV/nm

        # ---- nuclear stopping: (dE/dx)_n = N * 4 pi a_U Z1Z2 e^2 M1/(M1+M2) * s_n
        self.a_u = np.array([universal_screening_length(z) for z in self.z])
        self.gamma = 4.0 * M1 * self.m / (M1 + self.m) ** 2
        self.K_n = (self.n * 4 * math.pi * self.a_u * (Z1 * self.z * E2)
                    * (M1 / (M1 + self.m)) * 10.0)      # eV/nm per unit s_n

        # ---- alpha(E) table by matching screened Rutherford to ZBL S_n
        self.alpha_table = np.array([self._solve_alpha(e) for e in self.GRID_E])

    # -----------------------------------------------------------------
    def _solve_alpha(self, energy_kev: float) -> np.ndarray:
        """alpha per element so that the Rutherford transport = ZBL S_n."""
        eps = reduced_energy(np.array([energy_kev]), self.z, self.m)[:, None] \
            if False else reduced_energy(np.full_like(self.z, energy_kev), self.z, self.m)
        s_n = zbl_nuclear_stopping_reduced(eps)
        target = self.K_n * s_n                     # eV/nm, per element
        pref = math.pi * self.gamma * (Z1 * self.z * E2) ** 2 / (4.0 * energy_kev * 1e3)
        pref *= self.n * 10.0                       # eV/nm (units of A^2 folded in)
        # target = pref * [ln((1+a)/a) - 1/(1+a)]
        out = np.empty_like(target)
        for i, t in enumerate(np.atleast_1d(target)):
            f = t / pref[i]
            if f <= 0:
                out[i] = 1.0
                continue
            lo, hi = 1e-14, 1.0
            for _ in range(80):
                mid = math.sqrt(lo * hi)
                val = math.log((1 + mid) / mid) - 1.0 / (1 + mid)
                if val > f:
                    lo = mid
                else:
                    hi = mid
            out[i] = math.sqrt(lo * hi)
        return out

    def alpha(self, energy_kev):
        e = np.clip(np.asarray(energy_kev, dtype=float), 1e-3, 1e3)
        return np.exp(np.vstack([
            np.interp(np.log(e), np.log(self.GRID_E), np.log(self.alpha_table[:, i]))
            for i in range(len(self.z))]).T)

    # -----------------------------------------------------------------
    def stopping_electronic(self, energy_kev):
        return self.K_e * np.sqrt(np.clip(energy_kev, 1e-6, None))

    def stopping_nuclear(self, energy_kev):
        eps = reduced_energy(np.clip(energy_kev, 1e-6, None)[..., None], self.z, self.m)
        s_n = zbl_nuclear_stopping_reduced(eps)
        return np.sum(self.K_n * s_n, axis=-1)

    def transport_coefficients(self, energy_kev):
        """(mean-square angle per nm, hard-collision rate per nm) for a step."""
        e = np.clip(np.asarray(energy_kev, dtype=float), 1e-3, None)
        a = self.alpha(e)                                   # (..., n_el)
        base = 4 * math.pi * (Z1 * self.z * E2 / (4.0 * e[..., None] * 1e3)) ** 2
        # small-angle (theta < theta_c) transport cross-section ~ int theta^2 dsigma
        ms = base * 4.0 * (np.log((self.sc + a) / a) - self.sc / (self.sc + a))
        # hard collisions (theta > theta_c)
        hard = base * (1.0 / (self.sc + a) - 1.0 / (1.0 + a))
        theta2_per_nm = np.sum(self.n * ms, axis=-1) * 10.0     # rad^2 / nm
        rate_per_nm = np.sum(self.n * hard, axis=-1) * 10.0     # 1 / nm
        return theta2_per_nm, rate_per_nm

    def sample_hard_angle(self, energy_kev, rng):
        """Sample sin^2(theta/2) for a hard collision (element-resolved)."""
        e = np.clip(np.asarray(energy_kev, dtype=float), 1e-3, None)
        a = self.alpha(e)
        base = 4 * math.pi * (Z1 * self.z * E2 / (4.0 * e[..., None] * 1e3)) ** 2
        w = self.n * base * (1.0 / (self.sc + a) - 1.0 / (1.0 + a))
        w = w / np.clip(w.sum(axis=-1, keepdims=True), 1e-300, None)
        # pick element
        cdf = np.cumsum(w, axis=-1)
        u = rng.random(w.shape[:-1])[..., None]
        idx = np.argmax(u < cdf, axis=-1)
        a_sel = np.take_along_axis(a, idx[..., None], axis=-1)[..., 0]
        # inverse CDF of p(s) ~ 1/(s+a)^2 on [sc, 1]
        inv0, inv1 = 1.0 / (self.sc + a_sel), 1.0 / (1.0 + a_sel)
        uu = rng.random(a_sel.shape)
        s = 1.0 / (inv0 - uu * (inv0 - inv1)) - a_sel
        return np.clip(s, 0.0, 1.0)
