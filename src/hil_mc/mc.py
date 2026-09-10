"""Vectorised Monte-Carlo transport of 30 keV He+ through a layered stack.

Physics
-------
* electronic stopping : Lindhard-Scharff (S_e ~ v ~ sqrt(E)), Bragg additivity,
  globally scaled by one calibration constant fitted to the SRIM projected
  range of 30 keV He in Si (282.2 nm).
* nuclear stopping    : ZBL universal s_n(eps), Bragg additivity.
* angular deflection : screened Rutherford with a small-angle (theta < 5 deg)
  multiple-scattering term plus discrete hard collisions; the screening
  parameter alpha(E) is obtained by matching the ZBL nuclear stopping, so the
  lateral straggle follows SRIM.
* secondary electrons: the electronic energy loss is redistributed with an
  isotropic Gaussian kernel (sigma = se_sigma) - i.e. the SE1 delocalisation.
* substrate coupling: electronic loss inside the Si within one SE escape depth
  of the interface is recorded; a fraction se_transmission is re-injected into
  the resist (SE2 / proximity background).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from .materials import HeTransport, Material


@dataclass
class Layer:
    name: str
    z0: float            # nm, top of the layer
    z1: float            # nm, bottom
    material: Material
    transport: HeTransport
    is_resist: bool = False


@dataclass
class Stack:
    layers: List[Layer]
    name: str = "stack"
    is_suspended: bool = False

    @property
    def resist(self) -> Layer:
        return next(l for l in self.layers if l.is_resist)

    def layer_of(self, z: np.ndarray) -> np.ndarray:
        idx = np.full(z.shape, -1, dtype=int)
        for i, l in enumerate(self.layers):
            m = (z >= l.z0) & (z < l.z1)
            idx[m] = i
        return idx


@dataclass
class MCResult:
    n_ions: int
    box_nm: float
    pixel_nm: float
    # 2D maps (areal energy density per incident ion, eV/nm^2 per ion)
    direct_map: np.ndarray          # electronic loss inside the resist
    nuclear_map: np.ndarray
    substrate_map: np.ndarray       # SE2 source re-injected into the resist
    exposure_map: np.ndarray        # direct + SE1 spreading + SE2
    # depth profile inside the resist (eV/nm per ion vs depth)
    depth_edges: np.ndarray
    depth_profile: np.ndarray
    # substrate bookkeeping
    energy_in_resist: float         # eV per ion
    energy_in_substrate_total: float
    energy_backscattered_out: float
    backscatter_fraction: float
    backscatter_radii: np.ndarray
    backscatter_energies: np.ndarray
    transmission_fraction: float
    range_substrate: float          # mean stopping depth inside the substrate
    # defect-1: full-depth (all-layer) depth-dose curve, eV/nm per ion
    depth_edges_full: np.ndarray = None          # depth bin edges, nm
    depth_profile_full: np.ndarray = None        # eV/nm per ion vs depth
    # defect-2: end-of-range diagnostics
    n_alive_at_end: int = 0         # ions still alive when the step loop exited
    n_escaped_lateral: int = 0      # ions that left the lateral 2D-map box
    energy_transmitted_out: float = 0.0  # eV/ion carried out by deep transmission
    trajectories: List[np.ndarray] = field(default_factory=list)

    @property
    def grid(self):
        n = self.direct_map.shape[0]
        return (np.arange(n) - n / 2 + 0.5) * self.pixel_nm


# ---------------------------------------------------------------------------
def make_stack(resist: Material, substrate: Optional[Material],
               resist_thickness_nm: float = 40.0,
               substrate_thickness_nm: float = 670_000.0,
               membrane: Optional[Material] = None,
               membrane_thickness_nm: float = 0.0,
               e_scale: float = 1.0) -> Stack:
    """Build a Layer stack.

    resist   : 0 .. t_resist
    membrane : t_resist .. t_resist + t_mem   (optional, e.g. 20 nm SiNx)
    substrate: below the membrane; ``None`` means vacuum (suspended membrane).
    """
    layers = [
        Layer("resist", 0.0, resist_thickness_nm, resist,
              HeTransport(resist, e_scale=e_scale), is_resist=True)
    ]
    z = resist_thickness_nm
    if membrane is not None and membrane_thickness_nm > 0:
        layers.append(Layer("membrane", z, z + membrane_thickness_nm, membrane,
                            HeTransport(membrane, e_scale=e_scale)))
        z += membrane_thickness_nm
    if substrate is not None:
        layers.append(Layer("substrate", z, z + substrate_thickness_nm, substrate,
                            HeTransport(substrate, e_scale=e_scale)))
    st = Stack(layers)
    st.is_suspended = substrate is None
    return st


# ---------------------------------------------------------------------------
class HeMonteCarlo:
    def __init__(self, stack: Stack, energy_kev: float = 30.0, seed: int = 20240909,
                 step_nm: float = 1.0, box_nm: float = 80.0, pixel_nm: float = 0.25,
                 se_transmission: float = 0.35):
        self.stack = stack
        self.E0 = energy_kev
        self.step = step_nm
        self.box = box_nm
        self.pixel = pixel_nm
        self.nbins = int(round(2 * box_nm / pixel_nm))
        self.se_transmission = se_transmission
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def run(self, n_ions: int, keep_trajectories: int = 0,
             depth_max_nm: float = 600.0) -> MCResult:
        rng = self.rng
        nb = self.nbins
        resist_i = [i for i, l in enumerate(self.stack.layers) if l.is_resist][0]
        resist = self.stack.layers[resist_i]
        t_res = resist.z1
        # layer directly beneath the resist (Si bulk, SiNx membrane or nothing)
        sub_layer = (self.stack.layers[resist_i + 1]
                     if len(self.stack.layers) > resist_i + 1 else resist)
        z_int = resist.z1                       # resist / underlayer interface

        direct = np.zeros((nb, nb))
        nuclear = np.zeros((nb, nb))
        subsrc = np.zeros((nb, nb))
        depth_edges = np.linspace(0, t_res, 81)
        depth_hist = np.zeros(len(depth_edges) - 1)
        # defect 1: full-depth deposition histogram covering 0 .. depth_max_nm
        # (bins *all* electronic + nuclear loss regardless of layer, so the
        # model can produce the substrate depth-dose / Bragg-peak curve).
        depth_edges_full = np.linspace(0.0, depth_max_nm, 121)
        depth_hist_full = np.zeros(len(depth_edges_full) - 1)
        dw_full = depth_edges_full[1] - depth_edges_full[0]

        # per-layer accumulators
        E_in_resist = 0.0
        E_in_sub = 0.0
        E_back_out = 0.0
        E_trans_out = 0.0
        n_back = 0
        n_trans = 0
        n_escaped_lateral = 0
        back_r = []
        back_E = []
        stop_z = []
        escaped_lat = np.zeros(n_ions, dtype=bool)

        # state
        x = np.zeros(n_ions)
        y = np.zeros(n_ions)
        z = np.zeros(n_ions)
        ux = np.zeros(n_ions)
        uy = np.zeros(n_ions)
        uz = np.ones(n_ions)
        E = np.full(n_ions, self.E0)
        alive = np.ones(n_ions, dtype=bool)
        traj_idx = np.arange(min(keep_trajectories, n_ions))
        traj: Dict[int, list] = {int(i): [] for i in traj_idx}

        half = self.box
        for _ in range(200000):
            if not alive.any():
                break
            a = np.flatnonzero(alive)
            li = self.stack.layer_of(z[a])
            # -- step length (shorter near the end of range)
            ds = np.full(a.size, self.step)
            ds = np.minimum(ds, np.maximum(E[a] * 1e3 / 200.0, 0.25))  # >~200 eV/nm guard

            # -- stopping & scattering per layer
            dEe = np.zeros(a.size)
            dEn = np.zeros(a.size)
            th2 = np.zeros(a.size)
            rate = np.zeros(a.size)
            for k, layer in enumerate(self.stack.layers):
                m = li == k
                if not m.any():
                    continue
                tr = layer.transport
                ee = E[a][m]
                # stopping powers are eV/nm; the state variable E is in keV
                dEe[m] = tr.stopping_electronic(ee) * ds[m] * 1e-3
                dEn[m] = tr.stopping_nuclear(ee) * ds[m] * 1e-3
                t2, rt = tr.transport_coefficients(ee)
                th2[m] = t2 * ds[m]
                rate[m] = rt * ds[m]
            dEe = np.minimum(dEe, np.maximum(E[a] - 0.02, 0.0))
            dEn = np.minimum(dEn, np.maximum(E[a] - dEe - 0.02, 0.0))

            # -- record deposition inside the resist
            in_res = (li == resist_i)
            if in_res.any():
                ii = a[in_res]
                in_box = (np.abs(x[ii]) <= half) & (np.abs(y[ii]) <= half)
                if in_box.any():
                    jj = ii[in_box]
                    ix = np.clip(((x[jj] + half) / self.pixel).astype(int), 0, nb - 1)
                    iy = np.clip(((y[jj] + half) / self.pixel).astype(int), 0, nb - 1)
                    flat = ix * nb + iy
                    direct += np.bincount(flat, weights=dEe[in_res][in_box] * 1e3,
                                          minlength=nb * nb).reshape(nb, nb)
                    nuclear += np.bincount(flat, weights=dEn[in_res][in_box] * 1e3,
                                           minlength=nb * nb).reshape(nb, nb)
                ddz = np.clip(np.digitize(z[ii], depth_edges) - 1, 0, len(depth_hist) - 1)
                depth_hist += np.bincount(ddz, weights=dEe[in_res] * 1e3,
                                          minlength=len(depth_hist))
                E_in_resist += float(dEe[in_res].sum()) * 1e3

            # -- substrate: SE2 source within one escape depth of the interface
            in_sub = (z[a] >= z_int) & (z[a] < z_int + 6.0 * sub_layer.material.se_escape)
            if in_sub.any():
                ii = a[in_sub]
                w = dEe[in_sub] * 1e3 * np.exp(-(z[ii] - z_int) / sub_layer.material.se_escape)
                in_box = (np.abs(x[ii]) <= half) & (np.abs(y[ii]) <= half)
                if in_box.any():
                    jj = ii[in_box]
                    ix = np.clip(((x[jj] + half) / self.pixel).astype(int), 0, nb - 1)
                    iy = np.clip(((y[jj] + half) / self.pixel).astype(int), 0, nb - 1)
                    subsrc += np.bincount(ix * nb + iy, weights=w[in_box],
                                          minlength=nb * nb).reshape(nb, nb)
                E_in_sub += float(dEe[in_sub].sum()) * 1e3

            # -- defect 1: full-depth deposition (all layers, e + n loss) -----
            dfull = (dEe + dEn) * 1e3
            idx_full = np.clip(np.digitize(z[a], depth_edges_full) - 1,
                               0, len(depth_hist_full) - 1)
            depth_hist_full += np.bincount(idx_full, weights=dfull,
                                           minlength=len(depth_hist_full))

            # -- advance
            E[a] -= (dEe + dEn)
            #  small-angle multiple scattering
            sx = rng.normal(scale=np.sqrt(np.maximum(th2, 0)))
            sy = rng.normal(scale=np.sqrt(np.maximum(th2, 0)))
            #  hard collisions
            nh = rng.poisson(np.maximum(rate, 0))
            ang = np.zeros(a.size)
            azi = rng.uniform(0, 2 * np.pi, a.size)
            hm = nh > 0
            if hm.any():
                s2 = np.zeros(a.size)
                for k, layer in enumerate(self.stack.layers):
                    m = hm & (li == k)
                    if m.any():
                        s2[m] = layer.transport.sample_hard_angle(E[a][m], rng)
                ang = 2 * np.arcsin(np.sqrt(np.clip(s2, 0, 1)))
            self._rotate(ux, uy, uz, a, sx + ang * np.cos(azi), sy + ang * np.sin(azi))
            x[a] += ux[a] * ds
            y[a] += uy[a] * ds
            z[a] += uz[a] * ds

            for i in traj_idx:
                if alive[i]:
                    traj[int(i)].append((x[i], y[i], z[i]))

            # -- termination
            dead = E[a] <= 0.02
            if dead.any():
                ii = a[dead]
                # projected (resting) depth of every stopped ion, clipped at the
                # surface so backscattered ions contribute ~0 to the range.
                stop_z.extend(np.maximum(z[ii], 0.0).tolist())
                alive[ii] = False
            out_top = z[a] < -2.0
            if out_top.any():
                ii = a[out_top]
                r = np.hypot(x[ii], y[ii])
                back_r.extend(r.tolist())
                back_E.extend(E[ii].tolist())
                E_back_out += float(E[ii].sum()) * 1e3
                n_back += int(ii.size)
                alive[ii] = False
                # backscattered ions come to rest at the surface (z ~ 0)
                bt = out_top & ~dead
                stop_z.extend([0.0] * int(bt.sum()))
            # -- defect 2: lateral escape must NOT terminate the ion.  The 80 nm
            # box is only the domain of the 2D exposure map; an ion that
            # straggles outside it is still stopping in the substrate, so its
            # depth (range) and energy deposition are tracked to completion.
            # We merely count how many ions leave the lateral box (diagnostic).
            out_lat = (np.abs(x[a]) > half) | (np.abs(y[a]) > half)
            if out_lat.any():
                new_esc = out_lat & (~escaped_lat[a])
                n_escaped_lateral += int(new_esc.sum())
                escaped_lat[a[out_lat]] = True
            # -- genuine deep transmission (z > 4 um); should not occur for
            # 30 keV He in Si, but counted as a real transmission if it does.
            out_deep = z[a] > 4e3
            if out_deep.any():
                ii = a[out_deep]
                E_trans_out += float(E[ii].sum()) * 1e3
                n_trans += int(ii.size)
                alive[ii] = False

        # ---- diagnostics / normalise --------------------------------------
        n_alive_at_end = int(alive.sum())

        # ---- normalise maps to areal energy density per ion [eV/nm^2 per ion]
        norm = (self.pixel ** 2) * n_ions
        direct /= norm
        nuclear /= norm
        subsrc /= norm

        # ---- SE1 redistribution + SE2 injection -------------------------------
        sig_pix = resist.material.se_sigma / self.pixel
        spread = gaussian_filter(direct, sig_pix, mode="constant")
        # substrate SE2: enter at the interface, spread laterally (~3 nm) and
        # attenuate over se_lambda_z while travelling back through the resist
        sub_term = gaussian_filter(subsrc * self.se_transmission, 3.0 / self.pixel,
                                   mode="constant")
        frac_in_resist = 1.0 - np.exp(-t_res / sub_layer.material.se_lambda_z)
        sub_term *= frac_in_resist
        exposure = spread + nuclear * 0.0 + sub_term

        return MCResult(
            n_ions=n_ions, box_nm=self.box, pixel_nm=self.pixel,
            direct_map=direct, nuclear_map=nuclear, substrate_map=sub_term,
            exposure_map=exposure,
            depth_edges=depth_edges,
            depth_profile=depth_hist / max(n_ions, 1),
            energy_in_resist=E_in_resist / n_ions,
            energy_in_substrate_total=E_in_sub / n_ions,
            energy_backscattered_out=E_back_out / n_ions,
            backscatter_fraction=n_back / n_ions,
            backscatter_radii=np.array(back_r), backscatter_energies=np.array(back_E),
            transmission_fraction=n_trans / n_ions,
            range_substrate=float(np.mean(stop_z)) if stop_z else 0.0,
            depth_edges_full=depth_edges_full,
            depth_profile_full=depth_hist_full / max(n_ions, 1) / dw_full,
            n_alive_at_end=n_alive_at_end,
            n_escaped_lateral=n_escaped_lateral,
            energy_transmitted_out=E_trans_out / n_ions,
            trajectories=[np.array(traj[int(i)]) for i in traj_idx],
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _rotate(ux, uy, uz, idx, dx, dy):
        """Small rotation of the direction vector by (dx, dy) around the axes."""
        x = ux[idx]
        y = uy[idx]
        z = uz[idx]
        # rotate about y-axis by dx, then about x-axis by dy (small-angle exact)
        c1, s1 = np.cos(dx), np.sin(dx)
        x2 = x * c1 + z * s1
        z2 = -x * s1 + z * c1
        y2 = y
        c2, s2 = np.cos(dy), np.sin(dy)
        y3 = y2 * c2 - z2 * s2
        z3 = y2 * s2 + z2 * c2
        n = np.sqrt(x2 ** 2 + y3 ** 2 + z3 ** 2)
        ux[idx] = x2 / n
        uy[idx] = y3 / n
        uz[idx] = z3 / n
