"""Single-crystal (CIF) -> cluster chemistry + material parameters.

Everything here is derived from the CIF itself (no external database): the
molecule is re-assembled across periodic images, the bond graph is built from
covalent radii, and the Ti-oxo core is separated from the organic ligands by
graph partitioning after removing the metal atoms.
"""
from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cifio import (assemble_molecule, build_bonds, num,
                    read_structure, _strip_quotes)
from .materials import Material


@dataclass
class Ligand:
    formula: str
    count: int
    n_Ti: int
    ti_labels: List[str]
    binding: str
    note: str = ""


@dataclass
class StructureAnalysis:
    path: str
    data_name: str
    formula: str
    moiety: str
    fw: float
    crystal_system: str
    space_group: str
    it_number: int
    cell: Tuple[float, ...]
    volume: float
    z: int
    z_prime: float
    density_calc: float
    temperature_K: float
    wavelength: float
    r1: Optional[float]
    wr2: Optional[float]
    goof: Optional[float]
    completeness: Optional[float]
    theta_max: Optional[float]
    n_atoms_asu: int
    n_atoms_molecule: int
    ti_indices: List[int]
    ti_ti_distances: List[float]
    ti_o_bonds: Dict[str, List[Tuple[str, float]]]
    oxo: Dict[str, List[str]]
    ligands: List[Ligand]
    core_formula: str
    molecule_extent_nm: float
    rg_nm: float
    ti_mass_fraction: float
    elements: Dict[str, int]

    def to_dict(self):
        d = dict(self.__dict__)
        d["cell"] = list(self.cell)
        d["ligands"] = [dict(lig.__dict__) for lig in self.ligands]
        return d


def analyse_cif(path: str) -> StructureAnalysis:
    st = read_structure(path)
    bonds = build_bonds(st)
    order, xyz = assemble_molecule(st, bonds, seed=0)
    idx = {a: i for i, a in enumerate(order)}
    els = [st.atoms[i].element for i in order]
    labs = [st.atoms[i].label for i in order]
    lb = [(idx[i], idx[j], d) for i, j, d in bonds if i in idx and j in idx]
    adj: Dict[int, Dict[int, float]] = collections.defaultdict(dict)
    for i, j, d in lb:
        adj[i][j] = d
        adj[j][i] = d

    n = len(order)
    ti = [k for k in range(n) if els[k] == "Ti"]
    ox = [k for k in range(n) if els[k] == "O"]

    # -------- Ti environment
    ti_o: Dict[str, List[Tuple[str, float]]] = {}
    for t in ti:
        lab = labs[t]
        if lab in ti_o:                      # symmetry copies share the label
            continue
        pairs = sorted(((d, labs[j]) for j, d in adj[t].items() if els[j] == "O"))
        ti_o[lab] = [(olab, round(dist, 3)) for dist, olab in pairs]
    ti_ti = sorted(round(float(np.linalg.norm(xyz[a] - xyz[b])), 3)
                   for a, b in itertools.combinations(ti, 2))

    # -------- oxo classification
    oxo: Dict[str, List[str]] = {}
    for o in ox:
        nt = [labs[j] for j in adj[o] if els[j] == "Ti"]
        nc = [labs[j] for j in adj[o] if els[j] == "C"]
        if nc:
            continue
        key = {2: "mu2-O", 3: "mu3-O", 4: "mu4-O"}.get(len(nt), f"mu{len(nt)}-O")
        oxo.setdefault(key, [])
        if len(oxo[key]) < 8:
            oxo[key].append(f"{labs[o]}({','.join(sorted(set(nt)))})")

    # -------- ligands: connected components after deleting Ti
    seen = set()
    comps: List[set] = []
    for s in range(n):
        if els[s] == "Ti" or s in seen:
            continue
        comp = {s}
        q = [s]
        seen.add(s)
        while q:
            c = q.pop()
            for j in adj[c]:
                if els[j] == "Ti" or j in seen:
                    continue
                seen.add(j)
                comp.add(j)
                q.append(j)
        comps.append(comp)

    def ligand_chemistry(comp):
        """Rings, C=C bonds and donor type of one organic fragment."""
        carbons = [i for i in comp if els[i] == "C"]
        sub = {i: {j for j in adj[i] if els[j] == "C"} for i in carbons}
        rings = set()

        def dfs(path):
            last = path[-1]
            for nb in sub[last]:
                if len(path) == 6 and nb == path[0]:
                    rings.add(frozenset(path))
                elif nb not in path and len(path) < 6:
                    dfs(path + [nb])
        for s in carbons:
            dfs([s])
        # aromatic: 6-ring with all C-C between 1.33 and 1.45 A
        arom = 0
        for r in rings:
            ds = [adj[i][j] for i in r for j in adj[i] if j in r and els[j] == "C"]
            if ds and 1.33 <= np.mean(ds) <= 1.45:
                arom += 1
        cc = [d for i in carbons for j, d in adj[i].items() if els[j] == "C" and j > i]
        n_double = sum(1 for d in cc if d < 1.36)
        donors = [i for i in comp if els[i] == "O"]
        dco = [d for i in donors for j, d in adj[i].items() if els[j] == "C"]
        if dco and np.mean(dco) < 1.30:
            dtype = "carboxylate-type (C-O ~1.26 A)"
        elif dco:
            dtype = "phenolate/alkoxide-type (C-O ~1.35 A)"
        else:
            dtype = "oxo"
        return arom, n_double, dtype

    grouped: Dict[Tuple[str, Tuple[str, ...]], List[set]] = collections.defaultdict(list)
    for comp in comps:
        c = collections.Counter(els[i] for i in comp)
        formula = "".join(f"{k}{v if v > 1 else ''}" for k, v in sorted(c.items()))
        ti_attached = tuple(sorted({labs[j] for i in comp for j in adj[i] if els[j] == "Ti"}))
        grouped[(formula, ti_attached)].append((comp, ti_attached))

    ligands: List[Ligand] = []
    for (formula, tis), items in sorted(grouped.items()):
        comp = items[0][0]
        donors = [i for i in comp if els[i] == "O"]
        modes = collections.Counter(len([j for j in adj[i] if els[j] == "Ti"]) for i in donors)
        bind = "k:" + ",".join(f"{k}Ti x{v}" for k, v in sorted(modes.items()))
        arom, ndbl, dtype = ligand_chemistry(comp)
        bits = [dtype]
        if arom:
            bits.append(f"{arom} aromatic ring(s)")
        if ndbl:
            bits.append(f"{ndbl} C=C (allyl/vinyl)")
        note = "; ".join(bits)
        ligands.append(Ligand(formula=formula, count=len(items), n_Ti=len(tis),
                              ti_labels=list(tis), binding=bind, note=note))

    core = collections.Counter(els[i] for i in ti) + collections.Counter(
        els[i] for i in ox if not [j for j in adj[i] if els[j] == "C"])
    core_formula = "".join(f"{k}{v}" for k, v in sorted(core.items()))

    # -------- molecular size
    centre = xyz.mean(axis=0)
    rg = float(np.sqrt(((xyz - centre) ** 2).sum(axis=1).mean()))
    heavy = [i for i in range(n) if els[i] != "H"]
    d2 = ((xyz[heavy][:, None, :] - xyz[heavy][None, :, :]) ** 2).sum(-1)
    diameter = float(np.sqrt(d2.max()))
    extent = diameter

    sc = st.scalars
    z = int(num(sc.get("_cell_formula_units_Z")) or 1)
    n_sym = len(st.symops)
    formula = _strip_quotes(sc.get("_chemical_formula_sum", ""))
    elements = collections.Counter(els)

    return StructureAnalysis(
        path=path, data_name=st.block or st.name, formula=formula,
        moiety=_strip_quotes(sc.get("_chemical_formula_moiety", "")),
        fw=num(sc.get("_chemical_formula_weight")) or 0.0,
        crystal_system=_strip_quotes(sc.get("_space_group_crystal_system", "")),
        space_group=st.space_group, it_number=st.it_number,
        cell=st.cell_params, volume=st.volume, z=z,
        z_prime=z / max(n_sym, 1),
        density_calc=num(sc.get("_exptl_crystal_density_diffrn")) or 0.0,
        temperature_K=num(sc.get("_diffrn_ambient_temperature")) or 0.0,
        wavelength=num(sc.get("_diffrn_radiation_wavelength")) or 1.54184,
        r1=num(sc.get("_refine_ls_R_factor_gt")),
        wr2=num(sc.get("_refine_ls_wR_factor_ref")),
        goof=num(sc.get("_refine_ls_goodness_of_fit_ref")),
        completeness=num(sc.get("_diffrn_measured_fraction_theta_full")),
        theta_max=num(sc.get("_diffrn_reflns_theta_max")),
        n_atoms_asu=len(st.atoms) // n_sym, n_atoms_molecule=n,
        ti_indices=ti, ti_ti_distances=ti_ti, ti_o_bonds=ti_o, oxo=oxo,
        ligands=ligands, core_formula=core_formula,
        molecule_extent_nm=extent / 10.0, rg_nm=rg / 10.0,
        ti_mass_fraction=(6 * 47.867) / (num(sc.get("_chemical_formula_weight")) or 1.0),
        elements=dict(elements))


def material_from_cif(analysis: StructureAnalysis, density: float,
                      name: str = "resist film") -> Material:
    m = Material.from_formula(name, analysis.formula, density)
    m.molar_mass = analysis.fw or m.molar_mass
    return m
