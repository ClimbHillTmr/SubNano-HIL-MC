"""Minimal but robust CIF reader / crystal-structure builder.

Why a hand-written reader: gemmi 0.7.5 fails to assemble a Structure from the
Olex2/SHELXL output in ``data/4.cif`` (cell + atom sites are silently dropped),
so we parse the tags we need directly.  Everything below is dependency-free
(standard library only) apart from numpy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
_SU_RE = re.compile(r"^(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\((\d+)\)$")


def _strip_quotes(tok: str) -> str:
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    return tok


def tokenize_cif(text: str):
    """Yield ('tag'|'value'|'loop'|'data'|'blank', token) pairs."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if s.startswith(";"):  # multi-line text field
            i += 1
            buf: List[str] = []
            while i < n and not lines[i].strip().startswith(";"):
                buf.append(lines[i])
                i += 1
            i += 1
            yield "value", "\n".join(buf)
            continue
        if not s or s.startswith("#"):
            yield "blank", ""
            i += 1
            continue
        if s.lower().startswith("data_"):
            yield "data", s[5:]
            i += 1
            continue
        if s.lower() == "loop_":
            yield "loop", "loop_"
            i += 1
            continue
        for tok in re.findall(r"'(?:[^']*)'|\"(?:[^\"]*)\"|\S+", s):
            yield ("tag", tok) if tok.startswith("_") else ("value", tok)
        i += 1


def parse_cif(path: str) -> Dict[str, Any]:
    """Parse a CIF file into {scalars: {tag: value}, loops: [{tags: [], rows: []}]}.

    State machine: a ``loop_`` block ends at the first blank line, at the next
    ``loop_``, or at the next tag appearing *after* values have started.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    scalars: Dict[str, str] = {}
    loops: List[Dict[str, Any]] = []
    block_name = ""

    pending: Optional[str] = None
    loop_tags: List[str] = []
    loop_rows: List[List[str]] = []
    in_loop = False          # inside a loop_ block
    in_values = False        # values (not header tags) are being read

    def close_loop():
        nonlocal loop_tags, loop_rows, in_loop, in_values
        if in_loop and loop_tags:
            rows = [r for r in loop_rows if len(r) == len(loop_tags)]
            loops.append({"tags": list(loop_tags), "rows": rows})
        loop_tags, loop_rows = [], []
        in_loop, in_values = False, False

    for kind, tok in tokenize_cif(text):
        if kind == "data":
            close_loop()
            block_name = tok
            continue
        if kind == "blank":
            if in_loop:
                close_loop()
            continue
        if kind == "loop":
            close_loop()
            in_loop = True
            continue
        if kind == "tag":
            if in_loop and not in_values:
                loop_tags.append(tok)
            else:
                close_loop()
                pending = tok
            continue
        # value
        if in_loop:
            in_values = True
            if not loop_rows or len(loop_rows[-1]) == len(loop_tags):
                loop_rows.append([tok])
            else:
                loop_rows[-1].append(tok)
            continue
        if pending is not None:
            scalars[pending] = _strip_quotes(tok)
            pending = None

    close_loop()
    return {"scalars": scalars, "loops": loops, "block": block_name}


def get_loop(cif: Dict[str, Any], *tags: str) -> Optional[Dict[str, Any]]:
    """Return the loop that contains all of the requested tags."""
    for lp in cif["loops"]:
        if all(t in lp["tags"] for t in tags):
            return lp
        # also accept dot notation (_atom_site.fract_x)
        alt = [t.replace(".", "_") for t in lp["tags"]]
        if all(t in alt for t in tags):
            return lp
    return None


@dataclass
class Atom:
    label: str
    element: str
    frac: np.ndarray
    u_iso: float
    occupancy: float
    xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class CrystalStructure:
    name: str
    cell_params: Tuple[float, float, float, float, float, float]
    volume: float
    space_group: str
    it_number: int
    symops: List[str]
    atoms: List[Atom]
    scalars: Dict[str, str]
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    block: str = ""

    # -- geometry helpers ---------------------------------------------------
    def cartesian(self, frac: np.ndarray) -> np.ndarray:
        return frac @ self.matrix  # row-vector convention

    def frac_of(self, xyz: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.matrix, xyz)


def num(value: Optional[str]) -> Optional[float]:
    """'14.6464(4)' -> 14.6464 ; '?' -> None"""
    if value is None:
        return None
    v = _strip_quotes(value).strip()
    if v in ("?", ".", ""):
        return None
    m = _SU_RE.match(v)
    if m:
        return float(m.group(1))
    try:
        return float(v)
    except ValueError:
        return None


def su(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    m = _SU_RE.match(_strip_quotes(value).strip())
    if not m:
        return None
    ndec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
    return float(m.group(2)) * 10.0 ** (-ndec)


def cell_matrix(a, b, c, alpha, beta, gamma):
    al, be, ga = (np.radians(x) for x in (alpha, beta, gamma))
    v = a * b * c * np.sqrt(
        1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
        + 2 * np.cos(al) * np.cos(be) * np.cos(ga)
    )
    # rows are the basis vectors a, b, c in Cartesian space, so that
    #   cart = frac @ m      (frac = fractional, row-vector convention)
    m = np.array(
        [
            [a, 0.0, 0.0],
            [b * np.cos(ga), b * np.sin(ga), 0.0],
            [
                c * np.cos(be),
                c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga),
                v / (a * b * np.sin(ga)),
            ],
        ]
    )
    return m, v


def _apply_symop(op: str, frac: np.ndarray) -> np.ndarray:
    x, y, z = frac
    env = {"x": x, "y": y, "z": z}
    out = []
    for part in op.replace("'", "").split(","):
        expr = part.strip().lower()
        if not expr:
            continue
        # handle forms like  'x'  '-x'  '1/2-x'  '-y+1/2'  '2_666'
        expr = expr.replace(" ", "")
        token = ""
        sign = 1.0
        for ch in expr:
            if ch == "+":
                out.append(sign * float(token) if token not in ("",) else 0.0)
                token = ""
                sign = 1.0
            elif ch == "-":
                if token:
                    out.append(sign * float(token))
                token = ""
                sign = -1.0
            else:
                token += ch
        if token:
            try:
                out.append(sign * float(token))
            except ValueError:
                if token in env:
                    out.append(sign * env[token])
                else:
                    out.append(0.0)
        elif sign == -1.0:
            out.append(0.0)
        if len(out) >= 3:
            break
    while len(out) < 3:
        out.append(0.0)
    return np.array(out[:3], dtype=float)


def read_structure(path: str, expand_symmetry: bool = True) -> CrystalStructure:
    cif = parse_cif(path)
    sc = cif["scalars"]
    data_block = cif.get("block", "")

    a = num(sc.get("_cell_length_a"))
    b = num(sc.get("_cell_length_b"))
    c = num(sc.get("_cell_length_c"))
    al = num(sc.get("_cell_angle_alpha"))
    be = num(sc.get("_cell_angle_beta"))
    ga = num(sc.get("_cell_angle_gamma"))
    if None in (a, b, c, al, be, ga):
        raise ValueError("cell parameters missing in CIF")
    m, vol = cell_matrix(a, b, c, al, be, ga)
    vol = num(sc.get("_cell_volume")) or vol

    symops = ["x, y, z"]
    lp = get_loop(cif, "_space_group_symop_operation_xyz")
    if lp is None:
        lp = get_loop(cif, "_symmetry_equiv_pos_as_xyz")
    if lp is not None:
        tag = ("_space_group_symop_operation_xyz"
               if "_space_group_symop_operation_xyz" in lp["tags"]
               else "_symmetry_equiv_pos_as_xyz")
        idx = lp["tags"].index(tag)
        symops = [r[idx].strip("'\" ") for r in lp["rows"]]

    site = get_loop(cif, "_atom_site_label", "_atom_site_fract_x")
    if site is None:
        raise ValueError("atom_site loop not found")
    tg = site["tags"]

    def col(*names):
        for nm in names:
            if nm in tg:
                return tg.index(nm)
        return None

    i_lab = col("_atom_site_label", "_atom_site.label")
    i_typ = col("_atom_site_type_symbol", "_atom_site.type_symbol")
    i_x = col("_atom_site_fract_x", "_atom_site.fract_x")
    i_y = col("_atom_site_fract_y", "_atom_site.fract_y")
    i_z = col("_atom_site_fract_z", "_atom_site.fract_z")
    i_u = col("_atom_site_U_iso_or_equiv", "_atom_site.U_iso_or_equiv")
    i_occ = col("_atom_site_occupancy", "_atom_site.occupancy")

    atoms: List[Atom] = []
    for r in site["rows"]:
        if len(r) != len(tg):
            continue
        lab = r[i_lab]
        el = r[i_typ] if i_typ is not None else re.sub(r"\d", "", lab)
        el = re.sub(r"[^A-Za-z]", "", el)
        el = (el[:1].upper() + el[1:].lower()) if el else "?"
        fr = np.array([num(r[i_x]), num(r[i_y]), num(r[i_z])], dtype=float)
        u = num(r[i_u]) if i_u is not None else 0.0
        occ = num(r[i_occ]) if i_occ is not None else 1.0
        atoms.append(Atom(lab, el, fr, u or 0.0, occ or 1.0))

    # symmetry expansion (P-1 style: build the complete molecule)
    if expand_symmetry and len(symops) > 1:
        seen = set()
        full: List[Atom] = []
        for op in symops:
            for at in atoms:
                fr = _apply_symop(op, at.frac)
                key = (at.element, tuple(np.round(fr % 1.0, 4)))
                if key in seen:
                    continue
                seen.add(key)
                full.append(Atom(at.label, at.element, fr, at.u_iso, at.occupancy))
        atoms = full

    for at in atoms:
        at.xyz = at.frac @ m

    sg = _strip_quotes(sc.get("_space_group_name_H-M_alt", "")) or _strip_quotes(
        sc.get("_symmetry_space_group_name_H-M", "")
    )
    itn = num(sc.get("_space_group_IT_number")) or 0

    return CrystalStructure(
        name=_strip_quotes(sc.get("_chemical_formula_sum", "")) or "",
        block=data_block,
        cell_params=(a, b, c, al, be, ga),
        volume=vol,
        space_group=sg,
        it_number=int(itn),
        symops=symops,
        atoms=atoms,
        scalars=sc,
        matrix=m,
    )


# ---------------------------------------------------------------------------
# Bond graph
# ---------------------------------------------------------------------------

COV = {  # covalent radii (Angstrom), Cordero et al. 2008
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "Ti": 1.60,
    "Si": 1.11, "Zr": 1.75, "Hf": 1.75, "Zn": 1.22, "F": 0.57,
    "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}


def assemble_molecule(st: CrystalStructure, bonds, seed: int = 0):
    """Unwrap a molecule across periodic images (BFS with minimum-image bonds).

    Returns (idx_order, xyz_unwrapped, image_shifts).
    """
    import collections as _c

    adj = _c.defaultdict(list)
    for i, j, d in bonds:
        adj[i].append(j)
        adj[j].append(i)

    cell = st.matrix
    inv = np.linalg.inv(cell)
    seen = {seed: (st.atoms[seed].xyz.copy(), np.zeros(3))}
    queue = [seed]
    while queue:
        cur = queue.pop()
        pc, _ = seen[cur]
        for nb in adj[cur]:
            if nb in seen:
                continue
            delta = st.atoms[nb].xyz - st.atoms[cur].xyz
            fshift = np.round(delta @ inv)
            best = delta - fshift @ cell
            seen[nb] = (pc + best, -fshift)
            queue.append(nb)

    order = sorted(seen)
    xyz = np.array([seen[i][0] for i in order])
    return order, xyz


def build_bonds(st: CrystalStructure, tol: float = 0.45, max_bond: float = 2.6):
    """Distance-based bond graph with periodic images (3x3x3)."""
    n = len(st.atoms)
    xyz = np.array([a.xyz for a in st.atoms])
    els = [a.element for a in st.atoms]
    cell = st.matrix
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    ) @ cell
    bonds = []
    for i in range(n):
        ri = COV.get(els[i], 1.0)
        d = np.linalg.norm(xyz[i] - (xyz[i + 1:][:, None, :] - shifts[None, :, :]), axis=2)
        for off, j in enumerate(range(i + 1, n)):
            rj = COV.get(els[j], 1.0)
            cut = ri + rj + tol
            a = np.argmin(d[off])
            dmin = d[off][a]
            if dmin <= cut and dmin <= max_bond:
                bonds.append((i, j, float(dmin)))
    return bonds
