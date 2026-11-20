"""Atom correspondence between two structures of the same atoms.

Pure RDKit + numpy, no Qt, so every rule here is testable headless.

The mapping is ``mapping[ref_index] = current_index``: renumbering the current
molecule with that list puts its atoms in the reference's order.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

# Covalent-radius sum times this is a bond. 1.25 keeps an elongated TS bond
# (~1.3x) out while catching every ordinary single bond to hydrogen or a metal.
BOND_TOLERANCE = 1.25
# Upper bound on symmetric alternatives scored per matching step: a molecule
# with many equivalent heavy atoms (a fullerene) would otherwise enumerate for
# minutes. Hydrogens never enter the enumeration (see _assign_hydrogens).
MAX_SYMMETRIC_MATCHES = 2000
# Cap on (reference placement x current placement) pairs scored per piece.
MAX_PLACEMENT_TRIALS = 20000
MCS_TIMEOUT = 10

METHOD_CONNECTIVITY = "connectivity"
METHOD_DISTANCE = "distance"


class MappingError(ValueError):
    """The two structures cannot be put into correspondence."""


@dataclass
class Structure:
    """What the mapper needs from a molecule: elements, coordinates, bonds."""

    numbers: list[int]
    coords: np.ndarray
    bonds: list[tuple[int, int]]
    symbols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        self.bonds = sorted({(min(a, b), max(a, b)) for a, b in self.bonds if a != b})
        if not self.symbols:
            table = Chem.GetPeriodicTable()
            self.symbols = [table.GetElementSymbol(z) if z > 0 else "X" for z in self.numbers]

    def __len__(self) -> int:
        return len(self.numbers)

    def neighbours(self) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in self.numbers]
        for a, b in self.bonds:
            out[a].append(b)
            out[b].append(a)
        return out

    def formula(self) -> str:
        counts = Counter(self.symbols)
        order = (["C", "H"] if "C" in counts else []) + sorted(
            s for s in counts if s not in ("C", "H") or "C" not in counts
        )
        return "".join(f"{s}{counts[s] if counts[s] > 1 else ''}" for s in order)


# --------------------------------------------------------------------------
# Building structures
# --------------------------------------------------------------------------


def perceive_bonds(numbers: Sequence[int], coords: np.ndarray) -> list[tuple[int, int]]:
    """Distance-based connectivity, the same rule for every input."""
    table = Chem.GetPeriodicTable()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    radii = np.array([table.GetRcovalent(z) if z > 0 else 0.0 for z in numbers])
    if len(radii) < 2:
        return []
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    limit = (radii[:, None] + radii[None, :]) * BOND_TOLERANCE
    real = radii > 0
    close = (dist < limit) & real[:, None] & real[None, :]
    i, j = np.nonzero(np.triu(close, 1))
    return list(zip(i.tolist(), j.tolist()))


def structure_from_mol(mol: Chem.Mol, perceive: bool = False, conf_id: int = -1) -> Structure:
    if mol is None or mol.GetNumAtoms() == 0:
        raise MappingError("There is no molecule.")
    if mol.GetNumConformers() == 0:
        raise MappingError("The molecule has no 3D coordinates.")
    conf = mol.GetConformer(conf_id)
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    symbols = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.HasProp("xyz_original_symbol"):
            symbols.append(atom.GetProp("xyz_original_symbol"))
        else:
            symbols.append(atom.GetSymbol() if atom.GetAtomicNum() > 0 else "X")
    if perceive:
        bonds = perceive_bonds(numbers, coords)
    else:
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return Structure(numbers, coords, bonds, symbols)


def _number_for(symbol: str) -> int:
    clean = "".join(ch for ch in symbol if ch.isalpha())
    if not clean:
        return 0
    clean = clean[0].upper() + clean[1:].lower()
    try:
        return Chem.GetPeriodicTable().GetAtomicNumber(clean)
    except RuntimeError:
        return 0


def parse_xyz_frames(text: str) -> list[tuple[str, list[str], np.ndarray]]:
    """Every frame of a (multi-frame) XYZ file: (comment, symbols, coords)."""
    lines = text.splitlines()
    frames = []
    pos = 0
    while pos < len(lines):
        if not lines[pos].strip():
            pos += 1
            continue
        try:
            count = int(lines[pos].split()[0])
        except (ValueError, IndexError):
            raise MappingError(f"Line {pos + 1} should hold an atom count.") from None
        comment = lines[pos + 1] if pos + 1 < len(lines) else ""
        rows = lines[pos + 2 : pos + 2 + count]
        if len(rows) < count:
            raise MappingError(f"The frame starting at line {pos + 1} is cut short.")
        symbols, coords = [], []
        for row in rows:
            parts = row.split()
            try:
                coords.append([float(v) for v in parts[1:4]])
            except (ValueError, IndexError):
                raise MappingError(f"Cannot read the atom line: {row.strip()!r}") from None
            symbols.append(parts[0])
        frames.append((comment, symbols, np.array(coords, dtype=float).reshape(-1, 3)))
        pos += 2 + count
    if not frames:
        raise MappingError("The file holds no structure.")
    return frames


READABLE_EXTENSIONS = (".xyz", ".mol", ".sdf", ".sd", ".mol2", ".pdb")


def read_structures(path: str, perceive: bool = False) -> list[tuple[str, Structure]]:
    """Every structure in a file as (label, Structure).

    XYZ files carry no bonds, so theirs always come from the geometry; for the
    other formats ``perceive`` chooses between the file's bonds and geometry.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xyz":
        with open(path, encoding="utf-8", errors="replace") as handle:
            frames = parse_xyz_frames(handle.read())
        out = []
        for n, (comment, symbols, coords) in enumerate(frames, 1):
            numbers = [_number_for(s) for s in symbols]
            names = [
                s if z == 0 else Chem.GetPeriodicTable().GetElementSymbol(z)
                for s, z in zip(symbols, numbers)
            ]
            label = f"Frame {n}" + (f": {comment.strip()}" if comment.strip() else "")
            out.append(
                (
                    label,
                    Structure(numbers, coords, perceive_bonds(numbers, coords), names),
                )
            )
        return out
    if ext in (".sdf", ".sd"):
        mols = list(Chem.SDMolSupplier(path, removeHs=False, sanitize=False))
    elif ext == ".mol":
        mols = [Chem.MolFromMolFile(path, removeHs=False, sanitize=False)]
    elif ext == ".mol2":
        mols = [Chem.MolFromMol2File(path, removeHs=False, sanitize=False)]
    elif ext == ".pdb":
        mols = [Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)]
    else:
        raise MappingError(
            f"Cannot read {ext or 'this'} files. Use one of: " + ", ".join(READABLE_EXTENSIONS)
        )
    out = []
    for n, mol in enumerate(mols, 1):
        if mol is None:
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        label = f"Structure {n}" + (f": {name}" if name else "")
        out.append((label, structure_from_mol(mol, perceive=perceive)))
    if not out:
        raise MappingError("No structure could be read from the file.")
    return out


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def kabsch(moving: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |moving @ R.T + t - target|."""
    moving = np.asarray(moving, dtype=float)
    target = np.asarray(target, dtype=float)
    mc = moving.mean(axis=0)
    tc = target.mean(axis=0)
    if len(moving) < 2:
        return np.eye(3), tc - mc
    h = (moving - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, tc - rot @ mc


def apply_transform(coords: np.ndarray, rot: np.ndarray, shift: np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=float) @ rot.T + shift


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0:
        return 0.0
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def fitted_rmsd(moving: np.ndarray, target: np.ndarray) -> float:
    rot, shift = kabsch(moving, target)
    return rmsd(apply_transform(moving, rot, shift), target)


def assignment(cost: np.ndarray) -> list[int]:
    """Minimum-cost perfect matching on a square matrix: row i -> column out[i]."""
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    if n == 0:
        return []
    try:  # pragma: no cover - scipy is optional and only faster
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        out = [0] * n
        for r, c in zip(rows, cols):
            out[int(r)] = int(c)
        return out
    except ImportError:
        pass
    # Hungarian algorithm with potentials, O(n^3).
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = [0] * n
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def _graph(structure: Structure, atoms: Sequence[int]) -> Chem.Mol:
    """Element-and-connectivity-only molecule over ``atoms`` (in that order)."""
    rw = Chem.RWMol()
    where = {}
    for idx in atoms:
        atom = Chem.Atom(structure.numbers[idx])
        atom.SetNoImplicit(True)
        where[idx] = rw.AddAtom(atom)
    for a, b in structure.bonds:
        if a in where and b in where:
            rw.AddBond(where[a], where[b], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def _query(mol: Chem.Mol) -> Chem.Mol | None:
    if mol.GetNumAtoms() == 0:
        return None
    query = Chem.MolFromSmarts(Chem.MolToSmarts(mol))
    if query is not None:
        query.UpdatePropertyCache(strict=False)
    return query


def _common_piece(ref_graph: Chem.Mol, cur_graph: Chem.Mol) -> Chem.Mol | None:
    """Largest connected piece both graphs share, as a query (None: nothing)."""
    if (
        ref_graph.GetNumAtoms() == cur_graph.GetNumAtoms()
        and ref_graph.GetNumBonds() == cur_graph.GetNumBonds()
    ):
        whole = _query(ref_graph)
        if whole is not None and cur_graph.HasSubstructMatch(whole):
            return whole
    result = rdFMCS.FindMCS(
        [ref_graph, cur_graph],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        matchValences=False,
        timeout=MCS_TIMEOUT,
    )
    if result.numBonds < 1:
        return None
    query = Chem.MolFromSmarts(result.smartsString)
    if query is not None:
        query.UpdatePropertyCache(strict=False)
    return query


def _align_on(mapping: dict[int, int], ref: Structure, cur: Structure) -> np.ndarray:
    """Current coordinates moved onto the reference using the pairs so far."""
    if not mapping:
        shift = ref.coords.mean(axis=0) - cur.coords.mean(axis=0)
        return cur.coords + shift
    r_idx = list(mapping)
    c_idx = [mapping[i] for i in r_idx]
    rot, shift = kabsch(cur.coords[c_idx], ref.coords[r_idx])
    return apply_transform(cur.coords, rot, shift)


def _score(mapping: dict[int, int], ref: Structure, cur: Structure) -> float:
    r_idx = list(mapping)
    return fitted_rmsd(cur.coords[[mapping[i] for i in r_idx]], ref.coords[r_idx])


def _map_by_graph(
    seed: dict[int, int],
    ref: Structure,
    cur: Structure,
    atoms_ref: list[int],
    atoms_cur: list[int],
) -> dict[int, int]:
    """Pair atoms piece by piece through their largest common subgraphs.

    The first piece is the largest common subgraph; then that is removed and the
    rest searched again, so a reaction that joins or splits fragments keeps
    each fragment's own connectivity. Among symmetric ways to lay a piece on
    the current molecule, the one that superimposes best wins - together with
    the pairs already in ``seed``, which stay as they are.
    """
    mapping = dict(seed)
    while True:
        rest_ref = [i for i in atoms_ref if i not in mapping]
        used = set(mapping.values())
        rest_cur = [i for i in atoms_cur if i not in used]
        if len(rest_ref) < 2 or len(rest_cur) < 2:
            return mapping
        ref_graph = _graph(ref, rest_ref)
        cur_graph = _graph(cur, rest_cur)
        query = _common_piece(ref_graph, cur_graph)
        if query is None:
            return mapping
        # A piece smaller than the molecule can sit in several places on the
        # reference too (a 4-carbon chain in a 6-ring): try every placement on
        # both sides. Orderings only need trying on one side.
        cur_matches = cur_graph.GetSubstructMatches(
            query, uniquify=False, useChirality=False, maxMatches=MAX_SYMMETRIC_MATCHES
        )
        ref_matches = ref_graph.GetSubstructMatches(
            query, uniquify=True, useChirality=False, maxMatches=MAX_SYMMETRIC_MATCHES
        )
        if not ref_matches or not cur_matches:
            return mapping
        ref_matches = ref_matches[: max(1, MAX_PLACEMENT_TRIALS // len(cur_matches))]
        best, best_score = None, float("inf")
        for ref_match in ref_matches:
            for match in cur_matches:
                trial = dict(mapping)
                trial.update({rest_ref[r]: rest_cur[c] for r, c in zip(ref_match, match)})
                score = _score(trial, ref, cur)
                if score < best_score - 1e-9:
                    best, best_score = trial, score
        mapping = best


def _assign_rest(
    mapping: dict[int, int],
    ref: Structure,
    cur: Structure,
    atoms_ref: list[int],
    atoms_cur: list[int],
) -> None:
    """Pair whatever is left by element, nearest after superposition."""
    moved = _align_on(mapping, ref, cur)
    used = set(mapping.values())
    by_element_ref: dict[int, list[int]] = defaultdict(list)
    by_element_cur: dict[int, list[int]] = defaultdict(list)
    for i in atoms_ref:
        if i not in mapping:
            by_element_ref[ref.numbers[i]].append(i)
    for i in atoms_cur:
        if i not in used:
            by_element_cur[cur.numbers[i]].append(i)
    for z, r_list in by_element_ref.items():
        c_list = by_element_cur.get(z, [])
        if len(c_list) != len(r_list):
            raise MappingError("The element counts differ.")
        cost = np.linalg.norm(ref.coords[r_list][:, None, :] - moved[c_list][None, :, :], axis=-1)
        for r, c in zip(r_list, assignment(cost)):
            mapping[r] = c_list[c]


def _assign_hydrogens(mapping: dict[int, int], ref: Structure, cur: Structure) -> None:
    """Hydrogens follow the heavy atom they sit on; the rest go by distance.

    Hydrogens on one heavy atom are interchangeable by connectivity, so they are
    never enumerated as graph symmetries (a molecule with ten methyl groups has
    6^10 such permutations); each heavy atom's set is matched by distance
    instead. A hydrogen that moved to another atom - a proton transfer - is
    left over and paired by distance with the other moved hydrogens.
    """
    moved = _align_on(mapping, ref, cur)
    ref_n = ref.neighbours()
    cur_n = cur.neighbours()

    def parent(structure: Structure, neighbours: list[list[int]], h: int) -> int | None:
        heavy = [n for n in neighbours[h] if structure.numbers[n] != 1]
        if len(heavy) != 1:
            return None
        return heavy[0]

    ref_groups: dict[int, list[int]] = defaultdict(list)
    cur_groups: dict[int, list[int]] = defaultdict(list)
    for h in (i for i, z in enumerate(ref.numbers) if z == 1):
        p = parent(ref, ref_n, h)
        if p is not None and p in mapping:
            ref_groups[mapping[p]].append(h)
    for h in (i for i, z in enumerate(cur.numbers) if z == 1):
        p = parent(cur, cur_n, h)
        if p is not None:
            cur_groups[p].append(h)
    for cur_parent, r_list in ref_groups.items():
        c_list = cur_groups.get(cur_parent, [])
        if not c_list:
            continue
        cost = np.linalg.norm(ref.coords[r_list][:, None, :] - moved[c_list][None, :, :], axis=-1)
        if len(r_list) == len(c_list):
            for r, c in zip(r_list, assignment(cost)):
                mapping[r] = c_list[c]
            continue
        # A proton left or arrived here: keep the closest pairs only.
        k = min(len(r_list), len(c_list))
        square = np.full((max(len(r_list), len(c_list)),) * 2, 1e6)
        square[: len(r_list), : len(c_list)] = cost
        cols = assignment(square)
        pairs = sorted(
            ((cost[r, cols[r]], r, cols[r]) for r in range(len(r_list)) if cols[r] < len(c_list)),
        )[:k]
        for _, r, c in pairs:
            mapping[r_list[r]] = c_list[c]


@dataclass
class MappingResult:
    mapping: list[int]
    rmsd: float
    formed: list[tuple[int, int]]
    broken: list[tuple[int, int]]
    displacement: list[float]
    method: str


def check_same_atoms(ref: Structure, cur: Structure) -> None:
    if len(ref) != len(cur):
        raise MappingError(
            f"The reference has {len(ref)} atoms but the current molecule has {len(cur)}."
        )
    ref_count = Counter(ref.numbers)
    cur_count = Counter(cur.numbers)
    if ref_count != cur_count:
        table = Chem.GetPeriodicTable()
        diffs = []
        for z in sorted(set(ref_count) | set(cur_count)):
            if ref_count[z] != cur_count[z]:
                name = table.GetElementSymbol(z) if z > 0 else "X"
                diffs.append(f"{name}: reference {ref_count[z]}, current {cur_count[z]}")
        raise MappingError("The two structures are not the same atoms (" + "; ".join(diffs) + ").")


def find_mapping(
    ref: Structure,
    cur: Structure,
    method: str = METHOD_CONNECTIVITY,
    fixed: dict[int, int] | None = None,
) -> MappingResult:
    """Which current atom each reference atom is.

    ``fixed`` pins pairs (reference index -> current index) before anything else.
    """
    check_same_atoms(ref, cur)
    mapping: dict[int, int] = {}
    for r, c in (fixed or {}).items():
        if ref.numbers[r] != cur.numbers[c]:
            raise MappingError(f"Pinned pair {r + 1} -> {c + 1} joins two different elements.")
        mapping[r] = c
    if len(set(mapping.values())) != len(mapping):
        raise MappingError("Two pinned pairs use the same current atom.")
    all_ref = list(range(len(ref)))
    all_cur = list(range(len(cur)))
    if method == METHOD_CONNECTIVITY:
        heavy_ref = [i for i in all_ref if ref.numbers[i] != 1 and i not in mapping]
        used = set(mapping.values())
        heavy_cur = [i for i in all_cur if cur.numbers[i] != 1 and i not in used]
        mapping = _map_by_graph(mapping, ref, cur, heavy_ref, heavy_cur)
        heavy_ref_all = [i for i in all_ref if ref.numbers[i] != 1]
        heavy_cur_all = [i for i in all_cur if cur.numbers[i] != 1]
        _assign_rest(mapping, ref, cur, heavy_ref_all, heavy_cur_all)
        _assign_hydrogens(mapping, ref, cur)
        _assign_rest(mapping, ref, cur, all_ref, all_cur)
    elif method == METHOD_DISTANCE:
        # The two are already overlaid (frames of one path): no superposition.
        used = set(mapping.values())
        by_ref: dict[int, list[int]] = defaultdict(list)
        by_cur: dict[int, list[int]] = defaultdict(list)
        for i in all_ref:
            if i not in mapping:
                by_ref[ref.numbers[i]].append(i)
        for i in all_cur:
            if i not in used:
                by_cur[cur.numbers[i]].append(i)
        for z, r_list in by_ref.items():
            c_list = by_cur[z]
            cost = np.linalg.norm(
                ref.coords[r_list][:, None, :] - cur.coords[c_list][None, :, :], axis=-1
            )
            for r, c in zip(r_list, assignment(cost)):
                mapping[r] = c_list[c]
    else:
        raise MappingError(f"Unknown method {method!r}.")
    order = [mapping[i] for i in all_ref]
    if sorted(order) != all_cur:
        raise MappingError("The atoms could not be paired one to one.")
    return summarize(ref, cur, order, method)


def summarize(
    ref: Structure, cur: Structure, order: list[int], method: str = METHOD_CONNECTIVITY
) -> MappingResult:
    """RMSD, per-atom displacement and bond changes for a finished mapping."""
    reordered = cur.coords[order]
    rot, shift = kabsch(reordered, ref.coords)
    fitted = apply_transform(reordered, rot, shift)
    displacement = np.linalg.norm(fitted - ref.coords, axis=1).tolist()
    position = {c: r for r, c in enumerate(order)}
    cur_bonds = {tuple(sorted((position[a], position[b]))) for a, b in cur.bonds}
    ref_bonds = set(ref.bonds)
    return MappingResult(
        mapping=list(order),
        rmsd=rmsd(fitted, ref.coords),
        formed=sorted(ref_bonds - cur_bonds),
        broken=sorted(cur_bonds - ref_bonds),
        displacement=displacement,
        method=method,
    )


def renumber(mol: Chem.Mol, order: Sequence[int]) -> Chem.Mol:
    """The molecule with atom ``order[i]`` moved to index ``i``."""
    return Chem.RenumberAtoms(mol, [int(i) for i in order])


# --------------------------------------------------------------------------
# Superposition by index
# --------------------------------------------------------------------------


@dataclass
class SuperposeResult:
    coords: np.ndarray
    rmsd_before: float
    rmsd_after: float
    fitted_on: int


def superpose_by_index(
    cur: Structure,
    ref: Structure,
    atoms: Sequence[int] | None = None,
    heavy_only: bool = False,
) -> SuperposeResult:
    """Move the current coordinates onto the reference, atom i onto atom i.

    The fit uses ``atoms`` (default: all, or all heavy atoms) and the RMSD is
    then reported over every atom, so a fit on a fixed framework still shows
    how far the reacting part is from the reference.
    """
    if len(cur) != len(ref):
        raise MappingError(
            f"The reference has {len(ref)} atoms but the current molecule has {len(cur)}."
        )
    wrong = [i for i in range(len(cur)) if cur.numbers[i] != ref.numbers[i]]
    if wrong:
        shown = ", ".join(str(i + 1) for i in wrong[:8]) + (" ..." if len(wrong) > 8 else "")
        raise MappingError(
            f"Atom numbers do not correspond (different elements at {shown}). Match the indices first."
        )
    chosen = list(atoms) if atoms else list(range(len(cur)))
    if heavy_only:
        chosen = [i for i in chosen if cur.numbers[i] != 1] or chosen
    if not chosen:
        raise MappingError("No atoms to fit on.")
    rot, shift = kabsch(cur.coords[chosen], ref.coords[chosen])
    moved = apply_transform(cur.coords, rot, shift)
    return SuperposeResult(
        coords=moved,
        rmsd_before=rmsd(cur.coords, ref.coords),
        rmsd_after=rmsd(moved, ref.coords),
        fitted_on=len(chosen),
    )


def with_coordinates(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    """A copy of ``mol`` with its (first) conformer moved to ``coords``."""
    from rdkit.Geometry import Point3D

    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i, (x, y, z) in enumerate(np.asarray(coords, dtype=float)):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return out


def xyz_text(structure: Structure, coords: np.ndarray | None = None, comment: str = "") -> str:
    coords = structure.coords if coords is None else coords
    lines = [str(len(structure)), comment]
    for symbol, (x, y, z) in zip(structure.symbols, coords):
        lines.append(f"{symbol:<3}{x:>15.8f}{y:>15.8f}{z:>15.8f}")
    return "\n".join(lines) + "\n"


def parse_pins(text: str, n_ref: int, n_cur: int) -> dict[int, int]:
    """``"1:3, 5:8"`` (1-based reference:current) as a 0-based dict."""
    pins: dict[int, int] = {}
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.replace("-", ":").replace("=", ":").split(":")
        try:
            r, c = (int(p) for p in parts)
        except ValueError:
            raise MappingError(
                f"Cannot read the pinned pair {chunk!r}; write it as reference:current, e.g. 1:3."
            ) from None
        if not (1 <= r <= n_ref and 1 <= c <= n_cur):
            raise MappingError(f"Pinned pair {chunk!r} is outside the atom numbers.")
        if r - 1 in pins:
            raise MappingError(f"Reference atom {r} is pinned twice.")
        pins[r - 1] = c - 1
    return pins


def swap_order(count: int, first: int, second: int) -> list[int]:
    """Renumbering order that exchanges two atoms' numbers."""
    _check_index(count, first)
    _check_index(count, second)
    order = list(range(count))
    order[first], order[second] = order[second], order[first]
    return order


def move_order(count: int, atom: int, position: int) -> list[int]:
    """Renumbering order that moves one atom to ``position`` and shifts the rest."""
    _check_index(count, atom)
    _check_index(count, position)
    order = [i for i in range(count) if i != atom]
    order.insert(position, atom)
    return order


def _check_index(count: int, index: int) -> None:
    if not 0 <= index < count:
        raise MappingError(f"Atom number {index + 1} is outside 1-{count}.")
