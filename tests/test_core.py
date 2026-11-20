"""Mapping, renumbering and superposition on real RDKit molecules."""

import itertools

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from atom_index_mapper import core

from .helpers import bonds_after, diels_alder, embedded, rotation, scrambled


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "Cc1ccccc1",
        "OC(=O)c1ccccc1C(F)(F)F",
        "C1CCCCC1",
        "CC(C)(C)c1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1",
    ],
)
def test_a_shuffled_copy_maps_back_with_every_bond(smiles):
    ref = core.structure_from_mol(embedded(smiles), perceive=True)
    cur, _ = scrambled(ref, seed=3, noise=0.02)
    result = core.find_mapping(ref, cur)
    assert sorted(result.mapping) == list(range(len(ref)))
    # Symmetric molecules may come back as an equivalent labelling; what
    # matters is that the shape and every bond agree.
    assert result.rmsd < 0.08
    assert result.formed == [] and result.broken == []
    assert bonds_after(cur, result.mapping) == set(ref.bonds)


def test_methyl_hydrogens_follow_the_geometry_not_the_file_order():
    ref = core.structure_from_mol(embedded("CC(C)(C)C"), perceive=True)
    cur, where = scrambled(ref, seed=8)
    result = core.find_mapping(ref, cur)
    assert result.rmsd < 1e-6
    assert result.mapping == [where[i] for i in range(len(ref))]


def test_diels_alder_keeps_both_fragments_bonds():
    product, reactant = diels_alder()
    cur, where = scrambled(reactant, seed=5)
    result = core.find_mapping(product, cur)
    assert result.mapping == [where[i] for i in range(len(product))]
    assert result.formed == [(2, 3), (4, 5)]
    assert result.broken == []


def test_a_proton_transfer_moves_only_that_hydrogen():
    keto = core.structure_from_mol(embedded("CC(C)=O"), perceive=True)
    oxygen = keto.numbers.index(8)
    hydrogen = keto.numbers.index(1)
    coords = keto.coords.copy()
    axis = coords[oxygen] - coords[1]
    coords[hydrogen] = coords[oxygen] + axis / np.linalg.norm(axis) * 0.97
    enol = core.Structure(keto.numbers, coords, core.perceive_bonds(keto.numbers, coords))
    cur, where = scrambled(enol, seed=9)
    result = core.find_mapping(keto, cur)
    assert result.mapping == [where[i] for i in range(len(keto))]
    assert len(result.formed) == 1 and len(result.broken) == 1
    assert hydrogen in result.formed[0] and hydrogen in result.broken[0]


def test_distance_method_pairs_overlaid_frames_without_moving_them():
    ref = core.structure_from_mol(embedded("CCO"), perceive=True)
    perm = list(range(len(ref)))[::-1]
    cur = core.Structure([ref.numbers[i] for i in perm], ref.coords[perm] + 0.01, [])
    result = core.find_mapping(ref, cur, core.METHOD_DISTANCE)
    assert [perm[c] for c in result.mapping] == list(range(len(ref)))


def test_pinned_pairs_are_kept():
    ref = core.structure_from_mol(embedded("CC"), perceive=True)
    cur, where = scrambled(ref, seed=1)
    hydrogens = [i for i, z in enumerate(ref.numbers) if z == 1]
    wrong = where[hydrogens[1]]
    result = core.find_mapping(ref, cur, fixed={hydrogens[0]: wrong})
    assert result.mapping[hydrogens[0]] == wrong
    assert sorted(result.mapping) == list(range(len(ref)))


def test_pins_must_join_the_same_element():
    ref = core.structure_from_mol(embedded("CO"), perceive=True)
    with pytest.raises(core.MappingError, match="different elements"):
        core.find_mapping(ref, ref, fixed={ref.numbers.index(6): ref.numbers.index(8)})


def test_different_atoms_are_refused_with_the_difference():
    a = core.structure_from_mol(embedded("CCO"), perceive=True)
    b = core.Structure([7 if z == 8 else z for z in a.numbers], a.coords, a.bonds)
    with pytest.raises(
        core.MappingError, match="N: reference 0, current 1; O: reference 1, current 0"
    ):
        core.find_mapping(a, b)
    c = core.structure_from_mol(embedded("CC"), perceive=True)
    with pytest.raises(core.MappingError, match="has 8 atoms but the current molecule has 9"):
        core.find_mapping(c, a)


def test_renumber_moves_atoms_with_their_coordinates_and_properties():
    mol = embedded("CO")
    mol.GetAtomWithIdx(0).SetIntProp("_original_atom_id", 41)
    order = list(range(mol.GetNumAtoms()))[::-1]
    out = core.renumber(mol, order)
    last = mol.GetNumAtoms() - 1
    assert out.GetAtomWithIdx(last).GetIntProp("_original_atom_id") == 41
    assert np.allclose(
        list(out.GetConformer().GetAtomPosition(last)),
        list(mol.GetConformer().GetAtomPosition(0)),
    )


def test_superpose_by_index_recovers_a_rigid_motion():
    ref = core.structure_from_mol(embedded("CC(=O)N"), perceive=True)
    moved = ref.coords @ rotation(4).T + np.array([10.0, 0.0, -3.0])
    cur = core.Structure(ref.numbers, moved, ref.bonds)
    result = core.superpose_by_index(cur, ref)
    assert result.rmsd_before > 1
    assert result.rmsd_after < 1e-8
    assert result.fitted_on == len(ref)


def test_superpose_on_a_subset_fits_those_atoms_only():
    product, reactant = diels_alder()
    ring = [0, 1, 2, 5]
    moved = core.Structure(reactant.numbers, reactant.coords @ rotation(2).T + 4.0, reactant.bonds)
    result = core.superpose_by_index(moved, product, atoms=ring)
    assert np.allclose(result.coords[ring], product.coords[ring], atol=1e-6)
    assert result.rmsd_after > 0.1
    heavy = core.superpose_by_index(moved, product, heavy_only=True)
    assert heavy.fitted_on == 6


def test_superpose_refuses_unmatched_numbering():
    ref = core.structure_from_mol(embedded("CO"), perceive=True)
    flipped = core.Structure(ref.numbers[::-1], ref.coords[::-1], [])
    with pytest.raises(core.MappingError, match="Match the indices first"):
        core.superpose_by_index(flipped, ref)


def test_assignment_without_scipy_is_optimal(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 8):
        cost = rng.random((n, n))
        cols = core.assignment(cost)
        best = min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
        assert sorted(cols) == list(range(n))
        assert abs(sum(cost[i, cols[i]] for i in range(n)) - best) < 1e-12


def test_xyz_frames_and_dummy_atoms(tmp_path):
    path = tmp_path / "path.xyz"
    path.write_text(
        "3\nfirst\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n3\n\nO 0 0 0\nH 0.97 0 0\nX -0.24 0.93 0\n",
        encoding="utf-8",
    )
    frames = core.read_structures(str(path))
    assert [label for label, _ in frames] == ["Frame 1: first", "Frame 2"]
    assert frames[0][1].bonds == [(0, 1), (0, 2)]
    assert frames[1][1].numbers == [8, 1, 0]
    assert frames[1][1].symbols[2] == "X"


def test_a_cut_short_xyz_is_reported(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("3\n\nO 0 0 0\n", encoding="utf-8")
    with pytest.raises(core.MappingError, match="cut short"):
        core.read_structures(str(path))


def test_sdf_keeps_its_bonds_unless_geometry_is_asked_for(tmp_path):
    mol = embedded("CCO")
    path = tmp_path / "m.sdf"
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    ((_, from_file),) = core.read_structures(str(path))
    assert len(from_file.bonds) == mol.GetNumBonds()
    ((_, perceived),) = core.read_structures(str(path), perceive=True)
    assert perceived.bonds == from_file.bonds
    with pytest.raises(core.MappingError, match="Cannot read .cube"):
        core.read_structures(str(tmp_path / "x.cube"))


def test_pins_text():
    assert core.parse_pins("1:3, 5-8; 2=2", 9, 9) == {0: 2, 4: 7, 1: 1}
    assert core.parse_pins("  ", 3, 3) == {}
    for bad, message in (
        ("1:", "Cannot read"),
        ("1:10", "outside"),
        ("1:2,1:3", "twice"),
    ):
        with pytest.raises(core.MappingError, match=message):
            core.parse_pins(bad, 9, 9)


def test_formula_and_xyz_text():
    ref = core.structure_from_mol(embedded("CCO"), perceive=True)
    assert ref.formula() == "C2H6O"
    lines = core.xyz_text(ref, comment="x").splitlines()
    assert lines[0] == "9" and lines[1] == "x"


def test_swap_and_move_orders():
    assert core.swap_order(5, 1, 3) == [0, 3, 2, 1, 4]
    assert core.move_order(5, 3, 0) == [3, 0, 1, 2, 4]
    assert core.move_order(5, 0, 3) == [1, 2, 3, 0, 4]
    assert core.swap_order(3, 1, 1) == [0, 1, 2]
    with pytest.raises(core.MappingError, match="outside 1-5"):
        core.move_order(5, 0, 5)
