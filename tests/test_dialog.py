"""The window, driven with a fake host context and a real Qt."""

import json

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("PySide6.QtWidgets")

from rdkit import Chem

import atom_index_mapper
from atom_index_mapper import core

from .helpers import embedded, rotation


class FakeContext:
    def __init__(self, mol=None):
        self.current_molecule = mol
        self.undo = 0
        self.modified = 0
        self.selected = []
        self.windows = {}
        self.menu = {}
        self.messages = []
        self.camera_resets = 0

    def push_undo_checkpoint(self):
        self.undo += 1

    def mark_project_modified(self):
        self.modified += 1

    def reset_3d_camera(self):
        self.camera_resets += 1

    def show_status_message(self, message, timeout=3000):
        self.messages.append(message)

    def get_selected_atom_indices(self):
        return list(self.selected)

    def get_main_window(self):
        return None

    def get_window(self, key):
        return self.windows.get(key)

    def register_window(self, key, window):
        self.windows[key] = window

    def add_menu_action(self, path, callback):
        self.menu[path] = callback


@pytest.fixture
def warnings(monkeypatch):
    from atom_index_mapper import dialog

    seen = []
    monkeypatch.setattr(dialog.QMessageBox, "warning", lambda *args: seen.append(args[2]))
    return seen


def write_xyz(path, mol, order=None, transform=None):
    order = list(range(mol.GetNumAtoms())) if order is None else order
    coords = mol.GetConformer().GetPositions()[order]
    if transform is not None:
        coords = transform(coords)
    lines = [str(len(order)), "reference"]
    for i, (x, y, z) in zip(order, coords):
        lines.append(f"{mol.GetAtomWithIdx(i).GetSymbol()} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_dialog(context):
    from atom_index_mapper.dialog import AtomIndexMapperDialog

    return AtomIndexMapperDialog(context)


def test_match_renumbers_the_loaded_molecule_to_the_file(qapp, tmp_path, warnings):
    mol = embedded("CC(=O)O")
    for atom in mol.GetAtoms():
        atom.SetIntProp("_original_atom_id", 100 + atom.GetIdx())
    order = list(range(mol.GetNumAtoms()))[::-1]
    path = tmp_path / "ref.xyz"
    write_xyz(path, mol, order, lambda c: c @ rotation(3).T + 2.0)
    context = FakeContext(mol)
    dlg = make_dialog(context)
    assert dlg.load_reference(str(path))
    assert "Reference: C2H4O2, 8 atoms" in dlg.lbl_summary.text()
    result = dlg.match_indices()
    assert warnings == []
    assert result.mapping == order
    renumbered = context.current_molecule
    assert [a.GetIntProp("_original_atom_id") for a in renumbered.GetAtoms()] == [
        100 + i for i in order
    ]
    assert context.undo == 1 and context.modified == 1
    assert context.camera_resets == 0
    assert "8 of 8 atoms changed number" in dlg.lbl_result.text()
    assert dlg.table.rowCount() == 8
    assert dlg.table.item(0, 2).text() == str(order[0] + 1)

    # Matching again finds nothing to do and leaves the undo stack alone.
    dlg.match_indices()
    assert context.undo == 1
    assert "Already in the reference's order." in dlg.lbl_result.text()


def test_superimpose_moves_the_molecule_onto_the_file(qapp, tmp_path, warnings):
    mol = embedded("CC(=O)O")
    path = tmp_path / "ref.xyz"
    write_xyz(path, mol, transform=lambda c: c @ rotation(5).T + np.array([4.0, 1.0, -2.0]))
    context = FakeContext(mol)
    dlg = make_dialog(context)
    dlg.load_reference(str(path))
    result = dlg.superimpose()
    assert warnings == []
    assert result.rmsd_after < 1e-5
    ref = dlg.reference()
    assert np.allclose(
        context.current_molecule.GetConformer().GetPositions(), ref.coords, atol=1e-5
    )
    assert context.undo == 1
    assert context.camera_resets == 1
    assert "→ 0.000" in dlg.lbl_result.text()


def test_superimpose_on_selected_needs_three_atoms(qapp, tmp_path, warnings):
    mol = embedded("CCO")
    path = tmp_path / "ref.xyz"
    write_xyz(path, mol)
    context = FakeContext(mol)
    dlg = make_dialog(context)
    dlg.load_reference(str(path))
    dlg.cmb_fit.setCurrentIndex(dlg.cmb_fit.findData("selected"))
    context.selected = [0, 1]
    assert dlg.superimpose() is None
    assert "at least three" in warnings[-1]
    context.selected = [0, 1, 2]
    assert dlg.superimpose().fitted_on == 3


def test_mismatched_atoms_are_shown_and_refused(qapp, tmp_path, warnings):
    path = tmp_path / "ref.xyz"
    write_xyz(path, embedded("CCN"))
    context = FakeContext(embedded("CCO"))
    dlg = make_dialog(context)
    dlg.load_reference(str(path))
    assert "has 10 atoms but the current molecule has 9" in dlg.lbl_summary.text()
    assert dlg.match_indices() is None
    assert context.undo == 0
    assert warnings


def test_nothing_loaded_disables_the_buttons(qapp, tmp_path, warnings):
    path = tmp_path / "ref.xyz"
    write_xyz(path, embedded("CO"))
    dlg = make_dialog(FakeContext(None))
    dlg.load_reference(str(path))
    assert not dlg.btn_match.isEnabled()
    assert "Load a molecule with 3D coordinates first." in dlg.lbl_summary.text()


def test_a_trajectory_offers_its_frames_and_starts_on_the_last(qapp, tmp_path, warnings):
    mol = embedded("O")
    one = tmp_path / "a.xyz"
    write_xyz(one, mol)
    text = one.read_text(encoding="utf-8")
    path = tmp_path / "path.xyz"
    path.write_text(text * 3, encoding="utf-8")
    dlg = make_dialog(FakeContext(mol))
    dlg.load_reference(str(path))
    assert dlg.cmb_frame.count() == 3
    assert dlg.cmb_frame.currentIndex() == 2
    assert dlg.cmb_frame.isEnabled()


def test_an_unreadable_file_is_reported(qapp, tmp_path, warnings):
    path = tmp_path / "bad.xyz"
    path.write_text("not a count\n", encoding="utf-8")
    dlg = make_dialog(FakeContext(embedded("O")))
    assert not dlg.load_reference(str(path))
    assert "atom count" in warnings[-1]
    assert dlg.path == ""


def test_bad_pins_are_reported(qapp, tmp_path, warnings):
    mol = embedded("CO")
    path = tmp_path / "ref.xyz"
    write_xyz(path, mol)
    context = FakeContext(mol)
    dlg = make_dialog(context)
    dlg.load_reference(str(path))
    dlg.edit_pins.setText("1:99")
    assert dlg.match_indices() is None
    assert "outside" in warnings[-1]


def test_choices_are_remembered(qapp, tmp_path, warnings):
    mol = embedded("CO")
    path = tmp_path / "ref.xyz"
    write_xyz(path, mol)
    dlg = make_dialog(FakeContext(mol))
    dlg.load_reference(str(path))
    dlg.cmb_method.setCurrentIndex(dlg.cmb_method.findData(core.METHOD_DISTANCE))
    dlg.cmb_fit.setCurrentIndex(dlg.cmb_fit.findData("heavy"))
    dlg.chk_perceive.setChecked(True)
    dlg.close()
    with open(atom_index_mapper.SETTINGS_FILE, encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved == {
        "last_dir": str(tmp_path),
        "method": core.METHOD_DISTANCE,
        "perceive": True,
        "fit_on": "heavy",
    }
    again = make_dialog(FakeContext(mol))
    assert again.cmb_method.currentData() == core.METHOD_DISTANCE
    assert again.cmb_fit.currentData() == "heavy"
    assert again.chk_perceive.isChecked()


def test_initialize_registers_one_menu_entry_and_one_window(qapp):
    context = FakeContext(Chem.MolFromSmiles("C"))
    atom_index_mapper.initialize(context)
    assert list(context.menu) == ["3D Edit/Atom Index Mapper..."]
    first = context.menu["3D Edit/Atom Index Mapper..."]()
    second = atom_index_mapper.open_dialog()
    assert first is second
    first.close()
