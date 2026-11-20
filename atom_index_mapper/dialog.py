"""The mapper window. The decisions live in ``core``; this only wires them up."""

from __future__ import annotations

import logging
import os

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from . import PLUGIN_NAME, core, load_settings, save_settings

logger = logging.getLogger(__name__)

FILE_FILTER = "Structures (*.xyz *.mol *.sdf *.sd *.mol2 *.pdb);;All files (*)"
FIT_CHOICES = [
    ("all", "All atoms"),
    ("heavy", "Heavy atoms"),
    ("selected", "Selected atoms"),
]
METHOD_CHOICES = [
    (core.METHOD_CONNECTIVITY, "Connectivity (keep bonds)"),
    (core.METHOD_DISTANCE, "Distance (already overlaid)"),
]


class AtomIndexMapperDialog(QDialog):
    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.settings = load_settings()
        self.path = ""
        self.references: list[tuple[str, core.Structure]] = []
        self.setWindowTitle(PLUGIN_NAME)
        self.resize(560, 640)
        self._build()
        self._update_summary()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        layout = QVBoxLayout(self)

        ref_box = QGroupBox("Reference structure (the order to follow)")
        ref_form = QFormLayout(ref_box)
        row = QHBoxLayout()
        self.edit_path = QLineEdit()
        self.edit_path.setReadOnly(True)
        self.edit_path.setPlaceholderText("XYZ, MOL, SDF, MOL2 or PDB file")
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse)
        row.addWidget(self.edit_path, 1)
        row.addWidget(self.btn_browse)
        ref_form.addRow("File:", row)
        self.cmb_frame = QComboBox()
        self.cmb_frame.currentIndexChanged.connect(self._update_summary)
        ref_form.addRow("Structure:", self.cmb_frame)
        self.chk_perceive = QCheckBox("Take bonds from the geometry for both structures")
        self.chk_perceive.setToolTip(
            "XYZ files always get bonds from the geometry. Tick this when the loaded "
            "molecule's drawn bonds differ from what the geometry shows (metal contacts, "
            "a breaking bond drawn as a bond)."
        )
        self.chk_perceive.setChecked(bool(self.settings.get("perceive")))
        self.chk_perceive.toggled.connect(self._perceive_changed)
        ref_form.addRow(self.chk_perceive)
        self.lbl_summary = QLabel()
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextFormat(Qt.TextFormat.PlainText)
        ref_form.addRow(self.lbl_summary)
        layout.addWidget(ref_box)

        match_box = QGroupBox("Renumber the loaded molecule")
        match_form = QFormLayout(match_box)
        self.cmb_method = QComboBox()
        for key, label in METHOD_CHOICES:
            self.cmb_method.addItem(label, key)
        self._select(self.cmb_method, self.settings.get("method"))
        self.cmb_method.setToolTip(
            "Connectivity pairs atoms through the bonds the two structures share, so a "
            "reaction's endpoints keep every bond that does not change.\n"
            "Distance pairs each atom with the nearest one of the same element, for "
            "structures that already lie on top of each other."
        )
        match_form.addRow("Method:", self.cmb_method)
        self.edit_pins = QLineEdit()
        self.edit_pins.setPlaceholderText("optional, e.g. 1:3, 5:8  (reference : loaded)")
        self.edit_pins.setToolTip(
            "Pairs that must be kept, as 1-based reference:loaded atom numbers."
        )
        match_form.addRow("Pinned pairs:", self.edit_pins)
        self.btn_match = QPushButton("Match Indices")
        self.btn_match.setToolTip("Renumber the loaded molecule into the reference's atom order.")
        self.btn_match.clicked.connect(self.match_indices)
        match_form.addRow(self.btn_match)
        layout.addWidget(match_box)

        manual_box = QGroupBox("Change one number by hand")
        manual_form = QFormLayout(manual_box)
        self.lbl_selected = QLabel()
        manual_form.addRow("Selected atom:", self.lbl_selected)
        self.spin_number = QSpinBox()
        self.spin_number.setRange(1, 1)
        manual_form.addRow("New number:", self.spin_number)
        self.radio_swap = QRadioButton("Swap with the atom that has that number")
        self.radio_move = QRadioButton("Move it there and shift the atoms between")
        self.radio_swap.setChecked(self.settings.get("manual") != "move")
        self.radio_move.setChecked(self.settings.get("manual") == "move")
        manual_form.addRow(self.radio_swap)
        manual_form.addRow(self.radio_move)
        self.btn_manual = QPushButton("Apply Number")
        self.btn_manual.setToolTip("Select one atom in the 3D view, type its new number, apply.")
        self.btn_manual.clicked.connect(self.apply_number)
        manual_form.addRow(self.btn_manual)
        layout.addWidget(manual_box)
        # The host has no selection signal for plugins; the selection is read
        # on a short timer while the window is shown.
        self._selection_timer = QTimer(self)
        self._selection_timer.setInterval(400)
        self._selection_timer.timeout.connect(self._update_selection)

        fit_box = QGroupBox("Superimpose by atom number")
        fit_form = QFormLayout(fit_box)
        self.cmb_fit = QComboBox()
        for key, label in FIT_CHOICES:
            self.cmb_fit.addItem(label, key)
        self._select(self.cmb_fit, self.settings.get("fit_on"))
        fit_form.addRow("Fit on:", self.cmb_fit)
        self.btn_superpose = QPushButton("Superimpose")
        self.btn_superpose.setToolTip(
            "Rotate and move the loaded molecule onto the reference, atom i onto atom i, "
            "at minimum RMSD. The shape is not changed."
        )
        self.btn_superpose.clicked.connect(self.superimpose)
        fit_form.addRow(self.btn_superpose)
        layout.addWidget(fit_box)

        self.lbl_result = QLabel()
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setTextFormat(Qt.TextFormat.PlainText)
        self.lbl_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.lbl_result)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["#", "Element", "Was #", "Deviation (Å)", "Bond change"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    @staticmethod
    def _select(combo: QComboBox, key) -> None:
        index = combo.findData(key)
        if index >= 0:
            combo.setCurrentIndex(index)

    def event(self, event):
        # The loaded molecule can change while the window stays open.
        if event.type() == QEvent.Type.WindowActivate:
            self._update_summary()
        return super().event(event)

    def showEvent(self, event):
        self._selection_timer.start()
        self._update_selection()
        super().showEvent(event)

    def hideEvent(self, event):
        self._selection_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event):
        self._selection_timer.stop()
        self._remember()
        super().closeEvent(event)

    def _remember(self) -> None:
        self.settings["method"] = self.cmb_method.currentData()
        self.settings["fit_on"] = self.cmb_fit.currentData()
        self.settings["perceive"] = self.chk_perceive.isChecked()
        self.settings["manual"] = "move" if self.radio_move.isChecked() else "swap"
        save_settings(self.settings)

    # -------------------------------------------------------------- inputs
    def browse(self) -> None:
        start = self.settings.get("last_dir") or os.path.dirname(self.path)
        path, _ = QFileDialog.getOpenFileName(self, "Reference Structure", start or "", FILE_FILTER)
        if path:
            self.load_reference(path)

    def load_reference(self, path: str) -> bool:
        try:
            references = core.read_structures(path, perceive=self.chk_perceive.isChecked())
        except (core.MappingError, OSError) as exc:
            QMessageBox.warning(
                self, PLUGIN_NAME, f"Could not read {os.path.basename(path)}:\n{exc}"
            )
            return False
        self.path = path
        self.references = references
        self.settings["last_dir"] = os.path.dirname(path)
        self.edit_path.setText(path)
        self.cmb_frame.blockSignals(True)
        self.cmb_frame.clear()
        for label, structure in references:
            self.cmb_frame.addItem(f"{label}  ({len(structure)} atoms)")
        # The last frame of a trajectory is the optimised structure.
        self.cmb_frame.setCurrentIndex(len(references) - 1)
        self.cmb_frame.blockSignals(False)
        self.cmb_frame.setEnabled(len(references) > 1)
        self._remember()
        self._update_summary()
        return True

    def _perceive_changed(self) -> None:
        if self.path:
            index = self.cmb_frame.currentIndex()
            if self.load_reference(self.path) and 0 <= index < self.cmb_frame.count():
                self.cmb_frame.setCurrentIndex(index)
        else:
            self._update_summary()

    def reference(self) -> core.Structure:
        index = self.cmb_frame.currentIndex()
        if not self.references or index < 0:
            raise core.MappingError("Choose a reference structure file first.")
        return self.references[index][1]

    def loaded_molecule(self):
        mol = self.context.current_molecule if self.context is not None else None
        if mol is None or mol.GetNumAtoms() == 0:
            raise core.MappingError("Load a molecule with 3D coordinates first.")
        return mol

    def current(self) -> core.Structure:
        return core.structure_from_mol(
            self.loaded_molecule(), perceive=self.chk_perceive.isChecked()
        )

    def _update_summary(self) -> None:
        lines = []
        try:
            cur = self.current()
            lines.append(f"Loaded: {cur.formula()}, {len(cur)} atoms")
        except core.MappingError as exc:
            cur = None
            lines.append(f"Loaded: {exc}")
        if self.references and self.cmb_frame.currentIndex() >= 0:
            ref = self.reference()
            lines.append(f"Reference: {ref.formula()}, {len(ref)} atoms")
            if cur is not None:
                try:
                    core.check_same_atoms(ref, cur)
                except core.MappingError as exc:
                    lines.append(str(exc))
        self.lbl_summary.setText("\n".join(lines))
        ready = bool(self.references) and cur is not None
        self.btn_match.setEnabled(ready)
        self.btn_superpose.setEnabled(ready)

    # ------------------------------------------------------------- actions
    def match_indices(self) -> core.MappingResult | None:
        try:
            ref = self.reference()
            mol = self.loaded_molecule()
            cur = core.structure_from_mol(mol, perceive=self.chk_perceive.isChecked())
            pins = core.parse_pins(self.edit_pins.text(), len(ref), len(cur))
            result = core.find_mapping(ref, cur, self.cmb_method.currentData(), fixed=pins)
        except core.MappingError as exc:
            QMessageBox.warning(self, PLUGIN_NAME, str(exc))
            return None
        self._remember()
        moved = sum(1 for new, old in enumerate(result.mapping) if new != old)
        changes = self._bond_changes(result)
        text = [
            "Already in the reference's order."
            if moved == 0
            else f"Renumbered: {moved} of {len(result.mapping)} atoms changed number.",
            f"RMSD to the reference after superposition: {result.rmsd:.3f} Å.",
        ]
        if result.formed or result.broken:
            text.append(
                "Bonds only in the reference: "
                + (_pairs(result.formed) or "none")
                + ". Only in the loaded molecule: "
                + (_pairs(result.broken) or "none")
                + "."
            )
        else:
            text.append("Every bond is the same in both.")
        if moved:
            self._apply(
                core.renumber(mol, result.mapping),
                "Atoms renumbered to the reference order.",
            )
        self.lbl_result.setText("\n".join(text))
        self._fill_table(result.mapping, result.displacement, ref, changes)
        return result

    def _selected_atom(self):
        """(index, molecule) of the one selected atom, or (None, molecule)."""
        try:
            mol = self.loaded_molecule()
        except core.MappingError:
            return None, None
        try:
            picked = list(self.context.get_selected_atom_indices() or [])
        except Exception:  # pragma: no cover - host API guard
            logger.debug("%s: reading the selection failed", PLUGIN_NAME, exc_info=True)
            picked = []
        picked = [i for i in picked if 0 <= i < mol.GetNumAtoms()]
        return (picked[0] if len(picked) == 1 else None), mol

    def _update_selection(self) -> None:
        index, mol = self._selected_atom()
        count = mol.GetNumAtoms() if mol is not None else 0
        if self.spin_number.maximum() != max(count, 1):
            self.spin_number.setRange(1, max(count, 1))
        if index is None:
            picked = len(self.context.get_selected_atom_indices() or []) if mol else 0
            self.lbl_selected.setText(
                "none - select one atom in the 3D view"
                if picked == 0
                else f"{picked} atoms - select one"
            )
            self.btn_manual.setEnabled(False)
            self._shown_selection = None
            return
        symbol = mol.GetAtomWithIdx(index).GetSymbol()
        self.lbl_selected.setText(f"{index + 1} ({symbol})")
        self.btn_manual.setEnabled(True)
        if getattr(self, "_shown_selection", None) != index:
            # A new pick starts the number box from its current number.
            self._shown_selection = index
            self.spin_number.setValue(index + 1)

    def apply_number(self):
        index, mol = self._selected_atom()
        if index is None:
            QMessageBox.warning(self, PLUGIN_NAME, "Select exactly one atom in the 3D view.")
            return None
        target = self.spin_number.value() - 1
        count = mol.GetNumAtoms()
        try:
            if self.radio_move.isChecked():
                order = core.move_order(count, index, target)
            else:
                order = core.swap_order(count, index, target)
        except core.MappingError as exc:
            QMessageBox.warning(self, PLUGIN_NAME, str(exc))
            return None
        self._remember()
        if target == index:
            self.lbl_result.setText(f"Atom {index + 1} already has that number.")
            return order
        symbol = mol.GetAtomWithIdx(index).GetSymbol()
        if self.radio_move.isChecked():
            text = f"Atom {index + 1} ({symbol}) moved to {target + 1}; the atoms between shifted by one."
        else:
            other = mol.GetAtomWithIdx(target).GetSymbol()
            text = f"Atoms {index + 1} ({symbol}) and {target + 1} ({other}) swapped numbers."
        self._apply(core.renumber(mol, order), text)
        self._shown_selection = None
        self.lbl_result.setText(text)
        structure = core.structure_from_mol(self.context.current_molecule)
        self._fill_table(order, [0.0] * count, structure, {})
        return order

    def superimpose(self):
        try:
            ref = self.reference()
            mol = self.loaded_molecule()
            cur = core.structure_from_mol(mol)
            fit = self.cmb_fit.currentData()
            atoms = None
            if fit == "selected":
                atoms = list(self.context.get_selected_atom_indices() or [])
                if len(atoms) < 3:
                    raise core.MappingError("Select at least three atoms to fit on.")
            result = core.superpose_by_index(cur, ref, atoms=atoms, heavy_only=fit == "heavy")
        except core.MappingError as exc:
            QMessageBox.warning(self, PLUGIN_NAME, str(exc))
            return None
        self._remember()
        self._apply(
            core.with_coordinates(mol, result.coords),
            "Superimposed onto the reference.",
            moved=True,
        )
        self.lbl_result.setText(
            f"Superimposed using {result.fitted_on} atoms. RMSD over all atoms: "
            f"{result.rmsd_before:.3f} → {result.rmsd_after:.3f} Å."
        )
        deviation = np.linalg.norm(result.coords - ref.coords, axis=1).tolist()
        self._fill_table(list(range(len(ref))), deviation, ref, {})
        return result

    def _apply(self, mol, message: str, moved: bool = False) -> None:
        self.context.current_molecule = mol
        self.context.push_undo_checkpoint()
        try:
            self.context.mark_project_modified()
            if moved:
                # The molecule now sits in the reference's frame, possibly
                # far from where the camera was looking.
                self.context.reset_3d_camera()
            self.context.show_status_message(message, 4000)
        except Exception:  # pragma: no cover - older hosts
            logger.debug("%s: status update failed", PLUGIN_NAME, exc_info=True)
        self._update_summary()

    # --------------------------------------------------------------- table
    @staticmethod
    def _bond_changes(result: core.MappingResult) -> dict:
        changes: dict = {}
        for a, b in result.formed:
            changes.setdefault(a, []).append(f"+{b + 1}")
            changes.setdefault(b, []).append(f"+{a + 1}")
        for a, b in result.broken:
            changes.setdefault(a, []).append(f"−{b + 1}")
            changes.setdefault(b, []).append(f"−{a + 1}")
        return changes

    def _fill_table(self, order, deviation, ref: core.Structure, changes: dict) -> None:
        self.table.setRowCount(len(order))
        for row, old in enumerate(order):
            values = [
                str(row + 1),
                ref.symbols[row],
                str(old + 1),
                f"{deviation[row]:.3f}",
                " ".join(changes.get(row, [])),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 2, 3):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, col, item)


def _pairs(pairs) -> str:
    return ", ".join(f"{a + 1}-{b + 1}" for a, b in pairs)
