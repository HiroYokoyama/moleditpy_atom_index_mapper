import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from atom_index_mapper import core


def embedded(smiles, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def rotation(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def scrambled(structure, seed, noise=0.0):
    """The same atoms shuffled, rotated and moved; returns it and where each went."""
    rng = random.Random(seed)
    perm = list(range(len(structure)))
    rng.shuffle(perm)
    numbers = [structure.numbers[i] for i in perm]
    coords = structure.coords[perm] @ rotation(seed).T + np.array([3.0, -2.0, 5.0])
    if noise:
        coords = coords + np.random.default_rng(seed).normal(0, noise, coords.shape)
    where = {old: new for new, old in enumerate(perm)}
    return core.Structure(numbers, coords, core.perceive_bonds(numbers, coords)), where


def bonds_after(cur, order):
    position = {c: r for r, c in enumerate(order)}
    return {tuple(sorted((position[a], position[b]))) for a, b in cur.bonds}


def diels_alder():
    """Cyclohexene, and the same atoms with the C4-C5 unit pulled 1.4 A away."""
    product = core.structure_from_mol(embedded("C1=CCCCC1"), perceive=True)
    moving = [3, 4] + [
        h
        for c in (3, 4)
        for h in range(len(product))
        if product.numbers[h] == 1 and (min(c, h), max(c, h)) in product.bonds
    ]
    coords = product.coords.copy()
    away = coords[[3, 4]].mean(0) - coords[[0, 1, 2, 5]].mean(0)
    coords[moving] += away / np.linalg.norm(away) * 1.4
    reactant = core.Structure(product.numbers, coords, core.perceive_bonds(product.numbers, coords))
    return product, reactant
