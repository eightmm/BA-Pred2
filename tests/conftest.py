"""Tiny synthetic complex: a tripeptide (with PDB residue info) and a phenol placed 3 A off its +x face."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


def _embed(mol, seed):
    molh = Chem.AddHs(mol, addResidueInfo=mol.GetAtomWithIdx(0).GetPDBResidueInfo() is not None)
    assert AllChem.EmbedMolecule(molh, randomSeed=seed) == 0
    return Chem.RemoveHs(molh)


def _coords(mol):
    return np.array(mol.GetConformer().GetPositions(), dtype=float)


def _set_coords(mol, xyz):
    conf = mol.GetConformer()
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*map(float, p)))


def build_complex(rotation=None, translation=None):
    peptide = _embed(Chem.MolFromSequence("GAV"), seed=7)
    ligand = _embed(Chem.MolFromSmiles("c1ccccc1O"), seed=11)
    p = _coords(peptide)
    lig = _coords(ligand)
    lig = lig - lig.mean(0)
    lig[:, 0] += p[:, 0].max() - lig[:, 0].min() + 3.0
    lig[:, 1:] += p[:, 1:].mean(0)
    if rotation is not None:
        p = p @ rotation.T
        lig = lig @ rotation.T
    if translation is not None:
        p = p + translation
        lig = lig + translation
    _set_coords(peptide, p)
    _set_coords(ligand, lig)
    return peptide, ligand


def write_complex(tmp_path, tag, peptide, ligand):
    pdb = tmp_path / f"{tag}_protein.pdb"
    sdf = tmp_path / f"{tag}_ligand.sdf"
    pdb.write_text(Chem.MolToPDBBlock(peptide))
    w = Chem.SDWriter(str(sdf))
    w.write(ligand)
    w.close()
    return pdb, sdf


@pytest.fixture
def synthetic_complex(tmp_path):
    peptide, ligand = build_complex()
    return write_complex(tmp_path, "base", peptide, ligand)
