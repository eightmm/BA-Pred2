import numpy as np
import pandas as pd
import pytest
import torch
from conftest import build_complex, write_complex
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch

from bapred2.config import GraphConfig, ModelConfig
from bapred2.data.preprocess import ComplexPreprocessor, preprocess_manifest
from bapred2.model import model_from_sample

P_REL = ("protein", "intra", "protein")
L_REL = ("ligand", "intra", "ligand")
C_REL = ("protein", "contact", "ligand")


def _all_tensors(g):
    yield g["protein"].x
    yield g["protein"].pos_enc
    yield g["ligand"].x
    yield g["ligand"].pos_enc
    yield g[P_REL].edge_attr
    yield g[L_REL].edge_attr
    yield g[C_REL].edge_attr


def test_build_synthetic_complex_is_finite(synthetic_complex):
    pdb, sdf = synthetic_complex
    g = ComplexPreprocessor(GraphConfig()).build(pdb, sdf, 6.0, "syn")
    assert g["protein"].x.size(0) > 0 and g["ligand"].x.size(0) == 7
    assert g[C_REL].edge_index.size(1) > 0
    assert g[P_REL].edge_index.size(1) > 0 and g[L_REL].edge_index.size(1) >= 14
    for t in _all_tensors(g):
        assert torch.isfinite(t).all()
    # covalent/spatial flags are one-hot and exclusive (feature layout: 4 bond types, 3 flags, [cov, spatial], rbf)
    flags = g[L_REL].edge_attr[:, 7:9]
    assert torch.all(flags.sum(-1) == 1)


def test_rigid_transform_invariance(tmp_path):
    rot = Rotation.random(random_state=3).as_matrix()
    shift = np.array([12.0, -7.5, 3.25])
    base = write_complex(tmp_path, "a", *build_complex())
    moved = write_complex(tmp_path, "b", *build_complex(rotation=rot, translation=shift))
    pre = ComplexPreprocessor(GraphConfig())
    g0 = pre.build(*base, 6.0, "a")
    g1 = pre.build(*moved, 6.0, "b")
    assert g0[C_REL].edge_index.size(1) == g1[C_REL].edge_index.size(1)
    # PDB/SDF files carry 3-4 decimals, so rotated coordinates re-round by ~1e-3 A; RBF/cosine features inherit that.
    for t0, t1 in zip(_all_tensors(g0), _all_tensors(g1)):
        assert torch.allclose(t0, t1, atol=5e-3), "raw invariant features must not change under rigid motion"
    torch.manual_seed(0)
    model = model_from_sample(g0, ModelConfig(hidden_dim=32, prelude_layers=1)).eval()
    with torch.no_grad():
        y0 = model(Batch.from_data_list([g0]), recycles=4)
        y1 = model(Batch.from_data_list([g1]), recycles=4)
    assert torch.allclose(y0, y1, atol=2e-2)


def test_no_contact_raises(tmp_path):
    peptide, ligand = build_complex()
    conf = ligand.GetConformer()
    from rdkit.Geometry import Point3D
    for i in range(ligand.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + 50.0, p.y, p.z))
    pdb, sdf = write_complex(tmp_path, "far", peptide, ligand)
    with pytest.raises(ValueError, match="pocket|interface"):
        ComplexPreprocessor(GraphConfig()).build(pdb, sdf, 6.0, "far")


def test_preprocess_manifest_roundtrip(tmp_path, synthetic_complex):
    pdb, sdf = synthetic_complex
    peptide, ligand = build_complex()
    conf = ligand.GetConformer()
    from rdkit.Geometry import Point3D
    for i in range(ligand.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + 50.0, p.y, p.z))
    far_pdb, far_sdf = write_complex(tmp_path, "far", peptide, ligand)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        # pKd alias, float-looking id, missing alt path and pocket path
        {"id": "1e10", "protein_path": str(pdb), "ligand_path": str(sdf), "pKd": 6.0, "split": "train"},
        {"id": "0002", "protein_path": str(pdb), "pocket_path": str(pdb), "ligand_path": str(tmp_path / "missing.sdf"), "ligand_alt_path": str(sdf), "pKd": 5.0, "split": "val"},
        {"id": "0003", "protein_path": str(far_pdb), "ligand_path": str(far_sdf), "pKd": 4.0, "split": "test"},
    ]).to_csv(manifest, index=False)
    out = preprocess_manifest(manifest, tmp_path / "proc", GraphConfig(), workers=1)
    df = pd.read_csv(out, dtype={"id": str})
    assert list(df["id"]) == ["1e10", "0002"]
    assert df["ligand_used"].iloc[1].endswith("base_ligand.sdf")
    skipped = pd.read_csv(tmp_path / "proc" / "skipped.csv", dtype={"id": str})
    assert list(skipped["id"]) == ["0003"] and ("pocket" in skipped["reason"].iloc[0] or "interface" in skipped["reason"].iloc[0])
    assert (tmp_path / "proc" / "preprocess_config.yaml").exists()
    g = torch.load(df["graph_path"].iloc[0], weights_only=False)
    assert float(g.y) == 6.0
    # rerun reuses cached graphs
    out2 = preprocess_manifest(manifest, tmp_path / "proc", GraphConfig(), workers=1)
    assert set(pd.read_csv(out2)["ligand_used"]) == {"cached"}
