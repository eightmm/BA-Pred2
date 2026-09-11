"""v0.2: residue-template protein features, flag-gated model options, and Milestone-0 checkpoint compatibility."""
import torch
from conftest import build_complex, write_complex
from rdkit import Chem
from rdkit.Chem import AllChem
from test_model import make_graph
from torch_geometric.data import Batch

from bapred2.config import GraphConfig, ModelConfig, config_from_dict
from bapred2.data.features import N_ATOM_TOKENS, N_RES_TOKENS, RES_TOKENS, protein_tokens_and_flags
from bapred2.data.preprocess import ComplexPreprocessor, _load_protein, _select_pocket
from bapred2.model import feature_dims_from_sample, model_from_dims, model_from_sample

FLAG = {"hba": 0, "hbd": 1, "cation": 2, "anion": 3, "hydrophobic": 4}


def _peptide_pdb(tmp_path, seq="KDSHPY"):
    mol = Chem.MolFromSequence(seq)
    molh = Chem.AddHs(mol, addResidueInfo=True)
    assert AllChem.EmbedMolecule(molh, randomSeed=3) == 0
    mol = Chem.RemoveHs(molh)
    block = Chem.MolToPDBBlock(mol)
    # one crystal water 3 A off the first atom, then a zinc ion
    p = mol.GetConformer().GetAtomPosition(0)
    block = block.replace("END", f"HETATM 9001  O   HOH A 901    {p.x + 3:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.00           O  \nHETATM 9002 ZN    ZN A 902    {p.x:8.3f}{p.y + 3:8.3f}{p.z:8.3f}  1.00  0.00          ZN  \nEND")
    path = tmp_path / "pep.pdb"
    path.write_text(block)
    return path


def test_template_tokens_and_flags(tmp_path):
    prot = _load_protein(_peptide_pdb(tmp_path))
    idx = [a.GetIdx() for a in prot.GetAtoms() if a.GetAtomicNum() != 1]
    res_tok, atom_tok, flags = protein_tokens_and_flags(prot, idx)
    assert res_tok.shape == (len(idx),) and atom_tok.shape == (len(idx),) and flags.shape == (len(idx), 5)
    assert int(atom_tok.max()) < N_ATOM_TOKENS and int(res_tok.max()) < N_RES_TOKENS
    by_name = {}
    for k, i in enumerate(idx):
        info = prot.GetAtomWithIdx(i).GetPDBResidueInfo()
        by_name[(info.GetResidueName().strip(), info.GetName().strip())] = k

    def f(res, name, flag):
        return float(flags[by_name[(res, name)], FLAG[flag]])

    assert f("LYS", "NZ", "cation") == 1 and f("LYS", "NZ", "hbd") == 1
    assert f("ASP", "OD1", "anion") == 1 and f("ASP", "OD2", "hba") == 1
    assert f("SER", "OG", "hbd") == 1 and f("SER", "OG", "hba") == 1
    assert f("HIS", "ND1", "hbd") == 1 and f("HIS", "NE2", "hba") == 1
    assert f("TYR", "OH", "hbd") == 1 and f("TYR", "CE1", "hydrophobic") == 1
    assert f("ASP", "N", "hbd") == 1 and f("PRO", "N", "hbd") == 0
    assert int(res_tok[by_name[("ZN", "ZN")]]) == RES_TOKENS["METAL"] and f("ZN", "ZN", "cation") == 1
    assert int(res_tok[by_name[("HOH", "O")]]) == RES_TOKENS["XXX"]
    # water is excluded by the pocket selector when drop_water is on
    lig = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(lig, randomSeed=1)
    lig = Chem.RemoveHs(lig)
    conf, p0 = lig.GetConformer(), prot.GetConformer().GetAtomPosition(0)
    for i in range(lig.GetNumAtoms()):
        q = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, type(q)(q.x + p0.x, q.y + p0.y, q.z + p0.z))
    kept = _select_pocket(prot, lig, 8.0, drop_water=True)
    assert all(prot.GetAtomWithIdx(i).GetPDBResidueInfo().GetResidueName().strip() != "HOH" for i in kept)
    assert any(prot.GetAtomWithIdx(i).GetPDBResidueInfo().GetResidueName().strip() == "HOH" for i in _select_pocket(prot, lig, 8.0, drop_water=False))


def test_v02_graph_has_tokens_and_flags(tmp_path):
    pdb, sdf = write_complex(tmp_path, "a", *build_complex())
    g = ComplexPreprocessor(GraphConfig(protein_tokens=True, node_chem_flags=True)).build(pdb, sdf, 6.0, "a")
    assert g["protein"].x.size(-1) == 42 and g["ligand"].x.size(-1) == 42
    assert g["protein"].token_res.dtype == torch.long and g["protein"].token_atom.size(0) == g["protein"].x.size(0)
    g0 = ComplexPreprocessor(GraphConfig(protein_tokens=False, node_chem_flags=False, drop_water=False)).build(pdb, sdf, 6.0, "a")
    assert g0["protein"].x.size(-1) == 37 and not hasattr(g0["protein"], "token_res")
    cfg = ModelConfig(hidden_dim=32, protein_tokens=True, node_update="interpolate", bounded_scale=True, pre_readout_norm=True, readout_norm="block", core_dropout=0.0)
    model = model_from_sample(g, cfg)
    batch = Batch.from_data_list([g, g])
    pred, aux = model(batch, recycles=3, return_aux=True)
    assert pred.shape == (2,) and torch.isfinite(pred).all()
    assert set(aux["state_stats"]) >= {"delta_l", "norm_l", "cos_l"} and aux["state_stats"]["delta_l"].shape == (3,)
    all_cycles = model(batch, recycles=4, return_all_cycles=True)
    assert all_cycles.shape == (4, 2)
    all_cycles.sum().backward()
    assert model.res_embed.weight.grad is not None


def test_interpolate_update_is_bounded():
    sample = make_graph()
    cfg = ModelConfig(hidden_dim=32, node_update="interpolate", bounded_scale=True, dropout=0.0)
    model = model_from_sample(sample, cfg).eval()
    from bapred2.model.layers import scale_gain
    scale = scale_gain(model.core.q_scale, True)
    assert torch.all(scale > 0) and torch.all(scale < 1) and abs(float(scale.mean()) - 0.1) < 1e-3
    batch = Batch.from_data_list([sample, make_graph(1)])
    with torch.no_grad():
        _, aux = model(batch, recycles=12, return_aux=True)
    norms = aux["state_stats"]["norm_l"]
    assert torch.isfinite(norms).all() and float(norms[-1]) < 3 * float(norms[0])


def test_legacy_config_dict_loads_strict():
    """A Milestone-0 checkpoint stores no v0.2 keys; defaults must rebuild the identical architecture."""
    sample = make_graph()
    legacy_cfg = config_from_dict({"model": {"hidden_dim": 32, "prelude_layers": 1, "dropout": 0.1, "layerscale_init": 0.1, "use_endpoint_context": True}})
    torch.manual_seed(0)
    model = model_from_dims(feature_dims_from_sample(sample), legacy_cfg.model).eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    assert "core.q_scale" in state and not any("res_embed" in k or "out_norm" in k or "block_norms" in k for k in state)
    rebuilt = model_from_dims(feature_dims_from_sample(sample), legacy_cfg.model).eval()
    rebuilt.load_state_dict(state, strict=True)
    batch = Batch.from_data_list([sample])
    with torch.no_grad():
        assert torch.allclose(model(batch, recycles=3), rebuilt(batch, recycles=3))


def test_cycle_trace_matches_fixed_T():
    """Early exit at cycle t must equal a fixed run with recycles=t, otherwise adaptive stopping is not comparable."""
    sample = make_graph()
    torch.manual_seed(0)
    model = model_from_sample(sample, ModelConfig(hidden_dim=32, dropout=0.0)).eval()
    batch = Batch.from_data_list([sample, make_graph(1), make_graph(2, n_contacts=1)])
    with torch.no_grad():
        _, aux = model(batch, recycles=5, return_trace=True)
        for t in range(1, 6):
            assert torch.allclose(aux["cycle_pred"][t - 1], model(batch, recycles=t), atol=1e-5)
    assert aux["cycle_pred"].shape == (5, 3)
    for key in ("delta_p_graph", "delta_l_graph", "delta_q_graph"):
        assert aux[key].shape == (5, 3) and torch.isfinite(aux[key]).all()
    # the batch-mean of the per-graph deltas tracks the scalar diagnostic within pooling differences
    assert aux["delta_l_graph"].mean() > 0


def test_stop_indices_rule():
    import numpy as np

    from bapred2.evaluate import _stop_indices

    signal = np.array([[9.0, 9.0, 9.0], [0.5, 9.0, 9.0], [0.1, 0.4, 9.0], [0.1, 0.1, 9.0]])
    assert list(_stop_indices(signal, 1.0, 0)) == [1, 2, 3]  # first cycle below eps, else the last
    assert list(_stop_indices(signal, 0.2, 0)) == [2, 3, 3]
    assert list(_stop_indices(signal, 100.0, 1)) == [1, 1, 1]  # never stops before first_usable


def test_q_candidate_norm_bounds_interface_state():
    """v0.2b left ||q|| drifting upward; normalising the candidate must keep it flat across cycles."""
    sample = make_graph()
    base_cfg = ModelConfig(hidden_dim=32, dropout=0.0, node_update="interpolate", bounded_scale=True, pre_readout_norm=True, readout_norm="block")
    normed_cfg = ModelConfig(hidden_dim=32, dropout=0.0, node_update="interpolate", bounded_scale=True, pre_readout_norm=True, readout_norm="block", q_candidate_norm=True)
    batch = Batch.from_data_list([sample, make_graph(1)])
    torch.manual_seed(0)
    model = model_from_sample(sample, normed_cfg).eval()
    assert model.core.q_cand_norm is not None
    with torch.no_grad():
        _, aux = model(batch, recycles=16, return_aux=True)
    norms = aux["state_stats"]["norm_q"]
    assert float(norms[-1]) < 1.5 * float(norms[0])
    torch.manual_seed(0)
    plain = model_from_sample(sample, base_cfg).eval()
    assert plain.core.q_cand_norm is None
