import torch
from torch_geometric.data import Batch, HeteroData

from bapred2.config import Config, ModelConfig
from bapred2.model import feature_dims_from_sample, model_from_checkpoint, model_from_sample
from bapred2.train import make_checkpoint


def make_graph(seed=0, n_contacts=5):
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["protein"].x = torch.randn(5, 37, generator=g)
    data["protein"].pos_enc = torch.randn(5, 20, generator=g)
    data["ligand"].x = torch.randn(3, 37, generator=g)
    data["ligand"].pos_enc = torch.randn(3, 20, generator=g)
    p_rel = ("protein", "intra", "protein")
    l_rel = ("ligand", "intra", "ligand")
    c_rel = ("protein", "contact", "ligand")
    data[p_rel].edge_index = torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]])
    data[p_rel].edge_attr = torch.randn(8, 25, generator=g)
    data[l_rel].edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
    data[l_rel].edge_attr = torch.randn(4, 25, generator=g)
    data[c_rel].edge_index = torch.tensor([[0,1,2,3,4],[0,0,1,2,2]])[:, :n_contacts]
    data[c_rel].edge_attr = torch.randn(n_contacts, 31, generator=g)
    data.y = torch.tensor([6.5])
    data.sample_id = f"g{seed}"
    return data


def test_forward_and_recycle_sweep():
    sample = make_graph()
    model = model_from_sample(sample, ModelConfig(hidden_dim=64, prelude_layers=1))
    batch = Batch.from_data_list([sample, make_graph(1)])
    for r in [1, 2, 4, 8]:
        pred, aux = model(batch, recycles=r, return_aux=True)
        assert pred.shape == (2,)
        assert torch.isfinite(pred).all()
        assert aux["cycle_delta"].shape == (r,)


def test_backward():
    sample = make_graph()
    model = model_from_sample(sample, ModelConfig(hidden_dim=32, prelude_layers=1))
    batch = Batch.from_data_list([sample])
    loss = (model(batch, recycles=3) - batch.y.reshape(-1)).pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_low_contact_graph_in_batch():
    sample = make_graph()
    model = model_from_sample(sample, ModelConfig(hidden_dim=32, prelude_layers=1)).eval()
    batch = Batch.from_data_list([make_graph(2, n_contacts=1), sample])
    pred = model(batch, recycles=3)
    assert pred.shape == (2,) and torch.isfinite(pred).all()


def test_checkpoint_roundtrip(tmp_path):
    sample = make_graph()
    cfg = Config()
    cfg.model.hidden_dim, cfg.model.prelude_layers = 32, 1
    torch.manual_seed(0)
    model = model_from_sample(sample, cfg.model).eval()
    batch = Batch.from_data_list([sample, make_graph(1)])
    with torch.no_grad():
        ref = model(batch, recycles=4)
    path = tmp_path / "best.pt"
    torch.save(make_checkpoint(model, cfg, feature_dims_from_sample(sample), 1, {"rmse": 1.0}), path)
    ckpt = torch.load(path, weights_only=False)
    restored = model_from_checkpoint(ckpt).eval()
    with torch.no_grad():
        out = restored(batch, recycles=4)
    assert torch.allclose(ref, out)
    assert ckpt["config"]["model"]["hidden_dim"] == 32 and ckpt["feature_dims"]["interface_edge_dim"] == 31
