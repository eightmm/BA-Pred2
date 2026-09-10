from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

from .layers import IntraPropagation, RecurrentBindingBlock

P_REL = ("protein", "intra", "protein")
L_REL = ("ligand", "intra", "ligand")
C_REL = ("protein", "contact", "ligand")


class BAPred2(nn.Module):
    def __init__(
        self,
        protein_node_dim: int,
        ligand_node_dim: int,
        protein_edge_dim: int,
        ligand_edge_dim: int,
        interface_edge_dim: int,
        pos_dim: int,
        hidden_dim: int = 256,
        prelude_layers: int = 1,
        dropout: float = 0.1,
        layerscale_init: float = 0.1,
        use_endpoint_context: bool = True,
    ):
        super().__init__()
        d = hidden_dim
        self.p_node = nn.Linear(protein_node_dim, d)
        self.l_node = nn.Linear(ligand_node_dim, d)
        self.p_pos = nn.Linear(pos_dim, d)
        self.l_pos = nn.Linear(pos_dim, d)
        self.p_edge = nn.Linear(protein_edge_dim, d)
        self.l_edge = nn.Linear(ligand_edge_dim, d)
        self.q_edge = nn.Linear(interface_edge_dim, d)
        self.p_init_norm = nn.LayerNorm(d)
        self.l_init_norm = nn.LayerNorm(d)
        self.q_init_norm = nn.LayerNorm(d)
        self.p_prelude = nn.ModuleList([IntraPropagation(d, dropout, layerscale_init) for _ in range(prelude_layers)])
        self.l_prelude = nn.ModuleList([IntraPropagation(d, dropout, layerscale_init) for _ in range(prelude_layers)])
        self.core = RecurrentBindingBlock(d, dropout, layerscale_init, use_endpoint_context)
        self.interface_weight = nn.Linear(d, 1)
        self.readout = nn.Sequential(
            nn.LayerNorm(d * 4),
            nn.Linear(d * 4, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, data, recycles: int = 6, return_aux: bool = False):
        hp0 = self.p_init_norm(self.p_node(data["protein"].x) + self.p_pos(data["protein"].pos_enc))
        hl0 = self.l_init_norm(self.l_node(data["ligand"].x) + self.l_pos(data["ligand"].pos_enc))
        pe = self.p_edge(data[P_REL].edge_attr)
        le = self.l_edge(data[L_REL].edge_attr)
        q0 = self.q_init_norm(self.q_edge(data[C_REL].edge_attr))
        hp, hl, q = hp0, hl0, q0
        for pblock, lblock in zip(self.p_prelude, self.l_prelude):
            hp = pblock(hp, hp0, data[P_REL].edge_index, pe)
            hl = lblock(hl, hl0, data[L_REL].edge_index, le)
        deltas = []
        for _ in range(int(recycles)):
            old_hp, old_hl, old_q = hp, hl, q
            hp, hl, q = self.core(
                hp, hl, hp0, hl0, q, q0, data[C_REL].edge_index,
                data[P_REL].edge_index, pe, data[L_REL].edge_index, le,
            )
            if return_aux:
                delta = ((hp-old_hp).pow(2).mean().sqrt() + (hl-old_hl).pow(2).mean().sqrt() + (q-old_q).pow(2).mean().sqrt()) / 3.0
                deltas.append(delta.detach())
        pidx, lidx = data[C_REL].edge_index
        p_batch = data["protein"].batch
        l_batch = data["ligand"].batch
        batch_size = int(max(p_batch.max().item(), l_batch.max().item())) + 1
        edge_batch = l_batch[lidx]
        w = torch.sigmoid(self.interface_weight(q))
        q_pool = scatter(w * q, edge_batch, dim=0, dim_size=batch_size, reduce="sum")
        l_pool = global_add_pool(hl, l_batch, size=batch_size)
        p_strength = scatter(w, pidx, dim=0, dim_size=hp.size(0), reduce="sum")
        l_strength = scatter(w, lidx, dim=0, dim_size=hl.size(0), reduce="sum")
        p_contact = global_add_pool(p_strength * hp, p_batch, size=batch_size)
        l_contact = global_add_pool(l_strength * hl, l_batch, size=batch_size)
        pred = self.readout(torch.cat([l_pool, q_pool, p_contact, l_contact], dim=-1)).squeeze(-1)
        if return_aux:
            return pred, {"cycle_delta": torch.stack(deltas) if deltas else torch.empty(0, device=pred.device)}
        return pred


def feature_dims_from_sample(sample) -> dict[str, int]:
    """Raw feature widths the model must be built for; stored in checkpoints so eval needs no manifest sample."""
    return {
        "protein_node_dim": int(sample["protein"].x.size(-1)),
        "ligand_node_dim": int(sample["ligand"].x.size(-1)),
        "protein_edge_dim": int(sample[P_REL].edge_attr.size(-1)),
        "ligand_edge_dim": int(sample[L_REL].edge_attr.size(-1)),
        "interface_edge_dim": int(sample[C_REL].edge_attr.size(-1)),
        "pos_dim": int(sample["protein"].pos_enc.size(-1)),
    }


def model_from_dims(dims: dict[str, int], cfg) -> BAPred2:
    return BAPred2(
        **dims,
        hidden_dim=cfg.hidden_dim,
        prelude_layers=cfg.prelude_layers,
        dropout=cfg.dropout,
        layerscale_init=cfg.layerscale_init,
        use_endpoint_context=cfg.use_endpoint_context,
    )


def model_from_sample(sample, cfg) -> BAPred2:
    return model_from_dims(feature_dims_from_sample(sample), cfg)


def model_from_checkpoint(ckpt: dict, cfg=None) -> BAPred2:
    """Rebuild the exact architecture from a checkpoint produced by ``bapred2-train``.

    ``cfg`` (a ModelConfig) overrides the stored one only when explicitly passed; the stored feature dims are authoritative.
    """
    from bapred2.config import config_from_dict

    if "feature_dims" not in ckpt or "config" not in ckpt:
        raise KeyError("Checkpoint lacks 'feature_dims'/'config'; it was not written by bapred2-train >= 0.1.0")
    model_cfg = cfg if cfg is not None else config_from_dict(ckpt["config"]).model
    model = model_from_dims(ckpt["feature_dims"], model_cfg)
    model.load_state_dict(ckpt["model"])
    return model
