from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter

from bapred2.data.features import N_ATOM_TOKENS, N_RES_TOKENS

from .layers import IntraPropagation, RecurrentBindingBlock

P_REL = ("protein", "intra", "protein")
L_REL = ("ligand", "intra", "ligand")
C_REL = ("protein", "contact", "ligand")


def _mean_pairwise_cosine(h: torch.Tensor, max_nodes: int = 2048) -> torch.Tensor:
    """Oversmoothing probe: mean cosine between centred node states (subsampled for large batches)."""
    if h.size(0) > max_nodes:
        h = h[torch.randperm(h.size(0), device=h.device)[:max_nodes]]
    z = torch.nn.functional.normalize(h - h.mean(0, keepdim=True), dim=-1)
    return (z @ z.t()).mean()


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
        protein_tokens: bool = False,
        node_update: str = "residual",
        bounded_scale: bool = False,
        pre_readout_norm: bool = False,
        readout_norm: str = "concat",
        core_dropout: float | None = None,
        q_candidate_norm: bool = False,
    ):
        super().__init__()
        d = hidden_dim
        self.protein_tokens = protein_tokens
        self.pre_readout_norm = pre_readout_norm
        self.readout_norm = readout_norm
        core_dropout = dropout if core_dropout is None else core_dropout
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
        if protein_tokens:
            # nn.Embedding == one-hot x Linear(bias=False); ints in the graph instead of 198 one-hot floats per atom.
            self.res_embed = nn.Embedding(N_RES_TOKENS, d)
            self.atom_embed = nn.Embedding(N_ATOM_TOKENS, d)
        self.p_prelude = nn.ModuleList([IntraPropagation(d, dropout, layerscale_init, node_update, bounded_scale) for _ in range(prelude_layers)])
        self.l_prelude = nn.ModuleList([IntraPropagation(d, dropout, layerscale_init, node_update, bounded_scale) for _ in range(prelude_layers)])
        self.core = RecurrentBindingBlock(d, core_dropout, layerscale_init, use_endpoint_context, node_update, bounded_scale, q_candidate_norm)
        if pre_readout_norm:
            self.p_out_norm, self.l_out_norm, self.q_out_norm = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.interface_weight = nn.Linear(d, 1)
        if readout_norm == "block":
            self.block_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
            first = nn.Identity()
        elif readout_norm == "concat":
            first = nn.LayerNorm(d * 4)
        else:
            raise ValueError(f"unknown readout_norm {readout_norm!r}")
        self.readout = nn.Sequential(
            first,
            nn.Linear(d * 4, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def _encode(self, data):
        hp0 = self.p_node(data["protein"].x) + self.p_pos(data["protein"].pos_enc)
        if self.protein_tokens:
            hp0 = hp0 + self.res_embed(data["protein"].token_res) + self.atom_embed(data["protein"].token_atom)
        hp0 = self.p_init_norm(hp0)
        hl0 = self.l_init_norm(self.l_node(data["ligand"].x) + self.l_pos(data["ligand"].pos_enc))
        pe = self.p_edge(data[P_REL].edge_attr)
        le = self.l_edge(data[L_REL].edge_attr)
        q0 = self.q_init_norm(self.q_edge(data[C_REL].edge_attr))
        return hp0, hl0, pe, le, q0

    def _readout(self, hp, hl, q, data):
        pidx, lidx = data[C_REL].edge_index
        p_batch = data["protein"].batch
        l_batch = data["ligand"].batch
        batch_size = int(max(p_batch.max().item(), l_batch.max().item())) + 1
        if self.pre_readout_norm:
            hp, hl, q = self.p_out_norm(hp), self.l_out_norm(hl), self.q_out_norm(q)
        edge_batch = l_batch[lidx]
        w = torch.sigmoid(self.interface_weight(q))
        q_pool = scatter(w * q, edge_batch, dim=0, dim_size=batch_size, reduce="sum")
        l_pool = global_add_pool(hl, l_batch, size=batch_size)
        p_strength = scatter(w, pidx, dim=0, dim_size=hp.size(0), reduce="sum")
        l_strength = scatter(w, lidx, dim=0, dim_size=hl.size(0), reduce="sum")
        p_contact = global_add_pool(p_strength * hp, p_batch, size=batch_size)
        l_contact = global_add_pool(l_strength * hl, l_batch, size=batch_size)
        blocks = [l_pool, q_pool, p_contact, l_contact]
        if self.readout_norm == "block":
            blocks = [n(b) for n, b in zip(self.block_norms, blocks)]
        return self.readout(torch.cat(blocks, dim=-1)).squeeze(-1)

    @staticmethod
    def _per_graph_rms(diff: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        """RMS of ``diff`` per graph, same normalisation as the global ``x.pow(2).mean().sqrt()``."""
        sq = diff.pow(2).sum(-1)
        total = scatter(sq, batch, dim=0, dim_size=batch_size, reduce="sum")
        count = scatter(torch.ones_like(sq), batch, dim=0, dim_size=batch_size, reduce="sum").clamp_min(1.0)
        return (total / (count * diff.size(-1))).sqrt()

    def forward(self, data, recycles: int = 6, return_aux: bool = False, return_all_cycles: bool = False, return_trace: bool = False):
        """Returns pred [B]; with ``return_all_cycles`` the per-cycle predictions [T, B] instead.

        ``return_aux`` adds ``cycle_delta`` [T] (mean RMS state change) and ``state_stats`` (per-state norms/deltas/
        cosine per cycle). ``return_trace`` additionally records, per cycle and per graph in the batch, the readout
        (``cycle_pred`` [T, B]) and the state changes (``delta_{p,l,q}_graph`` [T, B]) so an evaluator can stop each
        complex at its own convergence point; it implies ``return_aux``.
        """
        want_aux = return_aux or return_trace
        need_cycle_pred = return_all_cycles or return_trace
        hp0, hl0, pe, le, q0 = self._encode(data)
        hp, hl, q = hp0, hl0, q0
        for pblock, lblock in zip(self.p_prelude, self.l_prelude):
            hp = pblock(hp, hp0, data[P_REL].edge_index, pe)
            hl = lblock(hl, hl0, data[L_REL].edge_index, le)
        deltas, stats, cycle_preds = [], {k: [] for k in ("delta_p", "delta_l", "delta_q", "norm_p", "norm_l", "norm_q", "cos_p", "cos_l")}, []
        graph_stats = {k: [] for k in ("delta_p_graph", "delta_l_graph", "delta_q_graph")}
        if return_trace:
            p_batch, l_batch = data["protein"].batch, data["ligand"].batch
            batch_size = int(max(p_batch.max().item(), l_batch.max().item())) + 1
            q_batch = l_batch[data[C_REL].edge_index[1]]
        for _ in range(int(recycles)):
            old_hp, old_hl, old_q = hp, hl, q
            hp, hl, q = self.core(
                hp, hl, hp0, hl0, q, q0, data[C_REL].edge_index,
                data[P_REL].edge_index, pe, data[L_REL].edge_index, le,
            )
            if want_aux:
                with torch.no_grad():
                    dp, dl, dq = (hp - old_hp).pow(2).mean().sqrt(), (hl - old_hl).pow(2).mean().sqrt(), (q - old_q).pow(2).mean().sqrt()
                    deltas.append((dp + dl + dq) / 3.0)
                    for k, v in (("delta_p", dp), ("delta_l", dl), ("delta_q", dq), ("norm_p", hp.norm(dim=-1).mean()), ("norm_l", hl.norm(dim=-1).mean()),
                                 ("norm_q", q.norm(dim=-1).mean()), ("cos_p", _mean_pairwise_cosine(hp)), ("cos_l", _mean_pairwise_cosine(hl))):
                        stats[k].append(v.float())
            if return_trace:
                with torch.no_grad():
                    graph_stats["delta_p_graph"].append(self._per_graph_rms(hp - old_hp, p_batch, batch_size))
                    graph_stats["delta_l_graph"].append(self._per_graph_rms(hl - old_hl, l_batch, batch_size))
                    graph_stats["delta_q_graph"].append(self._per_graph_rms(q - old_q, q_batch, batch_size))
            if need_cycle_pred:
                cycle_preds.append(self._readout(hp, hl, q, data))
        if return_all_cycles:
            pred = torch.stack(cycle_preds, dim=0)
        else:
            pred = cycle_preds[-1] if cycle_preds else self._readout(hp, hl, q, data)
        if want_aux:
            aux = {"cycle_delta": torch.stack(deltas) if deltas else torch.empty(0, device=pred.device)}
            aux["state_stats"] = {k: torch.stack(v) if v else torch.empty(0, device=pred.device) for k, v in stats.items()}
            if return_trace:
                aux["cycle_pred"] = torch.stack(cycle_preds, dim=0) if cycle_preds else torch.empty(0, device=pred.device)
                for k, v in graph_stats.items():
                    aux[k] = torch.stack(v, dim=0) if v else torch.empty(0, device=pred.device)
            return pred, aux
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
        protein_tokens=cfg.protein_tokens,
        node_update=cfg.node_update,
        bounded_scale=cfg.bounded_scale,
        pre_readout_norm=cfg.pre_readout_norm,
        readout_norm=cfg.readout_norm,
        core_dropout=cfg.core_dropout,
        q_candidate_norm=cfg.q_candidate_norm,
    )


def model_from_sample(sample, cfg) -> BAPred2:
    if cfg.protein_tokens and not hasattr(sample["protein"], "token_res"):
        raise ValueError("model.protein_tokens=true but the graphs carry no protein.token_res; preprocess with graph.protein_tokens=true")
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
