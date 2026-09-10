from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.utils import scatter


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def init_scale(dim: int, init: float, bounded: bool) -> nn.Parameter:
    """LayerScale parameter. ``bounded`` stores the logit so the effective gain is sigmoid(theta) in (0, 1). Kept as a
    bare Parameter (not a submodule) so Milestone-0 checkpoints keep their state_dict keys."""
    value = math.log(init / (1.0 - init)) if bounded else init
    return nn.Parameter(torch.full((dim,), float(value)))


def scale_gain(param: torch.Tensor, bounded: bool) -> torch.Tensor:
    return torch.sigmoid(param) if bounded else param


def gated_update(h: torch.Tensor, delta: torch.Tensor, gate: torch.Tensor, scale: torch.Tensor, mode: str) -> torch.Tensor:
    """``residual``: h + s*g*delta (unbounded accumulation). ``interpolate``: h + s*g*(delta - h), i.e. delta is a
    candidate state and h moves toward it by at most s*g, mirroring the interface update (SPEC 9.1)."""
    if mode == "residual":
        return h + scale * gate * delta
    if mode == "interpolate":
        return h + scale * gate * (delta - h)
    raise ValueError(f"unknown node_update mode {mode!r}")


class IntraPropagation(nn.Module):
    """Input-anchored gated propagation over a static molecular graph."""

    def __init__(self, dim: int, dropout: float, layerscale_init: float, node_update: str = "residual", bounded_scale: bool = False):
        super().__init__()
        self.node_update = node_update
        self.bounded_scale = bounded_scale
        self.norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)
        self.score = MLP(dim * 3, dim, 1, dropout)
        self.value = MLP(dim * 2, dim, dim, dropout)
        self.update = MLP(dim * 3, dim * 2, dim, dropout)
        self.gate = nn.Linear(dim * 3, dim)
        self.scale = init_scale(dim, layerscale_init, bounded_scale)

    def forward(self, h, h0, edge_index, edge_emb):
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        x = self.norm(h)
        e = self.edge_norm(edge_emb)
        pair = torch.cat([x[src], x[dst], e], dim=-1)
        score = torch.sigmoid(self.score(pair))
        denom = scatter(score, dst, dim=0, dim_size=h.size(0), reduce="sum").clamp_min(1e-6)
        alpha = score / denom[dst]
        msg = alpha * self.value(torch.cat([x[src], e], dim=-1))
        agg = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="sum")
        u = torch.cat([x, agg, h0], dim=-1)
        delta = self.update(u)
        gate = torch.sigmoid(self.gate(u))
        return gated_update(h, delta, gate, scale_gain(self.scale, self.bounded_scale), self.node_update)


class RecurrentBindingBlock(nn.Module):
    """One shared cycle: interface refinement -> cross message -> intra propagation."""

    def __init__(self, dim: int, dropout: float, layerscale_init: float, use_endpoint_context: bool = True, node_update: str = "residual", bounded_scale: bool = False):
        super().__init__()
        self.use_endpoint_context = use_endpoint_context
        self.node_update = node_update
        self.bounded_scale = bounded_scale
        self.p_norm = nn.LayerNorm(dim)
        self.l_norm = nn.LayerNorm(dim)
        self.q_norm = nn.LayerNorm(dim)
        q_in_dim = dim * (6 if use_endpoint_context else 4)
        self.q_candidate = MLP(q_in_dim, dim * 2, dim, dropout)
        self.q_gate = nn.Linear(q_in_dim, dim)
        self.q_scale = init_scale(dim, layerscale_init, bounded_scale)
        self.pl_score = MLP(dim * 3, dim, 1, dropout)
        self.lp_score = MLP(dim * 3, dim, 1, dropout)
        self.pl_value = MLP(dim * 2, dim, dim, dropout)
        self.lp_value = MLP(dim * 2, dim, dim, dropout)
        self.p_cross_update = MLP(dim * 3, dim * 2, dim, dropout)
        self.l_cross_update = MLP(dim * 3, dim * 2, dim, dropout)
        self.p_cross_gate = nn.Linear(dim * 3, dim)
        self.l_cross_gate = nn.Linear(dim * 3, dim)
        self.p_cross_scale = init_scale(dim, layerscale_init, bounded_scale)
        self.l_cross_scale = init_scale(dim, layerscale_init, bounded_scale)
        self.p_intra = IntraPropagation(dim, dropout, layerscale_init, node_update, bounded_scale)
        self.l_intra = IntraPropagation(dim, dropout, layerscale_init, node_update, bounded_scale)

    @staticmethod
    def _normalized_sigmoid(score, dst, dim_size):
        s = torch.sigmoid(score)
        denom = scatter(s, dst, dim=0, dim_size=dim_size, reduce="sum").clamp_min(1e-6)
        return s / denom[dst]

    def forward(self, hp, hl, hp0, hl0, q, q0, cross_edge_index, p_edge_index, p_edge_emb, l_edge_index, l_edge_emb):
        pidx, lidx = cross_edge_index
        pn, ln, qn = self.p_norm(hp), self.l_norm(hl), self.q_norm(q)
        pieces = [pn[pidx], ln[lidx], qn, q0]
        if self.use_endpoint_context:
            pctx = scatter(qn, pidx, dim=0, dim_size=hp.size(0), reduce="mean")
            lctx = scatter(qn, lidx, dim=0, dim_size=hl.size(0), reduce="mean")
            pieces += [pctx[pidx], lctx[lidx]]
        qu = torch.cat(pieces, dim=-1)
        qcand = self.q_candidate(qu)
        qgate = torch.sigmoid(self.q_gate(qu))
        q = q + scale_gain(self.q_scale, self.bounded_scale) * qgate * (qcand - q)
        qn = self.q_norm(q)
        pair = torch.cat([pn[pidx], ln[lidx], qn], dim=-1)
        a_pl = self._normalized_sigmoid(self.pl_score(pair), lidx, hl.size(0))
        a_lp = self._normalized_sigmoid(self.lp_score(pair), pidx, hp.size(0))
        m_pl = a_pl * self.pl_value(torch.cat([pn[pidx], qn], dim=-1))
        m_lp = a_lp * self.lp_value(torch.cat([ln[lidx], qn], dim=-1))
        agg_l = scatter(m_pl, lidx, dim=0, dim_size=hl.size(0), reduce="sum")
        agg_p = scatter(m_lp, pidx, dim=0, dim_size=hp.size(0), reduce="sum")
        pu = torch.cat([pn, agg_p, hp0], dim=-1)
        lu = torch.cat([ln, agg_l, hl0], dim=-1)
        hp = gated_update(hp, self.p_cross_update(pu), torch.sigmoid(self.p_cross_gate(pu)), scale_gain(self.p_cross_scale, self.bounded_scale), self.node_update)
        hl = gated_update(hl, self.l_cross_update(lu), torch.sigmoid(self.l_cross_gate(lu)), scale_gain(self.l_cross_scale, self.bounded_scale), self.node_update)
        hp = self.p_intra(hp, hp0, p_edge_index, p_edge_emb)
        hl = self.l_intra(hl, hl0, l_edge_index, l_edge_emb)
        return hp, hl, q
