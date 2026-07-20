"""Encoders (MLP baseline, HeteroSAGE) + ESMM two-head link decoder.
Inductive: node reps come from features + message passing (no per-node lookup),
so unseen val/test nodes score from features + their PIT neighbourhood."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace as dc_replace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv

from .config import ModelConfig
from .graph_build import NODE_TYPES, INTERACTION_RELS, OUTCOME_NEG_RELS, enabled_attribute_rels
from .match import MATCH_TAB_DIM

# explicit person-job match features: 4 text-emb + MATCH_TAB_DIM tabular
MATCH_TEXT_DIM = 4
MATCH_DIM = MATCH_TEXT_DIM + MATCH_TAB_DIM


# interaction relations = untied per-direction weights (vs attribute rels)
_INTERACTION_BASE = ({r for _, r, _ in INTERACTION_RELS}
                     | {r for _, r, _ in OUTCOME_NEG_RELS}
                     | {"satisfied"})   # +Tianchi hybrid satisfied edge


def _is_interaction_rel(rel: Tuple[str, str, str]) -> bool:
    r = rel[1]
    return (r[4:] if r.startswith("rev_") else r) in _INTERACTION_BASE


def _dir_conv_dict(edge_types, mk) -> Dict:
    """Per-relation conv modules for HeteroConv (one conv per relation)."""
    convs: Dict = {}
    for rel in edge_types:
        convs[rel] = mk()
    return convs


@dataclass
class GraphMeta:
    num_nodes: Dict[str, int]
    edge_types: List[Tuple[str, str, str]]   # hetero relations incl. reverses (per config)
    node_offset: Dict[str, int]
    n_total: int


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------

class InputEncoder(nn.Module):
    """Per-node-type Linear projecting heterogeneous input dims to ``hidden``."""
    def __init__(self, in_dims: Dict[str, int], hidden: int):
        super().__init__()
        self.lin = nn.ModuleDict({t: nn.Linear(in_dims[t], hidden) for t in in_dims})

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {t: F.relu(self.lin[t](x[t])) for t in x}


class MonotonicCalibrator(nn.Module):
    """Monotone piecewise-linear map on a scalar logit (LiRank isotonic layer),
    identity-initialised."""
    def __init__(self, n_knots: int = 16, logit_range: float = 8.0):
        super().__init__()
        self.K = n_knots
        knots = torch.linspace(-logit_range, logit_range, n_knots)
        self.register_buffer("knots", knots)                  # [K] fixed
        self.seg = float(knots[1] - knots[0])
        d0 = math.log(math.expm1(self.seg))                   # softplus(d0)=seg -> slope 1
        self.deltas = nn.Parameter(torch.full((n_knots - 1,), d0))   # [K-1]
        self.base = nn.Parameter(torch.tensor(float(knots[0])))      # value at first knot

    def _knot_values(self) -> torch.Tensor:
        inc = F.softplus(self.deltas)                         # [K-1] >= 0
        return self.base + torch.cat([inc.new_zeros(1), torch.cumsum(inc, 0)])  # [K] monotone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self._knot_values()
        k = self.knots
        idx = torch.clamp(((x - k[0]) / self.seg).floor().long(), 0, self.K - 2)
        x0 = k[idx]; v0 = v[idx]
        slope = (v[idx + 1] - v[idx]) / self.seg              # boundary slope handles tails
        return v0 + slope * (x - x0)


_COMBINE_MULT = {"concat": 2}


def _combine(zm, zj, mode: str):
    """Combine seeker/job embeddings for the decoder head (concat -> [zm || zj])."""
    if mode == "concat":
        return torch.cat([zm, zj], dim=-1)
    raise ValueError(f"unknown head_combine {mode!r} (expected concat)")


class LinkDecoder(nn.Module):
    def __init__(self, dim: int, kind: str,
                 head_calibrator: str = "none", n_knots: int = 16, logit_range: float = 8.0,
                 pair_dim: int = 0, head_combine: str = "concat"):
        super().__init__()
        self.kind = kind
        self.pair_dim = pair_dim    # per-pair match features appended to esmm heads
        self.head_combine = head_combine
        cw = _COMBINE_MULT[head_combine]        # concat=2
        if kind == "esmm":
            # ESMM two heads: head_accept = pAccept logit, head_cond = p(pass|accept) logit
            def _head():
                return nn.Sequential(
                    nn.Linear(cw * dim + pair_dim, dim), nn.ReLU(),
                    nn.Dropout(0.2), nn.Linear(dim, 1)
                )
            self.head_accept = _head()
            self.head_cond = _head()
            if head_calibrator == "lirank":     # co-trained monotone per-head layer
                self.cal_accept = MonotonicCalibrator(n_knots, logit_range)
                self.cal_cond = MonotonicCalibrator(n_knots, logit_range)
            else:
                self.cal_accept = self.cal_cond = nn.Identity()

    def forward(self, zm: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        return self.net(_combine(zm, zj, self.head_combine)).squeeze(-1)

    def forward_heads(self, zm: torch.Tensor, zj: torch.Tensor,
                      pair_feats: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ESMM two-head decode -> (accept_logit, pass|accept_logit); pair_feats appended to both heads."""
        assert self.kind == "esmm"
        comb = _combine(zm, zj, self.head_combine)
        base = [comb] if pair_feats is None else [comb, pair_feats]
        feat_a = feat_c = torch.cat(base, dim=-1)
        a = self.cal_accept(self.head_accept(feat_a).squeeze(-1))
        c = self.cal_cond(self.head_cond(feat_c).squeeze(-1))
        return a, c


# ---------------------------------------------------------------------------
# MLP (no message passing)
# ---------------------------------------------------------------------------

class MLPModel(nn.Module):
    def __init__(self, in_dims, mc: ModelConfig, pair_dim: int = 0):
        super().__init__()
        self.enc = InputEncoder(in_dims, mc.hidden_dim)
        self.mlp = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(mc.hidden_dim, mc.hidden_dim), nn.ReLU(),
                nn.Dropout(mc.dropout), nn.Linear(mc.hidden_dim, mc.out_dim),
            ) for t in in_dims
        })
        self.decoder = LinkDecoder(mc.out_dim, mc.decoder, head_calibrator=mc.head_calibrator,
                                   n_knots=mc.lirank_n_knots, logit_range=mc.lirank_logit_range,
                                   pair_dim=pair_dim,
                                   head_combine=getattr(mc, "head_combine", "concat"))

    def encode(self, x, edge_index_dict=None):
        h = self.enc(x)
        return {t: self.mlp[t](h[t]) for t in h}

    def forward(self, x, edge_index_dict, m_idx, j_idx):
        z = self.encode(x)
        return self.decoder(z["seeker"][m_idx], z["job"][j_idx])

    def decode_pairs(self, z, m_idx, j_idx):
        return self.decoder(z["seeker"][m_idx], z["job"][j_idx])

    def decode_pairs_heads(self, z, m_idx, j_idx, pair_feats=None):
        return self.decoder.forward_heads(z["seeker"][m_idx], z["job"][j_idx], pair_feats)


# ---------------------------------------------------------------------------
# Heterogeneous GraphSAGE
# ---------------------------------------------------------------------------

class HeteroSAGE(nn.Module):
    def __init__(self, in_dims, mc: ModelConfig, meta: GraphMeta, pair_dim: int = 0):
        super().__init__()
        self.mc = mc
        self.enc = InputEncoder(in_dims, mc.hidden_dim)
        ct = getattr(mc, "conv_type", "sage")

        def _mk_conv():
            d = mc.hidden_dim
            if ct == "gatv2":                       # dynamic attention, out=d
                return GATv2Conv(d, d, heads=mc.attn_heads, concat=False,
                                 add_self_loops=False,   # bipartite edge types
                                 residual=getattr(mc, "gat_residual", False))
            return SAGEConv(d, d)                    # default: mean aggregation

        self.convs = nn.ModuleList()
        for _ in range(mc.num_layers):
            conv = HeteroConv(_dir_conv_dict(meta.edge_types, _mk_conv), aggr="sum")
            self.convs.append(conv)
        self.out = nn.ModuleDict({t: nn.Linear(mc.hidden_dim, mc.out_dim) for t in in_dims})
        self.decoder = LinkDecoder(mc.out_dim, mc.decoder, head_calibrator=mc.head_calibrator,
                                   n_knots=mc.lirank_n_knots, logit_range=mc.lirank_logit_range,
                                   pair_dim=pair_dim,
                                   head_combine=getattr(mc, "head_combine", "concat"))
        self.dropout = mc.dropout

    def encode(self, x, edge_index_dict):
        h = self.enc(x)
        for conv in self.convs:
            hn = conv(h, edge_index_dict)
            hn = {t: F.relu(v) for t, v in hn.items()}
            # node types not updated this layer keep previous h
            hn = {t: hn.get(t, h[t]) for t in h}
            hn = {t: F.dropout(v, p=self.dropout, training=self.training) for t, v in hn.items()}
            h = hn
        return {t: self.out[t](h[t]) for t in h}

    def forward(self, x, edge_index_dict, m_idx, j_idx):
        z = self.encode(x, edge_index_dict)
        return self.decoder(z["seeker"][m_idx], z["job"][j_idx])

    def decode_pairs(self, z, m_idx, j_idx):
        return self.decoder(z["seeker"][m_idx], z["job"][j_idx])

    def decode_pairs_heads(self, z, m_idx, j_idx, pair_feats=None):
        return self.decoder.forward_heads(z["seeker"][m_idx], z["job"][j_idx], pair_feats)


# ---------------------------------------------------------------------------
# Controlled parallel-reference PJF — p8 as the EXACT combination of p1 and p3
# ---------------------------------------------------------------------------

class ParallelRefPJF(nn.Module):
    """Decoupled p8: content channel (== p1 MLP) parallel to graph channel (== p3
    content-off HeteroSAGE), fused by concat at a shared decoder. Splits the input
    features internally so each channel gets exactly its reference's input."""

    def __init__(self, in_dims, mc: ModelConfig, meta: GraphMeta, pair_dim: int = 0, content_dims=None):
        super().__init__()
        assert content_dims is not None, "ParallelRefPJF needs the pre-inflation content_dims"
        self.content_dims = dict(content_dims)                  # {seeker, job} content base width
        d = mc.out_dim
        self._d = d                                             # per-channel out width
        sub = dc_replace(mc, decoder="dot")                     # sub-encoders carry no decoder params
        # channel a == p1: content MLP
        self.content = MLPModel({t: content_dims[t] for t in ("seeker", "job")}, sub)
        # channel g == p3: content-off graph GNN (seeker/job base -> 1, keep degree block)
        g_in = {"seeker": 1 + (in_dims["seeker"] - content_dims["seeker"]),
                "job":    1 + (in_dims["job"]    - content_dims["job"]),
                **{t: in_dims[t] for t in in_dims if t not in ("seeker", "job")}}
        self.graph = HeteroSAGE(g_in, sub, meta)
        self.decoder = LinkDecoder(2 * d, mc.decoder, head_calibrator=mc.head_calibrator,
                                   n_knots=mc.lirank_n_knots, logit_range=mc.lirank_logit_range,
                                   pair_dim=pair_dim,
                                   head_combine=getattr(mc, "head_combine", "concat"))

    def _split(self, x):
        cd = self.content_dims
        xc = {t: x[t][:, :cd[t]] for t in ("seeker", "job")}
        def _g(t):
            one = torch.ones(x[t].size(0), 1, device=x[t].device, dtype=x[t].dtype)
            return torch.cat([one, x[t][:, cd[t]:]], dim=-1)
        xg = {"seeker": _g("seeker"), "job": _g("job"),
              **{t: x[t] for t in x if t not in ("seeker", "job")}}
        return xc, xg

    def encode(self, x, edge_index_dict):
        xc, xg = self._split(x)
        za = self.content.encode(xc)                            # content channel (p1)
        zg = self.graph.encode(xg, edge_index_dict)             # graph channel (p3)
        return {t: torch.cat([za[t], zg[t]], dim=-1) for t in ("seeker", "job")}

    def forward(self, x, edge_index_dict, m_idx, j_idx):
        z = self.encode(x, edge_index_dict)
        return self.decoder(z["seeker"][m_idx], z["job"][j_idx])

    def decode_pairs(self, z, m_idx, j_idx):
        return self.decoder(z["seeker"][m_idx], z["job"][j_idx])

    def decode_pairs_heads(self, z, m_idx, j_idx, pair_feats=None):
        return self.decoder.forward_heads(z["seeker"][m_idx], z["job"][j_idx], pair_feats)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def build_model(cfg, in_dims: Dict[str, int], meta: GraphMeta, content_dims=None) -> nn.Module:
    mc = cfg.model
    # pre-degree content dims for the parallel_ref channel split
    base_in_dims = dict(in_dims)
    # PIT degree NODE features: per-relation [in,out], appended to every node type
    if getattr(cfg.feature, "pit_degree_nodes", False):
        _dd = 2 * len(INTERACTION_RELS)
        in_dims = {t: d + _dd for t, d in in_dims.items()}
    # PIT attribute degree: per-type out-degree column on focal nodes (seeker/job)
    if getattr(cfg.feature, "pit_attr_degree_nodes", False):
        _asrc: Dict[str, int] = {}
        for s, _, _ in enabled_attribute_rels(cfg.feature):
            _asrc[s] = _asrc.get(s, 0) + 1
        in_dims = {t: d + _asrc.get(t, 0) for t, d in in_dims.items()}
    # PIT company out-degree: one column on the company node (applied last)
    if getattr(cfg.feature, "use_company_degree_feats", False):
        in_dims = {t: (d + 1 if t == "company" else d) for t, d in in_dims.items()}
    # per-pair decoder features on esmm heads (order matches Trainer._pair_feats)
    pair_dim = 0
    if mc.decoder == "esmm":
        if getattr(cfg.feature, "use_match_feats", False):
            pair_dim += MATCH_DIM
        pair_dim += getattr(cfg.feature, "ext_pair_dim", 0)   # externally-supplied (Tianchi match); 0 for Tech
    if mc.kind == "parallel_ref":
        # content_dims = pre-degree base (passed explicitly, else base_in_dims)
        cd = content_dims if content_dims is not None else base_in_dims
        return ParallelRefPJF(in_dims, mc, meta, pair_dim=pair_dim, content_dims=cd)
    if mc.kind == "mlp":
        return MLPModel(in_dims, mc, pair_dim=pair_dim)
    return HeteroSAGE(in_dims, mc, meta, pair_dim=pair_dim)
