"""Experiment configuration: one ExperimentConfig fully determines a run.

Presets at the bottom define the shipped experiment suites.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional

# ---------------------------------------------------------------------------
# Knob value spaces (mirror the spec)
# ---------------------------------------------------------------------------
EntityFeature = Literal["bow", "tabular_textemb"]
ModelKind = Literal["mlp", "sage", "parallel_ref"]
GraphMode = Literal["hetero"]

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """User-disjoint global-timeline split (strict-PIT, fully inductive)."""
    val_pct: float = 0.70
    test_pct: float = 0.90
    min_date: str = "2022-01-01"
    max_date: str = "2025-06-01"


@dataclass(frozen=True)
class FeatureConfig:
    entity: EntityFeature = "bow"
    # attribute edges; the cold-start lever
    use_attribute_edges: bool = True
    # message-pass the interaction edges (considers/accepts/screens); off = attribute-only GNN
    use_interaction_edges: bool = True
    # which attribute relations to include (skill/title/company)
    use_skill_edges: bool = True
    use_title_edges: bool = True
    use_company_edges: bool = True
    # job text-embedding source (Tech encoder axis): "job_text_emb" or a frozen-encoder tag
    # qwen3/qwen3_4b/qwen3_8b/me5 loading data/tech_prepared/<tag>_{seeker,job}_emb.pt
    text_emb: str = "job_text_emb"
    # tabular numeric profile features (PIT as-of split cutoff) for tabular_textemb
    use_tabular: bool = True
    # explicit person-job match features appended to the ESMM head input (leak-free; esmm only)
    use_match_feats: bool = False
    # leak-safe PIT in/out degree as node features (hetero SAGE/GAT only)
    pit_degree_nodes: bool = False
    # leak-safe PIT attribute per-type out-degree as focal-node features (log1p per enabled
    # skill/title/company relation); hetero SAGE/GAT only, requires use_attribute_edges
    pit_attr_degree_nodes: bool = False
    # leak-safe PIT company out-degree (log1p #jobs posted) as a company-node feature;
    # requires use_company_edges=True; hetero SAGE/GAT only
    use_company_degree_feats: bool = False
    # externally-supplied per-pair feature width added to the ESMM decoder pair_dim (Tianchi)
    ext_pair_dim: int = 0
    # append frozen Qwen3-Embedding node-content vectors to seeker/job features (Tianchi; leak-safe)
    use_qwen_content: bool = False
    # which frozen encoder's node-content files to load (Tianchi): qwen3/qwen3_4b/qwen3_8b/me5;
    # selects file prefix <emb_model>_{seeker,job}_emb.pt. Tech selects via text_emb instead.
    emb_model: str = "qwen3"
    # drop content (text-emb) node features: seeker/job base -> a single constant
    content_off: bool = False
    # Tianchi-only: append a leak-safe interaction+attribute degree block to seeker/job features
    # so Tianchi arms mirror Tech's content ⊕ degree structure
    tc_degree_nodes: bool = False
    # Tianchi controlled p3: collapse seeker/job base to a ones(1) column before the degree append
    tc_ones_collapse: bool = False


@dataclass(frozen=True)
class ModelConfig:
    kind: ModelKind = "sage"
    graph_mode: GraphMode = "hetero"
    # per-relation graph conv: "sage" (SAGEConv mean, SpMM fast-path) or "gatv2" (attention)
    conv_type: Literal["sage", "gatv2"] = "sage"
    # gatv2 only: add a residual self connection (GATv2Conv has no root weight); restores the
    # self term SAGEConv carries, for a clean "attention vs SAGE-mean" contrast
    gat_residual: bool = False
    hidden_dim: int = 128
    out_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    decoder: Literal["esmm"] = "esmm"
    # gatv2 attention heads (must divide hidden_dim)
    attn_heads: int = 4
    # LiRank co-trained per-head monotone calibration on each head logit (esmm only; identity-init)
    head_calibrator: Literal["none", "lirank"] = "none"
    lirank_n_knots: int = 16
    lirank_logit_range: float = 8.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12                  # MAX epochs; early stopping usually ends sooner
    lr: float = 5e-3
    weight_decay: float = 1e-5
    # chronological chunks for the leak-safe PIT mechanism (larger = closer to exact PIT)
    n_time_chunks_train: int = 24
    n_time_chunks_eval: int = 12
    device: str = "auto"              # auto -> mps if available else cpu
    seed: int = 0
    # cap supervision size for tractable representative runs (None = use all)
    max_train_pos: Optional[int] = None
    max_eval_pos: Optional[int] = None
    max_eval_exposed: Optional[int] = None    # cap exposed-AUC rows (esp. for exact PIT)
    # --- early stopping / checkpoint selection ---
    patience: int = 5                 # stop after this many non-improving val epochs
    min_delta: float = 1e-4
    # val selection metric (early stop + best ckpt): "val_loss" (lower better),
    # "auc_exposed_all" or "auc_sampled_all" (higher better)
    select_metric: str = "val_loss"
    val_select_max_pos: int = 4000    # cap val-selection set for cheap per-epoch eval
    # training objective: funnel-aware ESMM (real negs; head1=pAccept, head2=p(pass|accept),
    # loss = BCE(head1, accept) + BCE(head1*head2, pass)); needs model.decoder="esmm"
    objective: str = "esmm"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # multi-seed averaging: each config is run once per seed, reported mean±std
    seeds: tuple = (0, 1, 2)
    save_checkpoint: bool = True
    # report TEST metrics with exact per-row PIT (per anchor timestamp)
    exact_test_pit: bool = True

    # cohort definition: cf_n_prior = prior exposure count of the job
    cohort_prior_event: Literal["exposure"] = "exposure"
    ranking_ks: tuple = (5, 10, 20)
    # retrieval-metric negatives drawn only from jobs co-available with the user (PIT,
    # no future); eval-only, exposed-AUC unaffected. False = old lifespan-blind pool
    eval_coavail_pool: bool = True

    def __post_init__(self):
        # ESMM is a coupled decoder+objective; fail fast on a half-configured pair.
        _esmm_arch_obj = ("esmm",)
        if self.train.objective in _esmm_arch_obj and self.model.decoder != "esmm":
            raise ValueError(f"objective='{self.train.objective}' requires model.decoder='esmm'")
        if self.model.decoder == "esmm" and self.train.objective not in _esmm_arch_obj:
            raise ValueError("model.decoder='esmm' requires train.objective='esmm'")
        # calibration knobs both require the ESMM funnel heads
        if self.model.head_calibrator != "none" and self.model.decoder != "esmm":
            raise ValueError("model.head_calibrator='lirank' requires model.decoder='esmm'")
        if (self.model.kind in ("sage", "parallel_ref")
                and not self.feature.use_interaction_edges
                and not self.feature.use_attribute_edges):
            raise ValueError("a GNN needs at least one of use_interaction_edges / use_attribute_edges")
        if self.feature.use_match_feats:
            _mok = self.model.kind == "mlp" or (
                self.model.kind in ("sage", "parallel_ref")
                and self.model.graph_mode == "hetero")
            if not (self.model.decoder == "esmm" and _mok):
                raise ValueError("match pair features require decoder='esmm' + "
                                 "(kind='mlp', hetero sage, or parallel_ref)")
        if self.feature.pit_degree_nodes and not (
                self.model.graph_mode == "hetero"
                and self.model.kind in ("sage", "parallel_ref")):
            raise ValueError("pit_degree_nodes requires a hetero SAGE GNN")
        if self.feature.pit_attr_degree_nodes:
            if not (self.model.graph_mode == "hetero"
                    and self.model.kind in ("sage", "parallel_ref")):
                raise ValueError("pit_attr_degree_nodes requires a hetero SAGE GNN")
            if not self.feature.use_attribute_edges:
                raise ValueError("pit_attr_degree_nodes requires use_attribute_edges=True")
        if self.feature.use_company_degree_feats:
            if not (self.model.graph_mode == "hetero"
                    and self.model.kind in ("sage", "parallel_ref")):
                raise ValueError("use_company_degree_feats requires a hetero SAGE GNN")
            if not self.feature.use_company_edges:
                raise ValueError("use_company_degree_feats requires use_company_edges=True "
                                 "(posted_by must be in the PIT snapshot to count company out-degree)")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def _cfg(name: str, **over) -> ExperimentConfig:
    """Build a config from nested-dotted overrides, e.g. model__kind='mlp'."""
    sub = {"feature": {}, "model": {}, "split": {}, "train": {}}
    top = {}
    for k, v in over.items():
        if "__" in k:
            grp, key = k.split("__", 1)
            sub[grp][key] = v
        else:
            top[k] = v
    return ExperimentConfig(
        name=name,
        feature=FeatureConfig(**sub["feature"]),
        model=ModelConfig(**sub["model"]),
        split=SplitConfig(**sub["split"]),
        train=TrainConfig(**sub["train"]),
        **top,
    )


def default_template(name: str = "default_template", **overrides) -> ExperimentConfig:
    """The default experiment template (spec-driven). Pass dotted overrides to tweak."""
    base = dict(
        # --- Model: static MLP, ESMM funnel decoder, LiRank calibration ---
        model__kind="mlp",
        model__decoder="esmm",
        model__head_calibrator="lirank",
        # --- Features (spec): text embeddings ⊕ tabular node feats + match ---
        feature__entity="tabular_textemb",
        feature__use_match_feats=True,
        # --- Training: real-negative ESMM, 24 chunks ---
        train__objective="esmm",
        train__n_time_chunks_train=24,
        # --- Early stopping: exposed val AUC on real pos/neg, cap 50, patience 8 (spec) ---
        train__select_metric="auc_exposed_all",
        train__epochs=50,
        train__patience=8,
        train__val_select_max_pos=0,
        # --- Eval: 1-day exact PIT test; user-grouped ranking @ {5,10,20} ---
        exact_test_pit=True,
        ranking_ks=(5, 10, 20),
    )
    base.update(overrides)            # caller overrides win over the template defaults
    return _cfg(name, **base)


def default_template_hetero(name: str = "default_template_hetero", **overrides) -> ExperimentConfig:
    """Hetero-SAGE (GNN) variant of default_template: content + attribute-edges + PIT-degree node feats."""
    base = dict(
        # --- Model: hetero-SAGE encoder, ESMM funnel decoder, LiRank calibration ---
        model__kind="sage",
        model__graph_mode="hetero",
        model__decoder="esmm",
        model__head_calibrator="lirank",
        # --- Features: content ⊕ tabular + attribute edges + PIT degree/attr/company node feats ---
        feature__entity="tabular_textemb",
        feature__use_match_feats=True,
        feature__content_off=False,           # CONTENT text-emb node features on
        feature__use_attribute_edges=True,    # ATTRIBUTE edges on
        feature__use_company_edges=True,      # company posted_by edges on (for company degree)
        feature__pit_degree_nodes=True,       # PIT interaction in/out degree node features on
        feature__pit_attr_degree_nodes=True,  # PIT attribute per-relation out-degree (focal nodes)
        feature__use_company_degree_feats=True,  # PIT company out-degree node feature
        # --- Training: real-negative ESMM, 24 chunks ---
        train__objective="esmm",
        train__n_time_chunks_train=24,
        # --- Early stopping: exposed val AUC on real pos/neg, cap 50, patience 8 (spec) ---
        train__select_metric="auc_exposed_all",
        train__epochs=50,
        train__patience=8,
        train__val_select_max_pos=0,
        # --- Eval: 1-day exact PIT test; user-grouped ranking @ {5,10,20} ---
        exact_test_pit=True,
        ranking_ks=(5, 10, 20),
    )
    base.update(overrides)            # caller overrides win over the template defaults
    return _cfg(name, **base)


def paired_encoder_test() -> List[ExperimentConfig]:
    """Paired Tech↔Tianchi encoder study — controlled content-placement family (p1/p2/p3)."""
    QWEN = dict(feature__entity="tabular_textemb", feature__text_emb="qwen3",
                feature__content_off=False)
    return [
        default_template("p1_mlp", feature__text_emb="qwen3"),                          # A1
        default_template_hetero("p2_sage_con", **QWEN),                                 # A2
        default_template_hetero("p3_sage_coff", feature__text_emb="qwen3",              # A3 (Qwen match)
                                feature__content_off=True),
    ]


def controlled_p8_test() -> List[ExperimentConfig]:
    """Controlled p8 = parallel combination of p1 (content MLP) and p3 (content-off graph GNN)."""
    return [
        default_template_hetero("p8_parallel", model__kind="parallel_ref",
                                feature__entity="tabular_textemb", feature__text_emb="qwen3",
                                feature__content_off=False),
    ]


def model_grid_test() -> List[ExperimentConfig]:
    """Model-axis grid over the controlled p2/p3/p8 family — the hetero-GATv2 corner."""
    QWEN = dict(feature__entity="tabular_textemb", feature__text_emb="qwen3", feature__content_off=False)
    GAT1H = dict(model__conv_type="gatv2", model__attn_heads=1, model__gat_residual=True)
    variants = [
        ("hetgat",   "hetero", dict(**GAT1H)),
    ]
    out: List[ExperimentConfig] = []
    for sfx, gmode, mov in variants:
        base = dict(model__graph_mode=gmode, seeds=(0, 1, 2, 3, 4), **mov)
        out.append(default_template_hetero(f"p2_{sfx}", **QWEN, **base))                     # coupled
        out.append(default_template_hetero(f"p3_{sfx}", feature__text_emb="qwen3",           # content-off
                                           feature__content_off=True, **base))
        out.append(default_template_hetero(f"p8_{sfx}", model__kind="parallel_ref",          # p1 ∥ p3
                                           **QWEN, **base))
    return out
