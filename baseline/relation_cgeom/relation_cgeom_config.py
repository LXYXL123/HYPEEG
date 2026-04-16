"""
Relation-CGeom configuration for downstream EEG classification.
"""

from typing import Dict, List, Optional, Tuple

from pydantic import Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseLoggingArgs, BaseModelArgs, BaseTrainingArgs


class RelationCGeomDataArgs(BaseDataArgs):
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class RelationCGeomModelArgs(BaseModelArgs):
    pretrained_path: Optional[str] = None

    output_dims: int = 320
    hidden_dims: int = 64
    depth: int = 10
    bands: Tuple[Tuple[float, float], ...] = ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))

    use_raw_branch: bool = True
    use_complex_branch: bool = True
    use_hyperbolic_branch: bool = True
    share_band_encoder: bool = False
    band_fusion_type: str = "concat_linear"

    hyperbolic_depth: int = 1
    hyperbolic_curvature: float = 1.0
    learnable_curvature: bool = True

    use_raw_residual: bool = True
    use_complex_residual: bool = False
    use_hyper_relation: bool = True
    use_gated_raw_residual: bool = True
    use_gated_complex_residual: bool = False
    local_fusion_type: str = "add"
    relation_condition_type: str = "residual"
    relation_gate_scale: float = 0.1
    relation_film_scale: float = 0.1

    # Setup-Conditioned Hierarchical Relation-CGeom. Disabled by default so
    # existing relation_cgeom configs keep the original forward path.
    use_setup_conditioned: bool = False
    setup_meta_dim: int = 128
    setup_ctx_dim: int = 128
    setup_dim: int = 128
    setup_channel_vocab_size: int = 128
    setup_condition_scale: float = 0.1
    use_variable_channel_frontend: bool = False
    per_channel_stem_depth: int = 2
    channel_attn_heads: int = 4

    mask_mode: str = "binomial"
    downstream_mask: Optional[str] = "all_true"


class RelationCGeomTrainingArgs(BaseTrainingArgs):
    max_epochs: int = 50

    weight_decay: float = 0.01
    max_grad_norm: float = 3.0

    lr_schedule: str = "onecycle"
    max_lr: float = 5e-4
    encoder_lr_scale: float = 0.1
    warmup_epochs: int = 5
    warmup_scale: float = 1e-2
    pct_start: float = 0.2
    min_lr: float = 1e-6

    use_amp: bool = False
    freeze_encoder: bool = False

    label_smoothing: float = 0.1


class RelationCGeomLoggingArgs(BaseLoggingArgs):
    experiment_name: str = "relation_cgeom"
    run_dir: str = "assets/run"

    use_cloud: bool = True
    cloud_backend: str = "wandb"
    project: Optional[str] = "relation_cgeom"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: ["relation_cgeom"])

    log_step_interval: int = 1
    ckpt_interval: int = 1


class RelationCGeomConfig(AbstractConfig):
    model_type: str = "relation_cgeom"
    fs: int = 256

    data: RelationCGeomDataArgs = Field(default_factory=RelationCGeomDataArgs)
    model: RelationCGeomModelArgs = Field(default_factory=RelationCGeomModelArgs)
    training: RelationCGeomTrainingArgs = Field(default_factory=RelationCGeomTrainingArgs)
    logging: RelationCGeomLoggingArgs = Field(default_factory=RelationCGeomLoggingArgs)

    def validate_config(self) -> bool:
        if self.model.output_dims <= 0:
            return False
        if self.model.hidden_dims <= 0:
            return False
        if self.model.depth <= 0:
            return False
        if not self.model.use_complex_branch:
            return False
        if self.model.hyperbolic_depth <= 0:
            return False
        if self.model.hyperbolic_curvature <= 0:
            return False
        if self.model.relation_gate_scale < 0:
            return False
        if self.model.relation_film_scale < 0:
            return False
        if self.model.use_setup_conditioned and not self.model.use_raw_branch:
            return False
        if self.model.use_setup_conditioned and not self.model.use_hyper_relation:
            return False
        if self.model.setup_meta_dim <= 0:
            return False
        if self.model.setup_ctx_dim <= 0:
            return False
        if self.model.setup_dim <= 0:
            return False
        if self.model.setup_channel_vocab_size <= 0:
            return False
        if self.model.setup_condition_scale < 0:
            return False
        if self.model.use_variable_channel_frontend and not self.model.use_setup_conditioned:
            return False
        if self.model.per_channel_stem_depth < 0:
            return False
        if self.model.channel_attn_heads <= 0:
            return False
        if self.model.band_fusion_type not in ["concat_linear", "gated_sum"]:
            return False
        if self.model.local_fusion_type not in ["add", "concat_linear"]:
            return False
        if self.model.relation_condition_type not in ["residual", "gate", "film"]:
            return False
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False
        if self.model.downstream_mask not in [None, "all_true", "all_false", "binomial", "continuous", "mask_last"]:
            return False
        return True
