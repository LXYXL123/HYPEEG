"""
CGeom configuration for downstream EEG classification.
"""

from typing import Dict, Optional, List

from pydantic import Field

from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class CGeomDataArgs(BaseDataArgs):
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class CGeomModelArgs(BaseModelArgs):
    pretrained_path: Optional[str] = None

    output_dims: int = 320
    hidden_dims: int = 64
    depth: int = 10
    mask_mode: str = "binomial"
    downstream_mask: Optional[str] = "all_true"


class CGeomTrainingArgs(BaseTrainingArgs):
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


class CGeomLoggingArgs(BaseLoggingArgs):
    experiment_name: str = "cgeom"
    run_dir: str = "assets/run"

    use_cloud: bool = True
    cloud_backend: str = "wandb"
    project: Optional[str] = "cgeom"
    entity: Optional[str] = None

    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: ["cgeom"])

    log_step_interval: int = 1
    ckpt_interval: int = 1


class CGeomConfig(AbstractConfig):
    model_type: str = "cgeom"
    fs: int = 256

    data: CGeomDataArgs = Field(default_factory=CGeomDataArgs)
    model: CGeomModelArgs = Field(default_factory=CGeomModelArgs)
    training: CGeomTrainingArgs = Field(default_factory=CGeomTrainingArgs)
    logging: CGeomLoggingArgs = Field(default_factory=CGeomLoggingArgs)

    def validate_config(self) -> bool:
        if self.model.output_dims <= 0:
            return False
        if self.model.hidden_dims <= 0:
            return False
        if self.model.depth <= 0:
            return False
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False
        if self.model.downstream_mask not in [None, "all_true", "all_false", "binomial", "continuous", "mask_last"]:
            return False
        return True
