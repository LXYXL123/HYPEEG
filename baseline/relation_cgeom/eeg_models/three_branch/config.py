from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ThreeBranchConfig:
    input_channels: int
    num_classes: int
    sampling_rate: float = 256.0
    repr_dims: int = 320
    hidden_dims: int = 64
    depth: int = 10
    use_raw_branch: bool = True
    use_complex_branch: bool = True
    use_hyperbolic_branch: bool = True
    bands: Tuple[Tuple[float, float], ...] = ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
    share_band_encoder: bool = False
    band_fusion_type: str = 'concat_linear'
    tf_fusion_type: str = 'concat_linear'
    global_fusion_type: str = 'concat_linear'
    head_fusion_type: str = 'concat_linear'
    hyperbolic_depth: int = 1
    hyperbolic_curvature: float = 1.0
    learnable_curvature: bool = True
    dropout: float = 0.5
