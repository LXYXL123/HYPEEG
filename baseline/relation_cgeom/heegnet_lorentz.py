from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


def _load_heegnet_lorentz_backend():
    repo_root = Path(__file__).resolve().parents[2]
    heegnet_root = repo_root / "HEEGNet-main"
    if not heegnet_root.exists():
        raise FileNotFoundError(f"HEEGNet-main not found at: {heegnet_root}")
    if str(heegnet_root) not in sys.path:
        sys.path.insert(0, str(heegnet_root))

    from lib.lorentz.manifold import CustomLorentz  # type: ignore
    from lib.lorentz.layers.LMLR import LorentzMLR  # type: ignore

    return CustomLorentz, LorentzMLR


class EuclideanToLorentz(nn.Module):
    """Project Euclidean global vectors into HEEGNet's Lorentz model.

    x_euc: [B, D] or [B, N, D]
    x_h: [B, H] or [B, N, H]
    """

    def __init__(self, input_dim: int, lorentz_dim: int, manifold):
        super().__init__()
        if lorentz_dim < 3:
            raise ValueError(f"lorentz_dim must be >= 3, got {lorentz_dim}")
        self.manifold = manifold
        self.pre_norm = nn.LayerNorm(input_dim)
        self.space_proj = nn.Linear(input_dim, lorentz_dim - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.space_proj.weight.dtype)
        space = self.space_proj(self.pre_norm(x))
        space = F.normalize(space, dim=-1, eps=1e-6)
        return self.manifold.add_time(space)


class LorentzToEuclidean(nn.Module):
    """Project Lorentz points back to Euclidean features via the tangent space.

    x_h: [B, H] or [B, N, H]
    x_euc: [B, D] or [B, N, D]
    """

    def __init__(self, lorentz_dim: int, output_dim: int, manifold):
        super().__init__()
        if lorentz_dim < 3:
            raise ValueError(f"lorentz_dim must be >= 3, got {lorentz_dim}")
        self.manifold = manifold
        self.space_norm = nn.LayerNorm(lorentz_dim - 1)
        self.out_proj = nn.Linear(lorentz_dim - 1, output_dim)

    def forward(self, x_h: torch.Tensor, *, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        tangent = self.manifold.logmap0(x_h)
        space = tangent.narrow(-1, 1, tangent.size(-1) - 1)
        x_euc = self.out_proj(self.space_norm(space))
        if output_dtype is not None:
            x_euc = x_euc.to(dtype=output_dtype)
        return x_euc


class LorentzCentroidRelation(nn.Module):
    """Minimal HEEGNet-style Lorentz relation fusion for global tokens.

    raw_global/spec_global/setup_token: [B, D]
    relation_h: [B, H]
    relation_global: [B, D]
    """

    def __init__(
        self,
        repr_dims: int,
        lorentz_dim: int,
        curvature: float = 1.0,
        learnable_curvature: bool = True,
    ):
        super().__init__()
        CustomLorentz, _ = _load_heegnet_lorentz_backend()
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_curvature)
        self.raw_proj = EuclideanToLorentz(repr_dims, lorentz_dim, self.manifold)
        self.spec_proj = EuclideanToLorentz(repr_dims, lorentz_dim, self.manifold)
        self.setup_proj = EuclideanToLorentz(repr_dims, lorentz_dim, self.manifold)
        self.out_proj = LorentzToEuclidean(lorentz_dim, repr_dims, self.manifold)

    def forward(
        self,
        raw_global: torch.Tensor,
        spec_global: torch.Tensor,
        setup_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_dtype = raw_global.dtype
        raw_h = self.raw_proj(raw_global)
        spec_h = self.spec_proj(spec_global)
        setup_h = self.setup_proj(setup_token)
        relation_h = self.manifold.centroid(torch.stack([raw_h, spec_h, setup_h], dim=1))
        relation_global = self.out_proj(relation_h, output_dtype=target_dtype)
        return relation_global, {
            "raw_h": raw_h.detach(),
            "spec_h": spec_h.detach(),
            "setup_h": setup_h.detach(),
            "relation_h": relation_h.detach(),
        }


class MultiHeadLorentzClassifier(nn.Module):
    """Dataset-specific Lorentz MLR heads over pooled main_global vectors.

    main_global: [B, D]
    main_global_h: [B, H]
    logits: [B, n_class]
    """

    def __init__(
        self,
        input_dim: int,
        lorentz_dim: int,
        head_configs: Dict[str, int],
        curvature: float = 1.0,
        learnable_curvature: bool = True,
        t_sne: bool = False,
    ):
        super().__init__()
        CustomLorentz, LorentzMLR = _load_heegnet_lorentz_backend()
        self.manifold = CustomLorentz(k=curvature, learnable=learnable_curvature)
        self.input_proj = EuclideanToLorentz(input_dim, lorentz_dim, self.manifold)
        self.heads = nn.ModuleDict({
            ds_name: LorentzMLR(self.manifold, lorentz_dim, n_class)
            for ds_name, n_class in head_configs.items()
        })
        self.t_sne = t_sne
        self.cls_feature = None

    def forward(self, x: torch.Tensor, montage: str) -> torch.Tensor:
        head_name = montage.split("/")[0]
        if head_name not in self.heads:
            raise ValueError(f"Head '{head_name}' not found. Available heads: {list(self.heads.keys())}")
        if self.t_sne:
            self.cls_feature = x.detach().clone()
        x_h = self.input_proj(x)
        return self.heads[head_name](x_h)
